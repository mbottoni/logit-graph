#!/usr/bin/env python3
"""Fit LG + ER/WS/BA on every Google+ ego network and rank them by GIC.

For each ``data/misc/gplus/*.edges`` file with ``MIN_NODES ≤ n ≤ MAX_NODES``,
fits the Logit-Graph model (AIC d̂ + σ̂) and three baselines (Erdős–Rényi,
Watts–Strogatz, Barabási–Albert), scores each by spectral GIC against the
real graph, and ranks them. Outputs per-graph reports under
``notebooks/refactors/runs/gplus/{graph}/`` and aggregate tables/plots.

Env-var overrides (all optional):
  LG_GPLUS_MAX_NODES     cap on |V| (default 300, set to "none" for no cap)
  LG_GPLUS_MIN_NODES     floor on |V| (default 50)
  LG_GPLUS_LG_ITER       LG max_iterations override (default 2000)
  LG_GPLUS_GRID_POINTS   baseline grid resolution (default 3)
  LG_GPLUS_N_RUNS        baseline ensemble size (default 1)
  LG_GPLUS_USE_CACHE     reload finished networks (0/1, default 1)
  LG_GPLUS_QUICK         set to 1 for smoke (MAX_NODES=150, LG_ITER=1000)

  make gic-gplus         full preset, ~3-5 min on 4 cores
  make gic-gplus-quick   smoke run (~30s on 4 cores)
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Optional


def _get_int(env: str, default: int) -> int:
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


def _get_optional_int(env: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(env)
    if raw is None:
        return default
    if raw.lower() in ("none", "", "null"):
        return None
    return int(raw)


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:5.1f}s"
    m, rem = divmod(s, 60)
    return f"{int(m):2d}m{int(rem):02d}s"


def main() -> None:
    # Unbuffered output so progress shows up live
    sys.stdout.reconfigure(line_buffering=True)
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    _here = Path(__file__).resolve().parent
    _repo_root = _here.parents[1]
    _src = _repo_root / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import platform_fit_utils as pfu  # noqa: E402
    from platform_fit_utils import (  # noqa: E402
        PlatformConfig,
        fit_one_network,
        load_cached_result,
        plot_aggregate_summary,
        setup_platform_logging,
        summarize_aggregates,
    )
    import pandas as pd  # noqa: E402

    quick = os.environ.get("LG_GPLUS_QUICK", "0") == "1"
    max_nodes_default = 150 if quick else 300
    lg_iter_default = 1000 if quick else 2000

    max_nodes = _get_optional_int("LG_GPLUS_MAX_NODES", max_nodes_default)
    min_nodes = _get_int("LG_GPLUS_MIN_NODES", 50)
    lg_iter_cap = _get_int("LG_GPLUS_LG_ITER", lg_iter_default)
    grid_points = _get_int("LG_GPLUS_GRID_POINTS", 3)
    n_runs = _get_int("LG_GPLUS_N_RUNS", 1)
    use_cache = os.environ.get("LG_GPLUS_USE_CACHE", "1") == "1"

    _original_lg_max = pfu.lg_max_iterations

    def _capped(n: int) -> int:
        return min(_original_lg_max(n), lg_iter_cap)

    pfu.lg_max_iterations = _capped

    cfg = PlatformConfig(
        platform="gplus",
        glob_pattern="misc/gplus/*.edges",
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        other_model_n_runs=n_runs,
        other_model_grid_points=grid_points,
        use_cache=use_cache,
        display_plots=False,
        data_root=_repo_root / "data",
        run_dir=_here / "runs",
    )

    banner = (
        f"gplus GIC ranking  min_nodes={min_nodes}  max_nodes={max_nodes}  "
        f"lg_iter≤{lg_iter_cap}  grid_points={grid_points}  n_runs={n_runs}  "
        f"cache={use_cache}  quick={quick}"
    )
    print(banner)

    graph_files, sizes_by_stem = _fast_discover(cfg)
    if not graph_files:
        print("No graphs matched the size window; nothing to do.")
        return

    sizes = [(stem, sizes_by_stem[stem]) for stem in (p.stem for p in graph_files)]
    avg_n = sum(n for _, n in sizes) / len(sizes)
    print(
        f"\nDiscovered {len(graph_files)} networks  "
        f"|V|: min={min(n for _, n in sizes)}  max={max(n for _, n in sizes)}  "
        f"mean={avg_n:.0f}"
    )

    # Pre-check the cache to estimate how many fits are actually needed
    pending = []
    cached_count = 0
    for p in graph_files:
        net_dir = cfg.run_dir / p.stem
        if use_cache and load_cached_result(net_dir, p.stem) is not None:
            cached_count += 1
        else:
            pending.append(p)
    eta_per_fit = max(5.0, 0.05 * avg_n) if pending else 0  # ~5s+ per graph
    print(
        f"Cache: {cached_count}/{len(graph_files)} hits, {len(pending)} fits pending "
        f"(rough ETA per fit ≈ {_fmt_secs(eta_per_fit)})"
    )
    print(
        f"Total wall-time estimate: {_fmt_secs(len(pending) * eta_per_fit)} "
        f"(serial; less with adaptive)\n"
    )

    logger = setup_platform_logging(cfg.run_dir)
    logger.info(
        "=== gplus GIC sweep  (%d networks, %d cached, %d to fit) ===",
        len(graph_files), cached_count, len(pending),
    )

    summary_rows = []
    fit_meta_rows = []
    failures = []
    t_start = time.perf_counter()
    fits_done = 0
    fit_times: list[float] = []

    for i, edge_path in enumerate(graph_files, start=1):
        graph_name = edge_path.stem
        net_dir = cfg.run_dir / graph_name
        elapsed = time.perf_counter() - t_start
        try:
            if use_cache:
                cached = load_cached_result(net_dir, graph_name)
                if cached is not None:
                    n_v = cached["meta"].get("n_nodes")
                    print(
                        f"[{i:3d}/{len(graph_files)}] {graph_name}  CACHE  "
                        f"n={n_v}  best={cached['meta'].get('best_model')}  "
                        f"GIC={cached['meta'].get('best_gic', float('nan')):.3f}  "
                        f"| elapsed {_fmt_secs(elapsed)}"
                    )
                    summary_rows.append(cached["summary"])
                    fit_meta_rows.append({**cached["meta"], "cached": True})
                    continue

            t_one = time.perf_counter()
            n_v = next(n for stem, n in sizes if stem == graph_name)
            avg_fit_so_far = sum(fit_times) / len(fit_times) if fit_times else eta_per_fit
            remaining = len(pending) - fits_done
            eta = remaining * avg_fit_so_far
            print(
                f"[{i:3d}/{len(graph_files)}] {graph_name}  FIT  n={n_v}  "
                f"(remaining {remaining}, ETA {_fmt_secs(eta)}, "
                f"avg/fit so far {_fmt_secs(avg_fit_so_far)})"
            )
            result = fit_one_network(edge_path, cfg, logger, i, len(graph_files))
            dt = time.perf_counter() - t_one
            fit_times.append(dt)
            fits_done += 1
            meta = result["meta"]
            print(
                f"           ↳ DONE  best={meta['best_model']}  "
                f"GIC={meta['best_gic']:.3f}  σ̂={meta.get('sigma_hat', 0):+.3f}  "
                f"d̂={meta.get('d_hat')}  in {_fmt_secs(dt)}"
            )
            summary_rows.append(result["summary"])
            fit_meta_rows.append(meta)
        except Exception as exc:
            print(f"[{i}/{len(graph_files)}] {graph_name}  FAILED  {exc}")
            traceback.print_exc()
            failures.append({"graph": graph_name, "error": str(exc)})

    total_elapsed = time.perf_counter() - t_start
    print(f"\nFit phase complete in {_fmt_secs(total_elapsed)} "
          f"({fits_done} fresh fits, {cached_count} cache hits, {len(failures)} failures)")

    if not summary_rows:
        print("No networks were processed — aborting summary step.")
        return

    summary_all = pd.concat(summary_rows, ignore_index=True)
    fit_meta = pd.DataFrame(fit_meta_rows)
    summary_all.to_csv(cfg.run_dir / "summary_all.csv", index=False)
    fit_meta.to_csv(cfg.run_dir / "fit_meta_all.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(cfg.run_dir / "failures.csv", index=False)

    gic_pivot, rank_pivot, mean_rank = summarize_aggregates(summary_all, cfg.run_dir)
    plot_aggregate_summary(fit_meta, mean_rank, cfg.run_dir, cfg.platform, display=False)

    print("\nMean GIC rank (lower = better fit):")
    print(mean_rank.to_string())

    print("\nPer-graph best model (by GIC):")
    cols = ["graph", "n_nodes", "n_edges", "d_hat", "sigma_hat", "best_model", "best_gic"]
    cols = [c for c in cols if c in fit_meta.columns]
    print(fit_meta[cols].sort_values("n_nodes").to_string(index=False))

    print(f"\nArtifacts in: {cfg.run_dir}")


def _peek_size(edge_path: Path) -> int:
    """Get |V| of an .edges file without networkx parsing."""
    nodes = set()
    with open(edge_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                nodes.add(parts[0])
                nodes.add(parts[1])
    return len(nodes)


def _fast_discover(cfg):
    """File-size pre-filter + cheap node-count peek, skipping networkx parsing.

    The reference ``platform_fit_utils.discover_graph_files`` loads every
    candidate via ``nx.read_edgelist`` to count nodes — for the gplus
    collection this scans ~130 files (incl. multi-MB ones above MAX_NODES)
    and takes minutes. Here we (1) drop files whose byte size implies
    n > 2·max_nodes (lots of slack) and (2) count unique node tokens
    in a single linear pass for the survivors.
    """
    paths = sorted(cfg.data_root.glob(cfg.glob_pattern))
    paths = [p for p in paths if p.suffix == ".edges" and p.is_file()]
    sizes: dict[str, int] = {}
    kept: list[tuple[Path, int]] = []
    if cfg.max_nodes is not None:
        # Empirically each edge in this dataset is ~20 bytes; n*(n-1)/2 edges
        # at upper bound → safe size threshold = 50 · n²·2 = 100 · n² bytes.
        max_bytes = 100 * cfg.max_nodes * cfg.max_nodes
    else:
        max_bytes = None
    for p in paths:
        if max_bytes is not None and p.stat().st_size > max_bytes:
            continue
        try:
            n = _peek_size(p)
        except OSError:
            continue
        if n < cfg.min_nodes:
            continue
        if cfg.max_nodes is not None and n > cfg.max_nodes:
            continue
        if n == 0:
            continue
        sizes[p.stem] = n
        kept.append((p, n))
    kept.sort(key=lambda x: x[1])
    return [p for p, _ in kept], sizes


if __name__ == "__main__":
    main()
