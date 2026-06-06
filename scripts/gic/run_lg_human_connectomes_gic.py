#!/usr/bin/env python3
"""Fit LG + ER/WS/BA on human-brain connectomes (OASIS-3) and rank by GIC.

The repo ships ~7000 .graphml files under ``data/brain_graph/`` across
multiple parcellation atlases:
  oasis3_graphmls_scale1/2/3  — OASIS-3, ~975 subjects, n ∈ {124, 170, 272}
  repeated_10_scale_33/60/125/250 — repeated parcellations, n ∈ {83, 129, 234, 463}

To stay under the 5-min budget we sample N graphs from a chosen scale
and run the same LG vs ER/WS/BA GIC ranking pipeline used in
``run_connectomes_gic.py`` (with KPM spectral density + parallel workers).

Env-var overrides:
  LG_HCONN_SCALE        atlas dir under brain_graph/  (default oasis3_graphmls_scale1)
  LG_HCONN_SAMPLE       number of subjects to sample  (default 100; "all" = no cap)
  LG_HCONN_MAX_NODES    cap on |V|                     (default 500)
  LG_HCONN_MIN_NODES    floor on |V|                   (default 20)
  LG_HCONN_LG_ITER      LG MCMC cap                    (default 1500)
  LG_HCONN_GRID_POINTS  baseline grid                  (default 3)
  LG_HCONN_N_RUNS       baseline reps                  (default 1)
  LG_HCONN_USE_CACHE    reload finished networks       (default 1)
  LG_HCONN_WORKERS      parallel proc count            (default cpu-1)
  LG_HCONN_SEED         RNG seed for the random sample (default 0)
  LG_HCONN_QUICK        set to 1 for smoke (scale_33, sample=20, fewer iter)

  make lg-gic-human-connectomes        ~3-5 min on 4 cores
  make lg-gic-human-connectomes-quick  ~30s
"""
from __future__ import annotations

import os
import random
import sys
import time
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
    if raw.lower() in ("none", "all", "", "null"):
        return None
    return int(raw)


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:5.1f}s"
    m, rem = divmod(s, 60)
    return f"{int(m):2d}m{int(rem):02d}s"


def _load_graphml_graph(path):
    import networkx as nx
    G = nx.read_graphml(path)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() == 0:
        raise ValueError(f"Empty graph loaded from {path}")
    if G.number_of_edges() == 0:
        raise ValueError(f"Edgeless graph loaded from {path}")
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


def _patch_load_edges():
    import platform_fit_utils as pfu
    pfu.load_edges = _load_graphml_graph


def main() -> None:
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

    _patch_load_edges()

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

    quick = os.environ.get("LG_HCONN_QUICK", "0") == "1"
    scale_default = "repeated_10_scale_33" if quick else "oasis3_graphmls_scale1"
    sample_default = 20 if quick else 100
    lg_iter_default = 800 if quick else 1500

    scale = os.environ.get("LG_HCONN_SCALE", scale_default)
    sample_size = _get_optional_int("LG_HCONN_SAMPLE", sample_default)
    max_nodes = _get_optional_int("LG_HCONN_MAX_NODES", 500)
    min_nodes = _get_int("LG_HCONN_MIN_NODES", 20)
    lg_iter_cap = _get_int("LG_HCONN_LG_ITER", lg_iter_default)
    grid_points = _get_int("LG_HCONN_GRID_POINTS", 3)
    n_runs = _get_int("LG_HCONN_N_RUNS", 1)
    use_cache = os.environ.get("LG_HCONN_USE_CACHE", "1") == "1"
    workers = _get_int("LG_HCONN_WORKERS", max(1, (os.cpu_count() or 2) - 1))
    seed = _get_int("LG_HCONN_SEED", 0)

    _original_lg_max = pfu.lg_max_iterations

    def _capped(n: int) -> int:
        return min(_original_lg_max(n), lg_iter_cap)

    pfu.lg_max_iterations = _capped

    cfg = PlatformConfig(
        platform=f"human_connectomes_{scale}",
        glob_pattern=f"brain_graph/{scale}/*.graphml",
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        other_model_n_runs=n_runs,
        other_model_grid_points=grid_points,
        use_cache=use_cache,
        display_plots=False,
        data_root=_repo_root / "data",
        run_dir=_here / "runs",
        seed=seed,
    )

    print(
        f"human connectomes GIC ranking  scale={scale}  sample={sample_size}  "
        f"min_nodes={min_nodes}  max_nodes={max_nodes}  lg_iter≤{lg_iter_cap}  "
        f"grid_points={grid_points}  cache={use_cache}  workers={workers}  "
        f"quick={quick}"
    )

    graph_files, sizes_by_stem = _discover(cfg, sample_size, seed)
    if not graph_files:
        print("No graphs matched; nothing to do.")
        return

    sizes = [(stem, sizes_by_stem[stem]) for stem in (p.stem for p in graph_files)]
    avg_n = sum(n for _, n in sizes) / len(sizes)
    print(
        f"\nSampled {len(graph_files)} subjects  "
        f"|V|: min={min(n for _, n in sizes)}  max={max(n for _, n in sizes)}  "
        f"mean={avg_n:.0f}"
    )

    pending = []
    cached_count = 0
    for p in graph_files:
        net_dir = cfg.run_dir / p.stem
        if use_cache and load_cached_result(net_dir, p.stem) is not None:
            cached_count += 1
        else:
            pending.append(p)
    print(
        f"Cache: {cached_count}/{len(graph_files)} hits, {len(pending)} fits pending\n"
    )

    logger = setup_platform_logging(cfg.run_dir)
    logger.info(
        "=== human connectomes GIC sweep  (%d networks, %d cached, %d to fit) ===",
        len(graph_files), cached_count, len(pending),
    )

    summary_rows = []
    fit_meta_rows = []
    failures = []
    t_start = time.perf_counter()

    pending_with_idx: list[tuple[int, Path]] = []
    for i, edge_path in enumerate(graph_files, start=1):
        graph_name = edge_path.stem
        net_dir = cfg.run_dir / graph_name
        if use_cache:
            cached = load_cached_result(net_dir, graph_name)
            if cached is not None:
                summary_rows.append(cached["summary"])
                fit_meta_rows.append({**cached["meta"], "cached": True})
                continue
        pending_with_idx.append((i, edge_path))

    if pending_with_idx:
        worker_count = max(1, min(workers, len(pending_with_idx)))
        print(
            f"Launching {worker_count} worker(s) for {len(pending_with_idx)} fresh fits ...\n"
        )
        if worker_count == 1:
            results_iter = [
                _fit_worker((str(p), _cfg_to_dict(cfg), lg_iter_cap, i, len(graph_files)))
                for i, p in pending_with_idx
            ]
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            tasks = [
                (str(p), _cfg_to_dict(cfg), lg_iter_cap, i, len(graph_files))
                for i, p in pending_with_idx
            ]
            results_iter = []
            with ProcessPoolExecutor(max_workers=worker_count) as pool:
                futures = {pool.submit(_fit_worker, t): t for t in tasks}
                for fut in as_completed(futures):
                    results_iter.append(fut.result())

        fits_done = 0
        for status, name, payload in results_iter:
            fits_done += 1
            elapsed = time.perf_counter() - t_start
            if status == "err":
                print(f"  [{fits_done}/{len(pending_with_idx)}] {name}  FAILED  {payload}")
                failures.append({"graph": name, "error": payload})
                continue
            meta = payload["meta"]
            print(
                f"  [{fits_done}/{len(pending_with_idx)}] {name[-40:]}  "
                f"n={meta.get('n_nodes')}  best={meta['best_model']}  "
                f"GIC={meta['best_gic']:.3f}  in {_fmt_secs(meta.get('elapsed_s', 0))}  "
                f"| wall {_fmt_secs(elapsed)}"
            )
            summary_rows.append(payload["summary"])
            fit_meta_rows.append(meta)

    total_elapsed = time.perf_counter() - t_start
    print(
        f"\nFit phase complete in {_fmt_secs(total_elapsed)} "
        f"({len(pending_with_idx) - len(failures)} fresh fits, "
        f"{cached_count} cache hits, {len(failures)} failures)"
    )

    if not summary_rows:
        print("No networks were processed.")
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

    print("\nBest-model counts:")
    wins = fit_meta["best_model"].value_counts()
    print(wins.to_string())

    print(f"\nArtifacts in: {cfg.run_dir}")


def _cfg_to_dict(cfg) -> dict:
    return {
        "platform": cfg.platform,
        "glob_pattern": cfg.glob_pattern,
        "min_nodes": cfg.min_nodes,
        "max_nodes": cfg.max_nodes,
        "d_candidates": list(cfg.d_candidates),
        "seed": cfg.seed,
        "other_model_n_runs": cfg.other_model_n_runs,
        "other_model_grid_points": cfg.other_model_grid_points,
        "display_plots": cfg.display_plots,
        "use_cache": cfg.use_cache,
        "data_root": str(cfg.data_root),
        "run_dir_parent": str(cfg.run_dir.parent),
    }


def _fit_worker(args):
    edge_path_str, cfg_dict, lg_iter_cap, i, total = args
    import sys as _sys
    import os as _os
    from pathlib import Path as _Path

    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ.setdefault(_v, "1")

    _here = _Path(__file__).resolve().parent
    _repo_root = _here.parents[1]
    for _p in (str(_repo_root / "src"), str(_here)):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    _patch_load_edges()
    import platform_fit_utils as pfu
    from platform_fit_utils import PlatformConfig, fit_one_network, setup_platform_logging

    cfg_kwargs = dict(cfg_dict)
    run_dir_parent = cfg_kwargs.pop("run_dir_parent")
    cfg_kwargs["data_root"] = _Path(cfg_kwargs["data_root"])
    cfg_kwargs["run_dir"] = _Path(run_dir_parent)
    cfg = PlatformConfig(**cfg_kwargs)

    _orig_lg_max = pfu.lg_max_iterations
    pfu.lg_max_iterations = lambda n: min(_orig_lg_max(n), lg_iter_cap)

    edge_path = _Path(edge_path_str)
    logger = setup_platform_logging(cfg.run_dir, name=f"worker_{_os.getpid()}")
    try:
        result = fit_one_network(edge_path, cfg, logger, i, total)
        return ("ok", edge_path.stem, {"meta": result["meta"], "summary": result["summary"]})
    except Exception as exc:
        import traceback as _tb
        return ("err", edge_path.stem, f"{exc}\n{_tb.format_exc()}")


def _discover(cfg, sample_size, seed):
    """Sample up to ``sample_size`` graphs from cfg.glob_pattern.

    All subjects in a given parcellation atlas have the same node count,
    so the size filter is effectively a sanity check. We don't pre-load
    every .graphml to peek size — too slow with 1000+ files. Instead we
    peek a single file to learn n, apply min/max bounds to the whole
    batch, then random-sample paths.
    """
    import networkx as nx

    paths = sorted(cfg.data_root.glob(cfg.glob_pattern))
    paths = [p for p in paths if p.suffix == ".graphml" and p.is_file()]
    if not paths:
        return [], {}

    # Atlas-wide n: peek one file.
    n_peek = 0
    try:
        G = _load_graphml_graph(paths[0])
        n_peek = G.number_of_nodes()
    except Exception as exc:
        print(f"  warning: failed to peek size of {paths[0].name}: {exc}")
    if cfg.max_nodes is not None and n_peek > cfg.max_nodes:
        print(f"  scale n={n_peek} exceeds max_nodes={cfg.max_nodes}; skipping all")
        return [], {}
    if cfg.min_nodes and n_peek < cfg.min_nodes:
        print(f"  scale n={n_peek} below min_nodes={cfg.min_nodes}; skipping all")
        return [], {}

    rng = random.Random(seed)
    if sample_size is None or sample_size >= len(paths):
        chosen = paths
    else:
        chosen = rng.sample(paths, sample_size)
    chosen.sort()  # for stable display + cache ordering

    sizes = {p.stem: n_peek for p in chosen}
    return chosen, sizes


if __name__ == "__main__":
    main()
