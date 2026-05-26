"""Shared batch-fit helpers for platform refactor notebooks."""
from __future__ import annotations

import json
import logging
import pickle
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from logit_graph import (
    GraphModelComparator,
    calculate_graph_attributes,
    estimate_sigma_from_graph,
    select_d_ensemble,
)

MODELS = ["LG", "ER", "WS", "BA"]
MODEL_COLORS = {
    "LG": "#2b6cb0",
    "ER": "#a0aec0",
    "WS": "#ed8936",
    "BA": "#38a169",
    "Original": "#718096",
}


@dataclass
class PlatformConfig:
    platform: str
    glob_pattern: str
    min_nodes: int = 0
    max_nodes: Optional[int] = 500
    d_candidates: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    seed: int = 0
    other_model_n_runs: int = 2
    other_model_grid_points: int = 5
    display_plots: bool = False
    use_cache: bool = True
    data_root: Path = field(default_factory=lambda: Path("..") / ".." / "data")
    run_dir: Path = field(default_factory=lambda: Path("runs"))

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir) / self.platform
        self.run_dir.mkdir(parents=True, exist_ok=True)


def setup_platform_logging(run_dir: Path, name: str = "platform_fit") -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(run_dir / "platform.log", mode="a", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def load_edges(path: Path) -> nx.Graph:
    G = nx.read_edgelist(path, nodetype=int)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() == 0:
        raise ValueError(f"Empty graph loaded from {path}")
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


def lg_max_iterations(n: int) -> int:
    if n <= 100:
        return 4000
    if n <= 300:
        return 8000
    if n <= 500:
        return 12000
    if n <= 700:
        return 16000
    return 20000


def discover_graph_files(cfg: PlatformConfig) -> list[Path]:
    paths = sorted(cfg.data_root.glob(cfg.glob_pattern))
    paths = [p for p in paths if p.suffix == ".edges" and p.is_file()]
    kept: list[tuple[Path, int]] = []
    skipped: list[tuple[str, str]] = []
    skipped_large: list[tuple[str, int]] = []

    for p in paths:
        try:
            G = load_edges(p)
        except (ValueError, OSError, nx.NetworkXError) as exc:
            skipped.append((p.name, str(exc)))
            continue
        n = G.number_of_nodes()
        if n < cfg.min_nodes:
            continue
        if cfg.max_nodes is not None and n > cfg.max_nodes:
            skipped_large.append((p.name, n))
            continue
        kept.append((p, n))

    kept.sort(key=lambda x: x[1])
    if skipped:
        print(f"Skipped {len(skipped)} unreadable/empty graphs")
        for name, reason in skipped[:5]:
            print(f"  {name}: {reason}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")
    if skipped_large:
        cap = cfg.max_nodes
        print(f"Skipped {len(skipped_large)} graphs with n > {cap} (MAX_NODES)")
        for name, n in skipped_large[:5]:
            print(f"  {name}: n={n}")
        if len(skipped_large) > 5:
            print(f"  ... and {len(skipped_large) - 5} more")
    return [p for p, _ in kept]


def print_discovery(cfg: PlatformConfig, graph_files: list[Path]) -> None:
    max_label = cfg.max_nodes if cfg.max_nodes is not None else "∞"
    print(f"PLATFORM={cfg.platform}  RUN_DIR={cfg.run_dir.resolve()}")
    print(
        f"Found {len(graph_files)} networks "
        f"(MIN_NODES={cfg.min_nodes}, MAX_NODES={max_label}, USE_CACHE={cfg.use_cache})"
    )
    for p in graph_files:
        G = load_edges(p)
        cached = "  [cached]" if cfg.use_cache and _cache_valid(cfg.run_dir / p.stem) else ""
        print(
            f"  {p.name:>30s}  n={G.number_of_nodes():>5d}  "
            f"|E|={G.number_of_edges():>7d}{cached}"
        )


def _cache_valid(net_dir: Path) -> bool:
    return (net_dir / "comparator.pkl").is_file() and (net_dir / "summary.csv").is_file()


def load_cached_result(net_dir: Path, graph_name: str) -> Optional[dict[str, Any]]:
    if not (net_dir / "comparator.pkl").is_file() or not (net_dir / "summary.csv").is_file():
        return None
    with open(net_dir / "comparator.pkl", "rb") as f:
        comparator = pickle.load(f)
    summary = pd.read_csv(net_dir / "summary.csv")
    meta_path = net_dir / "fit_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        scored = summary[summary["model"] != "Original"].dropna(subset=["gic_value"])
        best = scored.sort_values("gic_value").iloc[0]
        orig = summary[summary["model"] == "Original"].iloc[0]
        meta = {
            "graph": graph_name,
            "n_nodes": int(orig.get("nodes", np.nan)),
            "n_edges": int(orig.get("edges", np.nan)),
            "best_model": str(best["model"]),
            "best_gic": float(best["gic_value"]),
            "cached": True,
        }
        lg = comparator.fitted_graphs_data.get("LG", {}).get("metadata", {})
        if lg.get("d") is not None:
            meta["d_hat"] = int(lg["d"])
        if lg.get("sigma") is not None:
            meta["sigma_hat"] = float(lg["sigma"])
        _save_fit_meta(net_dir, meta)
    meta.setdefault("graph", graph_name)
    meta["cached"] = True
    return {"comparator": comparator, "summary": summary, "meta": meta}


def _save_fit_meta(net_dir: Path, meta: dict[str, Any]) -> None:
    (net_dir / "fit_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )


def _aic_table(aic_stats: dict[int, dict[str, Any]], d_hat: int) -> pd.DataFrame:
    rows = [{
        "d": d,
        "AIC": s["aic"],
        "ll": s["ll"],
        "sigma_hat(d)": s["sigma_hat"],
        "n_obs": int(s["n_obs"]),
        "selected": d == d_hat,
    } for d, s in aic_stats.items()]
    return pd.DataFrame(rows).set_index("d")


def _plot_aic_selection(ax: plt.Axes, aic_stats: dict[int, dict[str, Any]], d_hat: int) -> None:
    ds = sorted(aic_stats.keys())
    aics = [aic_stats[d]["aic"] for d in ds]
    colors = ["#2b6cb0" if d == d_hat else "#cbd5e0" for d in ds]
    bars = ax.bar([str(d) for d in ds], aics, color=colors, edgecolor="white")
    ax.set_xlabel("d")
    ax.set_ylabel("AIC (lower = better)")
    ax.set_title(f"AIC model selection  →  d̂={d_hat}")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, aics):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f}",
                ha="center", va="bottom", fontsize=8)


def _plot_gic_comparison(ax: plt.Axes, summary: pd.DataFrame) -> None:
    scored = summary[summary["model"] != "Original"].dropna(subset=["gic_value"])
    scored = scored.sort_values("gic_value")
    colors = [MODEL_COLORS.get(m, "#a0aec0") for m in scored["model"]]
    bars = ax.bar(scored["model"], scored["gic_value"], color=colors, edgecolor="white")
    ax.set_ylabel("GIC (lower = better)")
    ax.set_title("Spectral GIC by model")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, scored["gic_value"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)


def _plot_lg_convergence(
    ax: plt.Axes,
    gic_values: list[float],
    check_interval: int,
    best_iteration: int = -1,
) -> None:
    if not gic_values:
        ax.text(0.5, 0.5, "No LG GIC trace", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    xs = np.arange(len(gic_values)) * check_interval
    ax.plot(xs, gic_values, color="#2b6cb0", alpha=0.35, linewidth=1, label="GIC checks")
    if len(gic_values) >= 3:
        ma = pd.Series(gic_values).rolling(window=min(10, len(gic_values)), min_periods=1).mean()
        ax.plot(xs, ma, color="#c53030", linewidth=1.5, label="rolling mean")
    if best_iteration >= 0 and best_iteration < len(gic_values):
        ax.axvline(xs[best_iteration], color="#38a169", linestyle="--", linewidth=1,
                   label=f"best @ {xs[best_iteration]}")
    ax.set_xlabel("Gibbs iteration")
    ax.set_ylabel("GIC")
    ax.set_title("LG fit convergence")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)


def _plot_structural_comparison(ax: plt.Axes, summary: pd.DataFrame) -> None:
    cols = ["density", "avg_clustering", "assortativity"]
    subset = summary[summary["model"].isin(["Original"] + MODELS)].copy()
    if subset.empty:
        ax.set_axis_off()
        return
    x = np.arange(len(cols))
    width = 0.15
    models_present = [m for m in ["Original"] + MODELS if m in subset["model"].values]
    for i, model in enumerate(models_present):
        row = subset[subset["model"] == model].iloc[0]
        vals = [row[c] for c in cols]
        offset = (i - len(models_present) / 2) * width + width / 2
        ax.bar(x + offset, vals, width=width, label=model,
               color=MODEL_COLORS.get(model, "#a0aec0"), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=15, ha="right")
    ax.set_title("Structural statistics")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)


def save_network_report(
    graph_name: str,
    G_real: nx.Graph,
    d_hat: int,
    sigma_hat: float,
    aic_stats: dict[int, dict[str, Any]],
    comparator: GraphModelComparator,
    net_dir: Path,
    check_interval: int = 50,
    *,
    close: bool = True,
) -> tuple[Path, plt.Figure]:
    summary = comparator.summary_df.copy()
    lg_meta = comparator.fitted_graphs_data.get("LG", {}).get("metadata", {})
    gic_values = lg_meta.get("gic_values") or []
    best_iter = int(lg_meta.get("min_gic_iteration", -1))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"{graph_name}  |  n={G_real.number_of_nodes()}  |E|={G_real.number_of_edges()}  "
        f"d̂={d_hat}  σ̂={sigma_hat:+.3f}",
        fontsize=11,
    )

    _plot_aic_selection(axes[0, 0], aic_stats, d_hat)
    _plot_gic_comparison(axes[0, 1], summary)
    _plot_lg_convergence(axes[1, 0], gic_values, check_interval, best_iter)
    _plot_structural_comparison(axes[1, 1], summary)

    fig.tight_layout()
    out = net_dir / "fit_report.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if close:
        plt.close(fig)
    return out, fig


def _log_step(logger: logging.Logger, step: int, total: int, title: str, detail: str = "") -> None:
    msg = f"  STEP {step}/{total}  {title}"
    if detail:
        msg = f"{msg}  {detail}"
    logger.info(msg)


def fit_one_network(
    edge_path: Path,
    cfg: PlatformConfig,
    logger: logging.Logger,
    index: int,
    total: int,
) -> dict[str, Any]:
    graph_name = edge_path.stem
    t0 = time.perf_counter()
    logger.info("[%d/%d] %s", index, total, graph_name)

    net_dir = cfg.run_dir / graph_name
    net_dir.mkdir(parents=True, exist_ok=True)

    G_real = load_edges(edge_path)
    n, m = G_real.number_of_nodes(), G_real.number_of_edges()
    attrs = calculate_graph_attributes(G_real)
    _log_step(
        logger, 1, 4, "Load graph",
        f"n={n} |E|={m} density={attrs['density']:.4f} "
        f"clustering={attrs['avg_clustering']:.3f} assort={attrs['assortativity']:.3f}",
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    adj = nx.to_numpy_array(G_real)

    t_aic = time.perf_counter()
    d_hat, aic_stats = select_d_ensemble(
        graphs=[adj],
        d_candidates=cfg.d_candidates,
        feature_mode="incremental",
        extra_penalty_per_d=0.0,
    )
    aic_df = _aic_table(aic_stats, d_hat)
    aic_detail = "  ".join(f"d={d}:AIC={aic_stats[d]['aic']:.1f}" for d in sorted(aic_stats))
    _log_step(
        logger, 2, 4, "AIC select d",
        f"d̂={d_hat}  ({aic_detail})  [{time.perf_counter() - t_aic:.1f}s]",
    )

    t_sig = time.perf_counter()
    sigma_hat = float(estimate_sigma_from_graph(adj, d_hat, feature_mode="incremental"))
    _log_step(
        logger, 3, 4, "Estimate σ at d̂",
        f"σ̂={sigma_hat:+.4f}  [{time.perf_counter() - t_sig:.1f}s]",
    )

    check_interval = 50
    lg_params = {
        "max_iterations": lg_max_iterations(n),
        "patience": 500,
        "edge_delta": None,
        "min_gic_threshold": 5,
        "er_p": 0.05,
        "check_interval": check_interval,
    }
    _log_step(
        logger, 4, 4, "Model comparison",
        f"LG max_iter={lg_params['max_iterations']}  baselines=ER/WS/BA",
    )

    t_cmp = time.perf_counter()
    comparator = GraphModelComparator(
        d_list=[d_hat],
        lg_params=lg_params,
        other_model_n_runs=cfg.other_model_n_runs,
        dist_type="KL",
        verbose=True,
        other_models=["ER", "WS", "BA"],
        other_model_grid_points=cfg.other_model_grid_points,
        random_state=cfg.seed,
    ).compare(original_graph=G_real, graph_filepath=graph_name)

    summary = comparator.summary_df.copy()
    scored = summary[summary["model"] != "Original"].dropna(subset=["gic_value"])
    scored = scored.sort_values("gic_value")
    best = scored.iloc[0]
    gic_line = "  ".join(
        f"{row['model']}={row['gic_value']:.3f}" for _, row in scored.iterrows()
    )
    logger.info(
        "  RESULT  best=%s  GIC=%.3f  (%s)  [compare %.1fs | total %.1fs]",
        best["model"], best["gic_value"], gic_line,
        time.perf_counter() - t_cmp, time.perf_counter() - t0,
    )

    aic_df.round(6).to_csv(net_dir / "aic_table.csv")
    summary.round(6).to_csv(net_dir / "summary.csv", index=False)
    with open(net_dir / "comparator.pkl", "wb") as f:
        pickle.dump(comparator, f)

    report_path, fig = save_network_report(
        graph_name, G_real, d_hat, sigma_hat, aic_stats, comparator, net_dir,
        check_interval, close=not cfg.display_plots,
    )
    logger.info("  Saved %s", report_path.name)

    if cfg.display_plots:
        plt.show()
        plt.close(fig)

    meta = {
        "graph": graph_name,
        "n_nodes": n,
        "n_edges": m,
        "d_hat": int(d_hat),
        "sigma_hat": sigma_hat,
        "best_model": str(best["model"]),
        "best_gic": float(best["gic_value"]),
        "elapsed_s": time.perf_counter() - t0,
        "seed": cfg.seed,
        "cached": False,
    }
    _save_fit_meta(net_dir, meta)

    return {
        "comparator": comparator,
        "summary": summary,
        "meta": meta,
    }


def fit_all_networks(
    graph_files: list[Path],
    cfg: PlatformConfig,
) -> tuple[list[GraphModelComparator], pd.DataFrame, pd.DataFrame, list[dict]]:
    logger = setup_platform_logging(cfg.run_dir)
    max_label = cfg.max_nodes if cfg.max_nodes is not None else "∞"
    logger.info(
        "=== %s batch fit  (%d networks, MAX_NODES=%s, cache=%s) ===",
        cfg.platform, len(graph_files), max_label, cfg.use_cache,
    )

    comparators: list[GraphModelComparator] = []
    summary_rows: list[pd.DataFrame] = []
    fit_meta_rows: list[dict] = []
    failures: list[dict] = []
    n_cached = 0

    for i, edge_path in enumerate(graph_files, start=1):
        graph_name = edge_path.stem
        net_dir = cfg.run_dir / graph_name
        try:
            if cfg.use_cache:
                cached = load_cached_result(net_dir, graph_name)
                if cached is not None:
                    n_cached += 1
                    logger.info(
                        "[%d/%d] %s  CACHED  (n=%s, best=%s, GIC=%.3f)",
                        i, len(graph_files), graph_name,
                        cached["meta"].get("n_nodes"),
                        cached["meta"].get("best_model"),
                        cached["meta"].get("best_gic", float("nan")),
                    )
                    comparators.append(cached["comparator"])
                    summary_rows.append(cached["summary"])
                    fit_meta_rows.append({**cached["meta"], "cached": True})
                    continue

            result = fit_one_network(edge_path, cfg, logger, i, len(graph_files))
            comparators.append(result["comparator"])
            summary_rows.append(result["summary"])
            fit_meta_rows.append(result["meta"])
        except Exception as exc:
            logger.error("  FAILED %s: %s", graph_name, exc)
            traceback.print_exc()
            failures.append({"graph": graph_name, "error": str(exc)})

    if failures:
        pd.DataFrame(failures).to_csv(cfg.run_dir / "failures.csv", index=False)
        logger.warning("%d network(s) failed — see failures.csv", len(failures))

    if not summary_rows:
        raise RuntimeError(
            "No networks were processed — check DATA_ROOT / MIN_NODES / MAX_NODES."
        )

    summary_all = pd.concat(summary_rows, ignore_index=True)
    fit_meta = pd.DataFrame(fit_meta_rows)
    summary_all.to_csv(cfg.run_dir / "summary_all.csv", index=False)
    fit_meta.to_csv(cfg.run_dir / "fit_meta_all.csv", index=False)
    n_fitted = len(summary_rows) - n_cached
    logger.info(
        "Done — %d/%d networks (%d cached, %d fitted). Outputs in %s",
        len(summary_rows), len(graph_files), n_cached, n_fitted,
        cfg.run_dir.resolve(),
    )
    return comparators, summary_all, fit_meta, failures


def summarize_aggregates(summary_all: pd.DataFrame, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    gic_pivot = summary_all[summary_all["model"].isin(MODELS)].pivot_table(
        index="graph_filename", columns="model", values="gic_value", aggfunc="first",
    )
    gic_pivot = gic_pivot.reindex(columns=MODELS)
    gic_pivot.to_csv(run_dir / "gic_pivot.csv")

    rank_pivot = gic_pivot.rank(axis=1, method="average")
    rank_pivot.to_csv(run_dir / "gic_rank_pivot.csv")

    mean_rank = rank_pivot.mean(axis=0).sort_values()
    print("Mean GIC rank by model (lower = better):")
    print(mean_rank.round(3))
    print()
    print(gic_pivot.round(3).to_string())
    print()
    print(rank_pivot.round(1).to_string())
    return gic_pivot, rank_pivot, mean_rank


def plot_aggregate_summary(
    fit_meta: pd.DataFrame,
    mean_rank: pd.Series,
    run_dir: Path,
    platform: str,
    display: bool = True,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    mean_rank.plot(kind="bar", ax=axes[0], color="#2b6cb0", edgecolor="white")
    axes[0].set_ylabel("Mean GIC rank (lower = better)")
    axes[0].set_title(f"{platform} — mean model rank across networks")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].grid(axis="y", alpha=0.3)

    best_counts = fit_meta["best_model"].value_counts().reindex(MODELS, fill_value=0)
    colors = [MODEL_COLORS.get(m, "#a0aec0") for m in best_counts.index]
    axes[1].bar(best_counts.index, best_counts.values, color=colors, edgecolor="white")
    axes[1].set_ylabel("# networks")
    axes[1].set_title(f"{platform} — best model by GIC count")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = run_dir / "aggregate_model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if display:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved {out}")
    return out
