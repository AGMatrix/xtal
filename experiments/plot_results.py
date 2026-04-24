"""
plot_results.py — Generates paper-quality figures from evaluation CSVs.

Figures produced (saved to experiments/figures/):
  fig1_latency_cdf_{pattern}.pdf  : CDF of p99 latency with tail inset
  fig2_timeseries_burst.pdf       : p99 time series with mean ± std bands
  fig3_timeseries_failure.pdf     : p99 time series during failure injection
  fig4_sla_violation.pdf          : Grouped SLA violation rate bar chart
  fig5_shap_importance.pdf        : SHAP feature importances with std bars
  fig6_tail_amplification.pdf     : Tail amplification boxplot + stripplot
  fig7_recovery_time.pdf          : Recovery time after failure injection
  fig8_overhead_table.pdf         : p99 improvement summary table

Usage:
    python -m experiments.plot_results --raw experiments/results/eval_raw_<ts>.csv \
           --decisions logs/xtal_decisions_<ts>.csv
"""

from __future__ import annotations

import argparse
import pathlib
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger(__name__)

# ── Style ──────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "legend.fontsize":    8.5,
    "legend.framealpha":  0.85,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.dpi":         150,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "lines.linewidth":    1.8,
})

FIGURES_DIR = pathlib.Path("experiments/figures")
SLA_MS = 500.0

# Colorblind-safe palette (Wong 2011)
SYSTEM_COLORS = {
    "static-default":    "#E69F00",   # orange
    "static-hand-tuned": "#56B4E9",   # sky blue
    "reactive-scaler":   "#009E73",   # green
    "xtal":              "#CC79A7",   # violet (stands out)
}
SYSTEM_LABELS = {
    "static-default":    "Static Default",
    "static-hand-tuned": "Static Hand-Tuned",
    "reactive-scaler":   "Reactive Scaler",
    "xtal":              "XTAL (ours)",
}
SYSTEM_HATCHES = {
    "static-default":    "//",
    "static-hand-tuned": "\\\\",
    "reactive-scaler":   "xx",
    "xtal":              "",
}
SYSTEM_ORDER = ["static-default", "static-hand-tuned", "reactive-scaler", "xtal"]
PATTERN_TITLES = {
    "steady":  "Steady Load",
    "bursty":  "Bursty Load (10× spike)",
    "failure": "Failure Injection",
}


def _save(fig: plt.Figure, path: pathlib.Path) -> pathlib.Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return path


# ── Figure 1: CDF of p99 latency  ────────────────────────────────────────────
# Shows full CDF + zoomed tail inset, with SLA line and per-system p95 ticks.

def fig_latency_cdf(df: pd.DataFrame, pattern: str) -> pathlib.Path:
    sub = df[df["load_pattern"] == pattern]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))

    # Inset for the tail (90th–100th percentile region)
    ax_inset = ax.inset_axes([0.52, 0.08, 0.44, 0.42])

    for sys in SYSTEM_ORDER:
        vals = sub[sub["system"] == sys]["p99_ms"].dropna().values
        if len(vals) == 0:
            continue
        vals = np.sort(vals)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        color = SYSTEM_COLORS[sys]
        label = SYSTEM_LABELS[sys]

        ax.plot(vals, cdf, color=color, label=label, linewidth=1.8)
        ax_inset.plot(vals, cdf, color=color, linewidth=1.4)

        # Mark p99 on main plot
        p99_val = np.percentile(vals, 99)
        ax.axvline(p99_val, color=color, linewidth=0.6, linestyle=":", alpha=0.6)

    # SLA line
    for a in (ax, ax_inset):
        a.axvline(SLA_MS, color="#D62728", linestyle="--", linewidth=1.2,
                  label="SLA (500 ms)" if a is ax else None)

    ax.fill_betweenx([0, 1], 0, SLA_MS, alpha=0.04, color="#2CA02C", label="_")
    ax.set_xlabel("p99 Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(f"p99 Latency CDF — {PATTERN_TITLES.get(pattern, pattern)}")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right", fontsize=8)

    # Inset limits: focus on values > SLA
    all_vals = sub["p99_ms"].dropna().values
    tail_lo = np.percentile(all_vals, 85)
    ax_inset.set_xlim(tail_lo, np.percentile(all_vals, 99.5))
    ax_inset.set_ylim(0.85, 1.01)
    ax_inset.set_title("Tail region", fontsize=7)
    ax_inset.tick_params(labelsize=7)
    ax_inset.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.indicate_inset_zoom(ax_inset, edgecolor="grey")

    out = FIGURES_DIR / f"fig1_latency_cdf_{pattern}.pdf"
    return _save(fig, out)


# ── Figure 2/3: Time series with confidence bands  ───────────────────────────
# Aggregates across runs → mean ± std shading, with event annotations.

def fig_timeseries(df: pd.DataFrame, pattern: str, title: str, out_name: str,
                   burst_at: int = 60, burst_len: int = 10,
                   fail_at: int = 90, fail_len: int = 30) -> pathlib.Path:
    sub = df[df["load_pattern"] == pattern]
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.5), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    ax, ax2 = axes

    for sys in SYSTEM_ORDER:
        s = sub[sub["system"] == sys]
        if s.empty:
            continue
        grouped = s.groupby("elapsed_s")["p99_ms"]
        times = np.array(sorted(grouped.groups.keys()))
        means = grouped.mean().loc[times].values
        stds  = grouped.std(ddof=0).fillna(0).loc[times].values

        color = SYSTEM_COLORS[sys]
        ax.plot(times, means, color=color, label=SYSTEM_LABELS[sys], linewidth=1.8)
        ax.fill_between(times, means - stds, means + stds,
                        alpha=0.15, color=color)

    # SLA line
    ax.axhline(SLA_MS, color="#D62728", linestyle="--", linewidth=1.2, label="SLA (500 ms)")

    # Event shading
    if pattern == "bursty":
        ax.axvspan(burst_at, burst_at + burst_len, alpha=0.10, color="#D62728", zorder=0)
        ax.annotate("10× burst", xy=(burst_at + burst_len / 2, SLA_MS * 1.05),
                    ha="center", fontsize=7.5, color="#D62728",
                    arrowprops=dict(arrowstyle="->", color="#D62728", lw=0.8),
                    xytext=(burst_at + burst_len / 2, SLA_MS * 1.25))
    elif pattern == "failure":
        ax.axvspan(fail_at, fail_at + fail_len, alpha=0.10, color="#FF7F0E", zorder=0)
        ax.annotate("TM failure", xy=(fail_at + 2, SLA_MS * 1.05),
                    ha="left", fontsize=7.5, color="#FF7F0E",
                    arrowprops=dict(arrowstyle="->", color="#FF7F0E", lw=0.8),
                    xytext=(fail_at + 8, SLA_MS * 1.35))

    ax.set_ylabel("p99 Latency (ms)")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.set_ylim(bottom=0)

    # Bottom panel: buffer usage
    for sys in SYSTEM_ORDER:
        s = sub[sub["system"] == sys]
        if s.empty or "buffer_usage" not in s.columns:
            continue
        grouped2 = s.groupby("elapsed_s")["buffer_usage"]
        times = np.array(sorted(grouped2.groups.keys()))
        means2 = grouped2.mean().loc[times].values
        ax2.plot(times, means2, color=SYSTEM_COLORS[sys], linewidth=1.2, alpha=0.9)

    ax2.set_ylabel("Buffer\nUsage", fontsize=9)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    if pattern == "bursty":
        ax2.axvspan(burst_at, burst_at + burst_len, alpha=0.10, color="#D62728", zorder=0)
    elif pattern == "failure":
        ax2.axvspan(fail_at, fail_at + fail_len, alpha=0.10, color="#FF7F0E", zorder=0)

    fig.tight_layout(h_pad=0.5)
    out = FIGURES_DIR / out_name
    return _save(fig, out)


# ── Figure 4: Grouped SLA violation bar chart  ────────────────────────────────
# One grouped bar cluster per load pattern, error bars from multiple runs.

def fig_sla_violation(df: pd.DataFrame) -> pathlib.Path:
    patterns = [p for p in ["steady", "bursty", "failure"] if p in df["load_pattern"].unique()]
    n_sys = len(SYSTEM_ORDER)
    x = np.arange(len(patterns))
    width = 0.18
    offsets = np.linspace(-(n_sys - 1) / 2, (n_sys - 1) / 2, n_sys) * width

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    for i, sys in enumerate(SYSTEM_ORDER):
        rates, errs = [], []
        for pat in patterns:
            sub = df[(df["load_pattern"] == pat) & (df["system"] == sys)]
            if sub.empty:
                rates.append(0.0); errs.append(0.0); continue
            # Per-run violation rate → mean ± std
            # We don't have a run column, so estimate std from temporal variance
            viol = sub["sla_violated"].values
            rates.append(float(viol.mean()) * 100)
            # Bootstrap std estimate
            boot = [np.mean(np.random.choice(viol, len(viol), replace=True)) * 100
                    for _ in range(200)]
            errs.append(float(np.std(boot)))

        bars = ax.bar(x + offsets[i], rates, width,
                      label=SYSTEM_LABELS[sys],
                      color=SYSTEM_COLORS[sys],
                      hatch=SYSTEM_HATCHES[sys],
                      edgecolor="white",
                      linewidth=0.5)
        ax.errorbar(x + offsets[i], rates, yerr=errs,
                    fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)

        # Label bars > 1%
        for xpos, rate in zip(x + offsets[i], rates):
            if rate > 1.0:
                ax.text(xpos, rate + max(errs) + 0.5, f"{rate:.1f}",
                        ha="center", va="bottom", fontsize=6.5, fontweight="bold",
                        color=SYSTEM_COLORS[sys])

    ax.set_xticks(x)
    ax.set_xticklabels([PATTERN_TITLES.get(p, p) for p in patterns], fontsize=9)
    ax.set_ylabel("SLA Violation Rate (%)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_title("SLA Violation Rate by System and Load Pattern")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    out = FIGURES_DIR / "fig4_sla_violation.pdf"
    return _save(fig, out)


# ── Figure 5: SHAP feature importance  ───────────────────────────────────────
# Horizontal bar chart, sorted by mean |SHAP|, with std caps.

_FEAT_LABELS = {
    "input_buffer_usage":   "Input Buffer Usage",
    "p99_latency_ms":       "p99 Latency",
    "checkpoint_ms":        "Checkpoint Interval",
    "kafka_consumer_lag":   "Kafka Consumer Lag",
    "records_in_per_sec":   "Throughput (rec/s)",
    "p999_latency_ms":      "p999 Latency",
    "checkpoint_duration_ms": "Checkpoint Duration",
    "buffer_timeout_ms":    "Buffer Timeout",
    "kafka_linger_ms":      "Kafka Linger",
}


def fig_shap_importance(decisions_df: pd.DataFrame) -> pathlib.Path:
    shap_data: dict[str, list[float]] = {}
    for i in range(1, 4):
        fc, vc = f"shap_feat_{i}", f"shap_val_{i}"
        if fc not in decisions_df.columns:
            continue
        for _, row in decisions_df.iterrows():
            feat, val = row.get(fc), row.get(vc)
            if pd.notna(feat) and pd.notna(val):
                shap_data.setdefault(str(feat), []).append(abs(float(val)))

    if not shap_data:
        log.warning("No SHAP data found — skipping fig5")
        return FIGURES_DIR / "fig5_shap_importance.pdf"

    features = sorted(shap_data.keys(), key=lambda f: np.mean(shap_data[f]), reverse=True)
    means = np.array([np.mean(shap_data[f]) for f in features])
    stds  = np.array([np.std(shap_data[f]) for f in features])
    labels = [_FEAT_LABELS.get(f, f) for f in features]

    # Normalise to [0,1] for colour mapping
    norm = matplotlib.colors.Normalize(vmin=0, vmax=means[0])
    cmap = matplotlib.cm.get_cmap("YlOrRd")
    colors = [cmap(norm(m)) for m in means]

    fig, ax = plt.subplots(figsize=(4.8, 0.5 * len(features) + 1.0))
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, capsize=3,
                   edgecolor="white", linewidth=0.5, height=0.65,
                   error_kw={"elinewidth": 0.8, "ecolor": "#555555"})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("XTAL: Feature Importance for Control Decisions")
    ax.invert_yaxis()

    # Percentage labels
    total = means.sum()
    for bar, m in zip(bars, means):
        pct = 100 * m / total
        ax.text(bar.get_width() + stds[list(means).index(m)] + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%", va="center", fontsize=7.5)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Importance", shrink=0.6)
    fig.tight_layout()

    out = FIGURES_DIR / "fig5_shap_importance.pdf"
    return _save(fig, out)


# ── Figure 6: Tail amplification (boxplot + stripplot)  ──────────────────────

def fig_tail_amplification(df: pd.DataFrame) -> pathlib.Path:
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.2), sharey=True)
    patterns = ["steady", "bursty", "failure"]

    for ax, pattern in zip(axes, patterns):
        sub = df[df["load_pattern"] == pattern]
        data, colors, labels = [], [], []
        for sys in SYSTEM_ORDER:
            vals = sub[sub["system"] == sys]["tail_amplification"].dropna().values
            if len(vals) == 0:
                continue
            data.append(vals)
            colors.append(SYSTEM_COLORS[sys])
            labels.append(SYSTEM_LABELS[sys])

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 1.8},
                        whiskerprops={"linewidth": 1.0},
                        capprops={"linewidth": 1.0},
                        flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})
        for patch, color, hatch in zip(bp["boxes"], colors,
                                        [SYSTEM_HATCHES[s] for s in SYSTEM_ORDER
                                         if not df[(df["load_pattern"] == pattern) &
                                                   (df["system"] == s)].empty]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_hatch(hatch)

        # Jitter strip
        rng = np.random.default_rng(0)
        for j, (vals, color) in enumerate(zip(data, colors)):
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), j + 1) + jitter, vals,
                       color=color, s=6, alpha=0.5, zorder=3)

        # Median labels
        for j, vals in enumerate(data):
            med = np.median(vals)
            ax.text(j + 1, med, f" {med:.1f}×",
                    va="center", fontsize=7, color="black")

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=7.5)
        ax.set_title(PATTERN_TITLES.get(pattern, pattern), fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Tail Amplification (p999/p50)")

    fig.suptitle("Tail Amplification Ratio Across Systems and Load Patterns", fontsize=10)
    fig.tight_layout()

    out = FIGURES_DIR / "fig6_tail_amplification.pdf"
    return _save(fig, out)


# ── Figure 7: Recovery time after failure injection  ─────────────────────────

def fig_recovery_time(df: pd.DataFrame) -> pathlib.Path:
    sub = df[(df["load_pattern"] == "failure") & df["recovery_time_s"].notna()]
    if sub.empty:
        log.warning("No recovery_time_s data — skipping fig7")
        return FIGURES_DIR / "fig7_recovery_time.pdf"

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x_pos, labels, bar_colors, bar_hatches = [], [], [], []

    for i, sys in enumerate(SYSTEM_ORDER):
        vals = sub[sub["system"] == sys]["recovery_time_s"].dropna().values
        if len(vals) == 0:
            continue
        mean_r = vals.mean()
        std_r  = vals.std(ddof=0) if len(vals) > 1 else 0.0
        bar = ax.bar(len(x_pos), mean_r, color=SYSTEM_COLORS[sys],
                     hatch=SYSTEM_HATCHES[sys], edgecolor="white",
                     linewidth=0.5, width=0.6)
        ax.errorbar(len(x_pos), mean_r, yerr=std_r,
                    fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3)
        ax.text(len(x_pos), mean_r + std_r + 0.5, f"{mean_r:.0f}s",
                ha="center", va="bottom", fontsize=8)
        x_pos.append(len(x_pos))
        labels.append(SYSTEM_LABELS[sys])
        bar_colors.append(SYSTEM_COLORS[sys])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Recovery Time (s)")
    ax.set_title("Time to Recover from TaskManager Failure")

    # XTAL improvement annotation
    xtal_vals = sub[sub["system"] == "xtal"]["recovery_time_s"].dropna().values
    base_vals  = sub[sub["system"] == "static-default"]["recovery_time_s"].dropna().values
    if len(xtal_vals) and len(base_vals):
        speedup = base_vals.mean() / xtal_vals.mean()
        ax.annotate(f"XTAL is {speedup:.1f}× faster",
                    xy=(len(x_pos) - 1, xtal_vals.mean()),
                    xytext=(len(x_pos) - 1.8, base_vals.mean() * 0.8),
                    fontsize=8, color="#CC79A7",
                    arrowprops=dict(arrowstyle="->", color="#CC79A7", lw=1.0))

    fig.tight_layout()
    out = FIGURES_DIR / "fig7_recovery_time.pdf"
    return _save(fig, out)


# ── Figure 8: p99 improvement summary heatmap  ───────────────────────────────
# Reduction in mean p99 latency vs. Static Default (%).

def fig_improvement_heatmap(df: pd.DataFrame) -> pathlib.Path:
    patterns = [p for p in ["steady", "bursty", "failure"] if p in df["load_pattern"].unique()]
    systems  = SYSTEM_ORDER[1:]   # exclude reference system (static-default)

    baseline_mean: dict[str, float] = {}
    for pat in patterns:
        b = df[(df["load_pattern"] == pat) & (df["system"] == "static-default")]["p99_ms"]
        baseline_mean[pat] = b.mean() if not b.empty else np.nan

    matrix = np.zeros((len(systems), len(patterns)))
    for i, sys in enumerate(systems):
        for j, pat in enumerate(patterns):
            vals = df[(df["load_pattern"] == pat) & (df["system"] == sys)]["p99_ms"]
            if not vals.empty and not np.isnan(baseline_mean[pat]):
                matrix[i, j] = 100.0 * (1.0 - vals.mean() / baseline_mean[pat])

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=80, aspect="auto")

    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels([PATTERN_TITLES.get(p, p) for p in patterns], fontsize=9)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([SYSTEM_LABELS[s] for s in systems], fontsize=9)
    ax.set_title("p99 Latency Reduction vs. Static Default (%)")

    for i in range(len(systems)):
        for j in range(len(patterns)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if val > 55 else "black")

    fig.colorbar(im, ax=ax, label="Reduction (%)", shrink=0.85)
    fig.tight_layout()
    out = FIGURES_DIR / "fig8_improvement_heatmap.pdf"
    return _save(fig, out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="XTAL Paper Figure Generator")
    p.add_argument("--raw",       required=True, help="Path to eval_raw_*.csv")
    p.add_argument("--decisions", default=None,  help="Path to xtal_decisions_*.csv")
    args = p.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw)
    log.info("Loaded %d rows from %s", len(df), args.raw)

    # -- Figure 1: CDF per load pattern
    for pattern in df["load_pattern"].unique():
        fig_latency_cdf(df, pattern)

    # -- Figures 2 & 3: Time series
    fig_timeseries(df, "bursty",
                   "p99 Latency — Bursty Load (10× spike at t=60 s)",
                   "fig2_timeseries_burst.pdf")
    if "failure" in df["load_pattern"].values:
        fig_timeseries(df, "failure",
                       "p99 Latency — Failure Injection (TM killed at t=90 s)",
                       "fig3_timeseries_failure.pdf")

    # -- Figure 4: SLA violation grouped bar chart
    fig_sla_violation(df)

    # -- Figure 6: Tail amplification
    fig_tail_amplification(df)

    # -- Figure 7: Recovery time
    fig_recovery_time(df)

    # -- Figure 8: Improvement heatmap
    fig_improvement_heatmap(df)

    # -- Figure 5: SHAP (optional)
    if args.decisions:
        decisions_df = pd.read_csv(args.decisions)
        fig_shap_importance(decisions_df)

    log.info("All figures saved to %s/", FIGURES_DIR)


if __name__ == "__main__":
    main()
