"""
plot_results.py — Generates paper-quality figures from evaluation CSVs.

Figures produced (saved to experiments/figures/):
  fig1_latency_cdf.pdf      : CDF of p99 latency for all systems × load patterns
  fig2_timeseries_burst.pdf : p99 time series during bursty load
  fig3_timeseries_failure.pdf: p99 time series during failure injection
  fig4_sla_violation.pdf    : SLA violation rate bar chart
  fig5_shap_importance.pdf  : Mean SHAP feature importances from XTAL decision log
  fig6_tail_amplification.pdf: Tail amplification ratio (p999/p50)

Usage:
    python -m experiments.plot_results --raw experiments/results/eval_raw_<ts>.csv \
           --decisions logs/xtal_decisions_<ts>.csv
"""

from __future__ import annotations

import argparse
import pathlib
import logging

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

log = logging.getLogger(__name__)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "pdf.fonttype": 42,         # embeds fonts for ACM/IEEE submission
    "ps.fonttype": 42,
})

FIGURES_DIR = pathlib.Path("experiments/figures")

SYSTEM_COLORS = {
    "static-default":    "#e74c3c",
    "static-hand-tuned": "#f39c12",
    "reactive-scaler":   "#3498db",
    "xtal":              "#27ae60",
}
SYSTEM_LABELS = {
    "static-default":    "Static Default",
    "static-hand-tuned": "Static Hand-Tuned",
    "reactive-scaler":   "Reactive Scaler",
    "xtal":              "XTAL (ours)",
}
SYSTEM_ORDER = list(SYSTEM_COLORS.keys())


# ── Figure 1: CDF of p99 latency ─────────────────────────────────────────────

def fig_latency_cdf(df: pd.DataFrame, pattern: str = "bursty") -> pathlib.Path:
    sub = df[df["load_pattern"] == pattern]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    for sys in SYSTEM_ORDER:
        vals = sub[sub["system"] == sys]["p99_ms"].dropna()
        if vals.empty:
            continue
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf,
                color=SYSTEM_COLORS[sys],
                label=SYSTEM_LABELS[sys],
                linewidth=1.8)

    ax.axvline(500, color="black", linestyle="--", linewidth=1, label="SLA (500ms)")
    ax.set_xlabel("p99 Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title(f"p99 Latency CDF — {pattern.capitalize()} Load")
    ax.legend(loc="lower right")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / f"fig1_latency_cdf_{pattern}.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Saved %s", out)
    return out


# ── Figure 2/3: Time series ───────────────────────────────────────────────────

def fig_timeseries(df: pd.DataFrame, pattern: str, title: str, out_name: str) -> pathlib.Path:
    sub = df[df["load_pattern"] == pattern]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    for sys in SYSTEM_ORDER:
        s = sub[sub["system"] == sys].sort_values("elapsed_s")
        if s.empty:
            continue
        ax.plot(s["elapsed_s"], s["p99_ms"],
                color=SYSTEM_COLORS[sys],
                label=SYSTEM_LABELS[sys],
                linewidth=1.5, alpha=0.9)

    ax.axhline(500, color="black", linestyle="--", linewidth=1, label="SLA")

    if pattern == "bursty":
        ax.axvspan(60, 70, alpha=0.08, color="red", label="Burst (10×)")
    elif pattern == "failure":
        ax.axvspan(90, 90 + 30, alpha=0.08, color="orange", label="TM failure")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("p99 Latency (ms)")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / out_name
    fig.savefig(out)
    plt.close(fig)
    log.info("Saved %s", out)
    return out


# ── Figure 4: SLA violation rate bar chart ────────────────────────────────────

def fig_sla_violation(df: pd.DataFrame) -> pathlib.Path:
    patterns = df["load_pattern"].unique()
    n_patterns = len(patterns)
    fig, axes = plt.subplots(1, n_patterns, figsize=(3.5 * n_patterns, 3.2), sharey=True)
    if n_patterns == 1:
        axes = [axes]

    for ax, pattern in zip(axes, patterns):
        sub = df[df["load_pattern"] == pattern]
        viol_rates = []
        labels = []
        colors = []
        for sys in SYSTEM_ORDER:
            vals = sub[sub["system"] == sys]["sla_violated"]
            if vals.empty:
                continue
            viol_rates.append(vals.mean() * 100)
            labels.append(SYSTEM_LABELS[sys])
            colors.append(SYSTEM_COLORS[sys])

        bars = ax.bar(range(len(labels)), viol_rates, color=colors, width=0.6, edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{pattern.capitalize()}")
        ax.set_ylabel("SLA Violation Rate (%)" if ax == axes[0] else "")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, rate in zip(bars, viol_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{rate:.1f}%", ha="center", va="bottom", fontsize=7)

    fig.suptitle("SLA Violation Rate by System and Load Pattern", fontsize=11)
    fig.tight_layout()
    out = FIGURES_DIR / "fig4_sla_violation.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Saved %s", out)
    return out


# ── Figure 5: SHAP feature importance ────────────────────────────────────────

def fig_shap_importance(decisions_df: pd.DataFrame) -> pathlib.Path:
    shap_data = {}
    for i in range(1, 4):
        feat_col = f"shap_feat_{i}"
        val_col = f"shap_val_{i}"
        if feat_col not in decisions_df.columns:
            continue
        for _, row in decisions_df.iterrows():
            feat = row.get(feat_col)
            val = row.get(val_col)
            if pd.notna(feat) and pd.notna(val):
                shap_data.setdefault(feat, []).append(abs(float(val)))

    if not shap_data:
        log.warning("No SHAP data found in decisions log")
        return FIGURES_DIR / "fig5_shap_importance.pdf"

    features = sorted(shap_data.keys(), key=lambda f: np.mean(shap_data[f]), reverse=True)
    means = [np.mean(shap_data[f]) for f in features]
    stds = [np.std(shap_data[f]) for f in features]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    y_pos = range(len(features))
    ax.barh(y_pos, means, xerr=stds, color="#27ae60", alpha=0.85, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("XTAL: Feature Importance for Control Decisions")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "fig5_shap_importance.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Saved %s", out)
    return out


# ── Figure 6: Tail amplification ─────────────────────────────────────────────

def fig_tail_amplification(df: pd.DataFrame) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    data = []
    labels = []
    colors = []

    for sys in SYSTEM_ORDER:
        vals = df[df["system"] == sys]["tail_amplification"].dropna()
        if vals.empty:
            continue
        data.append(vals.values)
        labels.append(SYSTEM_LABELS[sys])
        colors.append(SYSTEM_COLORS[sys])

    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Tail Amplification (p999/p50)")
    ax.set_title("Tail Amplification Ratio Across Systems")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "fig6_tail_amplification.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Saved %s", out)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="XTAL Paper Figure Generator")
    p.add_argument("--raw", required=True, help="Path to eval_raw_*.csv")
    p.add_argument("--decisions", default=None, help="Path to xtal_decisions_*.csv")
    args = p.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw)
    log.info("Loaded %d rows from %s", len(df), args.raw)

    for pattern in df["load_pattern"].unique():
        fig_latency_cdf(df, pattern)

    fig_timeseries(df, "bursty", "p99 Latency — Bursty Load (10× spike at t=60s)", "fig2_timeseries_burst.pdf")
    if "failure" in df["load_pattern"].values:
        fig_timeseries(df, "failure", "p99 Latency — Failure Injection (TM killed at t=90s)", "fig3_timeseries_failure.pdf")

    fig_sla_violation(df)
    fig_tail_amplification(df)

    if args.decisions:
        decisions_df = pd.read_csv(args.decisions)
        fig_shap_importance(decisions_df)

    log.info("All figures saved to %s/", FIGURES_DIR)


if __name__ == "__main__":
    main()
