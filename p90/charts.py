"""Notebook charts (matplotlib). One axis per chart, legend for 2+ series, quiet grid."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
MODEL = "#2a78d6"      # series 1
BOOKIE = "#eb6834"     # series 2
TARGET = "#52514e"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def _pct_axis(ax, lo, hi, step=0.05):
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(lo, hi + 1e-9, step))
    ax.set_yticklabels([f"{t:.0%}" for t in ax.get_yticks()])
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)



def season_accuracy(by_season, title="Accuracy per season - every match counted"):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(by_season))
    w = 0.36
    lo = 0.40
    hi = max(0.62, float(np.ceil(max(by_season["accuracy"].max(), by_season["accuracy_bookie"].max()) * 20) / 20))
    _pct_axis(ax, lo, hi)
    b1 = ax.bar(x - w / 2 - 0.01, by_season["accuracy"], w, color=MODEL, label="Model")
    b2 = ax.bar(x + w / 2 + 0.01, by_season["accuracy_bookie"], w, color=BOOKIE, label="Bookmaker favourite")
    for xi, v in zip(x, by_season["accuracy"]):
        ax.text(xi - w / 2 - 0.01, v + 0.004, f"{v:.1%}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(by_season["fold"])
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def confidence(curve, conf=0.50, strong=0.65):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(curve["threshold"], curve["accuracy"], color=MODEL, lw=2, marker="o", ms=6)
    a1.axhline(0.60, color=TARGET, lw=1, ls="--")
    a1.text(curve["threshold"].min(), 0.603, "60%", color=TARGET, fontsize=8, va="bottom")
    for t in (conf, strong):
        r = curve.iloc[(curve["threshold"] - t).abs().argmin()]
        a1.annotate(f"{r['accuracy']:.1%}", (r["threshold"], r["accuracy"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=INK)
    a1.set_ylim(0.40, 1.0)
    a1.set_yticks(np.arange(0.4, 1.01, 0.1))
    a1.set_yticklabels([f"{t:.0%}" for t in a1.get_yticks()])
    a1.set_xticks(curve["threshold"])
    a1.set_xticklabels([f"{t:.0%}" for t in curve["threshold"]], fontsize=8)
    a1.set_xlabel("Only picks at least this sure")
    a1.set_title("Pick accuracy rises with confidence")
    a2.bar(curve["threshold"], curve["share"], width=0.035, color=MODEL)
    a2.set_ylim(0, 1)
    a2.set_yticks(np.arange(0, 1.01, 0.2))
    a2.set_yticklabels([f"{t:.0%}" for t in a2.get_yticks()])
    a2.set_xticks(curve["threshold"])
    a2.set_xticklabels([f"{t:.0%}" for t in curve["threshold"]], fontsize=8)
    a2.set_xlabel("Only picks at least this sure")
    a2.set_title("...but fewer matches qualify")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def calibration(cal):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, zorder=1)
    ax.scatter(cal["predicted"], cal["actual"], s=np.clip(cal["n"] / cal["n"].max() * 260, 40, 260),
               color=MODEL, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = np.arange(0, 1.01, 0.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:.0%}" for t in ticks])
    ax.set_yticklabels([f"{t:.0%}" for t in ticks])
    ax.set_xlabel("Model said")
    ax.set_ylabel("Really happened")
    ax.set_title("Calibration: 70% means 70%")
    fig.tight_layout()
    return fig


def league_accuracy(by_league, labels):
    d = by_league.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.26 * len(d))))
    y = np.arange(len(d))
    for yi, a, b in zip(y, d["accuracy"], d["accuracy_bookie"]):
        ax.plot([min(a, b), max(a, b)], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(d["accuracy_bookie"], y, s=46, color=BOOKIE, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Bookmaker favourite")
    ax.scatter(d["accuracy"], y, s=46, color=MODEL, edgecolor=SURFACE, linewidth=2, zorder=4, label="Model")
    ax.set_yticks(y)
    ax.set_yticklabels([labels(c) for c in d["league"]], fontsize=8)
    lo = np.floor(min(d["accuracy"].min(), d["accuracy_bookie"].min()) * 20) / 20
    hi = np.ceil(max(d["accuracy"].max(), d["accuracy_bookie"].max()) * 20) / 20
    ax.set_xlim(lo, hi)
    xt = np.arange(lo, hi + 1e-9, 0.05)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t:.0%}" for t in xt])
    ax.grid(axis="y", visible=False)
    ax.set_title("Accuracy by league - every match counted", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    fig.tight_layout()
    return fig


def acca_by_season(acc_season):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(acc_season))
    _pct_axis(ax, 0.0, 0.30)
    ax.bar(x, acc_season["hit"], 0.55, color=MODEL, label="Slips won")
    be = acc_season["break_even"].mean()
    ax.axhline(be, color=TARGET, lw=1, ls="--", label=f"Break-even ({be:.0%})")
    for xi, v, n in zip(x, acc_season["hit"], acc_season["slips"]):
        ax.text(xi, 0.012, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n{n} slips" for f, n in zip(acc_season["fold"], acc_season["slips"])], fontsize=8)
    ax.set_title("Acca slips won per season (slips of ~4.5-7 odds)", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    fig.tight_layout()
    return fig
