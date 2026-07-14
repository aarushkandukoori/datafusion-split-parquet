#!/usr/bin/env python3
"""
A summary view of a partition benchmark (complements the per-query bar chart
from plot.py). Reads test-data-<bench>.csv and produces two panels:

  Left : per-query runtime of the zipped variants relative to the single-file
         control, sorted -- the whole distribution in one glance (below 1.0 =
         partitioning is *faster*; the tail on the right is where it hurts).
  Right: control vs zipped absolute runtime (log-log) with a y=x line, so the
         magnitudes and the outliers are both visible.

Usage: python3 summary.py [tpch|tpcds]
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
BENCH = sys.argv[1] if len(sys.argv) > 1 else "tpch"
DISPLAY = {"tpch": "TPC-H", "tpcds": "TPC-DS"}.get(BENCH, BENCH.upper())
CSV = os.path.join(REPO, f"test-data-{BENCH}.csv")
OUT = os.path.join(REPO, f"{BENCH}_summary.png")
COLORS = {"b": "#DD8452", "c": "#55A868"}
LABEL = {"b": "2-way split", "c": "3-way split"}


def geomean(s):
    s = s.dropna()
    return float(np.exp(np.log(s).mean())) if len(s) else float("nan")


def main():
    df = pd.read_csv(CSV)
    med = (df.groupby(["test", "query"])["runtime"].median()
             .unstack("test"))                       # rows=query, cols=a/b/c
    variants = [v for v in ["b", "c"] if v in med.columns]

    ratio = pd.DataFrame({v: med[v] / med["a"] for v in variants})
    order = ratio["b"].sort_values().index          # sort queries by 2-way ratio

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- Panel L: sorted relative runtime -------------------------------
    x = np.arange(len(order))
    for v in variants:
        axL.scatter(x, ratio.loc[order, v].values, s=22, color=COLORS[v],
                    label=f"{LABEL[v]} (geomean {geomean(ratio[v]):.2f}x)")
    axL.axhline(1.0, color="#444", lw=1)
    axL.axhspan(axL.get_ylim()[0], 1.0, color="#55A868", alpha=0.06)
    axL.set_yscale("log")
    axL.set_yticks([0.7, 1.0, 1.5, 2, 3, 5, 8])
    axL.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axL.set_xlabel(f"{DISPLAY} queries, sorted by 2-way slowdown")
    axL.set_ylabel("runtime vs single-file control  (x, log scale)")
    axL.set_title("Below the line = partitioning is faster")
    # annotate the worst outlier
    worst_q = ratio["b"].idxmax()
    worst_rank = list(order).index(worst_q)
    axL.annotate(f"Q{worst_q}  ({ratio.loc[worst_q, 'b']:.1f}x)",
                 xy=(worst_rank, ratio.loc[worst_q, "b"]),
                 xytext=(worst_rank - len(x) * 0.28, ratio.loc[worst_q, "b"] * 0.85),
                 arrowprops=dict(arrowstyle="->", color="#333"), fontsize=9)
    axL.legend(loc="upper left")
    axL.grid(axis="y", alpha=0.3)

    # ---- Panel R: control vs zipped, log-log ----------------------------
    lo = max(1, med[["a", "b"]].min().min() * 0.7)
    hi = med[["a"] + variants].max().max() * 1.4
    axR.plot([lo, hi], [lo, hi], color="#444", lw=1, label="equal (y = x)")
    for v in variants:
        axR.scatter(med["a"], med[v], s=22, color=COLORS[v], alpha=0.8, label=LABEL[v])
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlim(lo, hi); axR.set_ylim(lo, hi)
    axR.set_xlabel("control runtime (ms) -- single file")
    axR.set_ylabel("zipped runtime (ms)")
    axR.set_title("Above the line = slower when partitioned")
    axR.annotate(f"Q{worst_q}", xy=(med.loc[worst_q, "a"], med.loc[worst_q, "b"]),
                 xytext=(med.loc[worst_q, "a"] * 1.4, med.loc[worst_q, "b"] * 0.75),
                 arrowprops=dict(arrowstyle="->", color="#333"), fontsize=9)
    axR.legend(loc="upper left")
    axR.grid(which="both", alpha=0.25)

    nq = len(ratio)
    faster = int((ratio["b"] < 0.98).sum())
    slower = int((ratio["b"] > 1.02).sum())
    fig.suptitle(
        f"{DISPLAY} sf=1 in DataFusion: vertical partitioning costs ~{geomean(ratio['b']):.2f}x on average "
        f"({nq} queries; {faster} faster, {slower} slower, {nq - faster - slower} within 2%)",
        fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
