#!/usr/bin/env python3
"""
Turn the benchmark output (test-data.csv from src/main.rs) into a clustered bar
chart comparing, per TPC-H query, the control (single parquet file) against the
vertically-partitioned + zipped variants.

main.rs writes columns: test, query, trial, runtime(ms), with tests:
    a = control (1 file)   b = 2-way column split   c = 3-way column split
Each (test, query) is run 3 trials; we summarize with the median.
"""

import os
import glob
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
BENCH = sys.argv[1] if len(sys.argv) > 1 else "tpch"
CSV = os.path.join(REPO, f"test-data-{BENCH}.csv")
OUT = os.path.join(REPO, f"{BENCH}_partition_runtime.png")

LABELS = {"a": "Control (1 file)", "b": "2-way split (zipped)", "c": "3-way split (zipped)"}
COLORS = {"a": "#4C72B0", "b": "#DD8452", "c": "#55A868"}
COLS = ["test", "query", "trial", "runtime"]


def load(path):
    """DataFusion may write a single file or a directory of part-*.csv, with or
    without a header. Handle all cases."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
    else:
        files = [path]
    if not files:
        sys.exit(f"No CSV found at {path} -- run the benchmark first (cargo run --release).")

    frames = []
    for f in files:
        head = pd.read_csv(f, nrows=0)
        has_header = "test" in [c.strip() for c in head.columns]
        frames.append(pd.read_csv(f) if has_header else pd.read_csv(f, header=None, names=COLS))
    df = pd.concat(frames, ignore_index=True)
    df["query"] = df["query"].astype(int)
    df["runtime"] = df["runtime"].astype(float)
    return df


def main():
    df = load(CSV)
    med = df.groupby(["test", "query"])["runtime"].median().reset_index()
    tests = [t for t in ["a", "b", "c"] if t in med["test"].unique()]
    queries = sorted(med["query"].unique())

    pivot = med.pivot(index="query", columns="test", values="runtime").reindex(queries)

    # ---- chart ----
    import numpy as np
    display = {"tpch": "TPC-H", "tpcds": "TPC-DS"}.get(BENCH, BENCH.upper())
    # With many queries a single row is unreadable, so split across rows of <=50.
    nrows = 1 if len(queries) <= 30 else (len(queries) + 49) // 50
    chunks = np.array_split(queries, nrows)
    width = 0.8 / len(tests)
    fig, axes = plt.subplots(nrows, 1, figsize=(min(20, 1 + 0.32 * max(len(c) for c in chunks)),
                                                4.2 * nrows), squeeze=False)
    for row, (ax, qs) in enumerate(zip(axes[:, 0], chunks)):
        x = np.arange(len(qs))
        for i, t in enumerate(tests):
            vals = pivot.loc[qs, t].values
            ax.bar(x + (i - (len(tests) - 1) / 2) * width, vals, width,
                   label=LABELS[t] if row == 0 else None, color=COLORS[t])
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{q}" for q in qs], rotation=90, fontsize=7)
        ax.set_ylabel("Runtime (ms, median)")
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_title(
        f"{display} sf=1 in DataFusion: control vs vertically-partitioned (zipped) parquet")
    axes[-1, 0].set_xlabel(f"{display} query")
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"Wrote {OUT}")

    # ---- text summary: how much did partitioning cost/save vs control? ----
    if "a" in tests:
        def geomean(s):
            s = s.dropna()
            return float(np.exp(np.log(s).mean())) if len(s) else float("nan")
        print("\nRelative runtime vs control (a), median per query; >1 = slower:")
        for t in tests:
            if t == "a":
                continue
            ratio = (pivot[t] / pivot["a"]).dropna()
            print(f"  {LABELS[t]}: geomean {geomean(ratio):.2f}x  "
                  f"(min {ratio.min():.2f}x @Q{ratio.idxmin()}, "
                  f"max {ratio.max():.2f}x @Q{ratio.idxmax()})")
            faster = (ratio < 0.98).sum()
            slower = (ratio > 1.02).sum()
            print(f"     faster on {faster} queries, slower on {slower}, "
                  f"~equal on {len(ratio) - faster - slower}")


if __name__ == "__main__":
    main()
