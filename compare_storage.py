#!/usr/bin/env python3
"""
Compare local-disk vs S3 (MinIO) benchmark runs produced by:
    STORAGE=local cargo run --release -- <bench>
    STORAGE=s3    cargo run --release -- <bench>
which write test-data-<bench>-local.csv and test-data-<bench>-s3.csv.

Reports, per storage variant (a=control, b=2-way, c=3-way), the median-of-trials
runtime summarized as a geomean, and the S3/local slowdown. Usage:
    python3 compare_storage.py [tpch|tpcds]
"""
import os
import sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
BENCH = sys.argv[1] if len(sys.argv) > 1 else "tpch"
LABEL = {"a": "control (1 file)", "b": "2-way split", "c": "3-way split"}


def geomean(s):
    s = s.dropna()
    s = s[s > 0]
    return float(np.exp(np.log(s).mean())) if len(s) else float("nan")


def load(storage):
    path = os.path.join(REPO, f"test-data-{BENCH}-{storage}.csv")
    if not os.path.exists(path):
        sys.exit(f"missing {path} -- run STORAGE={storage} ... {BENCH} first")
    df = pd.read_csv(path)
    # median runtime per (test, query)
    return df.groupby(["test", "query"])["runtime"].median().rename(storage)


def main():
    local = load("local")
    s3 = load("s3")
    m = pd.concat([local, s3], axis=1).reset_index()

    print(f"\n{BENCH.upper()} sf=1  ---  local disk vs S3 (MinIO on localhost)\n")
    print(f"{'variant':<18}{'local (ms)':>12}{'s3 (ms)':>12}{'s3/local':>11}")
    print("-" * 53)
    for t in ["a", "b", "c"]:
        sub = m[m["test"] == t]
        if sub.empty:
            continue
        gl, gs = geomean(sub["local"]), geomean(sub["s3"])
        print(f"{LABEL[t]:<18}{gl:>12.1f}{gs:>12.1f}{gs / gl:>10.2f}x")

    # overall totals (sum of per-query medians) -- a "whole workload" view
    print("-" * 53)
    tl, ts = m["local"].sum(), m["s3"].sum()
    print(f"{'TOTAL (sum ms)':<18}{tl:>12.0f}{ts:>12.0f}{ts / tl:>10.2f}x")

    # does partitioning cost *more* on S3? (extra files = extra requests)
    print("\nPartitioning overhead vs its own control, per storage:")
    for storage, col in [("local", "local"), ("s3", "s3")]:
        piv = m.pivot(index="query", columns="test", values=col)
        for t in ["b", "c"]:
            if t in piv and "a" in piv:
                r = (piv[t] / piv["a"]).replace([np.inf, -np.inf], np.nan)
                print(f"  {storage:<6} {LABEL[t]:<14} geomean {geomean(r):.2f}x vs control")


if __name__ == "__main__":
    main()
