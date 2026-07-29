#!/usr/bin/env python3
"""Chart local-disk vs S3 (MinIO) benchmark runs. Usage: python3 plot_storage.py [tpch|tpcds]"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
BENCH = sys.argv[1] if len(sys.argv) > 1 else "tpch"
DISP = {"tpch": "TPC-H", "tpcds": "TPC-DS"}.get(BENCH, BENCH.upper())
VARIANTS = {"a": "control\n(1 file)", "b": "2-way\nsplit", "c": "3-way\nsplit"}
C_LOCAL, C_S3 = "#4C72B0", "#DD8452"


def geomean(s):
    s = s.dropna(); s = s[s > 0]
    return float(np.exp(np.log(s).mean())) if len(s) else float("nan")


def med(storage):
    df = pd.read_csv(os.path.join(REPO, f"test-data-{BENCH}-{storage}.csv"))
    return df.groupby(["test", "query"])["runtime"].median()


local, s3 = med("local"), med("s3")
tests = [t for t in ["a", "b", "c"] if t in local.index.get_level_values("test")]

gl = {t: geomean(local.xs(t, level="test")) for t in tests}
gs = {t: geomean(s3.xs(t, level="test")) for t in tests}
# partitioning overhead vs that storage's OWN control
ov = {}
for name, series in [("local", local), ("s3", s3)]:
    piv = series.unstack("test")
    ov[name] = {t: geomean(piv[t] / piv["a"]) for t in tests if t != "a"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
x = np.arange(len(tests)); w = 0.38

# Panel 1: absolute runtime, local vs s3
b1 = ax1.bar(x - w/2, [gl[t] for t in tests], w, label="local disk", color=C_LOCAL)
b2 = ax1.bar(x + w/2, [gs[t] for t in tests], w, label="S3 (MinIO)", color=C_S3)
for t, xi in zip(tests, x):
    ax1.text(xi, max(gl[t], gs[t]) + 1.2, f"{gs[t]/gl[t]:.2f}x", ha="center", fontsize=10, color=C_S3, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels([VARIANTS[t] for t in tests])
ax1.set_ylabel("runtime (ms, geomean of queries)")
ax1.set_title("Reading over S3 vs local disk")
ax1.legend(); ax1.grid(axis="y", alpha=0.3)

# Panel 2: partitioning overhead vs own control
pv = [t for t in tests if t != "a"]
x2 = np.arange(len(pv))
ax2.bar(x2 - w/2, [ov["local"][t] for t in pv], w, label="local disk", color=C_LOCAL)
ax2.bar(x2 + w/2, [ov["s3"][t] for t in pv], w, label="S3 (MinIO)", color=C_S3)
ax2.axhline(1.0, color="#444", lw=1)
for t, xi in zip(pv, x2):
    ax2.text(xi - w/2, ov["local"][t] + 0.01, f"{ov['local'][t]:.2f}x", ha="center", fontsize=9)
    ax2.text(xi + w/2, ov["s3"][t] + 0.01, f"{ov['s3'][t]:.2f}x", ha="center", fontsize=9, color=C_S3, fontweight="bold")
ax2.set_xticks(x2); ax2.set_xticklabels([VARIANTS[t] for t in pv])
ax2.set_ylabel("runtime vs single-file control (x)")
ax2.set_title("Cost of partitioning: free on disk, not on S3")
ax2.set_ylim(0.9, max(1.15, max(ov["s3"].values()) + 0.05))
ax2.legend(); ax2.grid(axis="y", alpha=0.3)

fig.suptitle(f"{DISP} sf=1: local disk vs S3 (MinIO on localhost) — extra files = extra object requests",
             fontsize=13, y=1.00)
fig.tight_layout()
out = os.path.join(REPO, f"{BENCH}_storage_compare.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("Wrote", out)
