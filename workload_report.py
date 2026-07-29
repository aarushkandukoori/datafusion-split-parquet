#!/usr/bin/env python3
"""
Report on a workload-harness run (see the `workload` subcommand in main.rs).

    python3 workload_report.py <tag>        e.g.  tpcds-b-s3-uniform

Reads workload-events-<tag>.csv and workload-columns-<tag>.csv and produces:
  - workload_report-<tag>.png : latency over time, latency distribution,
    cumulative object-store usage, and the most-touched columns
  - hot_columns-<tag>.csv     : every column of every table with touch count,
    share of queries, compressed size, and a hot/cold classification
  - a printed per-table "2-way table" of hot vs cold columns, plus the byte
    split between them (the input for designing a hot/cold partition scheme)
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "tpcds-b-s3-uniform"
BENCH = TAG.split("-")[0]
HOT_SHARE = 0.20  # touched in >= this share of queries -> "hot"

events = pd.read_csv(os.path.join(REPO, f"workload-events-{TAG}.csv"))
cols = pd.read_csv(os.path.join(REPO, f"workload-columns-{TAG}.csv"))
n_queries = len(events)

# ---- full column inventory (touched or not) from the control parquet files --
inventory = []
for path in sorted(glob.glob(os.path.join(REPO, "data", BENCH, "*_a0.parquet"))):
    table = os.path.basename(path).replace("_a0.parquet", "")
    md = pq.ParquetFile(path).metadata
    for i in range(md.num_columns):
        size = sum(
            md.row_group(rg).column(i).total_compressed_size
            for rg in range(md.num_row_groups)
        )
        name = md.schema.column(i).name
        inventory.append({"table": table, "column": name, "col_bytes": size})
inv = pd.DataFrame(inventory)

touches = (
    cols.groupby(["table", "column"])["query_nr"]
    .count()
    .rename("touches")
    .reset_index()
)
hot = inv.merge(touches, on=["table", "column"], how="left").fillna({"touches": 0})
hot["touches"] = hot["touches"].astype(int)
hot["share"] = hot["touches"] / n_queries
hot["class"] = np.where(hot["share"] >= HOT_SHARE, "hot", "cold")
hot = hot.sort_values(["touches", "col_bytes"], ascending=False)
hot.to_csv(os.path.join(REPO, f"hot_columns-{TAG}.csv"), index=False)

# ---- charts -----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
t = events["t_offset_ms"] / 1e3

ax = axes[0, 0]
ax.plot(t, events["latency_ms"], marker="o", ms=4, lw=1, color="#4C72B0")
ax.set_xlabel("time (s)")
ax.set_ylabel("query latency (ms)")
ax.set_title("Latency over the query stream")
ax.grid(alpha=0.3)

ax = axes[0, 1]
lat = np.sort(events.loc[events["ok"], "latency_ms"].values)
ax.plot(lat, np.arange(1, len(lat) + 1) / len(lat), color="#4C72B0")
for p in (0.5, 0.95):
    ax.axhline(p, color="#999", lw=0.6, ls="--")
ax.set_xlabel("latency (ms)")
ax.set_ylabel("fraction of queries")
ax.set_title(
    f"Latency CDF  (p50={np.percentile(lat, 50):.0f}ms, p95={np.percentile(lat, 95):.0f}ms)"
)
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(t, events["bytes_read"].cumsum() / 1048576.0, color="#DD8452", label="MiB read")
ax.set_xlabel("time (s)")
ax.set_ylabel("cumulative MiB read", color="#DD8452")
ax2 = ax.twinx()
ax2.plot(t, events["get_requests"].cumsum(), color="#55A868", label="GET requests")
ax2.set_ylabel("cumulative GET requests", color="#55A868")
ax.set_title("Object-store usage over the stream")
ax.grid(alpha=0.3)

ax = axes[1, 1]
top = hot.head(20).iloc[::-1]
labels = top["table"] + "." + top["column"]
colors = ["#DD8452" if c == "hot" else "#4C72B0" for c in top["class"]]
ax.barh(labels, top["touches"], color=colors)
ax.set_xlabel(f"queries touching the column (of {n_queries})")
ax.set_title("Most-touched columns")
ax.tick_params(axis="y", labelsize=7)
ax.grid(axis="x", alpha=0.3)

fig.suptitle(f"Workload run {TAG}: {n_queries} queries", fontsize=13)
fig.tight_layout()
out = os.path.join(REPO, f"workload_report-{TAG}.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
print(f"wrote hot_columns-{TAG}.csv")

# ---- the 2-way table: hot vs cold, per table and overall --------------------
print(f"\nHot = touched in >= {HOT_SHARE:.0%} of the {n_queries} queries\n")
total_hot = hot.loc[hot["class"] == "hot", "col_bytes"].sum()
total_all = hot["col_bytes"].sum()
print(
    f"OVERALL: {len(hot[hot['class'] == 'hot'])} hot columns hold "
    f"{total_hot / 1048576.0:.0f} MiB = {total_hot / total_all:.0%} of all table bytes; "
    f"{len(hot[hot['class'] == 'cold'])} cold columns hold the other {1 - total_hot / total_all:.0%}\n"
)
for table, grp in hot.groupby("table"):
    h = grp[grp["class"] == "hot"]
    c = grp[grp["class"] == "cold"]
    if h.empty:
        continue
    hb, tb = h["col_bytes"].sum(), grp["col_bytes"].sum()
    print(f"{table}  ({len(h)} hot / {len(c)} cold, hot = {hb / tb:.0%} of bytes)")
    for _, r in h.iterrows():
        print(f"   HOT  {r['touches']:>3}x {r['share']:>5.0%}  {r['column']}")
