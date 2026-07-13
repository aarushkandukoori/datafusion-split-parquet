#!/usr/bin/env python3
"""
Generate a TPC-H or TPC-DS dataset with DuckDB and vertically partition every
table into the layouts that src/main.rs expects under data/<benchmark>/:

    {table}_a0.parquet                      test "a" -> control (all columns, 1 file)
    {table}_b0.parquet, {table}_b1.parquet  test "b" -> 2-way column split
    {table}_c0.parquet, _c1, _c2            test "c" -> 3-way column split

Usage:
    python3 partition.py [tpch|tpcds] [scale_factor]     (defaults: tpch 1)

Correctness contract (see src/zip.rs::ZipStream::batch_zip): the zip is
*positional* -- row i of part 0 is glued to row i of part 1. If the parts don't
share identical row order and identical row-group boundaries, the reader
silently yields empty/wrong rows. We guarantee alignment by writing every part
of a table from the SAME `ORDER BY <primary key>` with the SAME ROW_GROUP_SIZE,
and we verify by reconstructing each table and comparing to the control.
"""

import os
import sys
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.abspath(__file__))
ROW_GROUP_SIZE = 122880          # DuckDB default; a multiple of DataFusion's 8192 batch size
SAMPLE_THRESHOLD = 1_500_000     # tables larger than this are verified on their first row group

# Per-benchmark config. `keys` maps table -> unique ordering key (primary key) so
# every part of a table lands rows in the same deterministic order. Any table not
# listed, or whose key columns are missing, falls back to ORDER BY all columns.
BENCHMARKS = {
    "tpch": {
        "generate": "CALL dbgen(sf={sf})",
        "keys": {
            "lineitem": ["l_orderkey", "l_linenumber"],
            "orders":   ["o_orderkey"],
            "customer": ["c_custkey"],
            "part":     ["p_partkey"],
            "supplier": ["s_suppkey"],
            "partsupp": ["ps_partkey", "ps_suppkey"],
            "nation":   ["n_nationkey"],
            "region":   ["r_regionkey"],
        },
    },
    "tpcds": {
        "generate": "CALL dsdgen(sf={sf})",
        "keys": {
            "call_center":            ["cc_call_center_sk"],
            "catalog_page":           ["cp_catalog_page_sk"],
            "catalog_returns":        ["cr_item_sk", "cr_order_number"],
            "catalog_sales":          ["cs_item_sk", "cs_order_number"],
            "customer":               ["c_customer_sk"],
            "customer_address":       ["ca_address_sk"],
            "customer_demographics":  ["cd_demo_sk"],
            "date_dim":               ["d_date_sk"],
            "household_demographics": ["hd_demo_sk"],
            "income_band":            ["ib_income_band_sk"],
            "inventory":              ["inv_date_sk", "inv_item_sk", "inv_warehouse_sk"],
            "item":                   ["i_item_sk"],
            "promotion":              ["p_promo_sk"],
            "reason":                 ["r_reason_sk"],
            "ship_mode":              ["sm_ship_mode_sk"],
            "store":                  ["s_store_sk"],
            "store_returns":          ["sr_item_sk", "sr_ticket_number"],
            "store_sales":            ["ss_item_sk", "ss_ticket_number"],
            "time_dim":               ["t_time_sk"],
            "warehouse":              ["w_warehouse_sk"],
            "web_page":               ["wp_web_page_sk"],
            "web_returns":            ["wr_item_sk", "wr_order_number"],
            "web_sales":              ["ws_item_sk", "ws_order_number"],
            "web_site":               ["web_site_sk"],
        },
    },
}


def contiguous_split(items, n):
    """Split a list into n contiguous chunks; first (len % n) chunks get one
    extra element. Matches the chunking semantics in src/zip.rs::split."""
    k, m = divmod(len(items), n)
    out, start = [], 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        out.append(items[start:start + size])
        start += size
    return out


def order_clause(cols, key):
    """Deterministic total order: use the primary key if all its columns exist,
    otherwise fall back to ordering by every column (still deterministic)."""
    if key and all(k in cols for k in key):
        return ", ".join(f'"{c}"' for c in key)
    return ", ".join(f'"{c}"' for c in cols)


def write_part(con, table, cols, order_by, path):
    collist = ", ".join(f'"{c}"' for c in cols)
    con.execute(
        f"COPY (SELECT {collist} FROM {table} ORDER BY {order_by}) "
        f"TO '{path}' (FORMAT parquet, ROW_GROUP_SIZE {ROW_GROUP_SIZE})"
    )


def generate(benchmark, sf, data_dir):
    cfg = BENCHMARKS[benchmark]
    os.makedirs(data_dir, exist_ok=True)
    con = duckdb.connect()
    ext = "tpch" if benchmark == "tpch" else "tpcds"
    print(f"Generating {benchmark.upper()} sf={sf} ...", flush=True)
    con.execute(f"INSTALL {ext}; LOAD {ext};")
    con.execute(cfg["generate"].format(sf=sf))

    tables = sorted(cfg["keys"].keys())
    meta = {}
    for t in tables:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info('{t}')").fetchall()]
        order_by = order_clause(cols, cfg["keys"].get(t))
        nrows = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {nrows:,} rows, {len(cols)} cols", flush=True)

        def p(name):
            return os.path.join(data_dir, name + ".parquet")

        write_part(con, t, cols, order_by, p(f"{t}_a0"))
        b = contiguous_split(cols, 2)
        write_part(con, t, b[0], order_by, p(f"{t}_b0"))
        write_part(con, t, b[1], order_by, p(f"{t}_b1"))
        c = contiguous_split(cols, 3)
        for i, grp in enumerate(c):
            write_part(con, t, grp, order_by, p(f"{t}_c{i}"))

        meta[t] = {"nrows": nrows}
    con.close()
    return tables, meta


# ---- verification -----------------------------------------------------------

def reconstruct_equals(data_dir, table, part_names, sample):
    """Positionally glue the parts together and compare to the a0 control,
    exactly as ZipStream does (concat columns of row-aligned batches)."""
    def path(name):
        return os.path.join(data_dir, name + ".parquet")

    if sample:
        a0 = pq.ParquetFile(path(f"{table}_a0")).read_row_group(0)
        parts = [pq.ParquetFile(path(p)).read_row_group(0) for p in part_names]
    else:
        a0 = pq.read_table(path(f"{table}_a0"))
        parts = [pq.read_table(path(p)) for p in part_names]

    nrows = parts[0].num_rows
    cols, order = {}, []
    for pt in parts:
        if pt.num_rows != nrows:
            print(f"    FAIL {table}: part row counts differ during zip")
            return False
        for name in pt.column_names:
            cols[name] = pt[name]
            order.append(name)
    recon = pa.table(cols)
    a0r = a0.select(order)
    for name in order:
        if not a0r[name].combine_chunks().equals(recon[name].combine_chunks()):
            print(f"    FAIL {table}: column '{name}' mismatch after zip")
            return False
    return True


def verify(data_dir, tables, meta):
    print("\nVerifying positional reconstruction (zip) == control ...", flush=True)
    all_ok = True
    for t in tables:
        sample = meta[t]["nrows"] > SAMPLE_THRESHOLD
        ok = (reconstruct_equals(data_dir, t, [f"{t}_b0", f"{t}_b1"], sample)
              and reconstruct_equals(data_dir, t, [f"{t}_c0", f"{t}_c1", f"{t}_c2"], sample))
        print(f"  [{'ok ' if ok else 'FAIL'}] {t}{' (sampled rg0)' if sample else ''}", flush=True)
        all_ok = all_ok and ok
    return all_ok


if __name__ == "__main__":
    benchmark = sys.argv[1] if len(sys.argv) > 1 else "tpch"
    sf = sys.argv[2] if len(sys.argv) > 2 else "1"
    if benchmark not in BENCHMARKS:
        sys.exit(f"Unknown benchmark '{benchmark}'; choose from {list(BENCHMARKS)}")
    data_dir = os.path.join(REPO, "data", benchmark)

    tables, meta = generate(benchmark, sf, data_dir)
    ok = verify(data_dir, tables, meta)

    files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    total_mb = sum(os.path.getsize(os.path.join(data_dir, f)) for f in files) / 1e6
    print(f"\nWrote {len(files)} parquet files, {total_mb:.0f} MB total, in {data_dir}")
    if not ok:
        print("VERIFICATION FAILED -- do not trust benchmark results.")
        sys.exit(1)
    print("All tables verified: zipped parts reconstruct the control exactly.")
