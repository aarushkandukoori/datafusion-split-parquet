// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use datafusion::{
    arrow::array::{Int64Array, StringArray, UInt32Array, UInt64Array},
    common::Result,
    dataframe::DataFrameWriteOptions,
};
use std::{collections::HashSet, fs, sync::Arc};

use chrono::Utc;
use datafusion::prelude::*;
use object_store::ObjectStore;
use url::Url;

pub mod zip;

pub struct TestResult {
    test: String,
    query: usize,
    trial: u32,
    runtime: i64,
}

impl TestResult {
    fn new(test: String, query: usize, trial: u32, runtime: i64) -> Self {
        Self {
            test,
            query,
            trial,
            runtime,
        }
    }
}

/// Configuration for one benchmark suite (TPC-H or TPC-DS).
struct BenchConfig {
    /// Suite name, used for logging and the output CSV filename.
    name: &'static str,
    /// Directory holding the partitioned parquet files (e.g. "data/tpch").
    data_dir: &'static str,
    /// Tables to register before running the queries.
    tables: Vec<&'static str>,
    /// (query number, SQL text) pairs to run.
    queries: Vec<(usize, String)>,
}

/// Load `q{n}.sql` for each n from `dir`. Missing files are skipped with a
/// warning rather than panicking, so a partial query set still runs.
fn load_queries(dir: &str, nums: impl Iterator<Item = usize>) -> Vec<(usize, String)> {
    nums.filter_map(|n| match fs::read_to_string(format!("{dir}/q{n}.sql")) {
        Ok(sql) => Some((n, sql)),
        Err(_) => {
            eprintln!("warning: missing query file {dir}/q{n}.sql, skipping");
            None
        }
    })
    .collect()
}

fn tpch_config() -> BenchConfig {
    BenchConfig {
        name: "tpch",
        data_dir: "data/tpch",
        tables: vec![
            "lineitem", "orders", "partsupp", "supplier", "nation", "region", "part", "customer",
        ],
        queries: load_queries("queries", 1..=22),
    }
}

fn tpcds_config() -> BenchConfig {
    BenchConfig {
        name: "tpcds",
        data_dir: "data/tpcds",
        tables: vec![
            "call_center", "catalog_page", "catalog_returns", "catalog_sales", "customer",
            "customer_address", "customer_demographics", "date_dim", "household_demographics",
            "income_band", "inventory", "item", "promotion", "reason", "ship_mode", "store",
            "store_returns", "store_sales", "time_dim", "warehouse", "web_page", "web_returns",
            "web_sales", "web_site",
        ],
        queries: load_queries("queries_tpcds", 1..=99),
    }
}

/// Register `table` into `ctx` for one storage variant:
///   "a" -> the control: a single parquet file read by DataFusion's native reader
///   "b" -> a 2-way vertical split, stitched back together by ZippedTableProvider
///   "c" -> a 3-way vertical split
async fn register_variant(
    ctx: &SessionContext,
    object_store: &Arc<dyn ObjectStore>,
    data_dir: &str,
    table: &str,
    test: &str,
) -> Result<()> {
    match test {
        "a" => {
            ctx.register_parquet(
                table,
                format!("{data_dir}/{table}_a0.parquet"),
                ParquetReadOptions::default(),
            )
            .await?;
        }
        "b" => {
            let provider = zip::ZippedTableProvider::try_new(
                Arc::clone(object_store),
                vec![
                    format!("{data_dir}/{table}_b0.parquet"),
                    format!("{data_dir}/{table}_b1.parquet"),
                ],
            )?;
            ctx.register_table(table, Arc::new(provider) as _)?;
        }
        "c" => {
            let provider = zip::ZippedTableProvider::try_new(
                Arc::clone(object_store),
                vec![
                    format!("{data_dir}/{table}_c0.parquet"),
                    format!("{data_dir}/{table}_c1.parquet"),
                    format!("{data_dir}/{table}_c2.parquet"),
                ],
            )?;
            ctx.register_table(table, Arc::new(provider) as _)?;
        }
        other => unreachable!("unknown test variant {other}"),
    }
    Ok(())
}

/// Run one (possibly multi-statement) query end to end and return its
/// wall-clock runtime in milliseconds. Statements are separated by ';'.
async fn run_query(ctx: &SessionContext, sql: &str) -> Result<i64> {
    let start = Utc::now();
    for segment in sql
        .split(';')
        .filter(|s| !s.split_whitespace().collect::<String>().is_empty())
    {
        let df = ctx.sql(segment).await?;
        df.collect().await?;
    }
    let end = Utc::now();
    Ok((end - start).num_milliseconds())
}

/// Run a benchmark suite across the three storage variants (a/b/c), `trials`
/// times each. Queries that DataFusion cannot plan or execute are logged and
/// skipped instead of aborting the whole run -- important for TPC-DS, where
/// DataFusion does not support all 99 queries.
async fn run_bench(
    cfg: &BenchConfig,
    trials: usize,
    queries: Option<Vec<usize>>,
) -> Result<Vec<TestResult>> {
    let subset = queries.map(|q| HashSet::from_iter(q));
    let mut results = Vec::new();
    let tests = ["a", "b", "c"];
    for test in tests {
        let ctx = SessionContext::new();
        // The object store reads the parquet files. Here it is the local file
        // system, but in a real system it could be S3, GCS, etc.
        let object_store: Arc<dyn ObjectStore> =
            Arc::new(object_store::local::LocalFileSystem::new());
        for table in &cfg.tables {
            register_variant(&ctx, &object_store, cfg.data_dir, table, test).await?;
        }
        // register object store provider so that urls like `file://` work
        let url = Url::try_from("file://").unwrap();
        ctx.register_object_store(&url, object_store);

        for t in 0..trials as u32 {
            for (q_num, sql) in cfg.queries.iter().filter(|(n, _)| {
                subset
                    .as_ref()
                    .map(|hs: &HashSet<usize>| hs.contains(n))
                    .unwrap_or(true)
            }) {
                println!("Starting {} Test {}, Q{} (#{})...", cfg.name, test, q_num, t + 1);
                match run_query(&ctx, sql).await {
                    Ok(runtime) => {
                        println!(
                            "Finished {} Test {}, Q{}: took {}ms",
                            cfg.name, test, q_num, runtime
                        );
                        results.push(TestResult::new(test.into(), *q_num, t, runtime));
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        let first_line = msg.lines().next().unwrap_or("");
                        println!(
                            "Skipped {} Test {}, Q{}: unsupported ({})",
                            cfg.name, test, q_num, first_line
                        );
                    }
                }
            }
        }
    }
    Ok(results)
}

// Kept as a lightweight debugging harness for selective/pruning experiments.
#[allow(dead_code)]
async fn smoke(trials: usize, queries: Option<Vec<usize>>) -> Result<Vec<TestResult>> {
    let subset_queries = queries.map(|q| HashSet::from_iter(q));
    let mut results = Vec::new();
    let table_names = [
        "lineitem", "orders", "partsupp", "supplier", "nation", "region", "part", "customer",
    ];
    let queries: Vec<String> = vec![
        "SELECT * FROM lineitem WHERE l_shipdate > '1993-11-09' AND l_shipdate < '1993-11-13'" // 0.12%
            .into(),
        "SELECT * FROM lineitem WHERE l_shipdate > '1993-11-09' AND l_shipdate < '1993-12-05'" // 1%
            .into(),
        "SELECT * FROM lineitem WHERE l_shipdate > '1993-11-09' AND l_shipdate < '1994-07-09'" // 10%
            .into(),
        "SELECT * FROM lineitem WHERE l_shipdate > '1993-11-09' AND l_shipdate < '1997-02-25'" // 50%
            .into(),
        "SELECT * FROM lineitem".into(), // everything
        // Prune row groups
        "SELECT * FROM lineitem WHERE l_orderkey = 1 AND l_shipmode = 'TRUCK'".into(),
        "SELECT l_orderkey, l_tax FROM lineitem".into(),
        "SELECT sum(ps_supplycost * ps_availqty) * 0.0001000000 FROM partsupp, supplier, nation WHERE ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'ALGERIA'"
            .into(),
        "SELECT ps_supplycost FROM partsupp, supplier WHERE ps_suppkey = s_suppkey"
            .into(),
        "SELECT * FROM orders WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > 313)".into(),
        "SELECT * FROM orders WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > 1)".into(),
    ];
    let tests = ["a", "b"];
    for test in tests {
        let ctx = SessionContext::default();
        // the object store is used to read the parquet files (in this case, it is
        // a local file system, but in a real system it could be S3, GCS, etc)
        let object_store: Arc<dyn ObjectStore> =
            Arc::new(object_store::local::LocalFileSystem::new());

        for table_name in table_names {
            // Create a custom table provider with our special index.
            if test == "a" {
                let parquet_options = ParquetReadOptions::default().parquet_pruning(true);
                ctx.register_parquet(
                    table_name,
                    format!("data/tpch/{}_a0.parquet", table_name),
                    parquet_options,
                )
                .await?;
            } else if test == "b" {
                let provider = if false {
                    Arc::new(zip::ZippedTableProvider::try_new(
                        Arc::clone(&object_store),
                        vec![format!("data/tpch/{}_a0.parquet", table_name)],
                    )?)
                } else {
                    Arc::new(zip::ZippedTableProvider::try_new(
                        Arc::clone(&object_store),
                        vec![
                            format!("data/tpch/{}_b0.parquet", table_name),
                            format!("data/tpch/{}_b1.parquet", table_name),
                        ],
                    )?)
                };
                ctx.register_table(table_name, Arc::clone(&provider) as _)?;
            } else if test == "c" {
                let provider = Arc::new(zip::ZippedTableProvider::try_new(
                    Arc::clone(&object_store),
                    vec![
                        format!("data/tpch/{}_c0.parquet", table_name),
                        format!("data/tpch/{}_c1.parquet", table_name),
                        format!("data/tpch/{}_c2.parquet", table_name),
                    ],
                )?);
                ctx.register_table(table_name, Arc::clone(&provider) as _)?;
            }
        }

        // register object store provider for urls like `file://` work
        let url = Url::try_from("file://").unwrap();
        ctx.register_object_store(&url, object_store);

        for t in 0..trials as u32 {
            for (i, q) in queries.iter().enumerate().filter(|p| {
                subset_queries
                    .as_ref()
                    .map(|hs: &HashSet<usize>| hs.contains(&(p.0 + 1)))
                    .unwrap_or(true)
            }) {
                println!("Starting Test {}, Q{} (#{})...", test, i + 1, t + 1);
                let start = Utc::now();
                for (_seg, query_segment) in q
                    .split(";")
                    .filter(|s| s.split_whitespace().collect::<String>() != "")
                    .enumerate()
                {
                    let df = ctx.sql(query_segment).await?;
                    df.clone().collect().await?;
                    df.explain(true, true)?.show().await?;
                }
                let end = Utc::now();
                let runtime = (end - start).num_milliseconds();
                println!("Finished Test {}, Q{}: took {}ms", test, i + 1, runtime);
                let point = TestResult::new(test.into(), i + 1, t, runtime);
                results.push(point);
            }
        }
    }
    Ok(results)
}

fn results_to_df(results: Vec<TestResult>) -> DataFrame {
    let tests = Arc::new(StringArray::from(
        results
            .iter()
            .map(|r| r.test.clone())
            .collect::<Vec<String>>(),
    ));
    let queries = Arc::new(UInt64Array::from(
        results.iter().map(|r| r.query as u64).collect::<Vec<u64>>(),
    ));
    let trials = Arc::new(UInt32Array::from(
        results.iter().map(|r| r.trial).collect::<Vec<u32>>(),
    ));
    let runtimes = Arc::new(Int64Array::from(
        results.iter().map(|r| r.runtime).collect::<Vec<i64>>(),
    ));
    DataFrame::from_columns(vec![
        ("test", tests),
        ("query", queries),
        ("trial", trials),
        ("runtime", runtimes),
    ])
    .expect("Couldn't pack results arrays into DataFrame")
}

#[tokio::main]
async fn main() -> Result<()> {
    // Pick the benchmark from the CLI: `cargo run --release -- tpcds`
    // (defaults to tpch to preserve the original behavior).
    let benchmark = std::env::args().nth(1).unwrap_or_else(|| "tpch".to_string());
    let cfg = match benchmark.as_str() {
        "tpch" => tpch_config(),
        "tpcds" => tpcds_config(),
        other => {
            eprintln!("unknown benchmark '{other}'; use 'tpch' or 'tpcds'");
            std::process::exit(1);
        }
    };
    println!(
        "Running {} benchmark ({} tables, {} queries loaded)",
        cfg.name,
        cfg.tables.len(),
        cfg.queries.len()
    );
    let results = run_bench(&cfg, 3, None).await?;
    let out = format!("test-data-{}.csv", cfg.name);
    let df = results_to_df(results);
    df.write_csv(&out, DataFrameWriteOptions::default(), None)
        .await?;
    println!("Wrote {out}");
    Ok(())
}
