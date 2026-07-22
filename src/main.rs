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
    common::{DataFusionError, Result},
    dataframe::DataFrameWriteOptions,
};
use std::{collections::HashSet, fs, sync::Arc};

use chrono::Utc;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::prelude::*;
use object_store::path::Path as ObjPath;
use object_store::ObjectStore;

pub mod counting_store;
pub mod zip;

/// Where the parquet files are read from. Chosen with the `STORAGE` env var:
/// `local` (default) reads from the local filesystem; `s3` reads from an
/// S3-compatible store (e.g. a local MinIO), configured with the S3_* env vars.
/// The store is wrapped so we can count requests + bytes (see counting_store).
struct Storage {
    store: Arc<dyn ObjectStore>,
    url: ObjectStoreUrl,
    s3: bool,
    bucket: String,
    stats: Arc<counting_store::StoreStats>,
}

impl Storage {
    fn from_env() -> Result<Self> {
        let stats = Arc::new(counting_store::StoreStats::default());
        if std::env::var("STORAGE").as_deref() == Ok("s3") {
            let endpoint =
                std::env::var("S3_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:9000".into());
            let bucket = std::env::var("S3_BUCKET").unwrap_or_else(|_| "bench".into());
            let key = std::env::var("S3_ACCESS_KEY").unwrap_or_else(|_| "minioadmin".into());
            let secret = std::env::var("S3_SECRET_KEY").unwrap_or_else(|_| "minioadmin".into());
            let inner: Arc<dyn ObjectStore> = Arc::new(
                object_store::aws::AmazonS3Builder::new()
                    .with_endpoint(endpoint)
                    .with_bucket_name(&bucket)
                    .with_region("us-east-1")
                    .with_access_key_id(key)
                    .with_secret_access_key(secret)
                    .with_allow_http(true)
                    .build()
                    .map_err(|e| DataFusionError::External(Box::new(e)))?,
            );
            let store: Arc<dyn ObjectStore> = Arc::new(
                counting_store::CountingObjectStore::new(inner, Arc::clone(&stats)),
            );
            let url = ObjectStoreUrl::parse(format!("s3://{bucket}"))?;
            Ok(Self { store, url, s3: true, bucket, stats })
        } else {
            let inner: Arc<dyn ObjectStore> =
                Arc::new(object_store::local::LocalFileSystem::new());
            let store: Arc<dyn ObjectStore> = Arc::new(
                counting_store::CountingObjectStore::new(inner, Arc::clone(&stats)),
            );
            Ok(Self {
                store,
                url: ObjectStoreUrl::parse("file://")?,
                s3: false,
                bucket: String::new(),
                stats,
            })
        }
    }

    /// Source string for the single-file control, passed to `register_parquet`.
    fn control_source(&self, cfg: &BenchConfig, table: &str) -> String {
        if self.s3 {
            format!("s3://{}/{}/{}_a0.parquet", self.bucket, cfg.name, table)
        } else {
            format!("{}/{}_a0.parquet", cfg.data_dir, table)
        }
    }

    /// Object-store key for a split file, passed to the ZippedTableProvider.
    fn key(&self, cfg: &BenchConfig, filename: &str) -> Result<ObjPath> {
        if self.s3 {
            Ok(ObjPath::from(format!("{}/{}", cfg.name, filename)))
        } else {
            let abs = std::fs::canonicalize(format!("{}/{}", cfg.data_dir, filename))?;
            ObjPath::from_absolute_path(abs).map_err(|e| DataFusionError::External(Box::new(e)))
        }
    }
}

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
    storage: &Storage,
    cfg: &BenchConfig,
    table: &str,
    test: &str,
) -> Result<()> {
    match test {
        "a" => {
            ctx.register_parquet(
                table,
                storage.control_source(cfg, table),
                ParquetReadOptions::default(),
            )
            .await?;
        }
        "b" => {
            let provider = zip::ZippedTableProvider::try_new(
                Arc::clone(&storage.store),
                storage.url.clone(),
                vec![
                    storage.key(cfg, &format!("{table}_b0.parquet"))?,
                    storage.key(cfg, &format!("{table}_b1.parquet"))?,
                ],
            )
            .await?;
            ctx.register_table(table, Arc::new(provider) as _)?;
        }
        "c" => {
            let provider = zip::ZippedTableProvider::try_new(
                Arc::clone(&storage.store),
                storage.url.clone(),
                vec![
                    storage.key(cfg, &format!("{table}_c0.parquet"))?,
                    storage.key(cfg, &format!("{table}_c1.parquet"))?,
                    storage.key(cfg, &format!("{table}_c2.parquet"))?,
                ],
            )
            .await?;
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
    let storage = Storage::from_env()?;
    let tests = ["a", "b", "c"];
    for test in tests {
        let ctx = SessionContext::new();
        ctx.register_object_store(storage.url.as_ref(), Arc::clone(&storage.store));
        for table in &cfg.tables {
            register_variant(&ctx, &storage, cfg, table, test).await?;
        }

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

/// Print `EXPLAIN ANALYZE` for one query under each storage variant, so we can
/// see which physical operators dominate and how much the ZipExec stitching
/// adds. Each query is run once to warm up before the analyzed run.
async fn profile_query(cfg: &BenchConfig, q_num: usize) -> Result<()> {
    let (_, sql) = cfg
        .queries
        .iter()
        .find(|(n, _)| *n == q_num)
        .unwrap_or_else(|| panic!("query Q{q_num} not found in {}", cfg.name));

    let storage = Storage::from_env()?;
    for test in ["a", "b", "c"] {
        let ctx = SessionContext::new();
        ctx.register_object_store(storage.url.as_ref(), Arc::clone(&storage.store));
        for table in &cfg.tables {
            register_variant(&ctx, &storage, cfg, table, test).await?;
        }

        let label = match test {
            "a" => "control (1 file)",
            "b" => "2-way split (zipped)",
            _ => "3-way split (zipped)",
        };
        println!("\n================ {} Q{} :: {} ================", cfg.name, q_num, label);
        for segment in sql
            .split(';')
            .filter(|s| !s.split_whitespace().collect::<String>().is_empty())
        {
            // Warm up once so the analyzed run reflects steady-state, not
            // one-time parquet metadata reads.
            ctx.sql(segment).await?.collect().await?;
            let explained = ctx.sql(segment).await?.explain(false, true)?;
            explained.show().await?;
        }
    }
    Ok(())
}

/// Count how the workload hits the object store (GET/HEAD requests and bytes
/// read) per storage variant, running every query once. This is the raw input
/// for modeling the $ cost of running on S3, which prices per request + per GB.
async fn collect_stats(cfg: &BenchConfig) -> Result<()> {
    let storage = Storage::from_env()?;
    let backend = if storage.s3 { "s3" } else { "local" };
    println!(
        "\n{} object-store interaction ({} backend, each query run once)\n",
        cfg.name, backend
    );
    println!(
        "{:<16}{:>10}{:>8}{:>10}{:>9}{:>14}",
        "variant", "GET reqs", "HEAD", "get_rngs", "ranges", "bytes read"
    );
    println!("{}", "-".repeat(67));
    for (test, label) in [
        ("a", "control 1-file"),
        ("b", "2-way split"),
        ("c", "3-way split"),
    ] {
        let before = storage.stats.snapshot();
        let ctx = SessionContext::new();
        ctx.register_object_store(storage.url.as_ref(), Arc::clone(&storage.store));
        for table in &cfg.tables {
            register_variant(&ctx, &storage, cfg, table, test).await?;
        }
        for (_q, sql) in &cfg.queries {
            let _ = run_query(&ctx, sql).await;
        }
        let d = storage.stats.snapshot() - before;
        let mib = d.bytes_read as f64 / (1024.0 * 1024.0);
        println!(
            "{:<16}{:>10}{:>8}{:>10}{:>9}{:>10.1} MiB",
            label,
            d.total_gets(),
            d.head_requests,
            d.get_ranges_calls,
            d.get_ranges_ranges,
            mib
        );
    }
    println!(
        "\nGET reqs = single GETs + each range in a get_ranges batch (upper bound on\nHTTP GETs, since object_store coalesces adjacent ranges). bytes read = actual\ntransfer. Metadata (HEAD + footer) is fetched once per table and cached."
    );
    Ok(())
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
    // Usage:
    //   cargo run --release [-- <bench>]              run the full benchmark
    //   cargo run --release -- <bench> profile <N>    EXPLAIN ANALYZE query N per variant
    // <bench> is tpch (default) or tpcds.
    let args: Vec<String> = std::env::args().collect();
    let benchmark = args.get(1).map(String::as_str).unwrap_or("tpch");
    let cfg = match benchmark {
        "tpch" => tpch_config(),
        "tpcds" => tpcds_config(),
        other => {
            eprintln!("unknown benchmark '{other}'; use 'tpch' or 'tpcds'");
            std::process::exit(1);
        }
    };

    if args.get(2).map(String::as_str) == Some("profile") {
        let q: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
            eprintln!("usage: {} profile <query_number>", cfg.name);
            std::process::exit(1);
        });
        profile_query(&cfg, q).await?;
        return Ok(());
    }

    if args.get(2).map(String::as_str) == Some("stats") {
        collect_stats(&cfg).await?;
        return Ok(());
    }

    println!(
        "Running {} benchmark ({} tables, {} queries loaded)",
        cfg.name,
        cfg.tables.len(),
        cfg.queries.len()
    );
    let results = run_bench(&cfg, 3, None).await?;
    let storage_tag = if std::env::var("STORAGE").as_deref() == Ok("s3") {
        "s3"
    } else {
        "local"
    };
    let out = format!("test-data-{}-{}.csv", cfg.name, storage_tag);
    let df = results_to_df(results);
    df.write_csv(&out, DataFrameWriteOptions::default(), None)
        .await?;
    println!("Wrote {out}");
    Ok(())
}
