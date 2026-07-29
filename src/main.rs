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
use datafusion::datasource::physical_plan::parquet::metadata::DFParquetMetadata;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::*;
use object_store::path::Path as ObjPath;
use object_store::ObjectStore;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::{Duration, Instant};

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

/// Per-(table, column) compressed size, read from the control files' parquet
/// footers. Used to estimate how many bytes each referenced column covers --
/// "inspect the plan before execution for the number of bytes per column".
async fn column_sizes(
    storage: &Storage,
    cfg: &BenchConfig,
) -> Result<HashMap<(String, String), u64>> {
    let mut sizes = HashMap::new();
    for table in &cfg.tables {
        let key = storage.key(cfg, &format!("{table}_a0.parquet"))?;
        let object_meta = storage
            .store
            .head(&key)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let md = DFParquetMetadata::new(storage.store.as_ref(), &object_meta)
            .fetch_metadata()
            .await?;
        let descr = md.file_metadata().schema_descr_ptr();
        for rg in md.row_groups() {
            for (i, col) in rg.columns().iter().enumerate() {
                *sizes
                    .entry((table.to_string(), descr.column(i).name().to_string()))
                    .or_insert(0u64) += col.compressed_size() as u64;
            }
        }
    }
    Ok(sizes)
}

/// Collect (table -> columns) referenced by a plan by walking its TableScans:
/// a scan's projected schema is exactly the set of columns it will read.
fn plan_columns(plan: &LogicalPlan, out: &mut BTreeMap<String, BTreeSet<String>>) {
    if let LogicalPlan::TableScan(scan) = plan {
        let entry = out.entry(scan.table_name.table().to_string()).or_default();
        for field in scan.projected_schema.fields() {
            entry.insert(field.name().clone());
        }
    }
    for input in plan.inputs() {
        plan_columns(input, out);
    }
}

struct WorkloadConfig {
    variant: String,        // a | b | c
    avg_interval_secs: f64, // average seconds between query arrivals
    duration_secs: f64,     // how long to keep firing
    dist: String,           // query-sampling distribution (uniform for now)
    arrival: String,        // fixed | poisson inter-arrival gaps
    seed: u64,              // RNG seed, so runs across variants are comparable
}

/// Fire a steady stream of randomly-sampled benchmark queries for a fixed
/// duration, logging per-query latency, object-store request/byte deltas, and
/// which table columns each query referenced. The column log is the input for
/// finding hot vs cold columns to design a better partition scheme.
async fn workload(cfg: &BenchConfig, wl: &WorkloadConfig) -> Result<()> {
    let storage = Storage::from_env()?;
    let backend = if storage.s3 { "s3" } else { "local" };
    let ctx = SessionContext::new();
    ctx.register_object_store(storage.url.as_ref(), Arc::clone(&storage.store));
    for table in &cfg.tables {
        register_variant(&ctx, &storage, cfg, table, &wl.variant).await?;
    }
    let sizes = column_sizes(&storage, cfg).await?;

    let mut rng = StdRng::seed_from_u64(wl.seed);
    let n = cfg.queries.len();
    let mut events: Vec<String> =
        vec!["t_offset_ms,query_nr,latency_ms,get_requests,bytes_read,ok".into()];
    let mut colrows: Vec<String> = vec!["t_offset_ms,query_nr,table,column,est_bytes".into()];
    let mut latencies: Vec<f64> = Vec::new();
    let (mut total_gets, mut total_bytes) = (0u64, 0u64);
    let mut touch: BTreeMap<(String, String), u64> = BTreeMap::new();

    println!(
        "workload: {} variant={} backend={} dist={} arrivals={} avg every {}s for {}s seed={}",
        cfg.name,
        wl.variant,
        backend,
        wl.dist,
        wl.arrival,
        wl.avg_interval_secs,
        wl.duration_secs,
        wl.seed
    );

    let start = Instant::now();
    let mut next_fire = 0.0f64;
    let mut fired = 0usize;
    while next_fire < wl.duration_secs {
        let now = start.elapsed().as_secs_f64();
        if now < next_fire {
            tokio::time::sleep(Duration::from_secs_f64(next_fire - now)).await;
        }
        // Sample a query. The match is the extension point for non-uniform
        // distributions (zipf, hot-set, ...).
        let idx = match wl.dist.as_str() {
            _ => rng.gen_range(0..n),
        };
        let (qnr, sql) = &cfg.queries[idx];

        // Untimed: which columns does this query read? Walk the optimized
        // plan of each statement. Note DataFusion executes DDL (e.g. CREATE
        // VIEW) eagerly at ctx.sql().await, so multi-statement queries plan
        // correctly here; DDL does no object-store IO, so the per-query
        // request/byte deltas below are unaffected.
        let mut cols: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for segment in sql
            .split(';')
            .filter(|s| !s.split_whitespace().collect::<String>().is_empty())
        {
            if let Ok(df) = ctx.sql(segment).await {
                if let Ok(plan) = df.into_optimized_plan() {
                    plan_columns(&plan, &mut cols);
                }
            }
        }

        let t_offset_ms = start.elapsed().as_millis();
        let before = storage.stats.snapshot();
        let qstart = Instant::now();
        let res = run_query(&ctx, sql).await;
        let latency_ms = qstart.elapsed().as_secs_f64() * 1e3;
        let d = storage.stats.snapshot() - before;
        let ok = res.is_ok();

        events.push(format!(
            "{},{},{:.1},{},{},{}",
            t_offset_ms,
            qnr,
            latency_ms,
            d.total_gets(),
            d.bytes_read,
            ok
        ));
        for (table, cs) in &cols {
            for c in cs {
                let est = sizes.get(&(table.clone(), c.clone())).copied().unwrap_or(0);
                colrows.push(format!("{t_offset_ms},{qnr},{table},{c},{est}"));
                *touch.entry((table.clone(), c.clone())).or_insert(0) += 1;
            }
        }
        if ok {
            latencies.push(latency_ms);
        }
        total_gets += d.total_gets();
        total_bytes += d.bytes_read;
        fired += 1;
        println!(
            "  t={:>6.1}s Q{:<3} {:>8.1}ms {:>5} GETs {:>11} B  cols={}",
            t_offset_ms as f64 / 1e3,
            qnr,
            latency_ms,
            d.total_gets(),
            d.bytes_read,
            cols.values().map(|s| s.len()).sum::<usize>()
        );

        // Open-loop arrival schedule: the next fire time advances by the gap
        // regardless of how long the query took, keeping the average rate; an
        // overrunning query just makes the next one fire immediately.
        let gap = match wl.arrival.as_str() {
            "poisson" => -wl.avg_interval_secs * (1.0 - rng.gen_range(0.0f64..1.0f64)).ln(),
            _ => wl.avg_interval_secs,
        };
        next_fire += gap;
    }

    let tag = format!("{}-{}-{}-{}", cfg.name, wl.variant, backend, wl.dist);
    fs::write(
        format!("workload-events-{tag}.csv"),
        events.join("\n") + "\n",
    )?;
    fs::write(
        format!("workload-columns-{tag}.csv"),
        colrows.join("\n") + "\n",
    )?;

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f64 {
        if latencies.is_empty() {
            0.0
        } else {
            latencies[((latencies.len() as f64 - 1.0) * p) as usize]
        }
    };
    println!(
        "\n{} queries fired in {:.0}s ({} ok) | latency p50={:.0}ms p95={:.0}ms max={:.0}ms | total {} GETs, {:.1} MiB read",
        fired,
        wl.duration_secs,
        latencies.len(),
        pct(0.5),
        pct(0.95),
        pct(1.0),
        total_gets,
        total_bytes as f64 / 1048576.0
    );
    let mut hot: Vec<_> = touch.iter().collect();
    hot.sort_by(|a, b| b.1.cmp(a.1));
    println!("\nhottest columns (touches across {fired} queries):");
    for ((t, c), cnt) in hot.iter().take(10) {
        println!("  {cnt:>4}x  {t}.{c}");
    }
    println!("\nwrote workload-events-{tag}.csv and workload-columns-{tag}.csv");
    println!("next: python3 workload_report.py {tag}");
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

    // cargo run --release -- <bench> workload [variant] [avg_interval_s]
    //   [duration_s] [dist] [arrival] [seed]
    // e.g. `tpcds workload b 2 60 uniform fixed 42`
    if args.get(2).map(String::as_str) == Some("workload") {
        let wl = WorkloadConfig {
            variant: args.get(3).cloned().unwrap_or_else(|| "a".into()),
            avg_interval_secs: args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2.0),
            duration_secs: args.get(5).and_then(|s| s.parse().ok()).unwrap_or(60.0),
            dist: args.get(6).cloned().unwrap_or_else(|| "uniform".into()),
            arrival: args.get(7).cloned().unwrap_or_else(|| "fixed".into()),
            seed: args.get(8).and_then(|s| s.parse().ok()).unwrap_or(42),
        };
        if !["a", "b", "c"].contains(&wl.variant.as_str()) {
            eprintln!("unknown variant '{}'; use a, b, or c", wl.variant);
            std::process::exit(1);
        }
        workload(&cfg, &wl).await?;
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
