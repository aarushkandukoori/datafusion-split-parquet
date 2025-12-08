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

async fn tpch(trials: usize, queries: Option<Vec<usize>>) -> Result<Vec<TestResult>> {
    let subset_queries = queries.map(|q| HashSet::from_iter(q));
    let mut results = Vec::new();
    let dump_results = false;
    let table_names = [
        "lineitem", "orders", "partsupp", "supplier", "nation", "region", "part", "customer",
    ];
    let queries: Vec<String> = (1..=22)
        .map(|q_num| {
            fs::read_to_string(format!("queries/q{}.sql", q_num)).expect("Couldn't open query file")
        })
        .collect();
    let tests = ["a", "b", "c"];
    for test in tests {
        let ctx = SessionContext::new();
        // the object store is used to read the parquet files (in this case, it is
        // a local file system, but in a real system it could be S3, GCS, etc)
        let object_store: Arc<dyn ObjectStore> =
            Arc::new(object_store::local::LocalFileSystem::new());

        for table_name in table_names {
            // Create a custom table provider with our special index.
            if test == "a" {
                let parquet_options = ParquetReadOptions::default();
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
                for (seg, query_segment) in q
                    .split(";")
                    .filter(|s| s.split_whitespace().collect::<String>() != "")
                    .enumerate()
                {
                    let df = ctx.sql(query_segment).await?;
                    if dump_results {
                        df.clone()
                            .write_parquet(
                                &format!("results/result_t{}_q{}_seg{}.parquet", test, i + 1, seg),
                                DataFrameWriteOptions::new(),
                                None,
                            )
                            .await?;
                        //df.explain(false, false)?.show().await?;
                    } else {
                        df.clone().collect().await?;
                        //df.explain(false, true)?.show().await?;
                    }
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
                for (seg, query_segment) in q
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
    let results = tpch(3, None).await?;
    let df = results_to_df(results);
    df.write_csv("test-data.csv", DataFrameWriteOptions::default(), None)
        .await?;
    Ok(())
}
