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
use std::{fs, sync::Arc};

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

async fn tpch(trials: usize) -> Result<Vec<TestResult>> {
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
                let parquet_options = ParquetReadOptions::default().parquet_pruning(true);
                ctx.register_parquet(
                    table_name,
                    format!("data/tpch/{}_a0.parquet", table_name),
                    parquet_options,
                )
                .await?;
            } else if test == "b" {
                let provider = Arc::new(zip::ZippedTableProvider::try_new(
                    Arc::clone(&object_store),
                    vec![
                        format!("data/tpch/{}_b0.parquet", table_name),
                        format!("data/tpch/{}_b1.parquet", table_name),
                    ],
                )?);
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
            for (i, q) in queries.iter().enumerate() {
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
                        println!("Explain:");
                        df.explain(true, true)?.show().await?;
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

async fn smoke(control: String, partitions: Vec<String>, dump: bool) -> Result<()> {
    let ctx = SessionContext::new();
    // the object store is used to read the parquet files (in this case, it is
    // a local file system, but in a real system it could be S3, GCS, etc)
    let object_store: Arc<dyn ObjectStore> = Arc::new(object_store::local::LocalFileSystem::new());

    ctx.register_parquet("whole_table", control, ParquetReadOptions::new())
        .await?;
    let provider = Arc::new(zip::ZippedTableProvider::try_new(
        Arc::clone(&object_store),
        partitions,
    )?);
    ctx.register_table("split_table", Arc::clone(&provider) as _)?;
    // register object store provider for urls like `file://` work
    let url = Url::try_from("file://").unwrap();
    ctx.register_object_store(&url, object_store);
    println!("Running smoke test for control table...");
    let mut start = Utc::now();
    if dump {
        ctx.sql("SELECT * FROM whole_table")
            .await?
            .write_parquet(
                "control_result.parquet",
                DataFrameWriteOptions::default(),
                None,
            )
            .await?;
    } else {
        ctx.sql("SELECT * FROM whole_table")
            .await?
            .collect()
            .await?;
    }
    let mut end = Utc::now();
    println!(
        "Finished smoke test for control table: took {}ms",
        (end - start).num_milliseconds()
    );

    println!("Running smoke test for split table...");
    start = Utc::now();
    if dump {
        ctx.sql("SELECT * FROM split_table")
            .await?
            .write_parquet(
                "split_result.parquet",
                DataFrameWriteOptions::default(),
                None,
            )
            .await?;
    } else {
        ctx.sql("SELECT * FROM split_table")
            .await?
            .collect()
            .await?;
    }
    end = Utc::now();
    let runtime = (end - start).num_milliseconds();
    println!("Finished smoke test for split table: took {}ms", runtime);

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
    /*
    smoke(
        "data/tpch/lineitem_sf10_a0.parquet".into(),
        vec![
            "data/tpch/lineitem_sf10_q0.parquet".into(),
            "data/tpch/lineitem_sf10_q1.parquet".into(),
        ],
        true,
    )
    .await
    */
    let results = tpch(1).await?;
    let df = results_to_df(results);
    df.write_csv("test-data.csv", DataFrameWriteOptions::default(), None)
        .await?;
    Ok(())
}
