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

use datafusion::{common::Result, dataframe::DataFrameWriteOptions};
use std::{fs, sync::Arc};

use chrono::Utc;
use datafusion::prelude::*;
use object_store::ObjectStore;
use url::Url;

pub mod zip;

async fn tpch() -> Result<()> {
    let dump_results = true;
    let table_names = ["lineitem"];
    let queries: Vec<String> = (1..=22)
        .map(|q_num| {
            fs::read_to_string(format!("queries/q{}.sql", q_num)).expect("Couldn't open query file")
        })
        .collect();
    let tests = ["a", "q"];
    for test in tests {
        let ctx = SessionContext::new();
        // the object store is used to read the parquet files (in this case, it is
        // a local file system, but in a real system it could be S3, GCS, etc)
        let object_store: Arc<dyn ObjectStore> =
            Arc::new(object_store::local::LocalFileSystem::new());

        for table_name in table_names {
            // Create a custom table provider with our special index.
            if test == "a" {
                ctx.register_parquet(
                    table_name,
                    format!("data/tpch/{}_sf10_a0.parquet", table_name),
                    ParquetReadOptions::new(),
                )
                .await?;
            } else {
                let provider = Arc::new(zip::ZippedTableProvider::try_new(
                    Arc::clone(&object_store),
                    vec![
                        format!("data/tpch/{}_sf10_q0.parquet", table_name),
                        format!("data/tpch/{}_sf10_q1.parquet", table_name),
                    ],
                )?);
                ctx.register_table(table_name, Arc::clone(&provider) as _)?;
            }
        }

        // register object store provider for urls like `file://` work
        let url = Url::try_from("file://").unwrap();
        ctx.register_object_store(&url, object_store);

        for (i, q) in queries.iter().take(1).enumerate() {
            println!("Starting Test {}, Q{}...", test, i + 1);
            let start = Utc::now();
            let df = ctx.sql(q).await?;
            if dump_results {
                df.clone()
                    .write_parquet(
                        &format!("results/result_t{}_q{}.parquet", test, i + 1),
                        DataFrameWriteOptions::new(),
                        None,
                    )
                    .await?;
                //df.explain(false, false)?.show().await?;
            } else {
                df.collect().await?;
            }
            let end = Utc::now();
            println!(
                "Finished Test {}, Q{}: took {}ms",
                test,
                i + 1,
                (end - start).num_milliseconds()
            );
        }
    }
    Ok(())
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
    println!(
        "Finished smoke test for split table: took {}ms",
        (end - start).num_milliseconds()
    );

    Ok(())
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
    tpch().await
}
