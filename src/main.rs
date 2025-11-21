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
    common::{Result, file_options::parquet_writer::ParquetWriterOptions},
    config::ParquetOptions,
    dataframe::DataFrameWriteOptions,
};
use std::{fs, sync::Arc};

use chrono::Utc;
use datafusion::prelude::*;
use object_store::ObjectStore;
use url::Url;

pub mod zip;

#[tokio::main]
async fn main() -> Result<()> {
    let dump_results = true;
    let table_names = [
        "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
    ];
    let queries: Vec<String> = (1..=22)
        .map(|q_num| {
            fs::read_to_string(format!("queries/q{}.sql", q_num)).expect("Couldn't open query file")
        })
        .collect();
    let tests = ["a", "q"];
    for test in tests {
        let session_cfg =
            SessionConfig::new().set_str("datafusion.optimizer.repartition_file_scans", "false");
        let ctx = SessionContext::new_with_config(session_cfg);
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

        for (i, q) in queries.iter().take(3).enumerate() {
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
