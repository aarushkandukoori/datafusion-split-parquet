use datafusion::common::Result;
use datafusion::prelude::{ParquetReadOptions, SessionContext};

/// Implement as a ListingTable
/// https://docs.rs/datafusion/latest/datafusion/datasource/listing/struct.ListingTable.html#example-read-a-directory-of-parquet-files-using-a-listingtable
#[tokio::main]
async fn main() -> Result<()> {
    let ctx = SessionContext::new();
    let df = ctx
        .read_parquet("data/control/table.parquet", ParquetReadOptions::new())
        .await?;
    df.show().await?;
    Ok(())
}
