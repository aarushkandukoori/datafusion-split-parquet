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

use datafusion::common::stats::Precision;
use datafusion::datasource::physical_plan::parquet::metadata::DFParquetMetadata;
use datafusion::physical_plan::metrics::MetricsSet;
use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use datafusion::catalog::Session;
use datafusion::common::instant::Instant;
use datafusion::common::{DFSchema, DataFusionError, Result, Statistics};

use datafusion::datasource::TableProvider;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::parquet::{
    DefaultParquetFileReaderFactory, ParquetAccessPlan, RowGroupAccess, RowGroupAccessPlanFilter,
};
use datafusion::datasource::physical_plan::{
    FileGroup, FileScanConfig, FileScanConfigBuilder, ParquetFileMetrics, ParquetSource,
};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{TableProviderFilterPushDown, TableType};
use datafusion::parquet::arrow::parquet_to_arrow_schema;

use datafusion::parquet::file::metadata::ParquetMetaData;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_optimizer::pruning::build_pruning_predicate;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::execution_plan::{Boundedness, CardinalityEffect, EmissionType};
use datafusion::physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder, Time};
use datafusion::physical_plan::{
    DisplayAs, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use datafusion::prelude::*;

use async_trait::async_trait;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::datasource::memory::DataSourceExec;
use futures::stream::Zip;
use futures::{Stream, StreamExt};
use object_store::ObjectStore;

#[derive(Debug)]
pub struct ZippedTableProvider {
    /// The file that is being read.
    zipped_files: Vec<ZippedFile>,
    /// Zipped schema of contained files
    schema: SchemaRef,
    /// The underlying object store
    object_store: Arc<dyn ObjectStore>,
    /// The object store URL the files live under (e.g. `file://` or `s3://bucket`)
    object_store_url: ObjectStoreUrl,
    metrics: ExecutionPlanMetricsSet,
    num_rows: i64,
}

impl ZippedTableProvider {
    /// Create a new ZippedTableProvider
    /// * `object_store` - the object store implementation to use for reading files
    pub async fn try_new(
        object_store: Arc<dyn ObjectStore>,
        object_store_url: ObjectStoreUrl,
        locations: Vec<object_store::path::Path>,
    ) -> Result<Self> {
        let mut zipped_files = Vec::with_capacity(locations.len());
        for location in locations {
            zipped_files.push(ZippedFile::try_new(object_store.as_ref(), location).await?);
        }
        let schema = SchemaRef::from(Schema::try_merge(
            zipped_files
                .iter()
                .map(|f| Arc::unwrap_or_clone(f.schema.clone())),
        )?);
        let num_rows = zipped_files
            .first()
            .map(|zf| {
                zf.metadata
                    .row_groups()
                    .iter()
                    .map(|rg| rg.num_rows())
                    .sum()
            })
            .unwrap_or(0);
        Ok(Self {
            zipped_files,
            schema,
            object_store,
            object_store_url,
            metrics: ExecutionPlanMetricsSet::new(),
            num_rows,
        })
    }

    /// convert filters like `a = 1`, `b = 2`
    /// to a single predicate like `a = 1 AND b = 2` suitable for execution
    fn filters_to_predicate(
        &self,
        state: &dyn Session,
        filters: &[Expr],
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let df_schema = DFSchema::try_from(self.schema())?;

        let predicate = conjunction(filters.to_vec());
        let predicate = predicate
            .map(|predicate| state.create_physical_expr(predicate, &df_schema))
            .transpose()?
            // if there are no filters, use a literal true to have a predicate
            // that always evaluates to true we can pass to the index
            .unwrap_or_else(|| datafusion::physical_expr::expressions::lit(true));
        Ok(predicate)
    }

    /// Returns a [`ParquetAccessPlan`] that specifies how to scan the
    /// parquet file.
    ///
    /// A `ParquetAccessPlan` specifies which row groups  and which rows within
    /// those row groups to scan.
    fn create_plan(&self, predicate: &Arc<dyn PhysicalExpr>) -> Result<ParquetAccessPlan> {
        // Create an initial plan that scans all row groups. If there are somehow no files
        // apart of this ZippedTable, create an empty access plan.
        let init_plan: ParquetAccessPlan = self
            .zipped_files
            .get(0)
            .map(|f| f.scan_all_plan())
            .unwrap_or(ParquetAccessPlan::new_none(0));

        let mut row_groups = RowGroupAccessPlanFilter::new(init_plan);

        let predicate_creation_errors =
            MetricBuilder::new(&self.metrics).global_counter("num_predicate_creation_errors");

        for zipped_file in &self.zipped_files {
            let file_metrics = ParquetFileMetrics::new(0, &zipped_file.file_name, &self.metrics);
            let pruning_predicate = build_pruning_predicate(
                predicate.clone(),
                &zipped_file.schema,
                &predicate_creation_errors,
            );
            if let Some(pruning_predicate) = pruning_predicate {
                row_groups.prune_by_statistics(
                    &zipped_file.schema,
                    zipped_file.metadata.file_metadata().schema_descr(),
                    zipped_file.metadata.row_groups(),
                    &pruning_predicate,
                    &file_metrics,
                );
            }
        }
        Ok(row_groups.build())
    }

    fn estimate_rows(&self, access_plan: &ParquetAccessPlan) -> usize {
        self.zipped_files
            .get(0)
            .map(|zf| {
                zf.metadata
                    .row_groups()
                    .iter()
                    .map(|rgm| rgm.num_rows())
                    .zip(access_plan.clone().into_inner())
                    .map(|(num_rows, should_scan)| match should_scan {
                        RowGroupAccess::Skip => 0 as usize,
                        RowGroupAccess::Scan => num_rows as usize,
                        RowGroupAccess::Selection(selection) => selection.row_count(),
                    })
                    .sum()
            })
            .unwrap_or(0)
    }
    fn estimate_bytes(&self, access_plan: &ParquetAccessPlan) -> usize {
        self.zipped_files
            .get(0)
            .map(|zf| {
                zf.metadata
                    .row_groups()
                    .iter()
                    .map(|rgm| (rgm.num_rows(), rgm.total_byte_size()))
                    .zip(access_plan.clone().into_inner())
                    .map(|((num_rows, total_bytes), should_scan)| match should_scan {
                        RowGroupAccess::Skip => 0 as usize,
                        RowGroupAccess::Scan => total_bytes as usize,
                        RowGroupAccess::Selection(selection) => {
                            ((selection.row_count() as f32 / num_rows as f32) * total_bytes as f32)
                                as usize
                        }
                    })
                    .sum()
            })
            .unwrap_or(0)
    }
}

/// Stores information needed to scan a file
#[derive(Debug)]
struct ZippedFile {
    /// File name
    file_name: String,
    /// The object store location (key) of the file
    location: object_store::path::Path,
    /// The size of the file
    file_size: u64,
    /// The pre-parsed parquet metadata for the file
    metadata: Arc<ParquetMetaData>,
    /// The arrow schema of the file
    schema: SchemaRef,
    /// (start, end) byte range for each row group
    row_group_ranges: Vec<(i64, i64)>,
}

struct Split<'a, T> {
    slice: &'a [T],
    len: usize,
    rem: usize,
}

impl<'a, T> Iterator for Split<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            return None;
        }
        let mut len = self.len;
        if self.rem > 0 {
            len += 1;
            self.rem -= 1;
        }
        let (chunk, rest) = self.slice.split_at(len);
        self.slice = rest;
        Some(chunk)
    }
}

pub fn split<T>(slice: &[T], n: usize) -> impl Iterator<Item = &[T]> {
    let len = slice.len() / n;
    let rem = slice.len() % n;
    Split { slice, len, rem }
}

impl ZippedFile {
    async fn try_new(
        store: &dyn ObjectStore,
        location: object_store::path::Path,
    ) -> Result<Self> {
        let file_name = location.filename().unwrap_or_default().to_string();

        // Read size + parquet metadata through the object store, so this works
        // uniformly over a local filesystem, S3, GCS, etc.
        let object_meta = store
            .head(&location)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let file_size = object_meta.size;

        let metadata = DFParquetMetadata::new(store, &object_meta)
            .fetch_metadata()
            .await?;
        let schema = SchemaRef::from(
            parquet_to_arrow_schema(
                metadata.file_metadata().schema_descr(),
                metadata.file_metadata().key_value_metadata(),
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?,
        );
        let row_group_ranges: Vec<(i64, i64)> = metadata
            .row_groups()
            .iter()
            .map(|rgm| {
                let start = rgm.file_offset().unwrap();
                // The compressed_size is basically the "real" size in the file
                (start, start + rgm.compressed_size())
            })
            .collect();

        Ok(Self {
            file_name,
            location,
            file_size,
            metadata,
            schema,
            row_group_ranges,
        })
    }

    /// Return a `ParquetAccessPlan` that scans all row groups in the file
    fn scan_all_plan(&self) -> ParquetAccessPlan {
        ParquetAccessPlan::new_all(self.metadata.num_row_groups())
    }
}

pub fn partition_file(
    path: String,
    file_size: u64,
    row_group_ranges: Vec<(i64, i64)>,
    partitions: usize,
    access_plan: ParquetAccessPlan,
) -> Vec<FileGroup> {
    let mut files: Vec<FileGroup> = vec![];
    let mut row_group_ranges = row_group_ranges.clone();
    // Sort them by their starting offset
    row_group_ranges.sort_by_key(|r| r.0);
    let ranges: Vec<(i64, i64)> = split(row_group_ranges.as_slice(), partitions)
        .map(|c| {
            // (offset of first row group, end of last row group)
            (c.first().unwrap().0, c.last().unwrap().1)
        })
        .collect();

    for r in ranges {
        files.push(FileGroup::new(vec![
            PartitionedFile::new_with_range(path.clone(), file_size, r.0, r.1)
                .with_extensions(Arc::new(access_plan.clone()) as _),
        ]));
    }

    files
}

#[derive(Debug, Clone)]
pub struct ZipPartitionInfo {
    path: String,
    file_size: u64,
    access_plan: ParquetAccessPlan,
    row_group_ranges: Vec<(i64, i64)>,
}

impl ZipPartitionInfo {
    fn new(
        path: String,
        file_size: u64,
        access_plan: ParquetAccessPlan,
        row_group_ranges: Vec<(i64, i64)>,
    ) -> Self {
        Self {
            path,
            file_size,
            access_plan,
            row_group_ranges,
        }
    }
}

/// Implement the TableProvider trait for ZippedTableProvider
/// so that we can query it as a table.
#[async_trait]
impl TableProvider for ZippedTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn statistics(&self) -> Option<Statistics> {
        Some(
            Statistics::new_unknown(&self.schema)
                .with_num_rows(Precision::Inexact(self.num_rows as usize)),
        )
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let target_partitions = state.config().target_partitions();
        let mut file_scans: Vec<(FileScanConfig, ZipPartitionInfo)> = vec![];
        // Every Parquet Scan needs to use the same access plan
        // so we can stitch entire row groups back together in the right order
        let predicate = self.filters_to_predicate(state, filters)?;
        // Figure out which row groups to scan based on the predicate
        let access_plan = self.create_plan(&predicate)?;
        // Estimate the number of rows that the operator will yield, important for optimizer
        let estimated_rows = self.estimate_rows(&access_plan);
        let estimated_byte_size = self.estimate_bytes(&access_plan);
        for zipped_file in &self.zipped_files {
            // Prepare for scanning
            let schema = zipped_file.schema.clone();
            let object_store_url = self.object_store_url.clone();
            let statistics = DFParquetMetadata::statistics_from_parquet_metadata(
                &zipped_file.metadata,
                &schema.clone(),
            )
            .unwrap_or(Statistics::new_unknown(&schema))
            .with_total_byte_size(Precision::Exact(estimated_byte_size))
            .with_num_rows(Precision::Exact(estimated_rows));
            //println!("creating file with schema = {}", schema.clone());
            //println!("creating file with statistics = {}", statistics.clone());
            // Configure a factory interface to avoid re-reading the metadata for each file
            let reader_factory =
                DefaultParquetFileReaderFactory::new(Arc::clone(&self.object_store));

            let file_source = Arc::new(
                ParquetSource::default()
                    .with_bloom_filter_on_read(false)
                    // provide the factory to create parquet reader without re-reading metadata
                    .with_parquet_file_reader_factory(Arc::new(reader_factory)),
                // We do not attach a predicate here, as all of our predicate
                // pushdown is precomputed in the `self.create_plan` call.
                // This is so that it is uniform accross all ZippedFiles that are
                // a part of this ZippedTable
            );
            // If given a projection, find out which columns of interest to the query reside within
            // this parquet file
            let file_projection = projection.map(|indices| {
                indices
                    .iter()
                    .filter_map(|i| schema.index_of(self.schema.field(*i).name()).ok())
                    .collect::<Vec<usize>>()
            });
            // We need to scan this file if the projection is None, or if there is a column in the
            // projection vector that is in this file
            if file_projection.clone().map(|proj| proj.len()).unwrap_or(1) > 0 {
                let zip_partition_info = ZipPartitionInfo::new(
                    zipped_file.location.to_string(),
                    zipped_file.file_size,
                    access_plan.clone(),
                    zipped_file.row_group_ranges.clone(),
                );

                let file_scan_config =
                    FileScanConfigBuilder::new(object_store_url, schema.clone(), file_source)
                        .with_limit(limit)
                        .with_statistics(statistics)
                        .with_projection_indices(file_projection)
                        .with_file_groups(partition_file(
                            zip_partition_info.path.clone(),
                            zip_partition_info.file_size,
                            zip_partition_info.row_group_ranges.clone(),
                            target_partitions,
                            zip_partition_info.access_plan.clone(),
                        ))
                        .build();
                file_scans.push((file_scan_config, zip_partition_info));
            }
        }

        let mut files_to_scan = file_scans.iter();
        let mut exec_plan: Arc<dyn ExecutionPlan> =
            match (files_to_scan.next(), files_to_scan.next()) {
                (Some(left), Some(right)) => Arc::new(ZipExec::try_new(
                    DataSourceExec::from_data_source(left.0.clone()),
                    DataSourceExec::from_data_source(right.0.clone()),
                )?),
                (Some(left), None) => DataSourceExec::from_data_source(left.0.clone()),
                (None, Some(right)) => DataSourceExec::from_data_source(right.0.clone()),
                (None, None) => Arc::new(EmptyExec::new(self.schema.clone())),
            };
        while let Some(file_scan) = files_to_scan.next() {
            let new_file_scan_op = DataSourceExec::from_data_source(file_scan.0.clone());
            exec_plan = Arc::new(ZipExec::try_new(exec_plan, new_file_scan_op)?);
        }
        Ok(exec_plan)
    }

    /// Tell DataFusion to push filters down to the scan method
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        // Inexact because the pruning can't handle all expressions and pruning
        // is not done at the row level -- there may be rows in returned files
        // that do not pass the filter
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }
}

#[derive(Debug, Clone)]
pub struct ZipExec {
    schema: SchemaRef,
    left_input: Arc<dyn ExecutionPlan>,
    right_input: Arc<dyn ExecutionPlan>,
    metrics: ExecutionPlanMetricsSet,
    cache: PlanProperties,
}

/// A timer that can be started and stopped.
#[derive(Debug, Clone)]
pub struct StartableTime {
    pub metrics: Time,
    // use for record each part cost time, will eventually add into 'metrics'.
    pub start: Option<Instant>,
}

impl StartableTime {
    pub fn start(&mut self) {
        if self.start.is_none() {
            self.start = Some(Instant::now());
        }
    }

    pub fn stop(&mut self) {
        if let Some(start) = self.start.take() {
            self.metrics.add_elapsed(start);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZipMetrics {
    time_elapsed_waiting: StartableTime,
    time_elapsed_zipping: StartableTime,
    time_elapsed_total: StartableTime,
    num_zips: Count,
}

impl ZipMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let time_elapsed_waiting = StartableTime {
            metrics: MetricBuilder::new(metrics).subset_time("time_elapsed_waiting", partition),
            start: None,
        };
        let time_elapsed_zipping = StartableTime {
            metrics: MetricBuilder::new(metrics).subset_time("time_elapsed_zipping", partition),
            start: None,
        };
        let time_elapsed_total = StartableTime {
            metrics: MetricBuilder::new(metrics).subset_time("time_elapsed_total", partition),
            start: None,
        };
        let num_zips = MetricBuilder::new(metrics).counter("num_zips", partition);
        Self {
            time_elapsed_waiting,
            time_elapsed_zipping,
            time_elapsed_total,
            num_zips,
        }
    }
}

/// Copy of private function used to calculate boundedness of JOIN children
fn boundedness_from_children<'a>(
    children: impl IntoIterator<Item = &'a Arc<dyn ExecutionPlan>>,
) -> Boundedness {
    let mut unbounded_with_finite_mem = false;

    for child in children {
        match child.boundedness() {
            Boundedness::Unbounded {
                requires_infinite_memory: true,
            } => {
                return Boundedness::Unbounded {
                    requires_infinite_memory: true,
                };
            }
            Boundedness::Unbounded {
                requires_infinite_memory: false,
            } => {
                unbounded_with_finite_mem = true;
            }
            Boundedness::Bounded => {}
        }
    }

    if unbounded_with_finite_mem {
        Boundedness::Unbounded {
            requires_infinite_memory: false,
        }
    } else {
        Boundedness::Bounded
    }
}

impl ZipExec {
    pub fn try_new(
        left_input: Arc<dyn ExecutionPlan>,
        right_input: Arc<dyn ExecutionPlan>,
    ) -> Result<Self> {
        let schema = SchemaRef::from(Schema::try_merge(vec![
            Arc::unwrap_or_clone(left_input.schema().clone()),
            Arc::unwrap_or_clone(right_input.schema().clone()),
        ])?);
        let cache = Self::compute_properties(&left_input, &right_input, schema.clone())?;
        Ok(Self {
            schema,
            left_input,
            right_input,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        })
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
    ) -> Result<PlanProperties> {
        // Get output partitioning:
        let output_partitioning = left.output_partitioning();
        Ok(PlanProperties::new(
            EquivalenceProperties::new(schema),
            output_partitioning.clone(),
            EmissionType::Incremental,
            boundedness_from_children([left, right]),
        ))
    }
}

impl DisplayAs for ZipExec {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let metrics_string = format!("{:?}", self.metrics.clone_inner().aggregate_by_name());
        write!(f, "ZipExec({})", metrics_string)
    }
}

impl ExecutionPlan for ZipExec {
    fn name(&self) -> &'static str {
        "ZipExec"
    }

    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        // Tell optimizer this operator doesn't reorder its input
        vec![true, true]
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left_input, &self.right_input]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn statistics(&self) -> Result<Statistics> {
        self.partition_statistics(None)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics> {
        if partition.is_some() {
            Ok(Statistics::new_unknown(&self.schema))
        } else {
            let left_stats = self.left_input.partition_statistics(partition)?;
            let right_stats = self.right_input.partition_statistics(partition)?;
            let mut combined_stats = Statistics::new_unknown(&self.schema)
                .with_num_rows(left_stats.num_rows)
                .with_total_byte_size(left_stats.total_byte_size.add(&right_stats.total_byte_size));
            combined_stats.column_statistics = Vec::new();
            for left_col in &left_stats.column_statistics {
                combined_stats.column_statistics.push(left_col.clone());
            }
            for right_col in &right_stats.column_statistics {
                combined_stats.column_statistics.push(right_col.clone());
            }
            Ok(combined_stats)
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let left_stream = self.left_input.execute(partition, Arc::clone(&context))?;
        let right_stream = self.right_input.execute(partition, Arc::clone(&context))?;
        let mut zip_metrics = ZipMetrics::new(&self.metrics, partition);
        zip_metrics.time_elapsed_total.start();
        Ok(Box::pin(ZipStream::new(
            Arc::clone(&self.schema),
            left_stream,
            right_stream,
            zip_metrics,
        )))
    }
}

pub struct ZipStream {
    schema: SchemaRef,
    zip_stream: Zip<SendableRecordBatchStream, SendableRecordBatchStream>,
    metrics: ZipMetrics,
}

impl ZipStream {
    fn new(
        schema: SchemaRef,
        left_input: SendableRecordBatchStream,
        right_input: SendableRecordBatchStream,
        metrics: ZipMetrics,
    ) -> Self {
        let zip_stream = left_input.zip(right_input);
        Self {
            schema,
            zip_stream,
            metrics,
        }
    }

    fn batch_zip(left_batch: &RecordBatch, right_batch: &RecordBatch) -> Result<RecordBatch> {
        let schema = SchemaRef::from(Schema::try_merge([
            Arc::unwrap_or_clone(left_batch.schema()),
            Arc::unwrap_or_clone(right_batch.schema()),
        ])?);
        let new_batch = if left_batch.num_rows() != right_batch.num_rows() {
            Ok(RecordBatch::new_empty(schema))
        } else {
            RecordBatch::try_new(
                Arc::clone(&schema),
                [left_batch.columns(), right_batch.columns()]
                    .concat()
                    .to_vec(),
            )
            .map_err(Into::into)
        };
        new_batch
    }
}

impl RecordBatchStream for ZipStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Stream for ZipStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll = match self.zip_stream.poll_next_unpin(cx) {
            Poll::Ready(Some((Ok(left_batch), Ok(right_batch)))) => {
                self.metrics.time_elapsed_waiting.stop();
                self.metrics.num_zips.add(1);
                self.metrics.time_elapsed_zipping.start();
                let zipped_batch = Self::batch_zip(&left_batch, &right_batch);
                self.metrics.time_elapsed_zipping.stop();
                Poll::Ready(Some(zipped_batch))
            }
            Poll::Ready(Some((Err(e), _))) => {
                self.metrics.time_elapsed_total.stop();
                self.metrics.time_elapsed_waiting.stop();
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(Some((_, Err(e)))) => {
                self.metrics.time_elapsed_total.stop();
                self.metrics.time_elapsed_waiting.stop();
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                self.metrics.time_elapsed_total.stop();
                self.metrics.time_elapsed_waiting.stop();
                Poll::Ready(None)
            }
            _ => {
                self.metrics.time_elapsed_waiting.start();
                Poll::Pending
            }
        };
        poll
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Same number of record batches
        self.zip_stream.size_hint()
    }
}
