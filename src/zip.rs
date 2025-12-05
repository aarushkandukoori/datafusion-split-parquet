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

use datafusion::physical_plan::metrics::MetricsSet;
use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use datafusion::catalog::Session;
use datafusion::common::instant::Instant;
use datafusion::common::{DFSchema, DataFusionError, Result, internal_datafusion_err};

use datafusion::datasource::TableProvider;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::parquet::{ParquetAccessPlan, RowGroupAccessPlanFilter};
use datafusion::datasource::physical_plan::{
    FileGroup, FileMeta, FileScanConfig, FileScanConfigBuilder, ParquetFileMetrics,
    ParquetFileReaderFactory, ParquetSource,
};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{TableProviderFilterPushDown, TableType};
use datafusion::parquet::arrow::arrow_reader::{
    ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use datafusion::parquet::arrow::async_reader::{AsyncFileReader, ParquetObjectReader};
use datafusion::parquet::file::metadata::ParquetMetaData;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::equivalence::join_equivalence_properties;
use datafusion::physical_optimizer::pruning::build_pruning_predicate;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder, Time};
use datafusion::physical_plan::{
    DisplayAs, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use datafusion::prelude::*;

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::datasource::memory::DataSourceExec;
use futures::future::BoxFuture;
use futures::stream::Zip;
use futures::{FutureExt, Stream, StreamExt};
use object_store::ObjectStore;

#[derive(Debug)]
pub struct ZippedTableProvider {
    /// The file that is being read.
    zipped_files: Vec<ZippedFile>,
    /// Zipped schema of contained files
    schema: SchemaRef,
    /// The underlying object store
    object_store: Arc<dyn ObjectStore>,
    metrics: ExecutionPlanMetricsSet,
}

impl ZippedTableProvider {
    /// Create a new ZippedTableProvider
    /// * `object_store` - the object store implementation to use for reading files
    pub fn try_new(
        object_store: Arc<dyn ObjectStore>,
        paths: Vec<impl AsRef<Path>>,
    ) -> Result<Self> {
        let zipped_files: Vec<ZippedFile> = paths
            .iter()
            .map(|path| ZippedFile::try_new(path))
            .collect::<Result<Vec<ZippedFile>>>()?;
        let schema = SchemaRef::from(Schema::try_merge(
            zipped_files
                .iter()
                .map(|f| Arc::unwrap_or_clone(f.schema.clone())),
        )?);

        Ok(Self {
            zipped_files,
            schema,
            object_store,
            metrics: ExecutionPlanMetricsSet::new(),
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
}

/// Stores information needed to scan a file
#[derive(Debug)]
struct ZippedFile {
    /// File name
    file_name: String,
    /// The path of the file
    path: PathBuf,
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
    fn try_new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        // Now, open the file and read its size and metadata
        let file_name = path
            .file_name()
            .ok_or_else(|| internal_datafusion_err!("Invalid path"))?
            .to_str()
            .ok_or_else(|| internal_datafusion_err!("Invalid filename"))?
            .to_string();
        let file_size = path.metadata()?.len();

        let file = File::open(path).map_err(|e| {
            DataFusionError::from(e).context(format!("Error opening file {path:?}"))
        })?;

        let options = ArrowReaderOptions::new();
        let reader = ParquetRecordBatchReaderBuilder::try_new_with_options(file, options)?;
        let metadata = reader.metadata().clone();
        let schema = reader.schema().clone();
        let row_group_ranges: Vec<(i64, i64)> = metadata
            .row_groups()
            .iter()
            .map(|rgm| {
                let start = rgm.file_offset().unwrap();
                // The compressed_size is basically the "real" size in the file
                (start, start + rgm.compressed_size())
            })
            .collect();
        // canonicalize after writing the file
        let path = std::fs::canonicalize(path)?;

        Ok(Self {
            file_name,
            path,
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
        for zipped_file in &self.zipped_files {
            // Prepare for scanning
            let schema = zipped_file.schema.clone();
            let object_store_url = ObjectStoreUrl::parse("file://")?;

            // Configure a factory interface to avoid re-reading the metadata for each file
            let reader_factory =
                CachedParquetFileReaderFactory::new(Arc::clone(&self.object_store))
                    .with_file(zipped_file);

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
                    zipped_file.path.display().to_string(),
                    zipped_file.file_size,
                    access_plan.clone(),
                    zipped_file.row_group_ranges.clone(),
                );
                let file_scan_config =
                    FileScanConfigBuilder::new(object_store_url, schema, file_source)
                        .with_limit(limit)
                        .with_projection(file_projection)
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
                    Some(Arc::new(left.1.clone())),
                    Some(Arc::new(right.1.clone())),
                )?),
                (Some(left), None) => DataSourceExec::from_data_source(left.0.clone()),
                (None, Some(right)) => DataSourceExec::from_data_source(right.0.clone()),
                (None, None) => Arc::new(EmptyExec::new(self.schema.clone())),
            };
        while let Some(file_scan) = files_to_scan.next() {
            let new_file_scan_op = DataSourceExec::from_data_source(file_scan.0.clone());
            exec_plan = Arc::new(ZipExec::try_new(
                exec_plan,
                new_file_scan_op,
                None,
                Some(Arc::new(file_scan.1.clone())),
            )?);
        }
        // Finally, put it all together into a DataSourceExec
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

/// A custom [`ParquetFileReaderFactory`] that handles opening parquet files
/// from object storage, and uses pre-loaded metadata.

#[derive(Debug)]
struct CachedParquetFileReaderFactory {
    /// The underlying object store implementation for reading file data
    object_store: Arc<dyn ObjectStore>,
    /// The parquet metadata for each file in the index, keyed by the file name
    /// (e.g. `file1.parquet`)
    metadata: HashMap<String, Arc<ParquetMetaData>>,
}

impl CachedParquetFileReaderFactory {
    fn new(object_store: Arc<dyn ObjectStore>) -> Self {
        Self {
            object_store,
            metadata: HashMap::new(),
        }
    }
    /// Add the pre-parsed information about the file to the factor
    fn with_file(mut self, zipped_file: &ZippedFile) -> Self {
        self.metadata.insert(
            zipped_file.file_name.clone(),
            Arc::clone(&zipped_file.metadata),
        );
        self
    }
}

impl ParquetFileReaderFactory for CachedParquetFileReaderFactory {
    fn create_reader(
        &self,
        _partition_index: usize,
        file_meta: FileMeta,
        metadata_size_hint: Option<usize>,
        _metrics: &ExecutionPlanMetricsSet,
    ) -> Result<Box<dyn AsyncFileReader + Send>> {
        // for this example we ignore the partition index and metrics
        // but in a real system you would likely use them to report details on
        // the performance of the reader.
        let filename = file_meta
            .location()
            .parts()
            .last()
            .expect("No path in location")
            .as_ref()
            .to_string();

        let object_store = Arc::clone(&self.object_store);
        let mut inner = ParquetObjectReader::new(object_store, file_meta.object_meta.location)
            .with_file_size(file_meta.object_meta.size);

        if let Some(hint) = metadata_size_hint {
            inner = inner.with_footer_size_hint(hint)
        };

        let metadata = self
            .metadata
            .get(&filename)
            .expect("metadata for file not found: {filename}");
        Ok(Box::new(ParquetReaderWithCache {
            filename,
            metadata: Arc::clone(metadata),
            inner,
            call_count: 0,
        }))
    }
}

/// wrapper around a ParquetObjectReader that caches metadata
struct ParquetReaderWithCache {
    filename: String,
    metadata: Arc<ParquetMetaData>,
    inner: ParquetObjectReader,
    call_count: i32,
}

impl AsyncFileReader for ParquetReaderWithCache {
    fn get_bytes(
        &mut self,
        range: Range<u64>,
    ) -> BoxFuture<'_, datafusion::parquet::errors::Result<Bytes>> {
        //println!("get_bytes: {} Reading range {:?}", self.filename, range);
        self.call_count += 1;
        self.inner.get_bytes(range)
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, datafusion::parquet::errors::Result<Vec<Bytes>>> {
        /*
        println!(
            "get_byte_ranges: {} Reading ranges {:?}",
            self.filename, ranges
        );
        */
        self.call_count += 1;
        self.inner.get_byte_ranges(ranges)
    }

    fn get_metadata(
        &mut self,
        _options: Option<&ArrowReaderOptions>,
    ) -> BoxFuture<'_, datafusion::parquet::errors::Result<Arc<ParquetMetaData>>> {
        //println!("get_metadata: {} returning cached metadata", self.filename);

        // return the cached metadata so the parquet reader does not read it
        let metadata = self.metadata.clone();
        async move { Ok(metadata) }.boxed()
    }
}

#[derive(Debug, Clone)]
pub struct ZipExec {
    schema: SchemaRef,
    left_input: Arc<dyn ExecutionPlan>,
    right_input: Arc<dyn ExecutionPlan>,
    left_info: Option<Arc<ZipPartitionInfo>>,
    right_info: Option<Arc<ZipPartitionInfo>>,
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
        left_info: Option<Arc<ZipPartitionInfo>>,
        right_info: Option<Arc<ZipPartitionInfo>>,
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
            left_info,
            right_info,
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
        let eq_properties = join_equivalence_properties(
            left.equivalence_properties().clone(),
            right.equivalence_properties().clone(),
            &JoinType::Full,
            schema,
            &[false, false],
            None,
            &[],
        )?;

        // Get output partitioning:
        let output_partitioning = left.output_partitioning();
        Ok(PlanProperties::new(
            eq_properties,
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
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(ZipExec::try_new(
            Arc::clone(&children[0]),
            Arc::clone(&children[1]),
            self.left_info.clone(),
            self.right_info.clone(),
        )?))
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
            16,
        )))
    }
}

pub struct ZipStream {
    schema: SchemaRef,
    left_stream: SendableRecordBatchStream,
    right_stream: SendableRecordBatchStream,
    left_buffer: VecDeque<RecordBatch>,
    right_buffer: VecDeque<RecordBatch>,
    metrics: ZipMetrics,
}

impl ZipStream {
    fn new(
        schema: SchemaRef,
        left_stream: SendableRecordBatchStream,
        right_stream: SendableRecordBatchStream,
        metrics: ZipMetrics,
        buffer_len: usize,
    ) -> Self {
        let left_buffer = VecDeque::with_capacity(buffer_len);
        let right_buffer = VecDeque::with_capacity(buffer_len);
        Self {
            schema,
            left_stream,
            right_stream,
            left_buffer,
            right_buffer,
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
        let mut left_none = false;
        let mut right_none = false;
        match self.left_stream.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(left_batch))) => {
                self.left_buffer.push_back(left_batch);
            }
            Poll::Ready(Some(Err(e))) => {
                self.metrics.time_elapsed_total.stop();
                self.metrics.time_elapsed_waiting.stop();
                return Poll::Ready(Some(Err(e)));
            }
            Poll::Ready(None) => {
                left_none = true;
            }
            Poll::Pending => {}
        }
        match self.right_stream.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(right_batch))) => {
                self.right_buffer.push_back(right_batch);
            }
            Poll::Ready(Some(Err(e))) => {
                self.metrics.time_elapsed_total.stop();
                self.metrics.time_elapsed_waiting.stop();
                return Poll::Ready(Some(Err(e)));
            }
            Poll::Ready(None) => {
                right_none = true;
            }
            Poll::Pending => {}
        }

        if let (Some(_), Some(_)) = (self.left_buffer.front(), self.right_buffer.front()) {
            let left_batch = self
                .left_buffer
                .pop_front()
                .expect("Panic, can't get left batch, this shouldn't happen");
            let right_batch = self
                .right_buffer
                .pop_front()
                .expect("Panic, can't get right batch, this shouldn't happen");
            self.metrics.time_elapsed_waiting.stop();
            self.metrics.num_zips.add(1);
            self.metrics.time_elapsed_zipping.start();
            let zipped_batch = Self::batch_zip(&left_batch, &right_batch);
            self.metrics.time_elapsed_zipping.stop();
            return Poll::Ready(Some(zipped_batch));
        } else if left_none && right_none {
            self.metrics.time_elapsed_waiting.stop();
            self.metrics.time_elapsed_total.stop();
            return Poll::Ready(None);
        }
        self.metrics.time_elapsed_waiting.start();
        Poll::Pending
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Same number of record batches
        self.left_stream.size_hint()
    }
}
