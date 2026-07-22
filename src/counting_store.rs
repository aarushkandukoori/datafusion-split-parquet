// A thin ObjectStore decorator that counts how a workload interacts with the
// store (requests + bytes), so we can model the $ cost of running on S3, which
// charges per request and per GB transferred. It only uses the public
// `ObjectStore` trait -- the object_store crate itself is untouched.

use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, Result,
};

/// Live counters, shared (Arc) with the code that reads them out.
#[derive(Debug, Default)]
pub struct StoreStats {
    /// Single-object GET requests (`get`, `get_opts`, `get_range`).
    pub get_requests: AtomicU64,
    /// `get_ranges()` calls (one call requests many ranges at once).
    pub get_ranges_calls: AtomicU64,
    /// Total ranges asked for across all `get_ranges()` calls (~ one GET each).
    pub get_ranges_ranges: AtomicU64,
    /// HEAD requests (object size / metadata lookups).
    pub head_requests: AtomicU64,
    /// LIST requests.
    pub list_requests: AtomicU64,
    /// Total bytes returned by the store (actual transfer).
    pub bytes_read: AtomicU64,
}

impl StoreStats {
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            get_requests: self.get_requests.load(Ordering::Relaxed),
            get_ranges_calls: self.get_ranges_calls.load(Ordering::Relaxed),
            get_ranges_ranges: self.get_ranges_ranges.load(Ordering::Relaxed),
            head_requests: self.head_requests.load(Ordering::Relaxed),
            list_requests: self.list_requests.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StatsSnapshot {
    pub get_requests: u64,
    pub get_ranges_calls: u64,
    pub get_ranges_ranges: u64,
    pub head_requests: u64,
    pub list_requests: u64,
    pub bytes_read: u64,
}

impl StatsSnapshot {
    /// GET requests actually reaching the store: single GETs plus each range
    /// inside a `get_ranges` batch (object_store may coalesce adjacent ranges,
    /// so treat this as an upper bound on real HTTP GETs).
    pub fn total_gets(&self) -> u64 {
        self.get_requests + self.get_ranges_ranges
    }
}

impl std::ops::Sub for StatsSnapshot {
    type Output = StatsSnapshot;
    fn sub(self, o: StatsSnapshot) -> StatsSnapshot {
        StatsSnapshot {
            get_requests: self.get_requests - o.get_requests,
            get_ranges_calls: self.get_ranges_calls - o.get_ranges_calls,
            get_ranges_ranges: self.get_ranges_ranges - o.get_ranges_ranges,
            head_requests: self.head_requests - o.head_requests,
            list_requests: self.list_requests - o.list_requests,
            bytes_read: self.bytes_read - o.bytes_read,
        }
    }
}

/// Wraps an object store and counts requests + bytes, delegating everything to
/// the inner store.
#[derive(Debug)]
pub struct CountingObjectStore {
    inner: Arc<dyn ObjectStore>,
    stats: Arc<StoreStats>,
}

impl CountingObjectStore {
    pub fn new(inner: Arc<dyn ObjectStore>, stats: Arc<StoreStats>) -> Self {
        Self { inner, stats }
    }
}

impl fmt::Display for CountingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CountingObjectStore({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for CountingObjectStore {
    // ---- read paths: count, then delegate to the inner store ----
    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        self.stats.get_requests.fetch_add(1, Ordering::Relaxed);
        let res = self.inner.get_opts(location, options).await?;
        self.stats
            .bytes_read
            .fetch_add(res.range.end - res.range.start, Ordering::Relaxed);
        Ok(res)
    }

    async fn get_range(&self, location: &Path, range: Range<u64>) -> Result<Bytes> {
        self.stats.get_requests.fetch_add(1, Ordering::Relaxed);
        let b = self.inner.get_range(location, range).await?;
        self.stats
            .bytes_read
            .fetch_add(b.len() as u64, Ordering::Relaxed);
        Ok(b)
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        self.stats.get_ranges_calls.fetch_add(1, Ordering::Relaxed);
        self.stats
            .get_ranges_ranges
            .fetch_add(ranges.len() as u64, Ordering::Relaxed);
        let v = self.inner.get_ranges(location, ranges).await?;
        let n: u64 = v.iter().map(|b| b.len() as u64).sum();
        self.stats.bytes_read.fetch_add(n, Ordering::Relaxed);
        Ok(v)
    }

    async fn head(&self, location: &Path) -> Result<ObjectMeta> {
        self.stats.head_requests.fetch_add(1, Ordering::Relaxed);
        self.inner.head(location).await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.stats.list_requests.fetch_add(1, Ordering::Relaxed);
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.stats.list_requests.fetch_add(1, Ordering::Relaxed);
        self.inner.list_with_delimiter(prefix).await
    }

    // ---- write / misc paths: pass through unchanged (workload is read-only) ----
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn delete(&self, location: &Path) -> Result<()> {
        self.inner.delete(location).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> Result<()> {
        self.inner.copy(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> Result<()> {
        self.inner.copy_if_not_exists(from, to).await
    }
}
