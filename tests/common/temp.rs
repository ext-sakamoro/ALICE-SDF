//! Temporary directories of the integration tests, removed with what is in them when dropped
//!
//! A test that panics still drops its locals while unwinding, so its directory goes too. Each
//! directory is new and unique (process id and a counter), so tests running in parallel never
//! share one.
//!
//! Include with `#[path = "common/temp.rs"] mod temp;` (not through `common`, so the files that
//! use only `common` do not compile it).
//!
//! Author: Moroya Sakamoto
#![allow(dead_code, reason = "each test file uses the part it needs")]

use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A new empty directory under the system temporary directory, removed when dropped
#[derive(Debug)]
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// `<temp>/<prefix>_<process id>_<n>`, empty
    pub fn new(prefix: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{prefix}_{}_{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temp dir");
        Self { path }
    }

    /// The directory
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A path inside the directory
    pub fn join(&self, name: impl AsRef<Path>) -> PathBuf {
        self.path.join(name)
    }
}

impl Deref for TempDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// A file path in its own [`TempDir`]: the directory, and the file, go when this is dropped
#[derive(Debug)]
pub struct TempPath {
    path: PathBuf,
    // dropped after `path` is no longer needed; declared last so it drops last
    _dir: TempDir,
}

impl TempPath {
    /// `name` inside a new [`TempDir`] of `prefix`
    pub fn new(prefix: &str, name: impl AsRef<Path>) -> Self {
        let dir = TempDir::new(prefix);
        Self {
            path: dir.join(name),
            _dir: dir,
        }
    }

    /// The directory holding the file
    pub fn dir(&self) -> &Path {
        self._dir.path()
    }
}

impl Deref for TempPath {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for TempPath {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}
