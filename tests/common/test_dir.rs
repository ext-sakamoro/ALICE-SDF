//! A scratch directory per test (the integration tests' copy of src/test_dir.rs)
//!
//! A fixed name under the system temporary directory is shared by every run on
//! the machine: two `cargo test` processes (or two CI jobs on one runner)
//! overwrite and delete each other's files. [`TestDir::new`] creates a fresh
//! directory with `create_dir`, which fails if the name is taken, and retries
//! with the next counter value, so no two live directories coincide. The
//! directory and its contents are removed on drop.
//!
//! `std::env::temp_dir` is a disallowed method (clippy.toml); this file is the
//! only place in `tests/` that calls it. The library's unit tests use
//! `src/test_dir.rs`, which an integration test cannot reach (it is
//! crate-private and compiled only under `cfg(test)`).

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

/// A directory that belongs to one test and is removed when dropped
#[allow(dead_code)]
pub struct TestDir {
    path: PathBuf,
}

#[allow(dead_code)]
impl TestDir {
    /// A new, empty directory whose name starts with `alice_sdf_<label>_`
    pub fn new(label: &str) -> Self {
        #[allow(
            clippy::disallowed_methods,
            reason = "the one call in tests/: every test directory is made here and is unique"
        )]
        let base = std::env::temp_dir();
        let pid = std::process::id();
        loop {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!("alice_sdf_{label}_{pid}_{n}"));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self { path },
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
                Err(e) => panic!("cannot create {}: {e}", path.display()),
            }
        }
    }

    /// The directory itself
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A path for `name` inside the directory
    pub fn join(&self, name: impl AsRef<Path>) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
