//! A scratch directory per test (compiled only for the library's unit tests)
//!
//! A fixed name under the system temporary directory is shared by every run on
//! the machine: two `cargo test` processes (or two CI jobs on one runner)
//! overwrite and delete each other's files. [`TestDir::new`] creates a fresh
//! directory with `create_dir`, which fails if the name is taken, and retries
//! with the next counter value, so no two live directories coincide. The
//! directory and its contents are removed on drop.
//!
//! `std::env::temp_dir` is a disallowed method (clippy.toml); this file is the
//! only place in `src/` that calls it. `tests/common/mod.rs` holds the same
//! helper for the integration tests.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

/// A directory that belongs to one test and is removed when dropped
pub struct TestDir {
    path: PathBuf,
}

impl TestDir {
    /// A new, empty directory whose name starts with `alice_sdf_<label>_`
    pub fn new(label: &str) -> Self {
        #[allow(
            clippy::disallowed_methods,
            reason = "the one call in src/: every test directory is made here and is unique"
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

#[cfg(test)]
mod tests {
    use super::TestDir;

    #[test]
    fn directories_are_distinct_and_removed() {
        let a = TestDir::new("test_dir_self");
        let b = TestDir::new("test_dir_self");
        assert_ne!(a.path(), b.path());
        assert!(a.path().is_dir() && b.path().is_dir());
        std::fs::write(a.join("f"), b"x").unwrap();
        let kept = a.path().to_path_buf();
        drop(a);
        assert!(!kept.exists());
        assert!(b.path().is_dir());
    }

    #[test]
    fn a_taken_name_is_skipped() {
        let a = TestDir::new("test_dir_taken");
        // the next name this process would pick is occupied by a stale directory
        let next = super::NEXT.load(std::sync::atomic::Ordering::Relaxed);
        let stale = a.path().with_file_name(format!(
            "alice_sdf_test_dir_taken_{}_{next}",
            std::process::id()
        ));
        std::fs::create_dir(&stale).unwrap();
        std::fs::write(stale.join("keep"), b"stale").unwrap();
        let b = TestDir::new("test_dir_taken");
        assert_ne!(b.path(), stale.as_path());
        assert_eq!(std::fs::read(stale.join("keep")).unwrap(), b"stale");
        std::fs::remove_dir_all(&stale).unwrap();
    }
}
