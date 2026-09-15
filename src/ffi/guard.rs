//! Panic isolation for the C ABI.
//!
//! A panic that reaches an `extern "C"` boundary aborts the whole process on
//! Rust 1.81+ (and was undefined behaviour before that), which takes the host
//! (Unity, Unreal, a Python interpreter) down with it. Every exported function
//! therefore runs its body through [`ffi_guard`]: a panic is caught inside the
//! function, its message is stored in a thread-local slot that the host can
//! read with `alice_sdf_last_error`, and the function returns the caller's
//! sentinel (null handle, `f32::MAX`, `0`, `false`, `SdfResult::Unknown`).
//!
//! Author: Moroya Sakamoto

use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};

thread_local! {
    /// Message of the most recent panic or error reported by an FFI call on
    /// this thread. Cleared by `alice_sdf_clear_last_error`.
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Record an error message for `alice_sdf_last_error`.
pub fn set_last_error(msg: impl Into<String>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(msg.into()));
}

/// Take the most recent error message (leaves the slot empty).
pub fn take_last_error() -> Option<String> {
    LAST_ERROR.with(|slot| slot.borrow_mut().take())
}

/// Clear the most recent error message.
pub fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Run `body`, converting a panic into `default` plus a recorded message.
///
/// The closure is treated as unwind-safe: every FFI body only touches its
/// arguments and the handle registries, and the registries tolerate a panic
/// while a lock is held (see `registry`), so no partially-updated state is
/// observable afterwards.
#[inline]
pub fn ffi_guard<T>(default: T, body: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(v) => v,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic with non-string payload".to_string());
            set_last_error(format!("alice-sdf FFI panic: {msg}"));
            default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_becomes_default_and_message() {
        clear_last_error();
        let v = ffi_guard(-1i32, || -> i32 { panic!("boom {}", 42) });
        assert_eq!(v, -1);
        let msg = take_last_error().expect("message recorded");
        assert!(msg.contains("boom 42"), "{msg}");
        assert!(take_last_error().is_none(), "take clears the slot");
    }

    #[test]
    fn success_leaves_slot_untouched() {
        clear_last_error();
        assert_eq!(ffi_guard(0, || 5), 5);
        assert!(take_last_error().is_none());
    }
}
