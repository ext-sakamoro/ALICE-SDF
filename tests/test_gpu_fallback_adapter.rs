//! `ALICE_SDF_TEST_FALLBACK_ADAPTER=1` gives a software adapter, or no GPU tests at all.
//!
//! With the variable set (the lavapipe CI lane), a context must be on a software adapter (device type `Cpu`): a run
//! on the real GPU must never read as a software run. A system without a fallback adapter (macOS has none for wgpu)
//! is refused (`GpuError::DeviceCreation` starting with `refusal::ADAPTER_REFUSED`), and this test fails saying so. Without the variable the test only
//! checks that this build has the guard.
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_sdf::compiled::GpuError;
use alice_sdf::gpu_context::{refusal, slot_guard, GpuContext};

#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "the constant is the feature this build has: the assertion is the check that it is on"
)]
fn the_fallback_adapter_is_a_software_one() {
    assert!(slot_guard::ENABLED, "built without gpu-slot-guard");
    if std::env::var("ALICE_SDF_TEST_FALLBACK_ADAPTER").as_deref() != Ok("1") {
        eprintln!(
            "ALICE_SDF_TEST_FALLBACK_ADAPTER is not 1: the fallback adapter is not exercised here"
        );
        return;
    }
    match GpuContext::request("fallback adapter test") {
        Ok(context) => {
            let info = context.adapter_info();
            println!(
                "[fallback adapter] name {} type {:?} backend {:?} driver {} {}",
                info.name, info.device_type, info.backend, info.driver, info.driver_info
            );
            assert_eq!(
                info.device_type,
                wgpu::DeviceType::Cpu,
                "not a software adapter: {}",
                info.name
            );
        }
        Err(e @ GpuError::NoAdapter) => {
            panic!(
                "this system offers no fallback (software) adapter, so no GPU test runs here: {e}"
            )
        }
        Err(GpuError::DeviceCreation(m)) if m.starts_with(refusal::ADAPTER_REFUSED) => {
            let e = m;
            panic!(
                "this system offers no fallback (software) adapter, so no GPU test runs here: {e}"
            )
        }
        Err(e) => panic!("request: {e}"),
    }
}
