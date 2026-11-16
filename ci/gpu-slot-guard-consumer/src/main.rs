//! Requests a GPU context the way an application does (CI builds this, it does not run it)
fn main() {
    match alice_sdf::gpu_context::GpuContext::request("consumer") {
        Ok(c) => println!("adapter: {}", c.adapter_info().name),
        Err(e) => println!("no GPU: {e}"),
    }
}
