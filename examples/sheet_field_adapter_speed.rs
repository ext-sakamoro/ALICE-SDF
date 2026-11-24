//! Measure this machine's GPU adapter for the sheet field march speed table and print its row.
//!
//! Run it on an idle device and add the printed line to `src/sheet_field/adapter_speeds.tsv`:
//!
//! ```text
//! cargo run --release --features gpu --example sheet_field_adapter_speed -- $(date -u +%F)
//! ```

fn main() {
    let Some(date) = std::env::args().nth(1).filter(|d| {
        d.len() == 10
            && d.bytes().enumerate().all(|(i, b)| {
                if i == 4 || i == 7 {
                    b == b'-'
                } else {
                    b.is_ascii_digit()
                }
            })
    }) else {
        eprintln!("usage: sheet_field_adapter_speed YYYY-MM-DD (the date the row is measured)");
        std::process::exit(2);
    };
    match alice_sdf::sheet_field::gpu::measure_adapter_speed(&date) {
        Ok(row) => {
            let full =
                row.ns_per_tick * f64::from(alice_sdf::sheet_field::gpu::MARCH_WORK_BUDGET) * 1e-9;
            eprintln!(
                "adapter speed: {:.2} ns per tick, full budget {full:.3} s, limit {:.3} s",
                row.ns_per_tick,
                alice_sdf::sheet_field::gpu::DISPATCH_WAIT.as_secs_f64() / 3.0
            );
            println!("{}", row.tsv());
        }
        Err(e) => {
            eprintln!("adapter speed: {e}");
            std::process::exit(1);
        }
    }
}
