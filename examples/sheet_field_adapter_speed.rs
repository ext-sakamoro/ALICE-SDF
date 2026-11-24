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
    // the row's speed is the slowest of 3 measurements, each the fastest of 5 runs: the slow side is the safe
    // direction for the refusal
    let mut rows = Vec::new();
    for _ in 0..3 {
        match alice_sdf::sheet_field::gpu::measure_adapter_speed(&date) {
            Ok(r) => {
                eprintln!(
                    "adapter speed: one measurement {:.2} ns per tick",
                    r.ns_per_tick
                );
                rows.push(r);
            }
            Err(e) => {
                eprintln!("adapter speed: {e}");
                std::process::exit(1);
            }
        }
    }
    rows.sort_by(|a, b| a.ns_per_tick.total_cmp(&b.ns_per_tick));
    match rows.pop().ok_or("no measurement") {
        Ok(row) => {
            eprintln!("adapter speed: the row takes the slowest of the 3 measurements (each the fastest of 5 runs)");
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
