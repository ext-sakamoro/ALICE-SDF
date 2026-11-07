//! `array_fault <kernel> <count> <tile> <runs>`: passes that differ from the first, and how
use alice_sdf::mesh::gpu_bvh::run_u32;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&a[1]).unwrap();
    let (count, tile, runs): (u32, u32, u32) = (
        a[2].parse().unwrap(),
        a[3].parse().unwrap(),
        a[4].parse().unwrap(),
    );
    let all = run_u32(&text, count, tile, runs).expect("kernel");
    let first = &all[0].0;
    let (mut differing, mut items) = (0, 0usize);
    let mut sample = String::new();
    for (got, _) in &all[1..] {
        let d: Vec<usize> = (0..first.len()).filter(|&i| first[i] != got[i]).collect();
        if !d.is_empty() {
            differing += 1;
            items += d.len();
            if sample.is_empty() {
                sample = format!(" first at {:?}", &d[..d.len().min(8)]);
            }
        }
    }
    let mut t: Vec<f64> = all.iter().map(|x| x.1 * 1e3).collect();
    t.sort_by(f64::total_cmp);
    println!(
        "runs {} differing {differing} items {items} median_ms {:.1}{sample}",
        runs - 1,
        t[t.len() / 2]
    );
}
