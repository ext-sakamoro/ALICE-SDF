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
    if a[1].contains("pair") {
        // each result is written with a check over its value and its own index: a pair that fails
        // the check was lost, torn or is another index's; a pair that passes and still differs from
        // the most common value was computed differently
        let (mut torn, mut stale, mut computed) = (0_u64, 0_u64, 0_u64);
        let mut from: std::collections::BTreeMap<(i64, i64), u64> =
            std::collections::BTreeMap::new();
        let n = first.len() / 2;
        for i in 0..n {
            let mut seen: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for (got, _) in &all {
                let (v, c) = (got[2 * i], got[2 * i + 1]);
                let good = (v.wrapping_mul(2_654_435_761)) ^ i as u32;
                if c == good {
                    *seen.entry(v).or_default() += 1;
                } else {
                    torn += 1;
                    // does it pass as some other index's pair: a stale result of the same position
                    let other = (v.wrapping_mul(2_654_435_761)) ^ c;
                    if (other as usize) < n {
                        stale += 1;
                        // how far away the pair came from, in tiles and within a tile
                        let tile_pairs = (tile / 2) as i64;
                        let delta = other as i64 - i as i64;
                        *from
                            .entry((delta.div_euclid(tile_pairs), delta.rem_euclid(tile_pairs)))
                            .or_insert(0_u64) += 1;
                    }
                }
            }
            if seen.len() > 1 {
                let most = seen.values().max().copied().unwrap_or(0);
                computed += seen
                    .values()
                    .filter(|c| **c != most)
                    .map(|c| u64::from(*c))
                    .sum::<u64>();
            }
        }
        let mut top: Vec<_> = from.into_iter().collect();
        top.sort_by_key(|x| std::cmp::Reverse(x.1));
        println!(
            "source of failing pairs (tiles away, offset in tile): {:?}",
            &top[..top.len().min(6)]
        );
        println!("pairs: failing the check {torn} (of them another index's {stale}), passing but computed differently {computed}");
    }
    let mut t: Vec<f64> = all.iter().map(|x| x.1 * 1e3).collect();
    t.sort_by(f64::total_cmp);
    println!(
        "runs {} differing {differing} items {items} median_ms {:.1}{sample}",
        runs - 1,
        t[t.len() / 2]
    );
}
