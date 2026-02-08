//! Nelder-Mead simplex optimizer (pure Rust, no dependencies)
//!
//! Minimizes a cost function `f: &[f32] -> f64` over a parameter vector.

/// Nelder-Mead simplex optimizer result
pub struct OptimizeResult {
    /// Best parameter vector found
    pub params: Vec<f32>,
    /// Cost at the best point
    pub cost: f64,
    /// Number of iterations performed
    pub iterations: u32,
}

/// Run Nelder-Mead optimization
///
/// - `initial`: starting parameter vector
/// - `step_sizes`: initial simplex step sizes per parameter
/// - `max_iter`: maximum iterations
/// - `cost_fn`: function to minimize
pub fn nelder_mead<F>(
    initial: &[f32],
    step_sizes: &[f32],
    max_iter: u32,
    cost_fn: F,
) -> OptimizeResult
where
    F: Fn(&[f32]) -> f64,
{
    let n = initial.len();
    let np1 = n + 1;

    // Reflection/expansion/contraction coefficients
    let alpha = 1.0_f64;
    let gamma = 2.0_f64;
    let rho = 0.5_f64;
    let sigma = 0.5_f64;

    // Initialize simplex: n+1 vertices
    let mut vertices: Vec<Vec<f32>> = Vec::with_capacity(np1);
    vertices.push(initial.to_vec());
    for i in 0..n {
        let mut v = initial.to_vec();
        v[i] += step_sizes[i];
        vertices.push(v);
    }

    // Evaluate costs
    let mut costs: Vec<f64> = vertices.iter().map(|v| cost_fn(v)).collect();

    let mut scratch = vec![0.0f32; n];
    let mut reflected = vec![0.0f32; n];
    let mut expanded = vec![0.0f32; n];
    let mut contracted = vec![0.0f32; n];

    let mut iter = 0u32;
    while iter < max_iter {
        iter += 1;

        // Sort vertices by cost
        let mut indices: Vec<usize> = (0..np1).collect();
        indices.sort_by(|&a, &b| costs[a].partial_cmp(&costs[b]).unwrap());

        let best_idx = indices[0];
        let worst_idx = indices[np1 - 1];
        let second_worst_idx = indices[np1 - 2];

        // Convergence check
        let cost_range = costs[worst_idx] - costs[best_idx];
        if cost_range < 1e-10 {
            break;
        }

        // Compute centroid (excluding worst)
        for s in scratch.iter_mut() {
            *s = 0.0;
        }
        for &idx in &indices[..n] {
            for (j, s) in scratch.iter_mut().enumerate() {
                *s += vertices[idx][j];
            }
        }
        for s in scratch.iter_mut() {
            *s /= n as f32;
        }

        // Reflection
        for j in 0..n {
            reflected[j] = scratch[j] + (alpha as f32) * (scratch[j] - vertices[worst_idx][j]);
        }
        let cost_r = cost_fn(&reflected);

        if cost_r < costs[second_worst_idx] && cost_r >= costs[best_idx] {
            // Accept reflection
            vertices[worst_idx].copy_from_slice(&reflected);
            costs[worst_idx] = cost_r;
            continue;
        }

        if cost_r < costs[best_idx] {
            // Try expansion
            for j in 0..n {
                expanded[j] = scratch[j] + (gamma as f32) * (reflected[j] - scratch[j]);
            }
            let cost_e = cost_fn(&expanded);
            if cost_e < cost_r {
                vertices[worst_idx].copy_from_slice(&expanded);
                costs[worst_idx] = cost_e;
            } else {
                vertices[worst_idx].copy_from_slice(&reflected);
                costs[worst_idx] = cost_r;
            }
            continue;
        }

        // Contraction
        for j in 0..n {
            contracted[j] = scratch[j] + (rho as f32) * (vertices[worst_idx][j] - scratch[j]);
        }
        let cost_c = cost_fn(&contracted);

        if cost_c < costs[worst_idx] {
            vertices[worst_idx].copy_from_slice(&contracted);
            costs[worst_idx] = cost_c;
            continue;
        }

        // Shrink: move all vertices toward best
        let best = vertices[best_idx].clone();
        for i in 0..np1 {
            if i == best_idx {
                continue;
            }
            for j in 0..n {
                vertices[i][j] = best[j] + (sigma as f32) * (vertices[i][j] - best[j]);
            }
            costs[i] = cost_fn(&vertices[i]);
        }
    }

    // Find best
    let mut best_idx = 0;
    for i in 1..np1 {
        if costs[i] < costs[best_idx] {
            best_idx = i;
        }
    }

    OptimizeResult {
        params: vertices[best_idx].clone(),
        cost: costs[best_idx],
        iterations: iter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimize_quadratic() {
        // Minimize (x-3)² + (y-5)²
        let result = nelder_mead(&[0.0, 0.0], &[1.0, 1.0], 1000, |p| {
            let dx = (p[0] - 3.0) as f64;
            let dy = (p[1] - 5.0) as f64;
            dx * dx + dy * dy
        });

        assert!(
            (result.params[0] - 3.0).abs() < 0.01,
            "x={}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 5.0).abs() < 0.01,
            "y={}",
            result.params[1]
        );
        assert!(result.cost < 0.001);
    }

    #[test]
    fn test_minimize_rosenbrock() {
        // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
        let result = nelder_mead(&[-1.0, -1.0], &[0.5, 0.5], 5000, |p| {
            let x = p[0] as f64;
            let y = p[1] as f64;
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        });

        assert!(
            (result.params[0] - 1.0).abs() < 0.1,
            "x={}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 1.0).abs() < 0.1,
            "y={}",
            result.params[1]
        );
    }
}
