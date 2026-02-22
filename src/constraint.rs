//! Parametric SDF constraint solver
//!
//! Assigns parametric variables to SDF node properties (radius, position, etc.)
//! and solves geometric constraints (distance, coincidence, fixed value) using
//! Gauss-Newton optimization.
//!
//! Author: Moroya Sakamoto

// ── Types ────────────────────────────────────────────────────

/// Identifier for a parametric variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId(pub u32);

/// A geometric constraint between parameters.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// What kind of constraint.
    pub kind: ConstraintKind,
    /// Target value for the constraint (residual = actual - target).
    pub target: f64,
}

/// Types of geometric constraints.
#[derive(Debug, Clone)]
pub enum ConstraintKind {
    /// Fix a parameter to a specific value.
    Fixed {
        /// Parameter to fix.
        param: ParamId,
    },
    /// Distance between two parameters equals target.
    Distance {
        /// First parameter.
        param_a: ParamId,
        /// Second parameter.
        param_b: ParamId,
    },
    /// Sum of two parameters equals target.
    Sum {
        /// First parameter.
        param_a: ParamId,
        /// Second parameter.
        param_b: ParamId,
    },
    /// Ratio of two parameters equals target (a / b = target).
    Ratio {
        /// Numerator parameter.
        param_a: ParamId,
        /// Denominator parameter.
        param_b: ParamId,
    },
}

/// Result of a constraint solve iteration.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Final parameter values.
    pub params: Vec<f64>,
    /// Final total residual (sum of squared constraint errors).
    pub residual: f64,
    /// Number of iterations performed.
    pub iterations: u32,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
}

// ── Solver ───────────────────────────────────────────────────

/// Gauss-Newton constraint solver for parametric SDFs.
pub struct ConstraintSolver {
    /// Current parameter values.
    params: Vec<f64>,
    /// Active constraints.
    constraints: Vec<Constraint>,
}

impl ConstraintSolver {
    /// Create a new solver with the given initial parameter values.
    pub fn new(params: Vec<f64>) -> Self {
        Self {
            params,
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to the solver.
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Fix a parameter to a value.
    pub fn fix(&mut self, param: ParamId, value: f64) {
        self.add_constraint(Constraint {
            kind: ConstraintKind::Fixed { param },
            target: value,
        });
    }

    /// Constrain the distance between two parameters.
    pub fn distance(&mut self, a: ParamId, b: ParamId, target: f64) {
        self.add_constraint(Constraint {
            kind: ConstraintKind::Distance {
                param_a: a,
                param_b: b,
            },
            target,
        });
    }

    /// Constrain the sum of two parameters.
    pub fn sum(&mut self, a: ParamId, b: ParamId, target: f64) {
        self.add_constraint(Constraint {
            kind: ConstraintKind::Sum {
                param_a: a,
                param_b: b,
            },
            target,
        });
    }

    /// Get the current parameter value.
    pub fn get(&self, id: ParamId) -> f64 {
        self.params[id.0 as usize]
    }

    /// Set a parameter value.
    pub fn set(&mut self, id: ParamId, value: f64) {
        self.params[id.0 as usize] = value;
    }

    /// Number of parameters.
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// Number of constraints.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Compute the residual for a single constraint.
    fn residual(&self, c: &Constraint) -> f64 {
        match &c.kind {
            ConstraintKind::Fixed { param } => self.params[param.0 as usize] - c.target,
            ConstraintKind::Distance { param_a, param_b } => {
                let a = self.params[param_a.0 as usize];
                let b = self.params[param_b.0 as usize];
                (a - b).abs() - c.target
            }
            ConstraintKind::Sum { param_a, param_b } => {
                let a = self.params[param_a.0 as usize];
                let b = self.params[param_b.0 as usize];
                (a + b) - c.target
            }
            ConstraintKind::Ratio { param_a, param_b } => {
                let a = self.params[param_a.0 as usize];
                let b = self.params[param_b.0 as usize];
                if b.abs() < 1e-12 {
                    a - c.target // degenerate: treat as fixed
                } else {
                    a / b - c.target
                }
            }
        }
    }

    /// Compute the Jacobian row for a constraint (∂residual/∂params).
    fn jacobian_row(&self, c: &Constraint) -> Vec<f64> {
        let n = self.params.len();
        let mut row = vec![0.0; n];

        match &c.kind {
            ConstraintKind::Fixed { param } => {
                row[param.0 as usize] = 1.0;
            }
            ConstraintKind::Distance { param_a, param_b } => {
                let a = self.params[param_a.0 as usize];
                let b = self.params[param_b.0 as usize];
                let sign = if a >= b { 1.0 } else { -1.0 };
                row[param_a.0 as usize] = sign;
                row[param_b.0 as usize] = -sign;
            }
            ConstraintKind::Sum { param_a, param_b } => {
                row[param_a.0 as usize] = 1.0;
                row[param_b.0 as usize] = 1.0;
            }
            ConstraintKind::Ratio { param_a, param_b } => {
                let b = self.params[param_b.0 as usize];
                if b.abs() < 1e-12 {
                    row[param_a.0 as usize] = 1.0;
                } else {
                    let a = self.params[param_a.0 as usize];
                    row[param_a.0 as usize] = 1.0 / b;
                    row[param_b.0 as usize] = -a / (b * b);
                }
            }
        }

        row
    }

    /// Solve the constraint system using Gauss-Newton iterations.
    pub fn solve(&mut self, max_iter: u32, tolerance: f64) -> SolveResult {
        let n = self.params.len();
        let m = self.constraints.len();

        if m == 0 {
            return SolveResult {
                params: self.params.clone(),
                residual: 0.0,
                iterations: 0,
                converged: true,
            };
        }

        let mut iterations = 0;

        for _ in 0..max_iter {
            iterations += 1;

            // Compute residuals
            let residuals: Vec<f64> = self.constraints.iter().map(|c| self.residual(c)).collect();
            let total_residual: f64 = residuals.iter().map(|r| r * r).sum();

            if total_residual < tolerance * tolerance {
                return SolveResult {
                    params: self.params.clone(),
                    residual: total_residual.sqrt(),
                    iterations,
                    converged: true,
                };
            }

            // Build J^T J and J^T r
            let mut jtj = vec![0.0; n * n];
            let mut jtr = vec![0.0; n];

            for (i, c) in self.constraints.iter().enumerate() {
                let row = self.jacobian_row(c);
                let r = residuals[i];
                for j in 0..n {
                    jtr[j] += row[j] * r;
                    for k in 0..n {
                        jtj[j * n + k] += row[j] * row[k];
                    }
                }
            }

            // Levenberg-Marquardt damping
            let lambda = 1e-6;
            for i in 0..n {
                jtj[i * n + i] += lambda;
            }

            // Solve (J^T J + λI) δ = J^T r via simple Gauss elimination
            let delta = solve_linear(n, &jtj, &jtr);

            // Update parameters
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                self.params[i] -= delta[i];
            }
        }

        let final_residual: f64 = self
            .constraints
            .iter()
            .map(|c| {
                let r = self.residual(c);
                r * r
            })
            .sum::<f64>()
            .sqrt();

        SolveResult {
            params: self.params.clone(),
            residual: final_residual,
            iterations,
            converged: final_residual < tolerance,
        }
    }
}

/// Simple Gaussian elimination for small systems.
fn solve_linear(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination
    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..=n {
                let idx_a = col * (n + 1) + j;
                let idx_b = max_row * (n + 1) + j;
                aug.swap(idx_a, idx_b);
            }
        }

        let pivot = aug[col * (n + 1) + col];
        if pivot.abs() < 1e-15 {
            continue;
        }

        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let pivot = aug[i * (n + 1) + i];
        if pivot.abs() < 1e-15 {
            continue;
        }
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / pivot;
    }

    x
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_constraint() {
        let mut solver = ConstraintSolver::new(vec![0.0]);
        solver.fix(ParamId(0), 5.0);
        let result = solver.solve(100, 1e-6);
        assert!(result.converged);
        assert!((result.params[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn two_fixed_constraints() {
        let mut solver = ConstraintSolver::new(vec![0.0, 0.0]);
        solver.fix(ParamId(0), 3.0);
        solver.fix(ParamId(1), 7.0);
        let result = solver.solve(100, 1e-6);
        assert!(result.converged);
        assert!((result.params[0] - 3.0).abs() < 1e-5);
        assert!((result.params[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn distance_constraint() {
        let mut solver = ConstraintSolver::new(vec![1.0, 5.0]);
        solver.distance(ParamId(0), ParamId(1), 3.0);
        solver.fix(ParamId(0), 1.0);
        let result = solver.solve(100, 1e-6);
        assert!(result.converged);
        assert!((result.params[0] - 1.0).abs() < 1e-4);
        assert!(((result.params[1] - result.params[0]).abs() - 3.0).abs() < 1e-4);
    }

    #[test]
    fn sum_constraint() {
        let mut solver = ConstraintSolver::new(vec![1.0, 2.0]);
        solver.sum(ParamId(0), ParamId(1), 10.0);
        solver.fix(ParamId(0), 4.0);
        let result = solver.solve(100, 1e-6);
        assert!(result.converged);
        assert!((result.params[0] - 4.0).abs() < 1e-4);
        assert!((result.params[1] - 6.0).abs() < 1e-4);
    }

    #[test]
    fn no_constraints() {
        let mut solver = ConstraintSolver::new(vec![1.0, 2.0, 3.0]);
        let result = solver.solve(100, 1e-6);
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.residual, 0.0);
    }

    #[test]
    fn solver_accessors() {
        let solver = ConstraintSolver::new(vec![1.0, 2.0]);
        assert_eq!(solver.param_count(), 2);
        assert_eq!(solver.constraint_count(), 0);
        assert_eq!(solver.get(ParamId(0)), 1.0);
        assert_eq!(solver.get(ParamId(1)), 2.0);
    }

    #[test]
    fn solver_set() {
        let mut solver = ConstraintSolver::new(vec![1.0, 2.0]);
        solver.set(ParamId(0), 42.0);
        assert_eq!(solver.get(ParamId(0)), 42.0);
    }

    #[test]
    fn convergence_report() {
        let mut solver = ConstraintSolver::new(vec![0.0]);
        solver.fix(ParamId(0), 1.0);
        let result = solver.solve(5, 1e-10);
        assert!(result.converged);
        assert!(result.residual < 1e-8);
    }

    #[test]
    fn multiple_constraints_system() {
        // 3 params, 3 constraints: fully determined
        let mut solver = ConstraintSolver::new(vec![0.0, 0.0, 0.0]);
        solver.fix(ParamId(0), 1.0);
        solver.sum(ParamId(0), ParamId(1), 5.0); // p1 = 4
        solver.sum(ParamId(1), ParamId(2), 10.0); // p2 = 6
        let result = solver.solve(100, 1e-6);
        assert!(result.converged, "residual={}", result.residual);
        assert!((result.params[0] - 1.0).abs() < 1e-3);
        assert!((result.params[1] - 4.0).abs() < 1e-3);
        assert!((result.params[2] - 6.0).abs() < 1e-3);
    }
}
