//! Matrix-Free Preconditioned Conjugate Gradient (PCG) and Contact Solvers in pure Rust.

pub struct PcgSolver {
    pub max_iter: usize,
    pub tol: f64,
}

impl PcgSolver {
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Solves A * x = b with diagonal Jacobi preconditioner where matvec is evaluated via closure.
    pub fn solve<F>(&self, n: usize, b: &[f64], diag_a: &[f64], mut matvec: F) -> (Vec<f64>, usize, f64)
    where
        F: FnMut(&[f64], &mut [f64]),
    {
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        let mut z = vec![0.0; n];
        let mut p = vec![0.0; n];
        let mut ap = vec![0.0; n];

        let inv_diag: Vec<f64> = diag_a
            .iter()
            .map(|&d| if d.abs() > 1e-12 { 1.0 / d } else { 1.0 })
            .collect();

        for i in 0..n {
            z[i] = inv_diag[i] * r[i];
            p[i] = z[i];
        }

        let mut rz_old: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();
        let norm_b: f64 = b.iter().map(|bi| bi * bi).sum::<f64>().sqrt().max(1e-16);

        let mut final_res = 1.0;
        let mut final_iter = 0;

        for iter in 0..self.max_iter {
            final_iter = iter + 1;
            matvec(&p, &mut ap);

            let p_ap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();
            if p_ap.abs() < 1e-20 {
                break;
            }

            let alpha = rz_old / p_ap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let norm_r: f64 = r.iter().map(|ri| ri * ri).sum::<f64>().sqrt();
            final_res = norm_r / norm_b;
            if final_res < self.tol {
                break;
            }

            for i in 0..n {
                z[i] = inv_diag[i] * r[i];
            }

            let rz_new: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();
            let beta = rz_new / rz_old;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz_old = rz_new;
        }

        (x, final_iter, final_res)
    }
}

/// Unilateral Contact Active-Set Newton Solver
pub struct ActiveSetContactSolver {
    pub max_iter: usize,
    pub penalty: f64,
}

impl ActiveSetContactSolver {
    pub fn new(max_iter: usize, penalty: f64) -> Self {
        Self { max_iter, penalty }
    }

    /// Evaluates normal contact gaps and returns active pairs (gap_n < 0).
    pub fn evaluate_active_set(&self, gaps_0: &[f64], u_a: &[f64], u_b: &[f64]) -> (Vec<bool>, Vec<f64>, Vec<f64>) {
        let n = gaps_0.len();
        let mut active = vec![false; n];
        let mut gaps = vec![0.0; n];
        let mut pressures = vec![0.0; n];

        for i in 0..n {
            let current_gap = gaps_0[i] + (u_b[i] - u_a[i]);
            gaps[i] = current_gap;
            if current_gap <= 0.0 {
                active[i] = true;
                pressures[i] = -self.penalty * current_gap;
            } else {
                active[i] = false;
                pressures[i] = 0.0;
            }
        }

        (active, gaps, pressures)
    }
}
