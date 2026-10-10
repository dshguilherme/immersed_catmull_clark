//! 3D isogeometric SIMP topology optimization of a cantilever, a port of the CPU
//! path of the MATLAB reference `src/fastformation/topopt_iga_3d.m` (validated in
//! `rust/tests/topopt_vs_matlab.rs`).
//!
//! Discretization: maximal-regularity B-splines of degree 2 on the box
//! [0, L] x [0, h] x [0, w], element-wise densities, SIMP with Emin = 1e-3,
//! sensitivity filter (Sigmund's heuristic) and optimality-criteria update with
//! move limit 0.2. Clamp at x = 0, downward load on control points near the
//! bottom-centre of the x = L face. Linear systems are solved with Jacobi PCG
//! (the MATLAB CPU path uses a direct solver; the tolerance is set tight enough to
//! reproduce it).

use crate::iga::{elasticity_element_matrices, lame, unravel, ElementMatrices, SpaceBox};
use crate::solver::PcgSolver;
use crate::sparse::{AssemblyPattern, CsrMatrix};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct CantileverTopOptConfig {
    pub nel: [usize; 3],
    pub size: [f64; 3],
    pub degree: usize,
    pub volfrac: f64,
    pub penal: f64,
    /// Filter radius in units of the mean element size.
    pub rmin: f64,
    pub max_iter: usize,
    pub young: f64,
    pub poisson: f64,
    pub emin: f64,
    pub move_limit: f64,
    pub pcg_tol: f64,
}

impl Default for CantileverTopOptConfig {
    fn default() -> Self {
        Self {
            nel: [16, 8, 4],
            size: [1.2, 0.6, 0.3],
            degree: 2,
            volfrac: 0.3,
            penal: 3.0,
            rmin: 1.5,
            max_iter: 40,
            young: 1.0,
            poisson: 0.3,
            emin: 1e-3,
            move_limit: 0.2,
            pcg_tol: 1e-10,
        }
    }
}

pub struct TopOptHistory {
    pub compliance: Vec<f64>,
    pub change: Vec<f64>,
    pub pcg_iterations: Vec<usize>,
    pub density: Vec<f64>,
}

pub struct CantileverTopOpt {
    pub cfg: CantileverTopOptConfig,
    pub sp: SpaceBox,
    em: ElementMatrices,
    pub force: Vec<f64>,
    pub free: Vec<bool>,
    filter: Vec<Vec<(usize, f64)>>,
    filter_sum: Vec<f64>,
    pattern: AssemblyPattern,
}

impl CantileverTopOpt {
    pub fn new(cfg: CantileverTopOptConfig) -> Self {
        let bounds = [[0.0, cfg.size[0]], [0.0, cfg.size[1]], [0.0, cfg.size[2]]];
        let sp = SpaceBox::new(&bounds, &cfg.nel, cfg.degree);
        let (lambda, mu) = lame(cfg.young, cfg.poisson);
        let em = elasticity_element_matrices(&sp, lambda, mu);

        // Clamp x = 0; load on control points with i1 = last, |i2 - 1| <= 1, |i3 - ncp3/2| <= 1 (1-based).
        let nd = &sp.ndof_dir;
        let mut free = vec![true; sp.ndof];
        let mut load = Vec::new();
        for i in 0..sp.ndof_sc {
            let s = unravel(i, nd);
            if s[0] == 0 {
                for c in 0..3 {
                    free[c * sp.ndof_sc + i] = false;
                }
            }
            let (i1, i2, i3) = (s[0] + 1, s[1] as f64 + 1.0, s[2] as f64 + 1.0);
            if i1 == nd[0] && (i2 - 1.0).abs() <= 1.0 && (i3 - nd[2] as f64 / 2.0).abs() <= 1.0 {
                load.push(i);
            }
        }
        let mut force = vec![0.0; sp.ndof];
        for &i in &load {
            force[sp.ndof_sc + i] = -1.0 / load.len() as f64;
        }

        // Sensitivity filter on element centres, radius rmin * mean(h).
        let h = sp.element_size();
        let r = cfg.rmin * (h[0] + h[1] + h[2]) / 3.0;
        let centres: Vec<[f64; 3]> = (0..sp.nel)
            .map(|e| {
                let s = unravel(e, &sp.nel_dir);
                [(s[0] as f64 + 0.5) * h[0], (s[1] as f64 + 0.5) * h[1], (s[2] as f64 + 0.5) * h[2]]
            })
            .collect();
        let filter: Vec<Vec<(usize, f64)>> = (0..sp.nel)
            .into_par_iter()
            .map(|e| {
                let ce = centres[e];
                (0..sp.nel)
                    .filter_map(|f| {
                        let cf = centres[f];
                        let d2 = (ce[0] - cf[0]).powi(2) + (ce[1] - cf[1]).powi(2) + (ce[2] - cf[2]).powi(2);
                        if d2 <= r * r {
                            Some((f, (r - d2.sqrt()).max(0.0)))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        let filter_sum = filter.iter().map(|row| row.iter().map(|(_, w)| w).sum()).collect();
        let pattern = AssemblyPattern::new(sp.ndof, &sp.connectivity, sp.nsh, sp.nel);
        Self { cfg, sp, em, force, free, filter, filter_sum, pattern }
    }

    fn assemble(&self, x: &[f64]) -> CsrMatrix {
        let scale: Vec<f64> = x.iter().map(|&xe| self.cfg.emin + (1.0 - self.cfg.emin) * xe.powf(self.cfg.penal)).collect();
        self.pattern.assemble(|e| self.em.of_element(e), &scale)
    }

    /// Solves K(x) u = F on the free DOFs, warm-started from `u0`.
    pub fn solve_state(&self, x: &[f64], u0: &[f64]) -> (Vec<f64>, usize, f64) {
        let k = self.assemble(x);
        let n = self.sp.ndof;
        let free = &self.free;
        // Solve for the correction from the warm start: K du = F - K u0.
        let mut ku0 = vec![0.0; n];
        k.matvec(u0, &mut ku0);
        let b: Vec<f64> = (0..n).map(|i| if free[i] { self.force[i] - ku0[i] } else { 0.0 }).collect();
        let nb: f64 = (0..n).filter(|&i| free[i]).map(|i| self.force[i] * self.force[i]).sum::<f64>().sqrt();
        let rb: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let diag: Vec<f64> = k.diagonal().iter().map(|&d| d.max(1e-12)).collect();
        let tol = if rb > 0.0 { (self.cfg.pcg_tol * nb / rb).min(0.5) } else { 1.0 };
        let mut pm = vec![0.0; n];
        let mut tmp = vec![0.0; n];
        let (du, iters, res) = PcgSolver::new(20 * n, tol).solve(n, &b, &diag, |p, ap| {
            for i in 0..n {
                pm[i] = if free[i] { p[i] } else { 0.0 };
            }
            k.matvec(&pm, &mut tmp);
            for i in 0..n {
                ap[i] = if free[i] { tmp[i] } else { 0.0 };
            }
        });
        let u: Vec<f64> = (0..n).map(|i| if free[i] { u0[i] + du[i] } else { 0.0 }).collect();
        (u, iters, res * rb / nb.max(1e-300))
    }

    /// Unpenalized element strain energies u_e^T K_e u_e.
    pub fn element_energies(&self, u: &[f64]) -> Vec<f64> {
        let nsh = self.sp.nsh;
        (0..self.sp.nel)
            .into_par_iter()
            .map(|e| {
                let ke = self.em.of_element(e);
                let ce = &self.sp.connectivity[e * nsh..(e + 1) * nsh];
                let mut s = 0.0;
                for a in 0..nsh {
                    let ua = u[ce[a]];
                    let mut row = 0.0;
                    for b in 0..nsh {
                        row += ke[a * nsh + b] * u[ce[b]];
                    }
                    s += ua * row;
                }
                s
            })
            .collect()
    }

    /// Runs the optimization (same stopping rule as MATLAB: change < 0.01 after 15 iterations).
    pub fn run<F: FnMut(usize, f64, f64, f64)>(&self, mut report: F) -> TopOptHistory {
        let cfg = &self.cfg;
        let nel = self.sp.nel;
        let mut x = vec![cfg.volfrac; nel];
        let mut u = vec![0.0; self.sp.ndof];
        let mut hist = TopOptHistory { compliance: Vec::new(), change: Vec::new(), pcg_iterations: Vec::new(), density: Vec::new() };
        for iter in 1..=cfg.max_iter {
            let x_old = x.clone();
            let (un, iters, _) = self.solve_state(&x, &u);
            u = un;
            let c: f64 = self.force.iter().zip(&u).map(|(f, v)| f * v).sum();
            let ee = self.element_energies(&u);
            let dc_raw: Vec<f64> = (0..nel).map(|e| -cfg.penal * (1.0 - cfg.emin) * x[e].powf(cfg.penal - 1.0) * ee[e]).collect();
            let dc: Vec<f64> = (0..nel)
                .map(|e| {
                    let s: f64 = self.filter[e].iter().map(|&(f, w)| w * x[f] * dc_raw[f]).sum();
                    s / (self.filter_sum[e] * x[e].max(1e-3))
                })
                .collect();
            // OC bisection (the update is the candidate of the last bisection step, as in MATLAB)
            let (mut l1, mut l2) = (0.0, 2.0 * dc.iter().map(|v| v.abs()).fold(0.0, f64::max));
            let mut cand = x_old.clone();
            while (l2 - l1) / (l1 + l2 + 1e-10) > 1e-4 {
                let lmid = 0.5 * (l1 + l2);
                for e in 0..nel {
                    let be = (-dc[e] / lmid).sqrt();
                    cand[e] = (x_old[e] * be).min(x_old[e] + cfg.move_limit).min(1.0).max(x_old[e] - cfg.move_limit).max(0.001);
                }
                let mean = cand.iter().sum::<f64>() / nel as f64;
                if mean > cfg.volfrac {
                    l1 = lmid;
                } else {
                    l2 = lmid;
                }
            }
            x = cand;
            let change = x.iter().zip(&x_old).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            hist.compliance.push(c.abs());
            hist.change.push(change);
            hist.pcg_iterations.push(iters);
            report(iter, c.abs(), x.iter().sum::<f64>() / nel as f64, change);
            if change < 0.01 && iter >= 15 {
                break;
            }
        }
        hist.density = x;
        hist
    }
}
