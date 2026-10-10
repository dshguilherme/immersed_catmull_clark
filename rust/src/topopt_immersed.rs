//! Immersed SIMP topology optimization inside a CAD B-Rep, a port of the MATLAB
//! reference `src/immersed/topopt_immersed_iga_3d.m` (validated in
//! `rust/tests/imm_topopt_vs_matlab.rs`).
//!
//! Background B-spline grid (p = 2) around the B-Rep; design domain = cells with volume
//! fraction >= 5%; element stiffness scale w_e (Emin + (1 - Emin) rho^p), Emin for cells
//! outside the design domain; sensitivity filter on the design domain; OC update with the
//! CAD-weighted volume constraint; bottom face clamped, central load on the top face.
//! The stabilization is selectable: the MATLAB legacy one (for parity) or the consistent
//! ghost penalty (recommended; see `immersed`). Solves with double-precision Jacobi PCG
//! (the MATLAB code runs single-precision PCG on the GPU).

use crate::cut_cell::TriangleMesh3D;
use crate::iga::{elasticity_element_matrices, lame, unravel, ElementMatrices, SpaceBox};
use crate::immersed::{element_weights, ghost_penalty, legacy_stabilization, padded_bounds, ParityRayCaster, Stabilization};
use crate::solver::PcgSolver;
use crate::sparse::{AssemblyPattern, CsrMatrix};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct ImmersedTopOptConfig {
    pub grid_res: [usize; 3],
    pub volfrac: f64,
    pub penal: f64,
    /// Filter radius in units of the mean cell size.
    pub rmin: f64,
    pub max_iter: usize,
    pub young: f64,
    pub poisson: f64,
    pub emin: f64,
    pub stabilization: Stabilization,
    pub pcg_tol: f64,
    pub pcg_maxit: usize,
}

impl Default for ImmersedTopOptConfig {
    fn default() -> Self {
        Self {
            grid_res: [28, 20, 16],
            volfrac: 0.35,
            penal: 3.0,
            rmin: 1.8,
            max_iter: 30,
            young: 1.0,
            poisson: 0.3,
            emin: 1e-4,
            stabilization: Stabilization::GhostPenalty { gamma: 1e-2 },
            pcg_tol: 1e-8,
            pcg_maxit: 20000,
        }
    }
}

pub struct ImmersedTopOpt {
    pub cfg: ImmersedTopOptConfig,
    pub sp: SpaceBox,
    pub weights: Vec<f64>,
    pub active: Vec<bool>,
    pub cad_volume: f64,
    em: ElementMatrices,
    pattern: AssemblyPattern,
    stab: CsrMatrix,
    pub force: Vec<f64>,
    pub free: Vec<bool>,
    filter: Vec<Vec<(usize, f64)>>,
    filter_sum: Vec<f64>,
}

pub struct ImmersedTopOptHistory {
    pub compliance: Vec<f64>,
    pub volume: Vec<f64>,
    pub change: Vec<f64>,
    pub pcg_iterations: Vec<usize>,
    pub density: Vec<f64>,
}

impl ImmersedTopOpt {
    pub fn new(brep: &TriangleMesh3D, cfg: ImmersedTopOptConfig) -> Self {
        let gb = padded_bounds(brep, 0.05);
        let res = cfg.grid_res;
        let sp = SpaceBox::new(&gb, &res, 2);
        let rc = ParityRayCaster::new(brep);
        let (weights, status) = element_weights(&rc, &gb, res, [4, 4, 4]);
        let active: Vec<bool> = weights.iter().map(|&w| w >= 0.05).collect();
        let cad_volume = weights.iter().zip(&active).filter(|(_, &a)| a).map(|(w, _)| w).sum();
        let (lambda, mu) = lame(cfg.young, cfg.poisson);
        let em = elasticity_element_matrices(&sp, lambda, mu);
        let pattern = AssemblyPattern::new(sp.ndof, &sp.connectivity, sp.nsh, sp.nel);
        let (r, c, v) = match cfg.stabilization {
            Stabilization::None => (vec![], vec![], vec![]),
            Stabilization::Legacy { gamma } => legacy_stabilization(&sp, &status, gamma),
            Stabilization::GhostPenalty { gamma } => ghost_penalty(&sp, &status, gamma, cfg.young),
        };
        let stab = CsrMatrix::from_triplets(sp.ndof, sp.ndof, &r, &c, &v);

        // Clamp bottom (i3 = 0); load on the top face centre (|i - mid| <= 2, mid = round(n/2), 1-based)
        let nd = sp.ndof_dir.clone();
        let mut free = vec![true; sp.ndof];
        let (mid_x, mid_y) = ((nd[0] as f64 / 2.0).round() as isize, (nd[1] as f64 / 2.0).round() as isize);
        let mut load = Vec::new();
        for i in 0..sp.ndof_sc {
            let s = unravel(i, &nd);
            if s[2] == 0 {
                for cc in 0..3 {
                    free[cc * sp.ndof_sc + i] = false;
                }
            }
            if s[2] == nd[2] - 1 && (s[0] as isize + 1 - mid_x).abs() <= 2 && (s[1] as isize + 1 - mid_y).abs() <= 2 {
                load.push(i);
            }
        }
        let mut force = vec![0.0; sp.ndof];
        for &i in &load {
            force[2 * sp.ndof_sc + i] = -1.0 / load.len() as f64;
        }

        // Filter on the design domain (offsets within r = rmin * mean(h))
        let h = sp.element_size();
        let r = cfg.rmin * (h[0] + h[1] + h[2]) / 3.0;
        let rmax = [0, 1, 2].map(|d| (r / h[d]).ceil() as isize);
        let mut offsets = Vec::new();
        for dk in -rmax[2]..=rmax[2] {
            for dj in -rmax[1]..=rmax[1] {
                for di in -rmax[0]..=rmax[0] {
                    let dist = ((di as f64 * h[0]).powi(2) + (dj as f64 * h[1]).powi(2) + (dk as f64 * h[2]).powi(2)).sqrt();
                    if dist <= r {
                        offsets.push((di, dj, dk, r - dist));
                    }
                }
            }
        }
        let filter: Vec<Vec<(usize, f64)>> = (0..sp.nel)
            .map(|e| {
                if !active[e] {
                    return Vec::new();
                }
                let s = unravel(e, &sp.nel_dir);
                offsets
                    .iter()
                    .filter_map(|&(di, dj, dk, w)| {
                        let (ni, nj, nk) = (s[0] as isize + di, s[1] as isize + dj, s[2] as isize + dk);
                        if ni < 0 || nj < 0 || nk < 0 || ni >= res[0] as isize || nj >= res[1] as isize || nk >= res[2] as isize {
                            return None;
                        }
                        let f = ni as usize + res[0] * (nj as usize + res[1] * nk as usize);
                        if active[f] { Some((f, w)) } else { None }
                    })
                    .collect()
            })
            .collect();
        let filter_sum = filter.iter().map(|row| { let s: f64 = row.iter().map(|x| x.1).sum(); if s < 1e-6 { 1.0 } else { s } }).collect();
        Self { cfg, sp, weights, active, cad_volume, em, pattern, stab, force, free, filter, filter_sum }
    }

    pub fn run<F: FnMut(usize, f64, f64, f64, usize)>(&self, mut report: F) -> ImmersedTopOptHistory {
        let cfg = &self.cfg;
        let nel = self.sp.nel;
        let n = self.sp.ndof;
        let mut x: Vec<f64> = self.active.iter().map(|&a| if a { cfg.volfrac } else { 0.0 }).collect();
        let mut u = vec![0.0; n];
        let mut hist = ImmersedTopOptHistory { compliance: vec![], volume: vec![], change: vec![], pcg_iterations: vec![], density: vec![] };
        let free = &self.free;
        let nf: f64 = (0..n).filter(|&i| free[i]).map(|i| self.force[i].powi(2)).sum::<f64>().sqrt();
        for iter in 1..=cfg.max_iter {
            let x_old = x.clone();
            let scale: Vec<f64> = (0..nel).map(|e| if self.active[e] { self.weights[e] * (cfg.emin + (1.0 - cfg.emin) * x[e].powf(cfg.penal)) } else { cfg.emin }).collect();
            let k = self.pattern.assemble(|e| self.em.of_element(e), &scale);
            let dk = k.diagonal();
            let ds = self.stab.diagonal();
            let diag: Vec<f64> = (0..n).map(|i| (dk[i] + ds[i]).max(1e-6)).collect();
            // warm-started correction solve
            let mut tmp = vec![0.0; n];
            let mut tmp2 = vec![0.0; n];
            let mut op = |p: &[f64], ap: &mut [f64], pm: &mut Vec<f64>| {
                for i in 0..n {
                    pm[i] = if free[i] { p[i] } else { 0.0 };
                }
                k.matvec(pm, &mut tmp);
                self.stab.matvec(pm, &mut tmp2);
                for i in 0..n {
                    ap[i] = if free[i] { tmp[i] + tmp2[i] } else { 0.0 };
                }
            };
            let mut pm = vec![0.0; n];
            let mut ku = vec![0.0; n];
            op(&u, &mut ku, &mut pm);
            let b: Vec<f64> = (0..n).map(|i| if free[i] { self.force[i] - ku[i] } else { 0.0 }).collect();
            let rb: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
            let tol = if rb > 0.0 { (cfg.pcg_tol * nf / rb).min(0.5) } else { 1.0 };
            let (du, its, _) = PcgSolver::new(cfg.pcg_maxit, tol).solve(n, &b, &diag, |p, ap| op(p, ap, &mut pm));
            for i in 0..n {
                u[i] = if free[i] { u[i] + du[i] } else { 0.0 };
            }
            let c: f64 = self.force.iter().zip(&u).map(|(f, v)| f * v).sum();
            let nsh = self.sp.nsh;
            let ee: Vec<f64> = (0..nel)
                .into_par_iter()
                .map(|e| {
                    let ke = self.em.of_element(e);
                    let ce = &self.sp.connectivity[e * nsh..(e + 1) * nsh];
                    let mut s = 0.0;
                    for a in 0..nsh {
                        let mut row = 0.0;
                        for bb in 0..nsh {
                            row += ke[a * nsh + bb] * u[ce[bb]];
                        }
                        s += u[ce[a]] * row;
                    }
                    s
                })
                .collect();
            let dc_raw: Vec<f64> = (0..nel).map(|e| -self.weights[e] * (cfg.penal * (1.0 - cfg.emin) * x[e].powf(cfg.penal - 1.0) * ee[e])).collect();
            let dc: Vec<f64> = (0..nel)
                .map(|e| self.filter[e].iter().map(|&(f, w)| w * x[f] * dc_raw[f]).sum::<f64>() / (self.filter_sum[e] * x[e].max(1e-3)))
                .collect();
            let act: Vec<usize> = (0..nel).filter(|&e| self.active[e]).collect();
            let (mut l1, mut l2) = (0.0, 2.0 * act.iter().map(|&e| dc[e].abs()).fold(0.0, f64::max));
            let target = cfg.volfrac * self.cad_volume;
            let mut cand: Vec<f64> = act.iter().map(|&e| x_old[e]).collect();
            while (l2 - l1) / (l1 + l2 + 1e-10) > 1e-4 {
                let lmid = 0.5 * (l1 + l2);
                for (k2, &e) in act.iter().enumerate() {
                    let be = (-dc[e] / lmid).sqrt();
                    cand[k2] = (x_old[e] * be).min(x_old[e] + 0.2).min(1.0).max(x_old[e] - 0.2).max(0.001);
                }
                let vol: f64 = act.iter().zip(&cand).map(|(&e, &xc)| self.weights[e] * xc).sum();
                if vol > target {
                    l1 = lmid;
                } else {
                    l2 = lmid;
                }
            }
            x = vec![0.0; nel];
            for (k2, &e) in act.iter().enumerate() {
                x[e] = cand[k2];
            }
            let change = act.iter().map(|&e| (x[e] - x_old[e]).abs()).fold(0.0, f64::max);
            let vol = act.iter().map(|&e| self.weights[e] * x[e]).sum::<f64>() / self.cad_volume;
            hist.compliance.push(c);
            hist.volume.push(vol);
            hist.change.push(change);
            hist.pcg_iterations.push(its);
            report(iter, c, vol, change, its);
        }
        hist.density = x;
        hist
    }
}
