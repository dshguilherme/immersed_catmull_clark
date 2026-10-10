//! Stress-driven adaptive octree refinement, a port of the MATLAB reference
//! `src/immersed/compute_amr_stress_indicators.m` and `adaptive_mesh_refinement_loop.m`
//! (validated in `rust/tests/amr_vs_matlab.rs`).
//!
//! The indicator is the MATLAB heuristic
//!   eta_e = sigma_vM,e * sqrt(h_e) * (1 + alpha * |sigma_vM,e - mean_nbr| / (mean_nbr + eps)),
//! with centroid von Mises stress of the trilinear element: it flags high-stress
//! regions, it is **not** an a-posteriori error estimator, so it carries no
//! guarantee of error reduction. Marking: Doerfler (bulk) on eta^2.

use crate::cut_cell::TriangleMesh3D;
use crate::octree::OctreeMesh3D;
use crate::octree_fem::OctreeFem;
use crate::solver::PcgSolver;

const XI: [f64; 8] = [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0];
const ETA: [f64; 8] = [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0];
const ZT: [f64; 8] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];

pub struct AmrOptions {
    pub alpha_jump: f64,
    pub theta_dorfler: f64,
    pub max_level_limit: usize,
}

impl Default for AmrOptions {
    fn default() -> Self {
        Self { alpha_jump: 1.5, theta_dorfler: 0.35, max_level_limit: 4 }
    }
}

/// Centroid von Mises stress per element, the indicator, and the marked leaves.
pub fn stress_indicators(fem: &OctreeFem, u_master: &[f64], opts: &AmrOptions) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let nn = fem.nodes.len();
    // u at all nodes: T * u_master
    let mut u = vec![0.0; 3 * nn];
    for i in 0..nn {
        for &(m, w) in &fem.t_rows[i] {
            for c in 0..3 {
                u[3 * i + c] += w * u_master[3 * m + c];
            }
        }
    }
    let (e_mod, nu) = (fem.young, fem.poisson);
    let lam = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e_mod / (2.0 * (1.0 + nu));
    let nel = fem.elem_nodes.len();
    let mut vm = vec![0.0; nel];
    for e in 0..nel {
        let b = fem.leaf_bounds[e];
        let h = [b[1] - b[0], b[3] - b[2], b[5] - b[4]];
        let mut eps = [0.0; 6];
        for a in 0..8 {
            let (dx, dy, dz) = (2.0 / h[0] * 0.125 * XI[a], 2.0 / h[1] * 0.125 * ETA[a], 2.0 / h[2] * 0.125 * ZT[a]);
            let n = fem.elem_nodes[e][a];
            let (ux, uy, uz) = (u[3 * n], u[3 * n + 1], u[3 * n + 2]);
            eps[0] += dx * ux;
            eps[1] += dy * uy;
            eps[2] += dz * uz;
            eps[3] += dz * uy + dy * uz;
            eps[4] += dz * ux + dx * uz;
            eps[5] += dy * ux + dx * uy;
        }
        let tr = eps[0] + eps[1] + eps[2];
        let s = [lam * tr + 2.0 * mu * eps[0], lam * tr + 2.0 * mu * eps[1], lam * tr + 2.0 * mu * eps[2], mu * eps[3], mu * eps[4], mu * eps[5]];
        vm[e] = (0.5 * ((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[2] - s[0]).powi(2) + 6.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))).sqrt();
    }
    // element neighbours sharing at least one node
    let mut node_elems: Vec<Vec<usize>> = vec![Vec::new(); nn];
    for (e, en) in fem.elem_nodes.iter().enumerate() {
        for &n in en {
            node_elems[n].push(e);
        }
    }
    let vmax = vm.iter().cloned().fold(0.0, f64::max);
    let eps_reg = 1e-4 * vmax + 1e-8;
    let eta: Vec<f64> = (0..nel)
        .map(|e| {
            let mut nb: Vec<usize> = fem.elem_nodes[e].iter().flat_map(|&n| node_elems[n].iter().copied()).filter(|&f| f != e).collect();
            nb.sort_unstable();
            nb.dedup();
            let sig_nbr = if nb.is_empty() { 0.0 } else { nb.iter().map(|&f| vm[f]).sum::<f64>() / nb.len() as f64 };
            let b = fem.leaf_bounds[e];
            let hc = ((b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4])).cbrt();
            vm[e] * hc.sqrt() * (1.0 + opts.alpha_jump * ((vm[e] - sig_nbr).abs() / (sig_nbr + eps_reg)))
        })
        .collect();
    // Doerfler marking (stable descending sort, as MATLAB sort 'descend')
    let mut order: Vec<usize> = (0..nel).collect();
    order.sort_by(|&a, &b| eta[b].partial_cmp(&eta[a]).unwrap());
    let total: f64 = eta.iter().map(|v| v * v).sum();
    let mut cum = 0.0;
    let mut n_mark = nel;
    for (k, &e) in order.iter().enumerate() {
        cum += eta[e] * eta[e];
        if cum >= opts.theta_dorfler * total {
            n_mark = k + 1;
            break;
        }
    }
    let marked = order[..n_mark.max(1)].iter().copied().filter(|&e| fem.levels[e] < opts.max_level_limit).collect();
    (vm, eta, marked)
}

pub struct AmrCycle {
    pub n_elements: usize,
    pub free_dofs: usize,
    pub compliance: f64,
    pub sigma_max: f64,
    pub marked: usize,
}

/// Runs `cycles` of solve -> indicator -> mark -> subdivide + balance. `bc(fem)`
/// returns (fixed master DOFs, master load vector). Returns the history and the
/// final mesh, displacement and von Mises stresses.
pub fn amr_loop<B>(mut octree: OctreeMesh3D, brep: Option<&TriangleMesh3D>, young: f64, poisson: f64, cycles: usize, opts: &AmrOptions, bc: B) -> (Vec<AmrCycle>, OctreeFem, Vec<f64>, Vec<f64>)
where
    B: Fn(&OctreeFem) -> (Vec<usize>, Vec<f64>),
{
    let mut history = Vec::new();
    let mut last = None;
    for cycle in 0..cycles {
        let fem = OctreeFem::build(&octree, brep, young, poisson, 1e-4, [3, 3, 3]);
        let (fixed, f) = bc(&fem);
        let n = f.len();
        let mut is_fixed = vec![false; n];
        for &d in &fixed {
            is_fixed[d] = true;
        }
        let b: Vec<f64> = (0..n).map(|i| if is_fixed[i] { 0.0 } else { f[i] }).collect();
        let diag: Vec<f64> = fem.k_master.diagonal().iter().map(|&d| if d.abs() < 1e-12 { 1.0 } else { d }).collect();
        let mut pm = vec![0.0; n];
        let mut tmp = vec![0.0; n];
        let (x, _, _) = PcgSolver::new(100 * n, 1e-13).solve(n, &b, &diag, |p, ap| {
            for i in 0..n {
                pm[i] = if is_fixed[i] { 0.0 } else { p[i] };
            }
            fem.k_master.matvec(&pm, &mut tmp);
            for i in 0..n {
                ap[i] = if is_fixed[i] { 0.0 } else { tmp[i] };
            }
        });
        let u: Vec<f64> = (0..n).map(|i| if is_fixed[i] { 0.0 } else { x[i] }).collect();
        let compliance = f.iter().zip(&u).map(|(a, b)| a * b).sum();
        let (vm, _, marked) = stress_indicators(&fem, &u, opts);
        history.push(AmrCycle {
            n_elements: fem.elem_nodes.len(),
            free_dofs: n - fixed.len(),
            compliance,
            sigma_max: vm.iter().cloned().fold(0.0, f64::max),
            marked: marked.len(),
        });
        let stop = marked.is_empty();
        if cycle + 1 < cycles && !stop {
            octree.subdivide_leaves(&marked);
        }
        last = Some((fem, u, vm));
        if stop {
            break;
        }
    }
    let (fem, u, vm) = last.expect("at least one cycle");
    (history, fem, u, vm)
}
