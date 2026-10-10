//! Immersed elasticity on a background B-spline grid: a port of the MATLAB
//! reference `src/immersed/solve_immersed_iga_3d.m` and its helpers
//! (`classify_background_cells`, `assemble_immersed_element_weights`,
//! `assemble_ghost_penalty_stabilization`), validated against MATLAB in
//! `rust/tests/immersed_vs_matlab.rs`.
//!
//! Method limitations inherited from the MATLAB reference (see README):
//! - cut cells use the full-cell stiffness scaled by the cell's volume fraction
//!   (sub-cell midpoint sampling), not an exact cut-cell integration;
//! - cells are classified from their 8 corners only, so a thin feature that cuts a
//!   cell without enclosing a corner is missed;
//! - `legacy_stabilization` is the MATLAB "ghost penalty", which is not a ghost
//!   penalty in the usual sense (see its documentation).

use crate::cut_cell::TriangleMesh3D;
use crate::iga::{elasticity_element_matrices, lame, unravel, SpaceBox};
use crate::solver::PcgSolver;
use crate::sparse::CsrMatrix;
use rayon::prelude::*;

/// Background cell classification.
pub const OUTSIDE: i8 = 0;
pub const INSIDE: i8 = 1;
pub const CUT: i8 = -1;

/// Inside/outside test by ray parity along +x, bit-for-bit the algorithm of the
/// MATLAB `inpolyhedron_fast` (barycentric test in the y-z projection; boundary
/// hits are counted inclusively).
pub struct ParityRayCaster {
    tris: Vec<[[f64; 3]; 3]>,
    yz_box: Vec<[f64; 4]>,
}

impl ParityRayCaster {
    pub fn new(mesh: &TriangleMesh3D) -> Self {
        let tris: Vec<[[f64; 3]; 3]> = mesh
            .triangles
            .iter()
            .map(|t| [mesh.vertices[t[0]], mesh.vertices[t[1]], mesh.vertices[t[2]]])
            .collect();
        let yz_box = tris
            .iter()
            .map(|[a, b, c]| [a[1].min(b[1]).min(c[1]), a[1].max(b[1]).max(c[1]), a[2].min(b[2]).min(c[2]), a[2].max(b[2]).max(c[2])])
            .collect();
        Self { tris, yz_box }
    }

    pub fn is_inside(&self, q: [f64; 3]) -> bool {
        let (qx, qy, qz) = (q[0], q[1], q[2]);
        let mut count = 0usize;
        for (t, bb) in self.tris.iter().zip(&self.yz_box) {
            if !(qy >= bb[0] && qy <= bb[1] && qz >= bb[2] && qz <= bb[3]) {
                continue;
            }
            let [a, b, c] = t;
            let v0 = [c[1] - a[1], c[2] - a[2]];
            let v1 = [b[1] - a[1], b[2] - a[2]];
            let v2 = [qy - a[1], qz - a[2]];
            let dot00 = v0[0] * v0[0] + v0[1] * v0[1];
            let dot01 = v0[0] * v1[0] + v0[1] * v1[1];
            let dot02 = v0[0] * v2[0] + v0[1] * v2[1];
            let dot11 = v1[0] * v1[0] + v1[1] * v1[1];
            let dot12 = v1[0] * v2[0] + v1[1] * v2[1];
            let inv = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-15);
            let u = (dot11 * dot02 - dot01 * dot12) * inv;
            let v = (dot00 * dot12 - dot01 * dot02) * inv;
            if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
                let x_hit = a[0] + v * (b[0] - a[0]) + u * (c[0] - a[0]);
                if x_hit > qx {
                    count += 1;
                }
            }
        }
        count % 2 == 1
    }
}

/// MATLAB-compatible `linspace(a, b, n)`.
fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    let step = (b - a) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| a + i as f64 * step).collect();
    v[n - 1] = b;
    v
}

/// Classifies the `nx * ny * nz` background cells (direction 0 fastest) from the
/// inside/outside status of their 8 corners.
pub fn classify_background_cells(rc: &ParityRayCaster, bounds: &[[f64; 2]; 3], res: [usize; 3]) -> Vec<i8> {
    let xs: Vec<Vec<f64>> = (0..3).map(|d| linspace(bounds[d][0], bounds[d][1], res[d] + 1)).collect();
    let nv = [res[0] + 1, res[1] + 1, res[2] + 1];
    let inside_v: Vec<bool> = (0..nv[0] * nv[1] * nv[2])
        .into_par_iter()
        .map(|i| {
            let (a, b, c) = (i % nv[0], (i / nv[0]) % nv[1], i / (nv[0] * nv[1]));
            rc.is_inside([xs[0][a], xs[1][b], xs[2][c]])
        })
        .collect();
    let mut status = vec![OUTSIDE; res[0] * res[1] * res[2]];
    for k in 0..res[2] {
        for j in 0..res[1] {
            for i in 0..res[0] {
                let mut s = 0;
                for (di, dj, dk) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)] {
                    if inside_v[(i + di) + nv[0] * ((j + dj) + nv[1] * (k + dk))] {
                        s += 1;
                    }
                }
                status[i + res[0] * (j + res[1] * k)] = match s {
                    8 => INSIDE,
                    0 => OUTSIDE,
                    _ => CUT,
                };
            }
        }
    }
    status
}

/// Volume fraction of every cell: 1 inside, 0 outside, sub-cell midpoint sampling
/// with `sub` points per direction for cut cells. Returns `(weights, status)`.
pub fn element_weights(rc: &ParityRayCaster, bounds: &[[f64; 2]; 3], res: [usize; 3], sub: [usize; 3]) -> (Vec<f64>, Vec<i8>) {
    let status = classify_background_cells(rc, bounds, res);
    let h: Vec<f64> = (0..3).map(|d| (bounds[d][1] - bounds[d][0]) / res[d] as f64).collect();
    let weights: Vec<f64> = status
        .par_iter()
        .enumerate()
        .map(|(e, &s)| match s {
            INSIDE => 1.0,
            OUTSIDE => 0.0,
            _ => {
                let idx = [e % res[0], (e / res[0]) % res[1], e / (res[0] * res[1])];
                let lo: Vec<f64> = (0..3).map(|d| bounds[d][0] + idx[d] as f64 * h[d]).collect();
                let hi: Vec<f64> = (0..3).map(|d| bounds[d][0] + (idx[d] + 1) as f64 * h[d]).collect();
                let dx: Vec<f64> = (0..3).map(|d| (hi[d] - lo[d]) / sub[d] as f64).collect();
                let mut n_in = 0usize;
                for c in 0..sub[2] {
                    for b in 0..sub[1] {
                        for a in 0..sub[0] {
                            let q = [
                                lo[0] + (a as f64 + 0.5) * dx[0],
                                lo[1] + (b as f64 + 0.5) * dx[1],
                                lo[2] + (c as f64 + 0.5) * dx[2],
                            ];
                            if rc.is_inside(q) {
                                n_in += 1;
                            }
                        }
                    }
                }
                n_in as f64 / (sub[0] * sub[1] * sub[2]) as f64
            }
        })
        .collect();
    (weights, status)
}

/// Port of the MATLAB `assemble_ghost_penalty_stabilization`, kept for parity.
///
/// It is **not** a ghost penalty in the usual sense (jump of normal derivatives
/// across faces). For every interior face between two active cells of which at
/// least one is cut, and for every displacement component, it:
/// 1. adds a block on the control points shared by both cells that, because of a
///    `repmat` orientation in the MATLAB code, only touches diagonal entries and
///    sums to zero in each row, i.e. contributes nothing beyond round-off;
/// 2. ties the cells' non-shared control points pairwise, in sorted index order,
///    with springs of stiffness `gamma * mean(h)`.
/// Returns triplets `(rows, cols, vals)`.
pub fn legacy_stabilization(sp: &SpaceBox, status: &[i8], gamma: f64) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let res = [sp.nel_dir[0], sp.nel_dir[1], sp.nel_dir[2]];
    let h = sp.element_size();
    let alpha = gamma * (h.iter().sum::<f64>() / 3.0);
    let lin = |i: usize, j: usize, k: usize| i + res[0] * (j + res[1] * k);
    let active = |e: usize| status[e] == INSIDE || status[e] == CUT;
    let cut = |e: usize| status[e] == CUT;
    let mut faces: Vec<(usize, usize)> = Vec::new();
    for (di, dj, dk) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
        for i in 0..res[0] - di {
            for j in 0..res[1] - dj {
                for k in 0..res[2] - dk {
                    let (e1, e2) = (lin(i, j, k), lin(i + di, j + dj, k + dk));
                    if active(e1) && active(e2) && (cut(e1) || cut(e2)) {
                        faces.push((e1, e2));
                    }
                }
            }
        }
    }
    let (mut ri, mut ci, mut vi) = (Vec::new(), Vec::new(), Vec::new());
    let nsh_sc = sp.nsh_sc;
    for &(e1, e2) in &faces {
        for c in 0..3 {
            let mut d1: Vec<usize> = sp.connectivity[e1 * sp.nsh + c * nsh_sc..e1 * sp.nsh + (c + 1) * nsh_sc].to_vec();
            let mut d2: Vec<usize> = sp.connectivity[e2 * sp.nsh + c * nsh_sc..e2 * sp.nsh + (c + 1) * nsh_sc].to_vec();
            d1.sort_unstable();
            d1.dedup();
            d2.sort_unstable();
            d2.dedup();
            let common: Vec<usize> = d1.iter().copied().filter(|x| d2.binary_search(x).is_ok()).collect();
            let only1: Vec<usize> = d1.iter().copied().filter(|x| d2.binary_search(x).is_err()).collect();
            let only2: Vec<usize> = d2.iter().copied().filter(|x| d1.binary_search(x).is_err()).collect();
            let n = common.len();
            if n > 0 {
                let nf = n as f64;
                for m in 0..n * n {
                    let (i, j) = (m % n, m / n);
                    let s = (alpha / nf) * ((if i == j { 1.0 } else { 0.0 }) - 1.0 / nf);
                    ri.push(common[i]);
                    ci.push(common[i]);
                    vi.push(s);
                }
            }
            let np = only1.len().min(only2.len());
            for k in 0..np {
                let (p1, p2) = (only1[k], only2[k]);
                ri.extend_from_slice(&[p1, p2, p1, p2]);
                ci.extend_from_slice(&[p1, p2, p2, p1]);
                vi.extend_from_slice(&[alpha, alpha, -alpha, -alpha]);
            }
        }
    }
    (ri, ci, vi)
}

/// Face-based ghost penalty for maximally smooth (C^{p-1}) splines:
///
///   j(u, v) = gamma * E * h^(2p-1) * sum_F int_F [d_n^p u] . [d_n^p v]
///
/// over interior faces between two active cells of which at least one is cut. For
/// C^{p-1} splines all lower normal-derivative jumps vanish, so the p-th one is the
/// only term; `j` vanishes on every global polynomial of degree <= p (consistent).
/// On the box grid the face integral factors into the 1D jump vector in the normal
/// direction times the 1D element mass matrices in the tangential directions, so it
/// is evaluated exactly. Returns triplets `(rows, cols, vals)`.
pub fn ghost_penalty(sp: &SpaceBox, status: &[i8], gamma: f64, young: f64) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    use crate::iga::basis::basis_funs_ders;
    let dim = sp.dim;
    let h = sp.element_size();
    let h_char = h.iter().sum::<f64>() / dim as f64;
    let res = &sp.nel_dir;
    let active = |e: usize| status[e] == INSIDE || status[e] == CUT;
    let cut = |e: usize| status[e] == CUT;

    // 1D tangential mass matrices per direction and element (physical).
    let mass: Vec<Vec<Vec<f64>>> = (0..dim)
        .map(|d| {
            let u = &sp.univ[d];
            let nl = u.nsh();
            (0..u.nel)
                .map(|e| {
                    let mut m = vec![0.0; nl * nl];
                    for q in 0..u.nquad {
                        let w = u.qw[e * u.nquad + q] * sp.lengths[d];
                        let base = (e * u.nquad + q) * nl;
                        for a in 0..nl {
                            for b in 0..nl {
                                m[a * nl + b] += w * u.shape[base + a] * u.shape[base + b];
                            }
                        }
                    }
                    m
                })
                .collect()
        })
        .collect();

    // 1D jumps of the p-th derivative across interior break k (between elements k-1, k):
    // (global function indices, jump values in physical units).
    let jumps: Vec<Vec<(Vec<usize>, Vec<f64>)>> = (0..dim)
        .map(|d| {
            let u = &sp.univ[d];
            let p = u.degree;
            let nl = u.nsh();
            let lp = sp.lengths[d].powi(p as i32);
            (0..u.nel)
                .map(|k| {
                    if k == 0 {
                        return (Vec::new(), Vec::new());
                    }
                    let x = u.breaks[k];
                    let left_first = u.connectivity[(k - 1) * nl];
                    let right_first = u.connectivity[k * nl];
                    let (_, dl) = basis_funs_ders(&u.knots, p, x, p, Some(left_first + p));
                    let (_, dr) = basis_funs_ders(&u.knots, p, x, p, Some(right_first + p));
                    let lo = left_first.min(right_first);
                    let hi = (left_first + p).max(right_first + p);
                    let idx: Vec<usize> = (lo..=hi).collect();
                    let vals: Vec<f64> = idx
                        .iter()
                        .map(|&g| {
                            let r_val = if g >= right_first && g <= right_first + p { dr[p][g - right_first] } else { 0.0 };
                            let l_val = if g >= left_first && g <= left_first + p { dl[p][g - left_first] } else { 0.0 };
                            (r_val - l_val) / lp
                        })
                        .collect();
                    (idx, vals)
                })
                .collect()
        })
        .collect();

    let (mut ri, mut ci, mut vi) = (Vec::new(), Vec::new(), Vec::new());
    for d in 0..dim {
        let p = sp.degree[d];
        let coef = gamma * young * h_char.powi(2 * p as i32 - 1);
        for e2 in 0..sp.nel {
            let s2 = unravel(e2, res);
            if s2[d] == 0 {
                continue;
            }
            let mut s1 = s2.clone();
            s1[d] -= 1;
            let mut lin1 = 0usize;
            let mut stride = 1usize;
            for dd in 0..dim {
                lin1 += s1[dd] * stride;
                stride *= res[dd];
            }
            if !(active(lin1) && active(e2) && (cut(lin1) || cut(e2))) {
                continue;
            }
            // Tensor factors: normal direction -> jump vector; tangential -> mass matrix.
            let (jidx, jval) = &jumps[d][s2[d]];
            let tang: Vec<usize> = (0..dim).filter(|&t| t != d).collect();
            // local index sets per direction
            let sets: Vec<Vec<usize>> = (0..dim)
                .map(|dd| {
                    if dd == d {
                        jidx.clone()
                    } else {
                        let u = &sp.univ[dd];
                        u.connectivity[s2[dd] * u.nsh()..(s2[dd] + 1) * u.nsh()].to_vec()
                    }
                })
                .collect();
            let sizes: Vec<usize> = sets.iter().map(|s| s.len()).collect();
            let nloc: usize = sizes.iter().product();
            let mut gidx = Vec::with_capacity(nloc);
            for a in 0..nloc {
                let sa = unravel(a, &sizes);
                let mut g = 0usize;
                let mut stride = 1usize;
                for dd in 0..dim {
                    g += sets[dd][sa[dd]] * stride;
                    stride *= sp.ndof_dir[dd];
                }
                gidx.push(g);
            }
            for a in 0..nloc {
                let sa = unravel(a, &sizes);
                for b in 0..nloc {
                    let sb = unravel(b, &sizes);
                    let mut v = coef * jval[sa[d]] * jval[sb[d]];
                    for &t in &tang {
                        let nl = sp.nsh_dir[t];
                        v *= mass[t][s2[t]][sa[t] * nl + sb[t]];
                    }
                    if v == 0.0 {
                        continue;
                    }
                    for c in 0..dim {
                        ri.push(c * sp.ndof_sc + gidx[a]);
                        ci.push(c * sp.ndof_sc + gidx[b]);
                        vi.push(v);
                    }
                }
            }
        }
    }
    (ri, ci, vi)
}

/// Cut-cell stabilization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stabilization {
    None,
    /// Port of the MATLAB heuristic (`legacy_stabilization`); for parity only. Its
    /// springs are not scaled by Young's modulus and badly over-stiffen thin parts.
    Legacy { gamma: f64 },
    /// Consistent face-based ghost penalty (`ghost_penalty`).
    GhostPenalty { gamma: f64 },
}

/// Which background face is clamped (strong Dirichlet on all components).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClampedFace {
    Bottom,
    Top,
    Left,
}

#[derive(Clone, Debug)]
pub struct ImmersedOptions {
    pub grid_res: [usize; 3],
    pub padding: f64,
    pub degree: usize,
    pub young: f64,
    pub poisson: f64,
    /// Cut-cell stabilization (default: consistent ghost penalty).
    pub stabilization: Stabilization,
    pub emin: f64,
    pub subcell_res: [usize; 3],
    /// Strong clamp of a background face (MATLAB reference behaviour); `None` when the
    /// supports come from labelled CAD faces instead (see `immersed_bc`).
    pub clamped_face: Option<ClampedFace>,
    /// External force vector (component-blocked); `None` = default central load on
    /// the top face (as in the MATLAB reference).
    pub force: Option<Vec<f64>>,
    pub pcg_tol: f64,
    pub pcg_maxit: usize,
}

impl Default for ImmersedOptions {
    fn default() -> Self {
        Self {
            grid_res: [24, 16, 12],
            padding: 0.05,
            degree: 2,
            young: 1.0,
            poisson: 0.3,
            stabilization: Stabilization::GhostPenalty { gamma: 1e-2 },
            emin: 1e-4,
            subcell_res: [4, 4, 4],
            clamped_face: Some(ClampedFace::Bottom),
            force: None,
            pcg_tol: 1e-8,
            pcg_maxit: 5000,
        }
    }
}

pub struct ImmersedProblem {
    pub sp: SpaceBox,
    pub grid_bounds: [[f64; 2]; 3],
    pub status: Vec<i8>,
    pub weights: Vec<f64>,
    pub stiffness: CsrMatrix,
    pub stabilization_frobenius: f64,
    pub force: Vec<f64>,
    pub free: Vec<bool>,
}

pub struct ImmersedSolution {
    pub u: Vec<f64>,
    pub compliance: f64,
    pub iterations: usize,
    pub residual: f64,
}

/// Padded bounding box of a mesh, as in the MATLAB reference.
pub fn padded_bounds(mesh: &TriangleMesh3D, padding: f64) -> [[f64; 2]; 3] {
    let mut b = [[f64::MAX, f64::MIN]; 3];
    for v in &mesh.vertices {
        for d in 0..3 {
            b[d][0] = b[d][0].min(v[d]);
            b[d][1] = b[d][1].max(v[d]);
        }
    }
    for d in 0..3 {
        let pad = padding * (b[d][1] - b[d][0]);
        b[d] = [b[d][0] - pad, b[d][1] + pad];
    }
    b
}

/// Builds the immersed system of `solve_immersed_iga_3d.m` (assembled form).
pub fn build_immersed_problem(mesh: &TriangleMesh3D, opts: &ImmersedOptions) -> ImmersedProblem {
    let grid_bounds = padded_bounds(mesh, opts.padding);
    let sp = SpaceBox::new(&grid_bounds, &opts.grid_res, opts.degree);
    let rc = ParityRayCaster::new(mesh);
    let (weights, status) = element_weights(&rc, &grid_bounds, opts.grid_res, opts.subcell_res);
    let (lambda, mu) = lame(opts.young, opts.poisson);
    let em = elasticity_element_matrices(&sp, lambda, mu);

    let (mut rows, mut cols, mut vals) = match opts.stabilization {
        Stabilization::None => (Vec::new(), Vec::new(), Vec::new()),
        Stabilization::Legacy { gamma } => legacy_stabilization(&sp, &status, gamma),
        Stabilization::GhostPenalty { gamma } => ghost_penalty(&sp, &status, gamma, opts.young),
    };
    let n_gp = vals.len();
    let stab = CsrMatrix::from_triplets(sp.ndof, sp.ndof, &rows[..n_gp], &cols[..n_gp], &vals[..n_gp]);
    let nsh = sp.nsh;
    rows.reserve(nsh * nsh * sp.nel);
    for e in 0..sp.nel {
        let s = opts.emin + (1.0 - opts.emin) * weights[e];
        let ke = em.of_element(e);
        let ce = &sp.connectivity[e * nsh..(e + 1) * nsh];
        for a in 0..nsh {
            for b in 0..nsh {
                rows.push(ce[a]);
                cols.push(ce[b]);
                vals.push(s * ke[a * nsh + b]);
            }
        }
    }
    let stiffness = CsrMatrix::from_triplets(sp.ndof, sp.ndof, &rows, &cols, &vals);

    // Boundary conditions and load, as in the MATLAB reference.
    let nd = &sp.ndof_dir;
    let sub = |i: usize| (i % nd[0], (i / nd[0]) % nd[1], i / (nd[0] * nd[1]));
    let mut free = vec![true; sp.ndof];
    for i in 0..sp.ndof_sc {
        let (i1, _, i3) = sub(i);
        let clamp = match opts.clamped_face {
            Some(ClampedFace::Bottom) => i3 == 0,
            Some(ClampedFace::Top) => i3 == nd[2] - 1,
            Some(ClampedFace::Left) => i1 == 0,
            None => false,
        };
        if clamp {
            for c in 0..3 {
                free[c * sp.ndof_sc + i] = false;
            }
        }
    }
    let force = match &opts.force {
        Some(f) => {
            assert_eq!(f.len(), sp.ndof, "force vector has the wrong length");
            f.clone()
        }
        None => {
            // MATLAB: mid = round(ncp/2) (1-based), |i - mid| <= 2 on the top face
            let mid_x = (nd[0] as f64 / 2.0).round() as isize;
            let mid_y = (nd[1] as f64 / 2.0).round() as isize;
            let load: Vec<usize> = (0..sp.ndof_sc)
                .filter(|&i| {
                    let (i1, i2, i3) = sub(i);
                    i3 == nd[2] - 1 && (i1 as isize + 1 - mid_x).abs() <= 2 && (i2 as isize + 1 - mid_y).abs() <= 2
                })
                .collect();
            let mut f = vec![0.0; sp.ndof];
            for &i in &load {
                f[2 * sp.ndof_sc + i] = -1.0 / load.len() as f64;
            }
            f
        }
    };
    ImmersedProblem { sp, grid_bounds, status, weights, stiffness, stabilization_frobenius: stab.frobenius(), force, free }
}

impl ImmersedProblem {
    /// Adds surface boundary-condition terms (see `immersed_bc::surface_boundary_terms`).
    pub fn add_surface_terms(&mut self, st: &crate::immersed_bc::SurfaceTerms) {
        self.stiffness = self.stiffness.plus_triplets(&st.rows, &st.cols, &st.vals);
        for (f, g) in self.force.iter_mut().zip(&st.force) {
            *f += g;
        }
    }
}

/// Solves the immersed system with Jacobi-preconditioned CG on the free DOFs.
pub fn solve_immersed_problem(pb: &ImmersedProblem, tol: f64, maxit: usize) -> ImmersedSolution {
    let n = pb.sp.ndof;
    let free = &pb.free;
    let b: Vec<f64> = (0..n).map(|i| if free[i] { pb.force[i] } else { 0.0 }).collect();
    let diag: Vec<f64> = pb.stiffness.diagonal().iter().map(|&d| d.max(1e-6)).collect();
    let mut pm = vec![0.0; n];
    let mut tmp = vec![0.0; n];
    let (u, iterations, residual) = PcgSolver::new(maxit, tol).solve(n, &b, &diag, |p, ap| {
        for i in 0..n {
            pm[i] = if free[i] { p[i] } else { 0.0 };
        }
        pb.stiffness.matvec(&pm, &mut tmp);
        for i in 0..n {
            ap[i] = if free[i] { tmp[i] } else { 0.0 };
        }
    });
    let u: Vec<f64> = (0..n).map(|i| if free[i] { u[i] } else { 0.0 }).collect();
    let compliance = pb.force.iter().zip(&u).map(|(f, x)| f * x).sum();
    ImmersedSolution { u, compliance, iterations, residual }
}

/// Convenience: build and solve (`solve_immersed_iga_3d` equivalent).
pub fn solve_immersed(mesh: &TriangleMesh3D, opts: &ImmersedOptions) -> (ImmersedProblem, ImmersedSolution) {
    let pb = build_immersed_problem(mesh, opts);
    let sol = solve_immersed_problem(&pb, opts.pcg_tol, opts.pcg_maxit);
    (pb, sol)
}
