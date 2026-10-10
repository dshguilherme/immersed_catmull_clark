//! Verification studies of the immersed method (measured, not pass/fail):
//! - cut-domain patch test (linear field, penalty Dirichlet on the immersed surface);
//! - manufactured-solution convergence on an immersed sphere;
//! - small-cut conditioning with and without stabilization.
//! See `bin/verification_studies.rs` for the driver and the interpretation.

use crate::cut_cell::TriangleMesh3D;
use crate::iga::{element_quad_points, element_shape_functions, eval_vector_at, lame};
use crate::immersed::{build_immersed_problem, padded_bounds, ImmersedOptions, ImmersedProblem, ParityRayCaster, Stabilization};
use crate::immersed_bc::dirichlet_penalty_fn;
use crate::sparse::CsrMatrix;
use std::f64::consts::PI;

/// Icosphere (subdivided icosahedron) of radius `r` around `c`, outward-oriented.
pub fn icosphere(c: [f64; 3], r: f64, levels: usize) -> TriangleMesh3D {
    let t = (1.0 + 5f64.sqrt()) / 2.0;
    let mut v: Vec<[f64; 3]> = vec![
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
    ];
    let mut f: Vec<[usize; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    let norm = |p: [f64; 3]| {
        let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        [p[0] / n, p[1] / n, p[2] / n]
    };
    v = v.into_iter().map(norm).collect();
    for _ in 0..levels {
        let mut cache = std::collections::HashMap::new();
        let mut mid = |a: usize, b: usize, v: &mut Vec<[f64; 3]>| -> usize {
            let key = (a.min(b), a.max(b));
            *cache.entry(key).or_insert_with(|| {
                v.push(norm([0.5 * (v[a][0] + v[b][0]), 0.5 * (v[a][1] + v[b][1]), 0.5 * (v[a][2] + v[b][2])]));
                v.len() - 1
            })
        };
        let mut nf = Vec::with_capacity(4 * f.len());
        for t in &f {
            let (a, b, cc) = (mid(t[0], t[1], &mut v), mid(t[1], t[2], &mut v), mid(t[2], t[0], &mut v));
            nf.extend([[t[0], a, cc], [t[1], b, a], [t[2], cc, b], [a, b, cc]]);
        }
        f = nf;
    }
    TriangleMesh3D::new(v.into_iter().map(|p| [c[0] + r * p[0], c[1] + r * p[1], c[2] + r * p[2]]).collect(), f)
}

/// Lanczos estimate of the extreme eigenvalues of the Jacobi-preconditioned operator
/// restricted to the free DOFs: returns (lambda_min, lambda_max) after `steps` steps.
pub fn lanczos_extremes<F: FnMut(&[f64], &mut [f64])>(n: usize, diag: &[f64], mut op: F, steps: usize) -> (f64, f64) {
    // symmetric scaling: B = D^{-1/2} A D^{-1/2}
    let s: Vec<f64> = diag.iter().map(|d| 1.0 / d.sqrt()).collect();
    let mut q: Vec<f64> = (0..n).map(|i| ((i * 7919 + 13) % 997) as f64 / 997.0 - 0.5).collect();
    let nq = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    q.iter_mut().for_each(|x| *x /= nq);
    let mut q_prev = vec![0.0; n];
    let (mut alpha, mut beta) = (Vec::new(), Vec::new());
    let mut b_prev = 0.0;
    let mut tmp = vec![0.0; n];
    let mut y = vec![0.0; n];
    for _ in 0..steps {
        let sq: Vec<f64> = (0..n).map(|i| s[i] * q[i]).collect();
        op(&sq, &mut tmp);
        for i in 0..n {
            y[i] = s[i] * tmp[i];
        }
        let a: f64 = (0..n).map(|i| y[i] * q[i]).sum();
        for i in 0..n {
            y[i] -= a * q[i] + b_prev * q_prev[i];
        }
        let b = y.iter().map(|x| x * x).sum::<f64>().sqrt();
        alpha.push(a);
        if b < 1e-14 {
            break;
        }
        beta.push(b);
        q_prev = q.clone();
        q = y.iter().map(|x| x / b).collect();
        b_prev = b;
    }
    let m = alpha.len();
    // Sturm-sequence bisection for the extreme eigenvalues of the tridiagonal matrix
    let count_below = |x: f64| -> usize {
        let mut cnt = 0;
        let mut d = alpha[0] - x;
        if d < 0.0 {
            cnt += 1;
        }
        for k in 1..m {
            let dd = if d.abs() < 1e-300 { 1e-300 } else { d };
            d = alpha[k] - x - beta[k - 1] * beta[k - 1] / dd;
            if d < 0.0 {
                cnt += 1;
            }
        }
        cnt
    };
    let bound = alpha.iter().cloned().fold(0.0, f64::max) + 2.0 * beta.iter().cloned().fold(0.0, f64::max) + 1.0;
    let find = |k: usize| -> f64 {
        let (mut lo, mut hi) = (-bound, bound);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if count_below(mid) > k { hi = mid } else { lo = mid }
        }
        0.5 * (lo + hi)
    };
    (find(0), find(m - 1))
}

/// Free-DOF operator of an immersed problem (stiffness incl. stabilization and surface terms).
pub fn free_operator<'a>(k: &'a CsrMatrix, free: &'a [bool]) -> impl FnMut(&[f64], &mut [f64]) + 'a {
    let n = free.len();
    let mut pm = vec![0.0; n];
    let mut tmp = vec![0.0; n];
    move |p: &[f64], ap: &mut [f64]| {
        for i in 0..n {
            pm[i] = if free[i] { p[i] } else { 0.0 };
        }
        k.matvec(&pm, &mut tmp);
        for i in 0..n {
            ap[i] = if free[i] { tmp[i] } else { p[i] };
        }
    }
}

/// Grid of points strictly inside the B-Rep (ray parity), `n` per bounding-box edge.
pub fn interior_samples(mesh: &TriangleMesh3D, n: usize) -> Vec<[f64; 3]> {
    let b = padded_bounds(mesh, 0.0);
    let rc = ParityRayCaster::new(mesh);
    let mut out = Vec::new();
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = [
                    b[0][0] + (i as f64 + 0.5) / n as f64 * (b[0][1] - b[0][0]),
                    b[1][0] + (j as f64 + 0.5) / n as f64 * (b[1][1] - b[1][0]),
                    b[2][0] + (k as f64 + 0.5) / n as f64 * (b[2][1] - b[2][0]),
                ];
                if rc.is_inside(p) {
                    out.push(p);
                }
            }
        }
    }
    out
}

pub struct StudyResult {
    pub cells: usize,
    pub ndof: usize,
    pub rel_l2_error: f64,
    pub pcg_iterations: usize,
}

/// Builds the immersed problem with penalty Dirichlet `g` on the whole surface and an
/// optional body force `f` integrated as sum_e w_e int_e f.v (consistent with the
/// volume-fraction stiffness), solves, and measures the relative RMS error against
/// `exact` at interior sample points.
pub fn immersed_dirichlet_problem<G, B, X>(mesh: &TriangleMesh3D, cells: usize, stab: Stabilization, penalty: f64, g: G, body: Option<B>, exact: X) -> (StudyResult, ImmersedProblem)
where
    G: Fn(&[f64; 3]) -> [f64; 3],
    B: Fn(&[f64; 3]) -> [f64; 3],
    X: Fn(&[f64; 3]) -> [f64; 3],
{
    let gb = padded_bounds(mesh, 0.05);
    let len: Vec<f64> = (0..3).map(|d| gb[d][1] - gb[d][0]).collect();
    let lmax = len.iter().cloned().fold(0.0, f64::max);
    let grid_res = [0, 1, 2].map(|d| ((cells as f64 * len[d] / lmax).round() as usize).max(3));
    let opts = ImmersedOptions { grid_res, clamped_face: None, stabilization: stab, ..ImmersedOptions::default() };
    let mut pb = build_immersed_problem(mesh, &opts);
    pb.force.iter_mut().for_each(|v| *v = 0.0);
    let h_min = pb.sp.element_size().iter().cloned().fold(f64::MAX, f64::min);
    let all: Vec<usize> = (0..mesh.triangles.len()).collect();
    let st = dirichlet_penalty_fn(&pb.sp, mesh, &all, penalty * opts.young / h_min, &g);
    pb.add_surface_terms(&st);
    if let Some(fb) = body {
        let sp = &pb.sp;
        for e in 0..sp.nel {
            let w_e = pb.weights[e];
            if w_e == 0.0 {
                continue;
            }
            let (x, w) = element_quad_points(sp, e);
            let (nshape, _) = element_shape_functions(sp, e);
            for q in 0..sp.nqn {
                let fq = fb(&x[q]);
                for a in 0..sp.nsh_sc {
                    let gi = sp.conn_sc[e * sp.nsh_sc + a];
                    for c in 0..3 {
                        pb.force[c * sp.ndof_sc + gi] += w_e * w[q] * nshape[q * sp.nsh_sc + a] * fq[c];
                    }
                }
            }
        }
    }
    let sol = crate::immersed::solve_immersed_problem(&pb, 1e-10, 50 * pb.sp.ndof);
    let pts = interior_samples(mesh, 24);
    let (mut e2, mut n2) = (0.0, 0.0);
    for p in &pts {
        let uh = eval_vector_at(&pb.sp, &sol.u, p);
        let ue = exact(p);
        for c in 0..3 {
            e2 += (uh[c] - ue[c]).powi(2);
            n2 += ue[c] * ue[c];
        }
    }
    (StudyResult { cells, ndof: pb.sp.ndof, rel_l2_error: (e2 / n2.max(1e-300)).sqrt(), pcg_iterations: sol.iterations }, pb)
}

/// Manufactured solution used on the sphere: u = 0.01 (sin(pi x) cos(pi y), sin(pi y) cos(pi z), sin(pi z) cos(pi x)),
/// with the matching body force f = -div sigma(u) for E = 1, nu = 0.3.
pub fn mms_fields() -> (impl Fn(&[f64; 3]) -> [f64; 3] + Copy, impl Fn(&[f64; 3]) -> [f64; 3] + Copy) {
    let (lam, mu) = lame(1.0, 0.3);
    let a = 0.01;
    let u = move |x: &[f64; 3]| -> [f64; 3] {
        let (s, c) = (|t: f64| (PI * t).sin(), |t: f64| (PI * t).cos());
        [a * s(x[0]) * c(x[1]), a * s(x[1]) * c(x[2]), a * s(x[2]) * c(x[0])]
    };
    // f_i = -(lam + mu) d_i div u - mu lap u_i
    let f = move |x: &[f64; 3]| -> [f64; 3] {
        let (s, c) = (|t: f64| (PI * t).sin(), |t: f64| (PI * t).cos());
        let p2 = PI * PI;
        // div u = a pi (cos(pi x) cos(pi y) + cos(pi y) cos(pi z) + cos(pi z) cos(pi x))
        let ddiv = [
            -a * p2 * (s(x[0]) * c(x[1]) + s(x[0]) * c(x[2])),
            -a * p2 * (c(x[0]) * s(x[1]) + s(x[1]) * c(x[2])),
            -a * p2 * (c(x[1]) * s(x[2]) + c(x[0]) * s(x[2])),
        ];
        let lap = [-2.0 * p2 * a * s(x[0]) * c(x[1]), -2.0 * p2 * a * s(x[1]) * c(x[2]), -2.0 * p2 * a * s(x[2]) * c(x[0])];
        [0, 1, 2].map(|i| -(lam + mu) * ddiv[i] - mu * lap[i])
    };
    (u, f)
}
