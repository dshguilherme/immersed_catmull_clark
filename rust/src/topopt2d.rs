//! 2D isogeometric SIMP topology optimization of a cantilever with element-wise or
//! spline (control-point) densities, and the direct strain-energy sensitivities.
//! Ports of `src/fastformation/topopt_iga_fast.m` and `fast_sensitivities.m`
//! (validated in `rust/tests/wq_vs_matlab.rs`).

use crate::iga::{basis_dense, elasticity_element_matrices, element_quad_points, element_shape_functions, element_vector_gradients, lame, load_vector, SpaceBox};
use crate::solver::PcgSolver;
use crate::sparse::AssemblyPattern;

/// Compliance sensitivities from a displacement field without forming element
/// matrices (MATLAB `fast_sensitivities`). `spline = false`: element densities
/// (returns one value per element); `true`: control-point densities on `sp`.
pub fn fast_sensitivities(sp: &SpaceBox, u: &[f64], x: &[f64], spline: bool, penal: f64, emin: f64, young: f64, poisson: f64) -> Vec<f64> {
    let (lam, mu) = lame(young, poisson);
    let dim = sp.dim;
    let mut out = vec![0.0; if spline { sp.ndof_sc } else { sp.nel }];
    for e in 0..sp.nel {
        let (_, w) = element_quad_points(sp, e);
        let grads = element_vector_gradients(sp, u, e);
        let n = if spline { Some(element_shape_functions(sp, e).0) } else { None };
        let mut ee = 0.0;
        for q in 0..sp.nqn {
            let g = &grads[q];
            let tr: f64 = (0..dim).map(|i| g[i][i]).sum();
            let mut epse = 0.0;
            for i in 0..dim {
                for j in 0..dim {
                    epse += (0.5 * (g[i][j] + g[j][i])).powi(2);
                }
            }
            let sed = lam * tr * tr + 2.0 * mu * epse;
            match &n {
                None => ee += w[q] * sed,
                Some(nv) => {
                    let rho: f64 = (0..sp.nsh_sc).map(|a| nv[q * sp.nsh_sc + a] * x[sp.conn_sc[e * sp.nsh_sc + a]]).sum();
                    let s = penal * rho.powf(penal - 1.0) * (1.0 - emin) * sed;
                    for a in 0..sp.nsh_sc {
                        out[sp.conn_sc[e * sp.nsh_sc + a]] -= w[q] * nv[q * sp.nsh_sc + a] * s;
                    }
                }
            }
        }
        if !spline {
            out[e] = -penal * x[e].powf(penal - 1.0) * (1.0 - emin) * ee;
        }
    }
    out
}

pub struct Cantilever2dResult {
    pub density: Vec<f64>,
    pub compliance: Vec<f64>,
}

/// MATLAB `topopt_iga_fast(nelx, nely, volfrac, penal, rmin, max_iter, density_type, degree)`
/// on the cantilever [0, 1] x [0, 0.5], clamped at x = 0, normalized load patch at the
/// centre of the free end. Densities are returned in MATLAB (direction 0 fastest) order.
pub fn topopt_iga_fast(nelx: usize, nely: usize, volfrac: f64, penal: f64, rmin: f64, max_iter: usize, spline: bool, degree: usize) -> Cantilever2dResult {
    let (l, h) = (1.0, 0.5);
    let sp = SpaceBox::new(&[[0.0, l], [0.0, h]], &[nelx, nely], degree);
    let (lam, mu) = lame(1.0, 0.3);
    let mut fixed = vec![false; sp.ndof];
    for i in sp.boundary_dofs(0) {
        fixed[i] = true;
    }
    let d = h / 10.0;
    let mut f = load_vector(&sp, |x| {
        let inside = x[0] >= l - d && x[1] <= h / 2.0 + d / 2.0 && x[1] >= h / 2.0 - d / 2.0;
        [0.0, if inside { -1.0 } else { 0.0 }, 0.0]
    });
    let fy: f64 = f[sp.ndof_sc..].iter().sum::<f64>().abs();
    if fy > 0.0 {
        f.iter_mut().for_each(|v| *v /= fy);
    }
    let em = elasticity_element_matrices(&sp, lam, mu);
    let pattern = AssemblyPattern::new(sp.ndof, &sp.connectivity, sp.nsh, sp.nel);
    let emin = 1e-3;
    let nel = nelx * nely;

    // element-density filter (element index units, strict < rmin)
    let filter: Vec<Vec<(usize, f64)>> = (0..nel)
        .map(|i| {
            let (xi, yi) = ((i % nelx) as f64, (i / nelx) as f64);
            (0..nel)
                .filter_map(|j| {
                    let dist = (((j % nelx) as f64 - xi).powi(2) + ((j / nelx) as f64 - yi).powi(2)).sqrt();
                    if dist < rmin { Some((j, rmin - dist)) } else { None }
                })
                .collect()
        })
        .collect();
    let hs: Vec<f64> = filter.iter().map(|r| r.iter().map(|x| x.1).sum()).collect();
    // spline density: element-centre interpolation P1 [nelx x ncpx], P2 [nely x ncpy]
    let (ncx, ncy) = (sp.ndof_dir[0], sp.ndof_dir[1]);
    let xc: Vec<f64> = (0..nelx).map(|e| 0.5 * (sp.univ[0].breaks[e] + sp.univ[0].breaks[e + 1])).collect();
    let yc: Vec<f64> = (0..nely).map(|e| 0.5 * (sp.univ[1].breaks[e] + sp.univ[1].breaks[e + 1])).collect();
    let (p1, _) = basis_dense(&sp.univ[0].knots, degree, &xc);
    let (p2, _) = basis_dense(&sp.univ[1].knots, degree, &yc);
    // rho_e(ex, ey) = sum_{a,b} P1[ex,a] x[a,b] P2[ey,b]
    let to_elem = |x: &[f64]| -> Vec<f64> {
        (0..nel)
            .map(|e| {
                let (ex, ey) = (e % nelx, e / nelx);
                let mut s = 0.0;
                for b in 0..ncy {
                    for a in 0..ncx {
                        s += p1[ex * ncx + a] * x[a + ncx * b] * p2[ey * ncy + b];
                    }
                }
                s
            })
            .collect()
    };
    let nvars = if spline { ncx * ncy } else { nel };
    let mut x = vec![volfrac; nvars];
    let v_target = volfrac * nel as f64;
    let v_eval = |x: &[f64]| -> f64 { if spline { to_elem(x).iter().sum() } else { x.iter().sum() } };
    let mut compliance = Vec::new();
    let mut u = vec![0.0; sp.ndof];
    let n = sp.ndof;
    let (mut change, mut iter) = (1.0, 0);
    let mv = 0.2;
    while iter < max_iter && change > 1e-3 {
        iter += 1;
        let rho_e = if spline { to_elem(&x) } else { x.clone() };
        let scale: Vec<f64> = rho_e.iter().map(|&r| emin + (1.0 - emin) * r.powf(penal)).collect();
        let k = pattern.assemble(|e| em.of_element(e), &scale);
        let b: Vec<f64> = (0..n).map(|i| if fixed[i] { 0.0 } else { f[i] }).collect();
        let diag: Vec<f64> = k.diagonal().iter().map(|&d| if d.abs() < 1e-14 { 1.0 } else { d }).collect();
        let mut pm = vec![0.0; n];
        let mut tmp = vec![0.0; n];
        let (sol, _, _) = PcgSolver::new(50 * n, 1e-13).solve(n, &b, &diag, |p, ap| {
            for i in 0..n {
                pm[i] = if fixed[i] { 0.0 } else { p[i] };
            }
            k.matvec(&pm, &mut tmp);
            for i in 0..n {
                ap[i] = if fixed[i] { 0.0 } else { tmp[i] };
            }
        });
        u = (0..n).map(|i| if fixed[i] { 0.0 } else { sol[i] }).collect();
        let c: f64 = f.iter().zip(&u).map(|(a, b)| a * b).sum();
        compliance.push(c);
        let nsh = sp.nsh;
        let ce: Vec<f64> = (0..nel)
            .map(|e| {
                let ke = em.of_element(e);
                let cn = &sp.connectivity[e * nsh..(e + 1) * nsh];
                (0..nsh).map(|a| u[cn[a]] * (0..nsh).map(|bb| ke[a * nsh + bb] * u[cn[bb]]).sum::<f64>()).sum()
            })
            .collect();
        let dc_elem: Vec<f64> = (0..nel).map(|e| -penal * (1.0 - emin) * rho_e[e].powf(penal - 1.0) * ce[e]).collect();
        let sens: Vec<f64> = if spline {
            // P1' * dC * P2
            (0..ncx * ncy)
                .map(|ab| {
                    let (a, bb) = (ab % ncx, ab / ncx);
                    -(0..nel).map(|e| p1[(e % nelx) * ncx + a] * dc_elem[e] * p2[(e / nelx) * ncy + bb]).sum::<f64>()
                })
                .collect()
        } else {
            (0..nel).map(|e| -(filter[e].iter().map(|&(j, w)| w * x[j] * dc_elem[j]).sum::<f64>()) / (hs[e] * x[e].max(1e-3))).collect()
        };
        let x_old = x.clone();
        let update = |lm: f64| -> Vec<f64> {
            (0..nvars).map(|i| (x_old[i] * (sens[i] / lm).max(0.0).sqrt()).min(x_old[i] + mv).min(1.0).max(x_old[i] - mv).max(1e-3)).collect()
        };
        let mut l1 = 0.0;
        let mut l2 = 2.0 * sens.iter().cloned().fold(f64::MIN, f64::max);
        while v_eval(&update(l2)) > v_target {
            l2 *= 2.0;
        }
        let mut x_new = x_old.clone();
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4 {
            let lmid = 0.5 * (l1 + l2);
            x_new = update(lmid);
            if v_eval(&x_new) > v_target {
                l1 = lmid;
            } else {
                l2 = lmid;
            }
        }
        change = x_new.iter().zip(&x_old).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        x = x_new;
    }
    Cantilever2dResult { density: x, compliance }
}
