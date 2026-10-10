//! Mathematical verification of the Rust IGA kernel (mirrors tests/testIgaKernel.m).

use immersed_iga::iga::{elasticity_element_matrices, element_quad_points, element_vector_values, lame, load_vector, SpaceBox};
use immersed_iga::solver::PcgSolver;
use immersed_iga::sparse::{assemble, CsrMatrix};
use std::f64::consts::PI;

fn stiffness(sp: &SpaceBox) -> CsrMatrix {
    let (lambda, mu) = lame(1.0, 0.3);
    let em = elasticity_element_matrices(sp, lambda, mu);
    assemble(sp.ndof, &sp.connectivity, sp.nsh, sp.nel, |e| em.of_element(e), None)
}

/// Solve K_ff u_f = rhs_f - K_fb u_b for the DOFs not in `fixed` (CG, tight tolerance).
fn solve_dirichlet(k: &CsrMatrix, rhs: &[f64], fixed: &[bool], u_fixed: &[f64]) -> Vec<f64> {
    let n = k.nrows;
    let mut ub = vec![0.0; n];
    for i in 0..n {
        if fixed[i] {
            ub[i] = u_fixed[i];
        }
    }
    let mut kub = vec![0.0; n];
    k.matvec(&ub, &mut kub);
    let b: Vec<f64> = (0..n).map(|i| if fixed[i] { 0.0 } else { rhs[i] - kub[i] }).collect();
    let diag: Vec<f64> = k.diagonal().iter().enumerate().map(|(i, &d)| if fixed[i] { 1.0 } else { d }).collect();
    let mut tmp = vec![0.0; n];
    let (x, _, res) = PcgSolver::new(20 * n, 1e-12).solve(n, &b, &diag, |p, ap| {
        let pm: Vec<f64> = (0..n).map(|i| if fixed[i] { 0.0 } else { p[i] }).collect();
        k.matvec(&pm, &mut tmp);
        for i in 0..n {
            ap[i] = if fixed[i] { p[i] } else { tmp[i] };
        }
    });
    assert!(res < 1e-11, "CG did not converge: {:e}", res);
    (0..n).map(|i| if fixed[i] { ub[i] } else { x[i] }).collect()
}

fn all_boundary(sp: &SpaceBox) -> Vec<bool> {
    let mut fixed = vec![false; sp.ndof];
    for side in 0..2 * sp.dim {
        for i in sp.boundary_dofs(side) {
            fixed[i] = true;
        }
    }
    fixed
}

#[test]
fn stiffness_is_symmetric_and_annihilates_rigid_body_modes() {
    for (bounds, nsub, p) in [
        (vec![[0.0, 1.0], [0.0, 0.5]], vec![4, 3], 3usize),
        (vec![[0.0, 1.2], [0.0, 0.6], [0.0, 0.3]], vec![3, 2, 2], 2usize),
    ] {
        let sp = SpaceBox::new(&bounds, &nsub, p);
        let k = stiffness(&sp);
        let fro = k.frobenius();
        for r in 0..k.nrows {
            for idx in k.indptr[r]..k.indptr[r + 1] {
                let c = k.indices[idx];
                assert!((k.data[idx] - k.get(c, r)).abs() < 1e-13 * fro);
            }
        }
        // rigid-body modes: translations and infinitesimal rotations at the Greville points
        let g = sp.greville_points();
        let dim = sp.dim;
        let mut modes: Vec<Vec<f64>> = Vec::new();
        for c in 0..dim {
            let mut m = vec![0.0; sp.ndof];
            for i in 0..sp.ndof_sc {
                m[c * sp.ndof_sc + i] = 1.0;
            }
            modes.push(m);
        }
        let planes: Vec<(usize, usize)> = if dim == 2 { vec![(0, 1)] } else { vec![(0, 1), (1, 2), (0, 2)] };
        for (a, b) in planes {
            let mut m = vec![0.0; sp.ndof];
            for i in 0..sp.ndof_sc {
                m[a * sp.ndof_sc + i] = -g[i][b];
                m[b * sp.ndof_sc + i] = g[i][a];
            }
            modes.push(m);
        }
        for m in &modes {
            let mut y = vec![0.0; sp.ndof];
            k.matvec(m, &mut y);
            let ny: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            let nm: f64 = m.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(ny / (fro * nm) < 1e-13, "rigid-body residual {:e}", ny / (fro * nm));
        }
    }
}

#[test]
fn patch_test_reproduces_linear_field() {
    for (bounds, nsub, p) in [
        (vec![[0.0, 2.0], [-1.0, 0.5]], vec![5, 4], 2usize),
        (vec![[0.0, 1.0], [0.0, 2.0], [0.5, 1.0]], vec![3, 4, 3], 3usize),
    ] {
        let sp = SpaceBox::new(&bounds, &nsub, p);
        let dim = sp.dim;
        let g = sp.greville_points();
        let a = |i: usize, j: usize| ((i * dim + j) as f64 + 1.0) * 1e-2 - 0.03;
        let mut u_exact = vec![0.0; sp.ndof];
        for c in 0..dim {
            for i in 0..sp.ndof_sc {
                u_exact[c * sp.ndof_sc + i] = (0..dim).map(|j| a(c, j) * g[i][j]).sum::<f64>() + 0.1 * (c as f64 + 1.0);
            }
        }
        let k = stiffness(&sp);
        let fixed = all_boundary(&sp);
        let u = solve_dirichlet(&k, &vec![0.0; sp.ndof], &fixed, &u_exact);
        let err: f64 = u.iter().zip(&u_exact).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
        let nrm: f64 = u_exact.iter().map(|y| y * y).sum::<f64>().sqrt();
        assert!(err / nrm < 1e-10, "patch test error {:e}", err / nrm);
    }
}

#[test]
fn manufactured_solution_converges_at_optimal_rate() {
    // Plane strain on the unit square, u = (sin(pi x) sin(pi y), 0), u = 0 on the boundary.
    let (lam, mu) = lame(1.0, 0.3);
    for p in [2usize, 3] {
        let mut errs = Vec::new();
        for nel in [4usize, 8, 16] {
            let sp = SpaceBox::new(&[[0.0, 1.0], [0.0, 1.0]], &[nel, nel], p);
            let k = stiffness(&sp);
            let f = load_vector(&sp, |x| {
                [
                    PI * PI * (lam + 3.0 * mu) * (PI * x[0]).sin() * (PI * x[1]).sin(),
                    -PI * PI * (lam + mu) * (PI * x[0]).cos() * (PI * x[1]).cos(),
                    0.0,
                ]
            });
            let fixed = all_boundary(&sp);
            let u = solve_dirichlet(&k, &f, &fixed, &vec![0.0; sp.ndof]);
            let mut e2 = 0.0;
            for e in 0..sp.nel {
                let (x, w) = element_quad_points(&sp, e);
                let uh = element_vector_values(&sp, &u, e);
                for q in 0..sp.nqn {
                    let ex = (PI * x[q][0]).sin() * (PI * x[q][1]).sin();
                    e2 += w[q] * ((uh[q][0] - ex).powi(2) + uh[q][1].powi(2));
                }
            }
            errs.push(e2.sqrt());
        }
        let rate = (errs[1] / errs[2]).log2();
        assert!(rate > p as f64 + 1.0 - 0.25, "p = {}: L2 errors {:?}, last rate {:.3}", p, errs, rate);
    }
}
