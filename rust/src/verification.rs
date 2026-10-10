//! Verification problems with known answers, shared by the `obstacle_course`
//! binary and the test suite. Each function returns the measured quantity; the
//! caller decides the tolerance.

use crate::cad_model::{BoundaryConditionType, CadAssembly, CadBody, LabeledBoundaryCondition, MaterialProperties};
use crate::cut_cell::TriangleMesh3D;
use crate::iga::{elasticity_element_matrices, element_quad_points, element_vector_values, eval_vector_at, lame, load_vector, SpaceBox};
use crate::immersed::{build_immersed_problem, solve_immersed_problem, ImmersedOptions, Stabilization};
use crate::immersed_bc::{surface_boundary_terms, surface_quadrature};
use crate::solver::PcgSolver;
use crate::sparse::{assemble, CsrMatrix};
use std::f64::consts::PI;

fn stiffness(sp: &SpaceBox, young: f64, poisson: f64) -> CsrMatrix {
    let (lambda, mu) = lame(young, poisson);
    let em = elasticity_element_matrices(sp, lambda, mu);
    assemble(sp.ndof, &sp.connectivity, sp.nsh, sp.nel, |e| em.of_element(e), None)
}

/// Solves `K_ff u_f = rhs_f - K_fb u_b` with PCG; returns the full vector.
pub fn solve_with_dirichlet(k: &CsrMatrix, rhs: &[f64], fixed: &[bool], u_fixed: &[f64], tol: f64) -> (Vec<f64>, f64) {
    let n = k.nrows;
    let ub: Vec<f64> = (0..n).map(|i| if fixed[i] { u_fixed[i] } else { 0.0 }).collect();
    let mut kub = vec![0.0; n];
    k.matvec(&ub, &mut kub);
    let b: Vec<f64> = (0..n).map(|i| if fixed[i] { 0.0 } else { rhs[i] - kub[i] }).collect();
    let diag: Vec<f64> = k.diagonal().iter().enumerate().map(|(i, &d)| if fixed[i] { 1.0 } else { d }).collect();
    let mut pm = vec![0.0; n];
    let mut tmp = vec![0.0; n];
    let (x, _, res) = PcgSolver::new(20 * n, tol).solve(n, &b, &diag, |p, ap| {
        for i in 0..n {
            pm[i] = if fixed[i] { 0.0 } else { p[i] };
        }
        k.matvec(&pm, &mut tmp);
        for i in 0..n {
            ap[i] = if fixed[i] { p[i] } else { tmp[i] };
        }
    });
    ((0..n).map(|i| if fixed[i] { ub[i] } else { x[i] }).collect(), res)
}

fn boundary_mask(sp: &SpaceBox) -> Vec<bool> {
    let mut fixed = vec![false; sp.ndof];
    for side in 0..2 * sp.dim {
        for i in sp.boundary_dofs(side) {
            fixed[i] = true;
        }
    }
    fixed
}

/// Patch test: a linear displacement imposed on the boundary control points must be
/// reproduced in the interior. Returns the relative error.
pub fn patch_test(bounds: &[[f64; 2]], nsub: &[usize], degree: usize) -> f64 {
    let sp = SpaceBox::new(bounds, nsub, degree);
    let dim = sp.dim;
    let g = sp.greville_points();
    let mut u_exact = vec![0.0; sp.ndof];
    for c in 0..dim {
        for i in 0..sp.ndof_sc {
            u_exact[c * sp.ndof_sc + i] = (0..dim).map(|j| ((c * dim + j) as f64 + 1.0) * 1e-2 * g[i][j]).sum::<f64>() + 0.1 * (c as f64 + 1.0);
        }
    }
    let k = stiffness(&sp, 1.0, 0.3);
    let (u, _) = solve_with_dirichlet(&k, &vec![0.0; sp.ndof], &boundary_mask(&sp), &u_exact, 1e-12);
    let err: f64 = u.iter().zip(&u_exact).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    err / u_exact.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Manufactured plane-strain solution u = (sin(pi x) sin(pi y), 0) on the unit square.
/// Returns the L2 errors for the given element counts per direction.
pub fn manufactured_solution_errors(degree: usize, nels: &[usize]) -> Vec<f64> {
    let (lam, mu) = lame(1.0, 0.3);
    nels.iter()
        .map(|&nel| {
            let sp = SpaceBox::new(&[[0.0, 1.0], [0.0, 1.0]], &[nel, nel], degree);
            let k = stiffness(&sp, 1.0, 0.3);
            let f = load_vector(&sp, |x| {
                [
                    PI * PI * (lam + 3.0 * mu) * (PI * x[0]).sin() * (PI * x[1]).sin(),
                    -PI * PI * (lam + mu) * (PI * x[0]).cos() * (PI * x[1]).cos(),
                    0.0,
                ]
            });
            let (u, _) = solve_with_dirichlet(&k, &f, &boundary_mask(&sp), &vec![0.0; sp.ndof], 1e-12);
            let mut e2 = 0.0;
            for e in 0..sp.nel {
                let (x, w) = element_quad_points(&sp, e);
                let uh = element_vector_values(&sp, &u, e);
                for q in 0..sp.nqn {
                    let ex = (PI * x[q][0]).sin() * (PI * x[q][1]).sin();
                    e2 += w[q] * ((uh[q][0] - ex).powi(2) + uh[q][1].powi(2));
                }
            }
            e2.sqrt()
        })
        .collect()
}

/// Immersed cantilever (box L x b x h, clamped at x = 0 by penalty, tip traction):
/// returns `(tip deflection, Timoshenko deflection, PCG iterations)`.
pub fn immersed_cantilever(grid_res: [usize; 3], stabilization: Stabilization) -> (f64, f64, usize) {
    let (l, b, h) = (4.0, 0.5, 0.5);
    let p_total = 1e-3;
    let mesh = TriangleMesh3D::new_box([0.0, 0.0, 0.0], [l, b, h]);
    let mat = MaterialProperties { name: "unit".into(), youngs_modulus: 1.0, poissons_ratio: 0.3, density: 1.0 };
    let mut body = CadBody::new("beam", mesh, mat);
    body.label_face_by_predicate("Clamp", |c, _| c[0] < 1e-9);
    body.label_face_by_predicate("Tip", move |c, _| c[0] > l - 1e-9);
    let mut asm = CadAssembly::new();
    asm.add_body(body);
    asm.add_boundary_condition(LabeledBoundaryCondition {
        target_face_name: "Clamp".into(),
        bc_type: BoundaryConditionType::Dirichlet { components: [true, true, true], values: [0.0; 3] },
    });
    asm.add_boundary_condition(LabeledBoundaryCondition {
        target_face_name: "Tip".into(),
        bc_type: BoundaryConditionType::NeumannTraction { traction: [0.0, -p_total / (b * h), 0.0] },
    });
    let opts = ImmersedOptions { grid_res, clamped_face: None, stabilization, ..ImmersedOptions::default() };
    let mut pb = build_immersed_problem(&asm.bodies[0].mesh, &opts);
    pb.force.iter_mut().for_each(|f| *f = 0.0);
    let st = surface_boundary_terms(&pb.sp, &asm, 0, 1e3);
    pb.add_surface_terms(&st);
    let sol = solve_immersed_problem(&pb, 1e-10, 20 * pb.sp.ndof);
    let face = &asm.bodies[0].faces["Tip"];
    let (mut s, mut a) = (0.0, 0.0);
    for (x, w, _) in surface_quadrature(&asm.bodies[0], &face.triangle_indices, 0.05) {
        s += w * eval_vector_at(&pb.sp, &sol.u, &x)[1];
        a += w;
    }
    let g = 1.0 / (2.0 * 1.3);
    let inertia = b * h.powi(3) / 12.0;
    let timoshenko = p_total * l.powi(3) / (3.0 * inertia) + p_total * l / (5.0 / 6.0 * g * b * h);
    (-s / a, timoshenko, sol.iterations)
}
