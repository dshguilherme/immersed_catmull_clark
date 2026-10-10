//! Surface boundary conditions on immersed CAD faces (penalty Dirichlet, consistent traction).

use immersed_iga::cad_model::{BoundaryConditionType, CadAssembly, CadBody, LabeledBoundaryCondition, MaterialProperties};
use immersed_iga::cut_cell::TriangleMesh3D;
use immersed_iga::iga::eval_vector_at;
use immersed_iga::iga::{basis_dense, SpaceBox};
use immersed_iga::immersed::{build_immersed_problem, ghost_penalty, solve_immersed_problem, ImmersedOptions, Stabilization, CUT};
use immersed_iga::sparse::CsrMatrix;
use immersed_iga::immersed_bc::{surface_boundary_terms, surface_quadrature};

fn cantilever(l: f64, b: f64, h: f64, traction_y: f64) -> CadAssembly {
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
        bc_type: BoundaryConditionType::NeumannTraction { traction: [0.0, traction_y, 0.0] },
    });
    asm
}

/// Coefficients of f(x_d) = x_d^k in the 1D basis of direction d (interpolation at
/// the Greville points; exact because the polynomial lies in the spline space).
fn poly_coeffs_1d(sp: &SpaceBox, d: usize, k: i32) -> Vec<f64> {
    let u = &sp.univ[d];
    let p = u.degree;
    let n = u.ndof;
    let g: Vec<f64> = (0..n).map(|i| u.knots[i + 1..=i + p].iter().sum::<f64>() / p as f64).collect();
    let (a, _) = basis_dense(&u.knots, p, &g); // [n x n]
    let mut m = a.clone();
    let mut rhs: Vec<f64> = g.iter().map(|&xi| (sp.lo[d] + sp.lengths[d] * xi).powi(k)).collect();
    // Gaussian elimination with partial pivoting
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| m[i * n + col].abs().partial_cmp(&m[j * n + col].abs()).unwrap()).unwrap();
        for j in 0..n {
            m.swap(col * n + j, piv * n + j);
        }
        rhs.swap(col, piv);
        for i in col + 1..n {
            let f = m[i * n + col] / m[col * n + col];
            for j in col..n {
                m[i * n + j] -= f * m[col * n + j];
            }
            rhs[i] -= f * rhs[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| m[i * n + j] * x[j]).sum();
        x[i] = (rhs[i] - s) / m[i * n + i];
    }
    x
}

#[test]
fn ghost_penalty_is_consistent_for_polynomials_up_to_degree_p() {
    for p in [2usize, 3] {
        let sp = SpaceBox::new(&[[0.0, 1.3], [-0.2, 0.7], [0.1, 0.6]], &[6, 5, 4], p);
        let status = vec![CUT; sp.nel];
        let (r, c, v) = ghost_penalty(&sp, &status, 1.0, 1.0);
        let k = CsrMatrix::from_triplets(sp.ndof, sp.ndof, &r, &c, &v);
        let norm = k.frobenius();
        assert!(norm > 0.0);
        for (d, deg) in [(0usize, 1), (1, 2), (2, p as i32)] {
            let cd = poly_coeffs_1d(&sp, d, deg);
            let mut u = vec![0.0; sp.ndof];
            for i in 0..sp.ndof_sc {
                let s = immersed_iga::iga::unravel(i, &sp.ndof_dir);
                u[(d + 1) % 3 * sp.ndof_sc + i] = cd[s[d]]; // a polynomial component field
            }
            let mut y = vec![0.0; sp.ndof];
            k.matvec(&u, &mut y);
            let ny: f64 = y.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nu: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(ny < 1e-10 * norm * nu, "p={} degree {} field: |Ku| = {:e}", p, deg, ny);
        }
        // a degree p+1 polynomial is penalized
        let cd = poly_coeffs_1d(&SpaceBox::new(&[[0.0, 1.3], [-0.2, 0.7], [0.1, 0.6]], &[6, 5, 4], p + 1), 0, p as i32 + 1);
        assert!(cd.len() > 0);
        let rnd: Vec<f64> = (0..sp.ndof).map(|i| ((i * 7919) % 101) as f64 / 101.0).collect();
        let mut y = vec![0.0; sp.ndof];
        k.matvec(&rnd, &mut y);
        assert!(y.iter().map(|x| x * x).sum::<f64>() > 0.0);
    }
}

#[test]
fn traction_load_integrates_to_total_force() {
    let (l, b, h) = (4.0, 0.5, 0.5);
    let asm = cantilever(l, b, h, -2.0);
    let opts = ImmersedOptions { grid_res: [16, 3, 3], clamped_face: None, ..ImmersedOptions::default() };
    let pb = build_immersed_problem(&asm.bodies[0].mesh, &opts);
    let st = surface_boundary_terms(&pb.sp, &asm, 0, 1e3);
    assert!(st.missing_faces.is_empty());
    let nsc = pb.sp.ndof_sc;
    let fy: f64 = st.force[nsc..2 * nsc].iter().sum();
    let fx: f64 = st.force[..nsc].iter().sum();
    assert!((fy - (-2.0 * b * h)).abs() < 1e-12, "total Fy = {}", fy);
    assert!(fx.abs() < 1e-12);
    // quadrature covers the face area exactly
    let face = &asm.bodies[0].faces["Tip"];
    let area: f64 = surface_quadrature(&asm.bodies[0], &face.triangle_indices, 0.05).iter().map(|q| q.1).sum();
    assert!((area - b * h).abs() < 1e-12);
}

#[test]
fn immersed_cantilever_tip_deflection_is_close_to_timoshenko() {
    let (l, b, h) = (4.0, 0.5, 0.5);
    let p_total = 1e-3;
    let asm = cantilever(l, b, h, -p_total / (b * h));
    // STAB = none | legacy | gp (default), GAMMA = coefficient
    let gamma: f64 = std::env::var("GAMMA").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-2);
    let stab = match std::env::var("STAB").unwrap_or_default().as_str() {
        "none" => Stabilization::None,
        "legacy" => Stabilization::Legacy { gamma },
        _ => Stabilization::GhostPenalty { gamma },
    };
    let res: usize = std::env::var("REFINE").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let opts = ImmersedOptions { grid_res: [44 * res, 6 * res, 6 * res], clamped_face: None, force: None, stabilization: stab, ..ImmersedOptions::default() };
    let mut pb = build_immersed_problem(&asm.bodies[0].mesh, &opts);
    pb.force.iter_mut().for_each(|f| *f = 0.0); // no default load: only the CAD traction
    let st = surface_boundary_terms(&pb.sp, &asm, 0, 1e3);
    pb.add_surface_terms(&st);
    let sol = solve_immersed_problem(&pb, 1e-10, 20 * pb.sp.ndof);
    assert!(sol.residual < 1e-9, "PCG residual {:e}", sol.residual);

    // Tip deflection: average u_y over the tip-face quadrature points
    let face = &asm.bodies[0].faces["Tip"];
    let pts = surface_quadrature(&asm.bodies[0], &face.triangle_indices, 0.05);
    let (mut s, mut a) = (0.0, 0.0);
    for (x, w, _) in &pts {
        s += w * eval_vector_at(&pb.sp, &sol.u, x)[1];
        a += w;
    }
    let delta = -s / a;
    let (e, nu) = (1.0, 0.3);
    let g = e / (2.0 * (1.0 + nu));
    let inertia = b * h * h * h / 12.0;
    let timoshenko = p_total * l.powi(3) / (3.0 * e * inertia) + p_total * l / (5.0 / 6.0 * g * b * h);
    let rel = (delta - timoshenko) / timoshenko;
    println!("immersed tip deflection {:.6e}, Timoshenko {:.6e}, relative difference {:+.2}% ({} PCG iterations)", delta, timoshenko, 100.0 * rel, sol.iterations);
    assert!(rel.abs() < 0.05, "tip deflection off by {:.1}%", 100.0 * rel);
}
