//! Rust multi-body contact vs the MATLAB reference (`tests/export_rust_contact_fixture.m`).

use immersed_iga::cad_model::{parse_json, JsonValue};
use immersed_iga::contact::{Assembly, BodySpec, ContactMode};
use immersed_iga::cut_cell::TriangleMesh3D;
use immersed_iga::octree_fem::OctreeBc;

fn nums(v: &JsonValue) -> Vec<f64> {
    match v {
        JsonValue::Number(x) => vec![*x],
        JsonValue::Array(a) => a.iter().flat_map(nums).collect(),
        JsonValue::Bool(b) => vec![*b as u8 as f64],
        _ => panic!("expected numbers"),
    }
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt();
    n / b.iter().map(|y| y * y).sum::<f64>().sqrt().max(1e-300)
}

#[test]
fn punch_contact_matches_matlab_reference() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/contact_punch.json");
    let d = parse_json(&std::fs::read_to_string(path).expect("run tests/export_rust_contact_fixture.m")).unwrap();
    let g = |k: &str| nums(d.get(k).unwrap_or_else(|| panic!("missing {}", k)));
    let el: Vec<[usize; 3]> = g("elements").chunks(3).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect();
    let mesh = |key: &str| TriangleMesh3D::new(g(key).chunks(3).map(|c| [c[0], c[1], c[2]]).collect(), el.clone());
    let (e, nu) = (g("E")[0], g("nu")[0]);
    let specs = vec![
        BodySpec { name: "Base".into(), brep: mesh("base_nodes"), young: e, poisson: nu, grid_res: [2, 2, 2], max_level: 0 },
        BodySpec { name: "Punch".into(), brep: mesh("punch_nodes"), young: e, poisson: nu, grid_res: [2, 2, 2], max_level: 0 },
    ];
    let asm = Assembly::setup(specs, Some(g("gap_tol")[0]), 0.1);
    assert_eq!(asm.total_dof, g("total_dof")[0] as usize);
    let ndof = g("ndof");
    assert_eq!(asm.bodies[0].ndof, ndof[0] as usize);
    assert_eq!(asm.bodies[1].ndof, ndof[1] as usize);
    assert_eq!(asm.interfaces.len(), 1);
    let it = &asm.interfaces[0];
    assert_eq!(it.pts_a.len(), g("n_pairs")[0] as usize);
    let fa: Vec<usize> = g("facets_A").iter().map(|&v| v as usize).collect();
    let fb: Vec<usize> = g("facets_B").iter().map(|&v| v as usize).collect();
    assert_eq!(it.facets_a, fa);
    assert_eq!(it.facets_b, fb);
    let flat = |v: &Vec<[f64; 3]>| v.iter().flatten().copied().collect::<Vec<f64>>();
    assert!(rel(&flat(&it.pts_a), &g("pts_A")) < 1e-14);
    assert!(rel(&flat(&it.normals), &g("normals")) < 1e-14);

    // BCs: base clamped on z = 0 (strong), punch loaded on z = 7
    let cz = |m: &TriangleMesh3D, f: usize| (m.vertices[m.triangles[f][0]][2] + m.vertices[m.triangles[f][1]][2] + m.vertices[m.triangles[f][2]][2]) / 3.0;
    let base_bottom: Vec<usize> = (0..el.len()).filter(|&f| cz(&asm.bodies[0].spec.brep, f).abs() < 1e-3).collect();
    let punch_top: Vec<usize> = (0..el.len()).filter(|&f| (cz(&asm.bodies[1].spec.brep, f) - 7.0).abs() < 1e-3).collect();
    let bcs = vec![
        vec![OctreeBc::DirichletStrong { facets: base_bottom, components: [true; 3], value: [0.0; 3] }],
        vec![OctreeBc::Traction { facets: punch_top, traction: [0.0, 0.0, -25.0] }],
    ];
    let gamma = g("gamma_c")[0];

    let uni = asm.solve(&bcs, ContactMode::Unilateral, gamma, 10, 1e-4, 1e-13);
    let u_ref = g("unilateral_u");
    // In unilateral mode the punch is held only by two normal (z) contact springs, so the
    // system is singular: the punch can translate in x/y and rotate (about z and about the
    // line through the two contact points) freely, and the MATLAB direct solve returns an
    // arbitrary member of the solution family. Compare what is determined: the clamped
    // base's displacement, the gaps and the active set.
    let nb = asm.bodies[0].ndof;
    let eu = rel(&uni.u[..nb], &u_ref[..nb]);
    let eu_full = rel(&uni.u, &u_ref);
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    println!(
        "base: |u_rust| = {:e}, |u_matlab| = {:e}, rel diff {:e}; punch: |u_rust| = {:e}, |u_matlab| = {:e}",
        norm(&uni.u[..nb]), norm(&u_ref[..nb]), rel(&uni.u[..nb], &u_ref[..nb]), norm(&uni.u[nb..]), norm(&u_ref[nb..])
    );
    println!("unilateral: {} iterations, active {:?}, gaps {:?}, base differs by {:e} (full vector incl. punch rigid modes {:e})", uni.iterations, uni.active, uni.gaps, eu, eu_full);
    assert_eq!(uni.iterations, g("unilateral_iterations")[0] as usize);
    assert!(uni.converged);
    let act_ref: Vec<bool> = g("unilateral_active").iter().map(|&v| v != 0.0).collect();
    assert_eq!(uni.active, act_ref);
    assert!((uni.gaps.iter().cloned().fold(f64::MAX, f64::min) - g("unilateral_min_gap")[0]).abs() < 1e-12);
    assert!(eu < 1e-7, "unilateral base displacement differs by {:e}", eu);

    let bon = asm.solve(&bcs, ContactMode::Bonded, gamma, 1, 1e-4, 1e-13);
    // Bonded with only two contact points still leaves the punch free to rotate about the
    // line through them (5 of 6 rigid modes removed): singular as well, so compare the base.
    let ub_ref = g("bonded_u");
    let eb = rel(&bon.u[..nb], &ub_ref[..nb]);
    println!("bonded: base differs by {:e}, gaps {:?}", eb, bon.gaps);
    assert!(eb < 1e-7, "bonded base displacement differs by {:e}", eb);
}
