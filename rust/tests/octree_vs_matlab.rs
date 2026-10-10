//! Rust octree / MPC / BC stack vs the MATLAB reference (`tests/export_rust_octree_fixture.m`).

use immersed_iga::cad_model::{parse_json, JsonValue};
use immersed_iga::cut_cell::TriangleMesh3D;
use immersed_iga::octree::{BoundingBox3D, OctreeMesh3D};
use immersed_iga::octree_fem::{apply_boundary_conditions, solve, OctreeBc, OctreeFem};

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

fn build(case: &JsonValue, brep: Option<&TriangleMesh3D>) -> OctreeFem {
    let g = |k: &str| nums(case.get(k).unwrap());
    let gb = g("grid_bounds");
    let res = g("grid_res");
    let bounds = BoundingBox3D::new([gb[0], gb[2], gb[4]], [gb[1], gb[3], gb[5]]);
    let mut oct = OctreeMesh3D::build(bounds, [res[0] as usize, res[1] as usize, res[2] as usize], g("max_level")[0] as usize, |_| false);
    let sub: Vec<usize> = g("subdivide").iter().map(|&v| v as usize).collect();
    oct.subdivide_leaves(&sub);
    oct.balance_2_to_1();
    OctreeFem::build(&oct, brep, g("E")[0], g("nu")[0], 1e-4, [3, 3, 3])
}

fn check_mesh(fem: &OctreeFem, case: &JsonValue, name: &str) {
    let g = |k: &str| nums(case.get(k).unwrap());
    assert_eq!(fem.elem_nodes.len(), g("n_elements")[0] as usize, "{}: leaves", name);
    assert_eq!(fem.nodes.len(), g("n_nodes")[0] as usize, "{}: nodes", name);
    assert_eq!(fem.n_master(), g("n_master")[0] as usize, "{}: master nodes", name);
    let lv: Vec<usize> = g("levels").iter().map(|&v| v as usize).collect();
    assert_eq!(fem.levels, lv, "{}: leaf levels/order", name);
    let en = g("elem_nodes"); // [n_el x 8] row-major
    for (e, row) in fem.elem_nodes.iter().enumerate() {
        for a in 0..8 {
            assert_eq!(row[a], en[e * 8 + a] as usize, "{}: element {} node {}", name, e, a);
        }
    }
    let mids: Vec<usize> = g("master_ids").iter().map(|&v| v as usize).collect();
    assert_eq!(fem.master_ids, mids, "{}: master ids", name);
    let w = g("weights");
    assert!(fem.weights.iter().zip(&w).all(|(a, b)| (a - b).abs() < 1e-12), "{}: weights", name);
    let n = 3 * fem.n_master();
    let v: Vec<f64> = (1..=n).map(|i| (0.37 * i as f64).sin()).collect();
    let mut y = vec![0.0; n];
    fem.k_master.matvec(&v, &mut y);
    let e = rel(&y, &g("k_probe"));
    assert!(e < 1e-12, "{}: K_master * v differs by {:e}", name, e);
}

#[test]
fn octree_stack_matches_matlab_reference() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/octree_cases.json");
    let d = parse_json(&std::fs::read_to_string(path).expect("run tests/export_rust_octree_fixture.m")).unwrap();
    let bn = nums(d.get("brep_nodes").unwrap());
    let be = nums(d.get("brep_elements").unwrap());
    let brep = TriangleMesh3D::new(
        bn.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
        be.chunks(3).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect(),
    );
    let cases = d.get("cases").unwrap().as_array().unwrap();

    // Case A: refined octree, no immersed geometry
    let fem_a = build(&cases[0], None);
    check_mesh(&fem_a, &cases[0], "A");

    // Case B: immersed box, strong Dirichlet (facets with |z + 1| < 2) and traction
    let fem_b = build(&cases[1], Some(&brep));
    check_mesh(&fem_b, &cases[1], "B");
    let cen_z: Vec<f64> = brep.triangles.iter().map(|t| (brep.vertices[t[0]][2] + brep.vertices[t[1]][2] + brep.vertices[t[2]][2]) / 3.0).collect();
    let cen_x: Vec<f64> = brep.triangles.iter().map(|t| (brep.vertices[t[0]][0] + brep.vertices[t[1]][0] + brep.vertices[t[2]][0]) / 3.0).collect();
    let dir: Vec<usize> = (0..brep.triangles.len()).filter(|&f| (cen_z[f] + 1.0).abs() < 2.0).collect();
    // The MATLAB Neumann filter |z - 11| < 2 selects no facet; MATLAB then applies the
    // traction to *all* facets (empty selection = all). Reproduce that explicitly.
    let all: Vec<usize> = (0..brep.triangles.len()).collect();
    let sys = apply_boundary_conditions(
        &fem_b,
        &brep,
        &[
            OctreeBc::DirichletStrong { facets: dir, components: [true; 3], value: [0.0; 3] },
            OctreeBc::Traction { facets: all, traction: [10.0, 0.0, 0.0] },
        ],
    );
    let g = |k: &str| nums(cases[1].get(k).unwrap());
    let fixed_ref: Vec<usize> = g("fixed_dofs").iter().map(|&v| v as usize).collect();
    assert_eq!(sys.fixed.iter().map(|x| x.0).collect::<Vec<_>>(), fixed_ref, "B: fixed DOFs");
    assert!(rel(&sys.force, &g("force")) < 1e-12, "B: load vector");
    let (u, it, res) = solve(&sys, 1e-12, 100000);
    let eu = rel(&u, &g("u_direct"));
    println!("case B: {} PCG iterations, residual {:e}, |u - u_matlab| / |u| = {:e}", it, res, eu);
    assert!(eu < 1e-8, "B: displacement differs by {:e}", eu);

    // Case B, second BC set: pressure on z = 2 and Robin springs on x = 0
    let top: Vec<usize> = (0..brep.triangles.len()).filter(|&f| (cen_z[f] - 2.0).abs() < 1e-6).collect();
    let left: Vec<usize> = (0..brep.triangles.len()).filter(|&f| cen_x[f].abs() < 1e-6).collect();
    let sys2 = apply_boundary_conditions(
        &fem_b,
        &brep,
        &[OctreeBc::Pressure { facets: top, pressure: 3.0 }, OctreeBc::Robin { facets: left, kn: 50.0, kt: 50.0 }],
    );
    assert!(rel(&sys2.force, &g("bc2_force")) < 1e-12, "B2: pressure load");
    let n = sys2.force.len();
    let v: Vec<f64> = (1..=n).map(|i| (0.37 * i as f64).sin()).collect();
    let mut y = vec![0.0; n];
    sys2.stiffness.matvec(&v, &mut y);
    assert!(rel(&y, &g("bc2_k_probe")) < 1e-12, "B2: K + K_robin");
}
