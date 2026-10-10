//! Rust port of src/catmull_clark vs MATLAB (`tests/export_rust_cc_fixture.m`).

use immersed_iga::cad_model::{parse_json, JsonValue};
use immersed_iga::catmull_clark::{barycentric_basis, projection_matrix, subdivide_quad_catmull_clark, subdivide_triangles_centroid, subdivision_matrix_1d};

fn nums(v: &JsonValue) -> Vec<f64> {
    match v {
        JsonValue::Number(x) => vec![*x],
        JsonValue::Array(a) => a.iter().flat_map(nums).collect(),
        _ => panic!("expected numbers"),
    }
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn catmull_clark_matches_matlab_reference() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/catmull_clark.json");
    let d = parse_json(&std::fs::read_to_string(path).expect("run tests/export_rust_cc_fixture.m")).unwrap();
    let g = |k: &str| nums(d.get(k).unwrap_or_else(|| panic!("missing {}", k)));
    let v3 = |k: &str| g(k).chunks(3).map(|c| [c[0], c[1], c[2]]).collect::<Vec<[f64; 3]>>();
    let q4 = |k: &str| g(k).chunks(4).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize]).collect::<Vec<[usize; 4]>>();
    let t3 = |k: &str| g(k).chunks(3).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect::<Vec<[usize; 3]>>();
    let flat = |v: &Vec<[f64; 3]>| v.iter().flatten().copied().collect::<Vec<f64>>();

    let (v1, f1) = subdivide_quad_catmull_clark(&v3("V"), &q4("F"));
    assert_eq!(f1, q4("F1"));
    assert!(close(&flat(&v1), &g("V1"), 1e-14), "first Catmull-Clark step vertices");
    let (v2, f2) = subdivide_quad_catmull_clark(&v1, &f1);
    assert_eq!(f2, q4("F2"));
    assert!(close(&flat(&v2), &g("V2"), 1e-14), "second Catmull-Clark step vertices");

    let (vt2, ft2) = subdivide_triangles_centroid(&v3("Vt"), &t3("Ft"));
    assert_eq!(ft2, t3("Ft2"));
    assert!(close(&flat(&vt2), &g("Vt2"), 1e-15));
    assert!(close(&barycentric_basis(&v3("Vt"), &t3("Ft"), &v3("P")), &g("B"), 1e-14));

    assert!(close(&projection_matrix(5, 3), &g("P1"), 1e-14));
    assert!(close(&projection_matrix(3, 3), &g("P2"), 1e-14));
    assert!(close(&projection_matrix(4, 2), &g("Q1"), 1e-14));
    let s = subdivision_matrix_1d();
    assert!(close(&s.iter().flatten().copied().collect::<Vec<_>>(), &g("Psub"), 0.0));
    // P3d = kron(Psub, kron(Psub, Psub)): sum of entries = (sum Psub)^3
    let sum1: f64 = s.iter().flatten().sum();
    assert!((sum1.powi(3) - g("P3d_sum")[0]).abs() < 1e-12);
}
