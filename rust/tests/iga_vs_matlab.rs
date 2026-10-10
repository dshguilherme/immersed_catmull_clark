//! Cross-validation of the Rust IGA kernel against the MATLAB reference implementation.
//! Fixture produced by `tests/export_rust_fixtures.m` (MATLAB).

use immersed_iga::cad_model::{parse_json, JsonValue};
use immersed_iga::iga::{elasticity_element_matrices, load_vector, SpaceBox};
use immersed_iga::sparse::assemble;

fn nums(v: &JsonValue) -> Vec<f64> {
    match v {
        JsonValue::Number(x) => vec![*x],
        JsonValue::Array(a) => a.iter().flat_map(nums).collect(),
        _ => panic!("expected numbers"),
    }
}

fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt().max(1e-300);
    num / den
}

#[test]
fn kernel_matches_matlab_reference() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/iga_kernel.json");
    let text = std::fs::read_to_string(path).expect("fixture missing: run tests/export_rust_fixtures.m in MATLAB");
    let data = parse_json(&text).unwrap();
    let lambda = data.get("lambda").unwrap().as_f64().unwrap();
    let mu = data.get("mu").unwrap().as_f64().unwrap();
    let cases = data.get("cases").unwrap().as_array().unwrap();
    assert_eq!(cases.len(), 6);
    for case in cases {
        let b = nums(case.get("bounds").unwrap());
        let dim = b.len() / 2;
        let bounds: Vec<[f64; 2]> = (0..dim).map(|d| [b[2 * d], b[2 * d + 1]]).collect();
        let nsub: Vec<usize> = nums(case.get("nsub").unwrap()).iter().map(|&v| v as usize).collect();
        let degree = case.get("degree").unwrap().as_f64().unwrap() as usize;

        let sp = SpaceBox::new(&bounds, &nsub, degree);
        assert_eq!(sp.ndof, case.get("ndof").unwrap().as_f64().unwrap() as usize);

        // connectivity (0-based, element-major) must be identical
        let conn: Vec<usize> = nums(case.get("connectivity").unwrap()).iter().map(|&v| v as usize).collect();
        assert_eq!(conn, sp.connectivity, "connectivity mismatch dim={} p={}", dim, degree);

        let em = elasticity_element_matrices(&sp, lambda, mu);
        assert_eq!(em.types.len(), case.get("ntypes").unwrap().as_f64().unwrap() as usize);
        let k = assemble(sp.ndof, &sp.connectivity, sp.nsh, sp.nel, |e| em.of_element(e), None);

        let fro_ref = case.get("frobenius").unwrap().as_f64().unwrap();
        assert!((k.frobenius() - fro_ref).abs() / fro_ref < 1e-13, "Frobenius norm dim={} p={}", dim, degree);

        let probes = nums(case.get("probes").unwrap()); // row-major [ndof x 3]
        for j in 1..=3usize {
            let v: Vec<f64> = (1..=sp.ndof).map(|i| (0.37 * i as f64 * j as f64).sin()).collect();
            let mut y = vec![0.0; sp.ndof];
            k.matvec(&v, &mut y);
            let yref: Vec<f64> = (0..sp.ndof).map(|i| probes[i * 3 + (j - 1)]).collect();
            let err = rel_err(&y, &yref);
            assert!(err < 1e-13, "K*v probe {} dim={} p={} err={:e}", j, dim, degree, err);
        }

        let f = load_vector(&sp, |x| {
            if dim == 2 {
                [x[0] * x[1], x[0].sin() + x[1] * x[1], 0.0]
            } else {
                [x[0] * x[1], x[0].sin() + x[2], x[1] * x[1]]
            }
        });
        let fref = nums(case.get("load").unwrap());
        let err = rel_err(&f, &fref);
        assert!(err < 1e-13, "load vector dim={} p={} err={:e}", dim, degree, err);
    }
}
