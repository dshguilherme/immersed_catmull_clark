//! Rust 3D cantilever topology optimization vs the MATLAB reference
//! (`tests/export_rust_topopt_fixture.m`, topopt_iga_3d CPU path).

use immersed_iga::cad_model::{parse_json, JsonValue};
use immersed_iga::topopt_iga::{CantileverTopOpt, CantileverTopOptConfig};

fn nums(v: &JsonValue) -> Vec<f64> {
    match v {
        JsonValue::Number(x) => vec![*x],
        JsonValue::Array(a) => a.iter().flat_map(nums).collect(),
        _ => panic!("expected numbers"),
    }
}

#[test]
fn topopt_matches_matlab_reference() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/topopt3d_8x4x2.json");
    let d = parse_json(&std::fs::read_to_string(path).expect("run tests/export_rust_topopt_fixture.m")).unwrap();
    let get = |k: &str| nums(d.get(k).unwrap());
    let nel = get("nel");
    let cfg = CantileverTopOptConfig {
        nel: [nel[0] as usize, nel[1] as usize, nel[2] as usize],
        volfrac: get("volfrac")[0],
        penal: get("penal")[0],
        rmin: get("rmin")[0],
        max_iter: get("max_iter")[0] as usize,
        pcg_tol: 1e-12,
        ..CantileverTopOptConfig::default()
    };
    let opt = CantileverTopOpt::new(cfg);
    let hist = opt.run(|_, _, _, _| {});
    let c_ref = get("compliance");
    assert_eq!(hist.compliance.len(), c_ref.len());
    for (k, (a, b)) in hist.compliance.iter().zip(&c_ref).enumerate() {
        assert!((a - b).abs() / b < 1e-7, "iteration {}: compliance {} vs MATLAB {}", k + 1, a, b);
    }
    let x_ref = get("xPhys");
    let dx = hist.density.iter().zip(&x_ref).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(dx < 1e-6, "final densities differ by {:e}", dx);
}
