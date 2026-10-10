//! Rust immersed topology optimization vs MATLAB `topopt_immersed_iga_3d`
//! (`tests/export_rust_imm_topopt_fixture.m`). MATLAB runs single-precision PCG on the
//! GPU, so agreement is checked to single-precision level.

use immersed_iga::cad_model::{parse_json, JsonValue};
use immersed_iga::cut_cell::TriangleMesh3D;
use immersed_iga::immersed::Stabilization;
use immersed_iga::topopt_immersed::{ImmersedTopOpt, ImmersedTopOptConfig};

fn nums(v: &JsonValue) -> Vec<f64> {
    match v {
        JsonValue::Number(x) => vec![*x],
        JsonValue::Array(a) => a.iter().flat_map(nums).collect(),
        _ => panic!("expected numbers"),
    }
}

#[test]
fn immersed_topopt_matches_matlab_reference() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/imm_topopt_tilted_box.json");
    let d = parse_json(&std::fs::read_to_string(path).expect("run tests/export_rust_imm_topopt_fixture.m")).unwrap();
    let g = |k: &str| nums(d.get(k).unwrap_or_else(|| panic!("missing {}", k)));
    let brep = TriangleMesh3D::new(
        g("nodes").chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
        g("elements").chunks(3).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect(),
    );
    let res = g("grid_res");
    let cfg = ImmersedTopOptConfig {
        grid_res: [res[0] as usize, res[1] as usize, res[2] as usize],
        volfrac: g("volfrac")[0],
        penal: g("penal")[0],
        rmin: g("rmin")[0],
        max_iter: g("max_iter")[0] as usize,
        stabilization: Stabilization::Legacy { gamma: g("gamma_gp")[0] },
        pcg_tol: 1e-10,
        ..ImmersedTopOptConfig::default()
    };
    let opt = ImmersedTopOpt::new(&brep, cfg);
    let hist = opt.run(|_, _, _, _, _| {});
    let c_ref = g("compliance");
    println!("rust compliance {:?}\nmatlab compliance {:?}", hist.compliance, c_ref);
    for (k, (a, b)) in hist.compliance.iter().zip(&c_ref).enumerate() {
        assert!((a - b).abs() / b < 1e-4, "iteration {}: {} vs MATLAB {}", k + 1, a, b);
    }
    let x_ref = g("xPhys");
    let dx = hist.density.iter().zip(&x_ref).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("max density difference {:e}", dx);
    assert!(dx < 1e-3, "final densities differ by {:e}", dx);
}
