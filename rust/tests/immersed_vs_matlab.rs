//! Rust immersed solver vs the MATLAB reference (`tests/export_rust_immersed_fixture.m`).

use immersed_iga::cad_model::{parse_json, JsonValue};
use immersed_iga::cut_cell::TriangleMesh3D;
use immersed_iga::immersed::{build_immersed_problem, solve_immersed_problem, ImmersedOptions, Stabilization};

fn nums(v: &JsonValue) -> Vec<f64> {
    match v {
        JsonValue::Number(x) => vec![*x],
        JsonValue::Array(a) => a.iter().flat_map(nums).collect(),
        _ => panic!("expected numbers"),
    }
}

fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    num / b.iter().map(|y| y * y).sum::<f64>().sqrt().max(1e-300)
}

#[test]
fn immersed_solver_matches_matlab_reference() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/immersed_tilted_box.json");
    let text = std::fs::read_to_string(path).expect("fixture missing: run tests/export_rust_immersed_fixture.m");
    let d = parse_json(&text).unwrap();
    let get = |k: &str| nums(d.get(k).unwrap_or_else(|| panic!("missing {}", k)));

    let nodes = get("nodes"); // [nv x 3] row-major
    let elems = get("elements");
    let vertices: Vec<[f64; 3]> = nodes.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let triangles: Vec<[usize; 3]> = elems.chunks(3).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect();
    let mesh = TriangleMesh3D::new(vertices, triangles);

    let res = get("grid_res");
    let opts = ImmersedOptions {
        grid_res: [res[0] as usize, res[1] as usize, res[2] as usize],
        degree: get("degree")[0] as usize,
        young: get("young")[0],
        poisson: get("poisson")[0],
        stabilization: Stabilization::Legacy { gamma: get("gamma_gp")[0] },
        emin: get("emin")[0],
        ..ImmersedOptions::default()
    };
    let pb = build_immersed_problem(&mesh, &opts);

    let gb = get("grid_bounds"); // [3 x 2] row-major
    for dd in 0..3 {
        assert!((pb.grid_bounds[dd][0] - gb[2 * dd]).abs() < 1e-14 && (pb.grid_bounds[dd][1] - gb[2 * dd + 1]).abs() < 1e-14);
    }
    let status: Vec<i8> = get("status").iter().map(|&v| v as i8).collect();
    assert_eq!(pb.status, status, "cell classification differs from MATLAB");
    let w = get("weights");
    assert!(pb.weights.iter().zip(&w).all(|(a, b)| (a - b).abs() < 1e-7), "cut-cell weights differ");

    let kgp_fro = get("kgp_frobenius")[0];
    assert!((pb.stabilization_frobenius - kgp_fro).abs() <= 1e-12 * kgp_fro.max(1.0), "stabilization norm {} vs {}", pb.stabilization_frobenius, kgp_fro);

    let n = pb.sp.ndof;
    let v: Vec<f64> = (1..=n).map(|i| (0.37 * i as f64).sin()).collect();
    let mut y = vec![0.0; n];
    pb.stiffness.matvec(&v, &mut y);
    let ky = rel_err(&y, &get("k_probe"));
    assert!(ky < 1e-12, "K*v differs: {:e}", ky);
    assert!(rel_err(&pb.force, &get("load")) < 1e-15);
    let free: Vec<bool> = get("free_mask").iter().map(|&v| v != 0.0).collect();
    assert_eq!(pb.free, free);

    let sol = solve_immersed_problem(&pb, 1e-12, 20 * n);
    let eu = rel_err(&sol.u, &get("u_direct"));
    let c_ref = get("compliance")[0];
    println!("PCG iterations {}, residual {:e}, |u - u_matlab|/|u| = {:e}, compliance {} vs {}", sol.iterations, sol.residual, eu, sol.compliance, c_ref);
    assert!(eu < 1e-6, "displacement differs from MATLAB direct solve: {:e}", eu);
    assert!((sol.compliance - c_ref).abs() / c_ref < 1e-9);
}
