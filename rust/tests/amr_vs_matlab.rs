//! Rust AMR loop vs the MATLAB reference (`tests/export_rust_amr_fixture.m`).

use immersed_iga::amr::{amr_loop, AmrOptions};
use immersed_iga::cad_model::{parse_json, JsonValue};
use immersed_iga::cut_cell::TriangleMesh3D;
use immersed_iga::octree::{BoundingBox3D, OctreeMesh3D};

fn nums(v: &JsonValue) -> Vec<f64> {
    match v {
        JsonValue::Number(x) => vec![*x],
        JsonValue::Array(a) => a.iter().flat_map(nums).collect(),
        _ => panic!("expected numbers"),
    }
}

#[test]
fn amr_loop_matches_matlab_reference() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/amr_lbracket.json");
    let d = parse_json(&std::fs::read_to_string(path).expect("run tests/export_rust_amr_fixture.m")).unwrap();
    let g = |k: &str| nums(d.get(k).unwrap_or_else(|| panic!("missing {}", k)));
    let brep = TriangleMesh3D::new(
        g("nodes").chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
        g("elements").chunks(3).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect(),
    );
    let gb = g("grid_bounds");
    let res = g("grid_res");
    let oct = OctreeMesh3D::build(BoundingBox3D::new([gb[0], gb[2], gb[4]], [gb[1], gb[3], gb[5]]), [res[0] as usize, res[1] as usize, res[2] as usize], 0, |_| false);
    let (ymin, xmax) = (gb[2], gb[1]);
    let bc = |fem: &immersed_iga::octree_fem::OctreeFem| {
        let mc: Vec<[f64; 3]> = fem.master_ids.iter().map(|&i| fem.nodes[i]).collect();
        let mut fixed = Vec::new();
        for (m, p) in mc.iter().enumerate() {
            if p[1] <= ymin + 1e-6 {
                fixed.extend([3 * m, 3 * m + 1, 3 * m + 2]);
            }
        }
        let load: Vec<usize> = (0..mc.len()).filter(|&m| mc[m][0] >= xmax - 1e-6).collect();
        let mut f = vec![0.0; 3 * mc.len()];
        for &m in &load {
            f[3 * m + 1] = -1.0 / load.len() as f64;
        }
        (fixed, f)
    };
    let (hist, fem, _, vm) = amr_loop(oct, Some(&brep), g("E")[0], g("nu")[0], g("cycles")[0] as usize, &AmrOptions::default(), bc);
    let ne: Vec<usize> = hist.iter().map(|c| c.n_elements).collect();
    let dofs: Vec<usize> = hist.iter().map(|c| c.free_dofs).collect();
    println!("elements per cycle {:?}, free DOFs {:?}", ne, dofs);
    assert_eq!(ne, g("n_elements").iter().map(|&v| v as usize).collect::<Vec<_>>());
    assert_eq!(dofs, g("dofs").iter().map(|&v| v as usize).collect::<Vec<_>>());
    for (k, (c, cr)) in hist.iter().zip(g("compliance")).enumerate() {
        assert!((c.compliance - cr).abs() / cr < 1e-8, "cycle {} compliance {} vs {}", k + 1, c.compliance, cr);
    }
    for (k, (c, sr)) in hist.iter().zip(g("sigma_max")).enumerate() {
        assert!((c.sigma_max - sr).abs() / sr < 1e-7, "cycle {} sigma_max {} vs {}", k + 1, c.sigma_max, sr);
    }
    let lv: Vec<usize> = g("final_levels").iter().map(|&v| v as usize).collect();
    assert_eq!(fem.levels, lv);
    let vref = g("final_vm");
    let dv = vm.iter().zip(&vref).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / vref.iter().cloned().fold(0.0, f64::max);
    assert!(dv < 1e-7, "final von Mises field differs by {:e}", dv);
}
