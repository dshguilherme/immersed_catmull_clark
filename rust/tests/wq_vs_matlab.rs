//! Rust WQ assembly, sensitivities and 2D topopt vs MATLAB (`tests/export_rust_wq_fixture.m`).

use immersed_iga::cad_model::{parse_json, JsonValue};
use immersed_iga::iga::{elasticity_element_matrices, lame, SpaceBox};
use immersed_iga::sparse::{assemble, CsrMatrix};
use immersed_iga::topopt2d::{fast_sensitivities, topopt_iga_fast};
use immersed_iga::wq::{wq_rules_1d, wq_stiffness, Density};

fn nums(v: &JsonValue) -> Vec<f64> {
    match v {
        JsonValue::Number(x) => vec![*x],
        JsonValue::Array(a) => a.iter().flat_map(nums).collect(),
        _ => panic!("expected numbers"),
    }
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt();
    n / b.iter().map(|y| y * y).sum::<f64>().sqrt().max(1e-300)
}

fn probe(k: &CsrMatrix) -> Vec<f64> {
    let v: Vec<f64> = (1..=k.nrows).map(|i| (0.37 * i as f64).sin()).collect();
    let mut y = vec![0.0; k.nrows];
    k.matvec(&v, &mut y);
    y
}

fn gauss(sp: &SpaceBox) -> CsrMatrix {
    let (l, m) = lame(1.0, 0.3);
    let em = elasticity_element_matrices(sp, l, m);
    assemble(sp.ndof, &sp.connectivity, sp.nsh, sp.nel, |e| em.of_element(e), None)
}

#[test]
fn wq_equals_gauss_for_uniform_density() {
    for (b, n, p) in [(vec![[0.0, 1.0], [0.0, 0.5]], vec![6, 4], 3usize), (vec![[0.0, 1.2], [0.0, 0.6], [0.0, 0.3]], vec![4, 3, 2], 2usize)] {
        let sp = SpaceBox::new(&b, &n, p);
        let kw = wq_stiffness(&sp, 1.0, 0.3, Density::None, 3.0, 1e-3);
        let kg = gauss(&sp);
        let e = rel(&probe(&kw), &probe(&kg));
        assert!(e < 1e-12, "WQ vs Gauss (dim {}): {:e}", sp.dim, e);
    }
}

#[test]
fn fastformation_matches_matlab_reference() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/wq_fastformation.json");
    let d = parse_json(&std::fs::read_to_string(path).expect("run tests/export_rust_wq_fixture.m")).unwrap();
    let g = |k: &str| nums(d.get(k).unwrap_or_else(|| panic!("missing {}", k)));
    let nel = g("nel");
    let p = g("degree")[0] as usize;
    let sp = SpaceBox::new(&[[0.0, 1.0], [0.0, 0.5]], &[nel[0] as usize, nel[1] as usize], p);

    let q = wq_rules_1d(&sp.univ[0].knots, p);
    assert!(rel(&q.points, &g("wq_points")) < 1e-15);
    assert!(rel(&q.weights[3][0], &g("wq_w11_first")) < 1e-11);
    assert!(rel(&q.weights[1][4], &g("wq_w10_mid")) < 1e-11);

    let (xe, xs) = (g("xe"), g("xs"));
    let ke = wq_stiffness(&sp, 1.0, 0.3, Density::Element(&xe), 3.0, 1e-3);
    let ks = wq_stiffness(&sp, 1.0, 0.3, Density::Spline(&xs), 3.0, 1e-3);
    assert!(rel(&probe(&ke), &g("Kfe_probe")) < 1e-12, "element-density WQ");
    assert!(rel(&probe(&ks), &g("Kfs_probe")) < 1e-12, "spline-density WQ");
    assert!((ke.frobenius() - g("Kfe_fro")[0]).abs() / g("Kfe_fro")[0] < 1e-12);
    let sp3 = SpaceBox::new(&[[0.0, 1.2], [0.0, 0.6], [0.0, 0.3]], &[4, 3, 2], 2);
    let k3 = wq_stiffness(&sp3, 1.0, 0.3, Density::Element(&g("xe3")), 3.0, 1e-3);
    assert!(rel(&probe(&k3), &g("Kf3_probe")) < 1e-12, "3D element-density WQ");

    let u = g("U");
    assert!(rel(&fast_sensitivities(&sp, &u, &xe, false, 3.0, 1e-3, 1.0, 0.3), &g("dCe")) < 1e-12, "element sensitivities");
    assert!(rel(&fast_sensitivities(&sp, &u, &xs, true, 3.0, 1e-3, 1.0, 0.3), &g("dCs")) < 1e-12, "spline sensitivities");

    for (spline, kc, kx) in [(false, "topo_e_c", "topo_e_x"), (true, "topo_s_c", "topo_s_x")] {
        let r = topopt_iga_fast(16, 8, 0.5, 3.0, 2.0, 5, spline, 3);
        let cref = g(kc);
        for (k, (a, b)) in r.compliance.iter().zip(&cref).enumerate() {
            assert!((a - b).abs() / b < 1e-8, "2D topopt (spline {}) iteration {}: {} vs {}", spline, k + 1, a, b);
        }
        let dx = r.density.iter().zip(&g(kx)).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(dx < 1e-6, "2D topopt (spline {}) densities differ by {:e}", spline, dx);
    }
}
