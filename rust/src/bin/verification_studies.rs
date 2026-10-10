//! Verification studies of the immersed method. Prints tables and writes
//! `benchmarks/verification_results.json`. These are measurements: they report what
//! the current method achieves, with no tolerances attached.
//!
//! S1  Cut-domain patch test: tilted box, u linear, penalty Dirichlet u = g on the whole
//!     immersed surface. A consistent method reproduces u to round-off.
//! S2  Manufactured solution on an immersed sphere (R = 0.4), body force + penalty
//!     Dirichlet; relative L2 error and observed rate under grid refinement
//!     (optimal for p = 2 would be 3).
//! S3  Small-cut conditioning: unit box with the last cell layer cut at fraction eps;
//!     Lanczos estimate of cond(D^-1/2 K D^-1/2) and PCG iterations, per stabilization.
//!
//! Usage: verification_studies [--quick]

use immersed_iga::cut_cell::TriangleMesh3D;
use immersed_iga::immersed::{build_immersed_problem, ImmersedOptions, Stabilization, CUT};
use immersed_iga::immersed_bc::dirichlet_penalty_fn;
use immersed_iga::verification_studies::{free_operator, icosphere, immersed_dirichlet_problem, lanczos_extremes, mms_fields};
use std::time::Instant;

fn tilted_box() -> TriangleMesh3D {
    let base = TriangleMesh3D::new_box([0.0; 3], [1.6, 0.8, 0.6]);
    let (a, b) = (17f64.to_radians(), 9f64.to_radians());
    let v = base
        .vertices
        .iter()
        .map(|p| {
            let rz = [a.cos() * p[0] - a.sin() * p[1], a.sin() * p[0] + a.cos() * p[1], p[2]];
            [rz[0] + 0.13, b.cos() * rz[1] - b.sin() * rz[2] + 0.07, b.sin() * rz[1] + b.cos() * rz[2] + 0.05]
        })
        .collect();
    TriangleMesh3D::new(v, base.triangles.clone())
}

fn stab_name(s: Stabilization) -> &'static str {
    match s {
        Stabilization::None => "none",
        Stabilization::Legacy { .. } => "legacy",
        Stabilization::GhostPenalty { .. } => "ghost penalty",
    }
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    let t0 = Instant::now();
    let stabs = [Stabilization::None, Stabilization::Legacy { gamma: 0.05 }, Stabilization::GhostPenalty { gamma: 1e-2 }];
    let mut json = String::from("{");

    // S1
    println!("S1  Cut-domain patch test (tilted box, linear u, penalty Dirichlet beta = 1e3 E/h)");
    println!("    {:<14} {:>6} {:>8} {:>12} {:>6}", "stabilization", "cells", "DOFs", "rel L2 err", "PCG");
    let lin = |x: &[f64; 3]| [0.010 * x[0] + 0.004 * x[1] - 0.002 * x[2] + 0.1, -0.003 * x[0] + 0.012 * x[1] + 0.001 * x[2] - 0.05, 0.002 * x[0] - 0.001 * x[1] + 0.008 * x[2] + 0.02];
    let mesh = tilted_box();
    json += "\"S1\": [";
    let cells_s1: &[usize] = if quick { &[8, 16] } else { &[8, 16, 24] };
    for &s in &stabs {
        for &c in cells_s1 {
            let (r, _) = immersed_dirichlet_problem(&mesh, c, s, 1e3, lin, None::<fn(&[f64; 3]) -> [f64; 3]>, lin);
            println!("    {:<14} {:>6} {:>8} {:>12.3e} {:>6}", stab_name(s), r.cells, r.ndof, r.rel_l2_error, r.pcg_iterations);
            json += &format!("{{\"stab\": \"{}\", \"cells\": {}, \"ndof\": {}, \"rel_l2\": {:e}, \"pcg\": {}}},", stab_name(s), r.cells, r.ndof, r.rel_l2_error, r.pcg_iterations);
        }
    }
    json.pop();
    json += "], ";

    // S2
    println!("\nS2  Manufactured solution on an immersed sphere (p = 2, optimal L2 rate 3)");
    println!("    {:<14} {:>6} {:>8} {:>12} {:>6} {:>6}", "stabilization", "cells", "DOFs", "rel L2 err", "rate", "PCG");
    let sphere = icosphere([0.5, 0.5, 0.5], 0.4, 3);
    let (u_ex, f_body) = mms_fields();
    json += "\"S2\": [";
    let cells_s2: &[usize] = if quick { &[6, 12] } else { &[6, 12, 24] };
    for &s in &[Stabilization::None, Stabilization::GhostPenalty { gamma: 1e-2 }] {
        let mut prev: Option<f64> = None;
        for &c in cells_s2 {
            let (r, _) = immersed_dirichlet_problem(&sphere, c, s, 1e3, u_ex, Some(f_body), u_ex);
            let rate = prev.map(|e| (e / r.rel_l2_error).log2());
            println!("    {:<14} {:>6} {:>8} {:>12.3e} {:>6} {:>6}", stab_name(s), r.cells, r.ndof, r.rel_l2_error, rate.map_or("-".into(), |v| format!("{:.2}", v)), r.pcg_iterations);
            json += &format!("{{\"stab\": \"{}\", \"cells\": {}, \"ndof\": {}, \"rel_l2\": {:e}, \"pcg\": {}}},", stab_name(s), r.cells, r.ndof, r.rel_l2_error, r.pcg_iterations);
            prev = Some(r.rel_l2_error);
        }
    }
    json.pop();
    json += "], ";

    // S3
    println!("\nS3  Small cuts: unit box, 8 cells per direction, outer cell layer cut at fraction eps");
    println!("    {:<14} {:>8} {:>10} {:>12} {:>6}", "stabilization", "eps", "min w_cut", "cond est.", "PCG");
    let n = 8usize;
    json += "\"S3\": [";
    let eps_list: &[f64] = if quick { &[0.3, 0.01] } else { &[0.3, 0.1, 0.03, 0.01, 1e-3] };
    for &s in &stabs {
        for &eps in eps_list {
            let p = (1.0 - eps) / (n as f64 - 2.0 * (1.0 - eps));
            let boxm = TriangleMesh3D::new_box([0.0; 3], [1.0; 3]);
            let opts = ImmersedOptions { grid_res: [n; 3], padding: p, clamped_face: None, stabilization: s, ..ImmersedOptions::default() };
            let mut pb = build_immersed_problem(&boxm, &opts);
            pb.force.iter_mut().for_each(|v| *v = 0.0);
            let h = pb.sp.element_size()[0];
            let all: Vec<usize> = (0..boxm.triangles.len()).collect();
            let st = dirichlet_penalty_fn(&pb.sp, &boxm, &all, 1e3 / h, lin);
            pb.add_surface_terms(&st);
            let wmin = pb.status.iter().zip(&pb.weights).filter(|(&st, _)| st == CUT).map(|(_, &w)| w).fold(1.0, f64::min);
            let diag: Vec<f64> = pb.stiffness.diagonal().iter().map(|&d| d.max(1e-300)).collect();
            let (lmin, lmax) = lanczos_extremes(pb.sp.ndof, &diag, free_operator(&pb.stiffness, &pb.free), 300);
            let sol = immersed_iga::immersed::solve_immersed_problem(&pb, 1e-10, 50 * pb.sp.ndof);
            let cond = lmax / lmin.max(1e-300);
            println!("    {:<14} {:>8.0e} {:>10.3} {:>12.3e} {:>6}", stab_name(s), eps, wmin, cond, sol.iterations);
            json += &format!("{{\"stab\": \"{}\", \"eps\": {:e}, \"min_cut_weight\": {:e}, \"cond\": {:e}, \"pcg\": {}}},", stab_name(s), eps, wmin, cond, sol.iterations);
        }
    }
    json.pop();
    json += "]}";
    let out = if std::path::Path::new("benchmarks").exists() { "benchmarks/verification_results.json" } else { "verification_results.json" };
    std::fs::write(out, json).expect("write results");
    println!("\nsaved {} ({:.0} s)", out, t0.elapsed().as_secs_f64());
}
