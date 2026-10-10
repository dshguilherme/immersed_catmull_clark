//! Verification run of the Rust stack. Every check compares a computed quantity with
//! a known answer under a stated tolerance. Capabilities that are ported (with MATLAB
//! parity, see `cargo test` / tests/*_vs_matlab.rs) but have no analytical check yet are
//! reported as UNVERIFIED instead of passing.

use immersed_iga::immersed::Stabilization;
use immersed_iga::verification::{immersed_cantilever, manufactured_solution_errors, patch_test};
use std::time::Instant;

enum Outcome {
    Pass,
    Fail,
    /// Ported with MATLAB parity (cargo test), but no analytical verification exists yet.
    NotVerified,
}

fn main() {
    println!("========================================================================");
    println!("  IMMERSED IGA (RUST) VERIFICATION RUN");
    println!("========================================================================");
    let t0 = Instant::now();
    let mut results: Vec<(&str, Outcome, String)> = Vec::new();

    // 1. Patch test (exact reproduction of linear fields), 3D, p = 2 and 3
    let e2 = patch_test(&[[0.0, 2.0], [0.0, 1.0], [0.0, 1.0]], &[4, 3, 3], 2);
    let e3 = patch_test(&[[0.0, 1.0], [0.0, 2.0], [0.5, 1.0]], &[3, 4, 3], 3);
    let ok = e2 < 1e-10 && e3 < 1e-10;
    results.push(("3D elasticity patch test (p = 2, 3)", if ok { Outcome::Pass } else { Outcome::Fail }, format!("rel. errors {:.1e}, {:.1e} (tol 1e-10)", e2, e3)));

    // 2. Manufactured solution: optimal L2 rate p + 1
    let mut msg = String::new();
    let mut ok = true;
    for p in [2usize, 3] {
        let err = manufactured_solution_errors(p, &[4, 8, 16]);
        let rate = (err[1] / err[2]).log2();
        ok &= rate > p as f64 + 1.0 - 0.25;
        msg += &format!("p={}: rate {:.2} (expect {}); ", p, rate, p + 1);
    }
    results.push(("Manufactured solution, L2 convergence rate", if ok { Outcome::Pass } else { Outcome::Fail }, msg));

    // 3. Immersed cantilever vs Timoshenko beam theory (ghost penalty)
    let (d, t, it) = immersed_cantilever([44, 6, 6], Stabilization::GhostPenalty { gamma: 1e-2 });
    let rel = (d - t) / t;
    results.push((
        "Immersed cantilever tip deflection vs Timoshenko",
        if rel.abs() < 0.05 { Outcome::Pass } else { Outcome::Fail },
        format!("{:+.2}% (tol 5%), {} PCG iterations", 100.0 * rel, it),
    ));

    // 4-6. Ported with MATLAB parity, but not verified against analytical solutions
    results.push(("Unilateral contact (active set) vs Hertz", Outcome::NotVerified, "ported (contact.rs, MATLAB parity); needs a contact patch test / real Hertz problem".into()));
    results.push(("Adaptive octree refinement: convergence rate", Outcome::NotVerified, "ported (amr.rs, MATLAB parity); indicator is a stress heuristic, rate not measured".into()));
    results.push(("Multi-body CAD assembly", Outcome::NotVerified, "ported (contact.rs bonded mode); needs a tied patch test".into()));

    println!();
    let (mut n_pass, mut n_fail, mut n_nv) = (0, 0, 0);
    for (name, outcome, detail) in &results {
        let tag = match outcome {
            Outcome::Pass => {
                n_pass += 1;
                "PASS      "
            }
            Outcome::Fail => {
                n_fail += 1;
                "FAIL      "
            }
            Outcome::NotVerified => {
                n_nv += 1;
                "UNVERIFIED"
            }
        };
        println!("  [{}] {:<48} {}", tag, name, detail);
    }
    println!("------------------------------------------------------------------------");
    println!("  {} passed, {} failed, {} unverified ({:.1} s)", n_pass, n_fail, n_nv, t0.elapsed().as_secs_f64());
    if n_fail > 0 {
        std::process::exit(1);
    }
}
