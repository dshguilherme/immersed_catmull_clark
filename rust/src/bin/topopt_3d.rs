//! 3D isogeometric SIMP topology optimization of a cantilever (port of the MATLAB
//! `topopt_iga_3d` CPU path; see `immersed_iga::topopt_iga`).
//!
//! Usage: topopt_3d [nelx nely nelz] [max_iter]   (default 24 12 6, 40 iterations)
//! Writes `benchmarks/topopt_result.json` (keys used by scripts/plot_publication_figures.py).

use immersed_iga::topopt_iga::{CantileverTopOpt, CantileverTopOptConfig};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse()).collect::<Result<_, _>>()?;
    let nel = if a.len() >= 3 { [a[0], a[1], a[2]] } else { [24, 12, 6] };
    let max_iter = a.get(3).copied().unwrap_or(40);
    let cfg = CantileverTopOptConfig { nel, max_iter, ..CantileverTopOptConfig::default() };
    println!("3D isogeometric topology optimization: {}x{}x{} elements, p = {}, volfrac {}, penal {}, rmin {}",
        nel[0], nel[1], nel[2], cfg.degree, cfg.volfrac, cfg.penal, cfg.rmin);
    let t0 = Instant::now();
    let opt = CantileverTopOpt::new(cfg.clone());
    println!("setup: {} DOFs, {:.0} ms", opt.sp.ndof, t0.elapsed().as_secs_f64() * 1e3);
    let mut t_iter = Instant::now();
    let hist = opt.run(|it, c, vol, change| {
        println!("  iter {:>3} | compliance {:>12.4} | volfrac {:.4} | change {:.4e} | {:.0} ms", it, c, vol, change, t_iter.elapsed().as_secs_f64() * 1e3);
        t_iter = Instant::now();
    });
    let total = t0.elapsed().as_secs_f64();
    println!("finished {} iterations in {:.2} s (PCG iterations per solve: {:?})", hist.compliance.len(), total, hist.pcg_iterations);
    let json = format!(
        "{{\"nelx\": {}, \"nely\": {}, \"nelz\": {}, \"solver\": \"immersed_iga::topopt_iga (B-spline p = {}, SIMP, OC)\", \"compliance\": {:?}, \"densities\": {:?}}}",
        nel[0], nel[1], nel[2], cfg.degree, hist.compliance, hist.density
    );
    let out = if std::path::Path::new("benchmarks").exists() { "benchmarks/topopt_result.json" } else { "topopt_result.json" };
    std::fs::write(out, json)?;
    println!("saved {}", out);
    Ok(())
}
