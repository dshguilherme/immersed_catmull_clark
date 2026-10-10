//! Immersed topology optimization inside a CAD body (port of the MATLAB
//! `topopt_immersed_iga_3d`, with the consistent ghost penalty by default).
//!
//! Usage: topopt_cad [model.obj] [--cells N] [--iters N] [--volfrac F] [--stab gp|legacy|none]
//! Without a model, a tilted box is used. The bottom face of the background grid is
//! clamped and a central load is applied on the top face (as in the MATLAB reference).
//! Writes `benchmarks/topopt_cad_result.json` (grid, densities, history).

use immersed_iga::cad_model::{CadBody, MaterialProperties};
use immersed_iga::cut_cell::TriangleMesh3D;
use immersed_iga::immersed::{padded_bounds, Stabilization};
use immersed_iga::topopt_immersed::{ImmersedTopOpt, ImmersedTopOptConfig};
use std::time::Instant;

fn tilted_box() -> TriangleMesh3D {
    let size = [1.6, 0.8, 0.6];
    let (a, b) = (17f64.to_radians(), 9f64.to_radians());
    let base = TriangleMesh3D::new_box([0.0; 3], size);
    let verts = base
        .vertices
        .iter()
        .map(|v| {
            let rz = [a.cos() * v[0] - a.sin() * v[1], a.sin() * v[0] + a.cos() * v[1], v[2]];
            let rx = [rz[0], b.cos() * rz[1] - b.sin() * rz[2], b.sin() * rz[1] + b.cos() * rz[2]];
            [rx[0] + 0.13, rx[1] + 0.07, rx[2] + 0.05]
        })
        .collect();
    TriangleMesh3D::new(verts, base.triangles.clone())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model: Option<String> = None;
    let (mut cells, mut iters, mut volfrac, mut stab) = (28usize, 30usize, 0.35f64, "gp".to_string());
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--cells" => cells = it.next().ok_or("--cells value")?.parse()?,
            "--iters" => iters = it.next().ok_or("--iters value")?.parse()?,
            "--volfrac" => volfrac = it.next().ok_or("--volfrac value")?.parse()?,
            "--stab" => stab = it.next().ok_or("--stab value")?,
            s if s.starts_with("--") => return Err(format!("unknown option {}", s).into()),
            s => model = Some(s.to_string()),
        }
    }
    let mesh = match &model {
        Some(path) => CadBody::parse_obj(&std::fs::read_to_string(path)?, MaterialProperties::default())?.mesh,
        None => tilted_box(),
    };
    let gb = padded_bounds(&mesh, 0.05);
    let len: Vec<f64> = (0..3).map(|d| gb[d][1] - gb[d][0]).collect();
    let lmax = len.iter().cloned().fold(0.0, f64::max);
    let grid_res = [0, 1, 2].map(|d| ((cells as f64 * len[d] / lmax).round() as usize).max(4));
    let stabilization = match stab.as_str() {
        "legacy" => Stabilization::Legacy { gamma: 0.05 },
        "none" => Stabilization::None,
        _ => Stabilization::GhostPenalty { gamma: 1e-2 },
    };
    let cfg = ImmersedTopOptConfig { grid_res, max_iter: iters, volfrac, stabilization, ..ImmersedTopOptConfig::default() };
    println!("Immersed topology optimization: grid {:?}, volfrac {}, {:?}", grid_res, volfrac, stabilization);
    let t0 = Instant::now();
    let opt = ImmersedTopOpt::new(&mesh, cfg);
    println!("  {} design cells (CAD volume {:.1} cells), {} DOFs, setup {:.1} s", opt.active.iter().filter(|&&a| a).count(), opt.cad_volume, opt.sp.ndof, t0.elapsed().as_secs_f64());
    let hist = opt.run(|k, c, v, ch, its| println!("  iter {:>3} | compliance {:.6e} | volume {:.3} | change {:.4} | PCG {}", k, c, v, ch, its));
    println!("finished in {:.1} s", t0.elapsed().as_secs_f64());
    let json = format!(
        "{{\"grid_res\": {:?}, \"grid_bounds\": {:?}, \"stabilization\": \"{:?}\", \"compliance\": {:?}, \"volume\": {:?}, \"densities\": {:?}}}",
        grid_res, gb, stabilization, hist.compliance, hist.volume, hist.density
    );
    let out = if std::path::Path::new("benchmarks").exists() { "benchmarks/topopt_cad_result.json" } else { "topopt_cad_result.json" };
    std::fs::write(out, json)?;
    println!("saved {}", out);
    Ok(())
}
