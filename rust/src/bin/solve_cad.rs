//! Immersed elasticity solve of a labelled CAD body.
//!
//! Reads a CAD assembly with labelled faces and boundary conditions (JSON, see
//! `benchmarks/cad_model_input.json`), embeds the first body in a background
//! B-spline grid and solves linear elasticity with the immersed solver
//! (`immersed_iga::immersed`): volume-fraction cut cells, ghost-penalty
//! stabilization, penalty Dirichlet and consistent tractions on the labelled
//! faces. Writes the displacement at the CAD vertices to `benchmarks/cad_solution.json`.
//!
//! Usage: solve_cad [input.json] [--cells N] [--degree P] [--penalty F]
//!                  [--stab gp|none|legacy] [--gamma G] [--out path]
//!   --cells N   cells along the longest bounding-box edge (default 24; at least 6 per direction)
//!
//! The immersed method is approximate near the boundary (volume-fraction cut cells), so
//! check results under grid refinement (`--cells`) before relying on them.

use immersed_iga::cad_model::{BoundaryConditionType, CadAssembly, CadBody, LabeledBoundaryCondition, MaterialProperties};
use immersed_iga::cut_cell::TriangleMesh3D;
use immersed_iga::iga::eval_vector_at;
use immersed_iga::immersed::{build_immersed_problem, padded_bounds, solve_immersed_problem, ImmersedOptions, Stabilization};
use immersed_iga::immersed_bc::surface_boundary_terms;
use std::time::Instant;

struct Args {
    input: Option<String>,
    cells: usize,
    degree: usize,
    penalty: f64,
    stab: String,
    gamma: f64,
    out: String,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args { input: None, cells: 24, degree: 2, penalty: 1e3, stab: "gp".into(), gamma: 1e-2, out: "benchmarks/cad_solution.json".into() };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or(format!("missing value for {}", name));
        match arg.as_str() {
            "--cells" => a.cells = val("--cells")?.parse().map_err(|e| format!("--cells: {}", e))?,
            "--degree" => a.degree = val("--degree")?.parse().map_err(|e| format!("--degree: {}", e))?,
            "--penalty" => a.penalty = val("--penalty")?.parse().map_err(|e| format!("--penalty: {}", e))?,
            "--stab" => a.stab = val("--stab")?,
            "--gamma" => a.gamma = val("--gamma")?.parse().map_err(|e| format!("--gamma: {}", e))?,
            "--out" => a.out = val("--out")?,
            s if s.starts_with("--") => return Err(format!("unknown option {}", s)),
            s => a.input = Some(s.to_string()),
        }
    }
    Ok(a)
}

/// Default problem: cantilever bar clamped at x = 0 with a downward tip traction.
fn default_assembly() -> CadAssembly {
    let mesh = TriangleMesh3D::new_box([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    let steel = MaterialProperties { name: "StructuralSteel".to_string(), youngs_modulus: 2.1e5, poissons_ratio: 0.30, density: 7850.0 };
    let mut bracket = CadBody::new("CantileverBracket", mesh, steel);
    bracket.label_face_by_predicate("ClampFace", |c, _n| c[0] < 1e-4);
    bracket.label_face_by_predicate("LoadFace", |c, _n| c[0] > 2.0 - 1e-4);
    let mut assy = CadAssembly::new();
    assy.add_body(bracket);
    assy.add_boundary_condition(LabeledBoundaryCondition {
        target_face_name: "ClampFace".to_string(),
        bc_type: BoundaryConditionType::Dirichlet { components: [true, true, true], values: [0.0, 0.0, 0.0] },
    });
    assy.add_boundary_condition(LabeledBoundaryCondition {
        target_face_name: "LoadFace".to_string(),
        bc_type: BoundaryConditionType::NeumannTraction { traction: [0.0, -100.0, 0.0] },
    });
    assy
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let t0 = Instant::now();
    let input = args.input.clone().or_else(|| {
        ["benchmarks/cad_model_input.json", "../benchmarks/cad_model_input.json"].iter().find(|p| std::path::Path::new(p).exists()).map(|s| s.to_string())
    });
    let assembly = match &input {
        Some(path) => {
            println!("[1/4] Reading labelled CAD model '{}'", path);
            CadAssembly::from_json(&std::fs::read_to_string(path)?)?
        }
        None => {
            println!("[1/4] No input file: using the default cantilever bracket");
            default_assembly()
        }
    };
    if assembly.bodies.len() > 1 {
        println!("      NOTE: {} bodies in the assembly; only the first is solved (multi-body contact is not ported yet)", assembly.bodies.len());
    }
    let body = &assembly.bodies[0];
    println!("      body '{}': {} vertices, {} triangles, E = {}, nu = {}", body.name, body.mesh.vertices.len(), body.mesh.triangles.len(), body.material.youngs_modulus, body.material.poissons_ratio);

    // Background grid: `cells` along the longest edge, proportional elsewhere.
    let padding = 0.05;
    let gb = padded_bounds(&body.mesh, padding);
    let len: Vec<f64> = (0..3).map(|d| gb[d][1] - gb[d][0]).collect();
    let lmax = len.iter().cloned().fold(0.0, f64::max);
    // at least 6 cells in every direction so thin parts still get interior cells
    let grid_res = [0, 1, 2].map(|d| ((args.cells as f64 * len[d] / lmax).round() as usize).max(6));
    let stabilization = match args.stab.as_str() {
        "none" => Stabilization::None,
        "legacy" => Stabilization::Legacy { gamma: args.gamma },
        "gp" => Stabilization::GhostPenalty { gamma: args.gamma },
        s => return Err(format!("unknown --stab '{}'", s).into()),
    };
    let opts = ImmersedOptions {
        grid_res,
        padding,
        degree: args.degree,
        young: body.material.youngs_modulus,
        poisson: body.material.poissons_ratio,
        stabilization,
        clamped_face: None,
        ..ImmersedOptions::default()
    };
    println!("[2/4] Background grid {:?}, degree {}, stabilization {:?}", grid_res, args.degree, stabilization);
    let mut pb = build_immersed_problem(&body.mesh, &opts);
    pb.force.iter_mut().for_each(|f| *f = 0.0); // loads come only from the labelled faces
    let n_cut = pb.status.iter().filter(|&&s| s == immersed_iga::immersed::CUT).count();
    let n_in = pb.status.iter().filter(|&&s| s == immersed_iga::immersed::INSIDE).count();
    println!("      cells: {} inside, {} cut, {} outside; {} DOFs", n_in, n_cut, pb.status.len() - n_in - n_cut, pb.sp.ndof);

    println!("[3/4] Integrating boundary conditions on the labelled faces (penalty factor {:.1e})", args.penalty);
    let st = surface_boundary_terms(&pb.sp, &assembly, 0, args.penalty);
    for name in &st.missing_faces {
        println!("      WARNING: face '{}' referenced by a boundary condition was not found", name);
    }
    let has_dirichlet = assembly.boundary_conditions.iter().any(|bc| matches!(bc.bc_type, BoundaryConditionType::Dirichlet { .. }) && body.faces.contains_key(&bc.target_face_name));
    if !has_dirichlet {
        return Err("no Dirichlet face on the solved body: the problem has rigid-body modes".into());
    }
    pb.add_surface_terms(&st);
    let total_force: Vec<f64> = (0..3).map(|c| pb.force[c * pb.sp.ndof_sc..(c + 1) * pb.sp.ndof_sc].iter().sum()).collect();
    println!("      net applied force (tractions) [{:.4e}, {:.4e}, {:.4e}]", total_force[0], total_force[1], total_force[2]);

    println!("[4/4] Solving (Jacobi PCG)");
    let t_solve = Instant::now();
    let sol = solve_immersed_problem(&pb, 1e-8, 20 * pb.sp.ndof);
    let solve_ms = t_solve.elapsed().as_secs_f64() * 1e3;
    let converged = sol.residual < 1e-8;
    println!("      {} after {} iterations (relative residual {:.2e}, {:.1} ms)", if converged { "converged" } else { "NOT converged" }, sol.iterations, sol.residual, solve_ms);

    let disp: Vec<[f64; 3]> = body.mesh.vertices.iter().map(|v| eval_vector_at(&pb.sp, &sol.u, v)).collect();
    let peak = disp.iter().map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()).fold(0.0, f64::max);
    println!("      compliance {:.6e}, peak displacement at CAD vertices {:.6e}", sol.compliance, peak);

    let fmt3 = |v: &[f64; 3], prec: usize| format!("[{:.*e}, {:.*e}, {:.*e}]", prec, v[0], prec, v[1], prec, v[2]);
    let nodes_json = body.mesh.vertices.iter().map(|v| fmt3(v, 8)).collect::<Vec<_>>().join(", ");
    let disp_json = disp.iter().map(|d| fmt3(d, 8)).collect::<Vec<_>>().join(", ");
    let total_ms = t0.elapsed().as_secs_f64() * 1e3;
    // peak_disp_mm keeps the previous key name; the value is in model length units.
    let json = format!(
        "{{\"body\": \"{}\", \"solver\": \"immersed_iga (p = {}, {:?})\", \"grid_res\": [{}, {}, {}], \"n_master_dofs\": {}, \"pcg_iters\": {}, \"residual\": {:.3e}, \"converged\": {}, \"compliance\": {:.10e}, \"peak_disp_mm\": {:.10e}, \"wall_time_ms\": {:.2}, \"nodes\": [{}], \"displacements\": [{}]}}",
        body.name, args.degree, stabilization, grid_res[0], grid_res[1], grid_res[2], pb.sp.ndof, sol.iterations, sol.residual, converged, sol.compliance, peak, total_ms, nodes_json, disp_json
    );
    if let Some(dir) = std::path::Path::new(&args.out).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    std::fs::write(&args.out, json)?;
    println!("Saved '{}' ({:.0} ms total)", args.out, total_ms);
    if !converged {
        return Err("PCG did not converge".into());
    }
    Ok(())
}
