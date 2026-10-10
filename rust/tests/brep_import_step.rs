//! STEP import through the Gmsh bridge vs the MATLAB importer (NIST CTC-01, as imported
//! by tests/test_brep_import.m: 1816 vertices, 3198 facets). Skipped when the model file
//! (git-ignored) or the Python/Gmsh bridge is unavailable.

use immersed_iga::brep_import::import_brep;
use std::path::PathBuf;

#[test]
fn nist_step_import_matches_matlab_counts() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("Models/NIST-PMI-STEP-Files/NIST-PMI-STEP-Files/AP203 geometry only/nist_ctc_01_asme1_rd.stp");
    if !p.exists() {
        eprintln!("skipped: {} not present", p.display());
        return;
    }
    match import_brep(&p) {
        Ok(m) => {
            assert_eq!(m.vertices.len(), 1816);
            assert_eq!(m.triangles.len(), 3198);
        }
        Err(e) if e.contains("could not run") => eprintln!("skipped: {}", e),
        Err(e) => panic!("{}", e),
    }
}
