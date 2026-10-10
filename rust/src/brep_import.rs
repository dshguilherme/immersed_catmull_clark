//! CAD B-Rep import into a triangle surface mesh (port of `src/brep/importBRep.m`).
//!
//! OBJ and STL (ASCII or binary) are parsed natively. STEP/IGES/BREP/MSH/VTK/VTU go
//! through the same Gmsh/Python bridge as MATLAB (`src/brep/brep_to_tri_mesh.py`),
//! which writes a temporary OBJ. The Python executable is taken from the environment
//! variable `IMMERSED_IGA_PYTHON` (default: `python` on the PATH).

use crate::cut_cell::TriangleMesh3D;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Parses OBJ text: `v x y z` vertices and polygonal `f` faces (fan-triangulated; the
/// `i/j/k` and negative-index forms are accepted).
pub fn parse_obj(text: &str) -> Result<TriangleMesh3D, String> {
    let mut verts: Vec<[f64; 3]> = Vec::new();
    let mut tris: Vec<[usize; 3]> = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let c: Vec<f64> = it.take(3).map(|s| s.parse::<f64>()).collect::<Result<_, _>>().map_err(|e| format!("line {}: {}", ln + 1, e))?;
                if c.len() != 3 {
                    return Err(format!("line {}: vertex needs 3 coordinates", ln + 1));
                }
                verts.push([c[0], c[1], c[2]]);
            }
            Some("f") => {
                let idx: Vec<usize> = it
                    .map(|tok| {
                        let i: i64 = tok.split('/').next().unwrap_or("").parse().map_err(|e| format!("line {}: {}", ln + 1, e))?;
                        let k = if i < 0 { verts.len() as i64 + i } else { i - 1 };
                        if k < 0 { Err(format!("line {}: bad index", ln + 1)) } else { Ok(k as usize) }
                    })
                    .collect::<Result<_, String>>()?;
                for k in 1..idx.len().saturating_sub(1) {
                    tris.push([idx[0], idx[k], idx[k + 1]]);
                }
            }
            _ => {}
        }
    }
    if tris.iter().flatten().any(|&i| i >= verts.len()) {
        return Err("face index out of range".into());
    }
    Ok(TriangleMesh3D::new(verts, tris))
}

/// Parses ASCII or binary STL, merging bit-identical vertices.
pub fn parse_stl(bytes: &[u8]) -> Result<TriangleMesh3D, String> {
    let mut raw: Vec<[f64; 3]> = Vec::new();
    let is_ascii = bytes.starts_with(b"solid") && std::str::from_utf8(bytes).map(|s| s.contains("facet")).unwrap_or(false);
    if is_ascii {
        let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if it.next() == Some("vertex") {
                let c: Vec<f64> = it.take(3).map(|s| s.parse::<f64>()).collect::<Result<_, _>>().map_err(|e| e.to_string())?;
                raw.push([c[0], c[1], c[2]]);
            }
        }
    } else {
        if bytes.len() < 84 {
            return Err("binary STL too short".into());
        }
        let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        if bytes.len() < 84 + 50 * n {
            return Err("binary STL truncated".into());
        }
        for t in 0..n {
            let base = 84 + 50 * t + 12;
            for v in 0..3 {
                let o = base + 12 * v;
                let f = |k: usize| f32::from_le_bytes([bytes[o + 4 * k], bytes[o + 4 * k + 1], bytes[o + 4 * k + 2], bytes[o + 4 * k + 3]]) as f64;
                raw.push([f(0), f(1), f(2)]);
            }
        }
    }
    if raw.len() % 3 != 0 {
        return Err("STL vertex count is not a multiple of 3".into());
    }
    let mut map: HashMap<[u64; 3], usize> = HashMap::new();
    let mut verts = Vec::new();
    let idx: Vec<usize> = raw
        .iter()
        .map(|p| {
            let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
            let next = verts.len();
            *map.entry(key).or_insert_with(|| {
                verts.push(*p);
                next
            })
        })
        .collect();
    let tris = idx.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    Ok(TriangleMesh3D::new(verts, tris))
}

fn bridge_script() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("src").join("brep").join("brep_to_tri_mesh.py"),
        PathBuf::from("src/brep/brep_to_tri_mesh.py"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// Imports a CAD model by extension (MATLAB `importBRep`).
pub fn import_brep(path: &Path) -> Result<TriangleMesh3D, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "obj" => parse_obj(&std::fs::read_to_string(path).map_err(|e| e.to_string())?),
        "stl" => parse_stl(&std::fs::read(path).map_err(|e| e.to_string())?),
        "stp" | "step" | "igs" | "iges" | "brep" | "msh" | "vtk" | "vtu" => {
            let script = bridge_script().ok_or("Gmsh bridge script src/brep/brep_to_tri_mesh.py not found")?;
            let python = std::env::var("IMMERSED_IGA_PYTHON").unwrap_or_else(|_| "python".to_string());
            let tmp = std::env::temp_dir().join(format!("immersed_iga_{}_{}.obj", std::process::id(), path.file_stem().and_then(|s| s.to_str()).unwrap_or("model")));
            let out = std::process::Command::new(&python)
                .arg(&script)
                .arg(path)
                .arg(&tmp)
                .output()
                .map_err(|e| format!("could not run '{}': {} (set IMMERSED_IGA_PYTHON)", python, e))?;
            if !out.status.success() {
                return Err(format!("Gmsh bridge failed: {}", String::from_utf8_lossy(&out.stderr)));
            }
            let text = std::fs::read_to_string(&tmp).map_err(|e| format!("bridge output missing: {}", e))?;
            let _ = std::fs::remove_file(&tmp);
            parse_obj(&text)
        }
        other => Err(format!("unsupported B-Rep extension '.{}'", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_polygons_are_fan_triangulated() {
        let m = parse_obj("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1 2/2 3/3 4/4\n").unwrap();
        assert_eq!(m.triangles, vec![[0, 1, 2], [0, 2, 3]]);
        assert!(parse_obj("v 0 0 0\nf 1 2 3\n").is_err());
    }

    #[test]
    fn stl_ascii_and_binary_agree() {
        let ascii = "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nfacet normal 0 0 1\nouter loop\nvertex 1 0 0\nvertex 1 1 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
        let a = parse_stl(ascii.as_bytes()).unwrap();
        let mut bin = vec![0u8; 80];
        bin.extend(2u32.to_le_bytes());
        for tri in [[[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]] {
            bin.extend([0u8; 12]);
            for v in tri {
                for c in v {
                    bin.extend(c.to_le_bytes());
                }
            }
            bin.extend([0u8; 2]);
        }
        let b = parse_stl(&bin).unwrap();
        assert_eq!(a.vertices.len(), 4);
        assert_eq!(a.triangles, b.triangles);
        assert_eq!(a.vertices, b.vertices);
    }
}
