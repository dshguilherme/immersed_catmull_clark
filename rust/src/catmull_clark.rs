//! Port of `src/catmull_clark` (validated in `rust/tests/catmull_clark_vs_matlab.rs`).
//!
//! - `subdivide_quad_catmull_clark`: one Catmull-Clark step on a quad mesh (face, edge
//!   and vertex points; boundary edges use midpoints and boundary vertices the
//!   3/4-1/8 curve rule, so corners are smoothed rather than kept sharp).
//! - `subdivision_matrix_1d`: the uniform cubic B-spline dyadic subdivision mask.
//! - `projection_matrix`: B-spline basis at element centres (on regular grids the
//!   Catmull-Clark limit functions are uniform B-splines; with open knot vectors the
//!   boundary functions are those of the open B-spline, not of Catmull-Clark).
//! - `subdivide_triangles_centroid` / `barycentric_basis`: the MATLAB `subdivide` and
//!   `basisFunctions` helpers. Note these are not Catmull-Clark: `subdivide` splits each
//!   triangle into 3 at its centroid and `basisFunctions` is linear interpolation.

use crate::iga::basis_dense;

/// One Catmull-Clark step. `v` are vertex coordinates, `f` quads (0-based). Output
/// layout as in MATLAB: updated vertices, then face points, then edge points.
pub fn subdivide_quad_catmull_clark(v: &[[f64; 3]], f: &[[usize; 4]]) -> (Vec<[f64; 3]>, Vec<[usize; 4]>) {
    let (nv, nf) = (v.len(), f.len());
    let add = |a: [f64; 3], b: [f64; 3]| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let scale = |a: [f64; 3], s: f64| [a[0] * s, a[1] * s, a[2] * s];
    let face_pts: Vec<[f64; 3]> = f.iter().map(|q| scale(add(add(add(v[q[0]], v[q[1]]), v[q[2]]), v[q[3]]), 0.25)).collect();
    // raw edges in MATLAB order: all faces' edge (1-2), then (2-3), (3-4), (4-1)
    let mut raw: Vec<[usize; 2]> = Vec::with_capacity(4 * nf);
    let mut raw_face = Vec::with_capacity(4 * nf);
    for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
        for (fi, q) in f.iter().enumerate() {
            let (x, y) = (q[a].min(q[b]), q[a].max(q[b]));
            raw.push([x, y]);
            raw_face.push(fi);
        }
    }
    let mut uniq = raw.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let edge_map: Vec<usize> = raw.iter().map(|e| uniq.binary_search(e).unwrap()).collect();
    let ne = uniq.len();
    let mut edge_faces: Vec<Vec<usize>> = vec![Vec::new(); ne];
    for (k, &e) in edge_map.iter().enumerate() {
        edge_faces[e].push(raw_face[k]);
    }
    let boundary: Vec<bool> = edge_faces.iter().map(|fs| fs.len() != 2).collect();
    let edge_pts: Vec<[f64; 3]> = (0..ne)
        .map(|e| {
            let [a, b] = uniq[e];
            if boundary[e] {
                scale(add(v[a], v[b]), 0.5)
            } else {
                let (f1, f2) = (edge_faces[e][0], edge_faces[e][1]);
                scale(add(add(add(v[a], v[b]), face_pts[f1]), face_pts[f2]), 0.25)
            }
        })
        .collect();
    let mut vert_faces: Vec<Vec<usize>> = vec![Vec::new(); nv];
    for (fi, q) in f.iter().enumerate() {
        for &x in q {
            vert_faces[x].push(fi);
        }
    }
    let mut vert_edges: Vec<Vec<usize>> = vec![Vec::new(); nv];
    for (e, &[a, b]) in uniq.iter().enumerate() {
        vert_edges[a].push(e);
        vert_edges[b].push(e);
    }
    let new_v: Vec<[f64; 3]> = (0..nv)
        .map(|x| {
            let adj = &vert_edges[x];
            let b_edges: Vec<usize> = adj.iter().copied().filter(|&e| boundary[e]).collect();
            if !b_edges.is_empty() {
                let nb: Vec<usize> = b_edges.iter().map(|&e| if uniq[e][0] == x { uniq[e][1] } else { uniq[e][0] }).collect();
                if nb.len() >= 2 {
                    add(scale(v[x], 0.75), scale(add(v[nb[0]], v[nb[1]]), 0.125))
                } else {
                    v[x]
                }
            } else {
                let n = vert_faces[x].len() as f64;
                let mut favg = [0.0; 3];
                for &fi in &vert_faces[x] {
                    favg = add(favg, face_pts[fi]);
                }
                favg = scale(favg, 1.0 / n);
                let mut emid = [0.0; 3];
                for &e in adj {
                    emid = add(emid, scale(add(v[uniq[e][0]], v[uniq[e][1]]), 0.5));
                }
                emid = scale(emid, 1.0 / adj.len() as f64);
                scale(add(add(favg, scale(emid, 2.0)), scale(v[x], n - 3.0)), 1.0 / n)
            }
        })
        .collect();
    let mut out_v = new_v;
    out_v.extend(face_pts);
    out_v.extend(edge_pts);
    let (fo, eo) = (nv, nv + nf);
    let mut out_f = Vec::with_capacity(4 * nf);
    for (fi, q) in f.iter().enumerate() {
        let cf = fo + fi;
        let e: Vec<usize> = (0..4).map(|k| eo + edge_map[k * nf + fi]).collect();
        out_f.push([q[0], e[0], cf, e[3]]);
        out_f.push([q[1], e[1], cf, e[0]]);
        out_f.push([q[2], e[2], cf, e[1]]);
        out_f.push([q[3], e[3], cf, e[2]]);
    }
    (out_v, out_f)
}

/// Uniform cubic B-spline dyadic subdivision mask `[5 x 4]` (row-major), mapping 4 coarse
/// control points to the 5 fine ones of the two half-intervals.
pub fn subdivision_matrix_1d() -> [[f64; 4]; 5] {
    let s = 1.0 / 8.0;
    [[4.0 * s, 4.0 * s, 0.0, 0.0], [s, 6.0 * s, s, 0.0], [0.0, 4.0 * s, 4.0 * s, 0.0], [0.0, s, 6.0 * s, s], [0.0, 0.0, 4.0 * s, 4.0 * s]]
}

/// B-spline basis of degree `p` (open knots, `nel` uniform elements) evaluated at the
/// element centres: row-major `[nel x (nel + p)]`.
pub fn projection_matrix(nel: usize, p: usize) -> Vec<f64> {
    let mut knots = vec![0.0; p + 1];
    knots.extend((1..nel).map(|i| i as f64 / nel as f64));
    knots.extend(std::iter::repeat(1.0).take(p + 1));
    let xc: Vec<f64> = (0..nel).map(|i| (i as f64 + 0.5) / nel as f64).collect();
    basis_dense(&knots, p, &xc).0
}

/// Splits every triangle into three at its centroid (MATLAB `subdivide`).
pub fn subdivide_triangles_centroid(v: &[[f64; 3]], f: &[[usize; 3]]) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut out_v = v.to_vec();
    let mut out_f = Vec::with_capacity(3 * f.len());
    for (k, t) in f.iter().enumerate() {
        out_v.push(std::array::from_fn(|d| (v[t[0]][d] + v[t[1]][d] + v[t[2]][d]) / 3.0));
        let c = v.len() + k;
        out_f.push([t[0], t[1], c]);
        out_f.push([t[1], t[2], c]);
        out_f.push([t[2], t[0], c]);
    }
    (out_v, out_f)
}

/// Barycentric (linear) basis at points `p` from the first triangle that contains the
/// point's projection (MATLAB `basisFunctions`): row-major `[p.len() x v.len()]`.
pub fn barycentric_basis(v: &[[f64; 3]], f: &[[usize; 3]], p: &[[f64; 3]]) -> Vec<f64> {
    let nv = v.len();
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let mut out = vec![0.0; p.len() * nv];
    for (i, &q) in p.iter().enumerate() {
        for t in f {
            let (v0, v1, v2) = (v[t[0]], v[t[1]], v[t[2]]);
            let (e0, e1, ep) = (sub(v1, v0), sub(v2, v0), sub(q, v0));
            let (d00, d01, d11, d20, d21) = (dot(e0, e0), dot(e0, e1), dot(e1, e1), dot(ep, e0), dot(ep, e1));
            let den = d00 * d11 - d01 * d01;
            if den == 0.0 {
                continue;
            }
            let b = (d11 * d20 - d01 * d21) / den;
            let c = (d00 * d21 - d01 * d20) / den;
            let a = 1.0 - b - c;
            if a >= -1e-9 && b >= -1e-9 && c >= -1e-9 {
                out[i * nv + t[0]] = a;
                out[i * nv + t[1]] = b;
                out[i * nv + t[2]] = c;
                break;
            }
        }
    }
    out
}
