//! Boundary conditions on immersed CAD surfaces, integrated over the surface
//! triangles with the actual spline trace (6-point degree-4 Dunavant rule).
//!
//! - Dirichlet: **penalty** method, `beta * int_G u.v` with `beta = factor * E / h`.
//!   It is not Nitsche (no consistency terms), so it converges to the constrained
//!   solution only as beta grows; the constraint error scales like 1/beta.
//! - Neumann traction / pressure: consistent load `int_G t.v`.
//! - Robin foundation: `int_G (k_n n n^T + k_t (I - n n^T)) u.v`.

use crate::cad_model::{BoundaryConditionType, CadAssembly, CadBody};
use crate::iga::{unravel, SpaceBox};
use crate::iga::basis::basis_funs;

/// Dunavant degree-4 rule on the reference triangle: barycentric points and weights (sum 1).
const TRI_RULE: [([f64; 3], f64); 6] = [
    ([0.108103018168070, 0.445948490915965, 0.445948490915965], 0.223381589678011),
    ([0.445948490915965, 0.108103018168070, 0.445948490915965], 0.223381589678011),
    ([0.445948490915965, 0.445948490915965, 0.108103018168070], 0.223381589678011),
    ([0.816847572980459, 0.091576213509771, 0.091576213509771], 0.109951743655322),
    ([0.091576213509771, 0.816847572980459, 0.091576213509771], 0.109951743655322),
    ([0.091576213509771, 0.091576213509771, 0.816847572980459], 0.109951743655322),
];

/// Nonzero scalar basis functions at a physical point: `(global index, value)`.
pub fn scalar_basis_at(sp: &SpaceBox, x: &[f64; 3]) -> Vec<(usize, f64)> {
    let dim = sp.dim;
    let mut first = [0usize; 3];
    let mut vals: Vec<Vec<f64>> = Vec::with_capacity(dim);
    for d in 0..dim {
        let u = &sp.univ[d];
        let xi = ((x[d] - sp.lo[d]) / sp.lengths[d]).clamp(0.0, 1.0);
        let (f, v, _) = basis_funs(&u.knots, u.degree, xi);
        first[d] = f;
        vals.push(v);
    }
    (0..sp.nsh_sc)
        .map(|a| {
            let s = unravel(a, &sp.nsh_dir);
            let mut w = 1.0;
            let mut g = 0usize;
            let mut stride = 1usize;
            for d in 0..dim {
                w *= vals[d][s[d]];
                g += (first[d] + s[d]) * stride;
                stride *= sp.ndof_dir[d];
            }
            (g, w)
        })
        .collect()
}

/// Accumulates surface stiffness per background element (all quadrature points that see
/// the same set of basis functions share one dense local block), which keeps memory
/// proportional to the number of cut elements instead of the number of surface points.
#[derive(Default)]
struct SurfaceAccumulator {
    blocks: std::collections::HashMap<usize, (Vec<usize>, Vec<f64>)>,
}

impl SurfaceAccumulator {
    /// Adds `w * K_s[ci][cj] * N_a N_b` for the basis functions `nz` at one point.
    fn add(&mut self, nz: &[(usize, f64)], ks: [[f64; 3]; 3], w: f64) {
        let n = nz.len();
        let key = nz[0].0; // first function identifies the element support
        let entry = self.blocks.entry(key).or_insert_with(|| (nz.iter().map(|x| x.0).collect(), vec![0.0; 9 * n * n]));
        let m = &mut entry.1;
        for ci in 0..3 {
            for cj in 0..3 {
                let k = ks[ci][cj] * w;
                if k == 0.0 {
                    continue;
                }
                for a in 0..n {
                    let ka = k * nz[a].1;
                    let row = (ci * n + a) * 3 * n + cj * n;
                    for b in 0..n {
                        m[row + b] += ka * nz[b].1;
                    }
                }
            }
        }
    }

    fn into_triplets(self, nsc: usize, st: &mut SurfaceTerms) {
        for (_, (idx, m)) in self.blocks {
            let n = idx.len();
            for ci in 0..3 {
                for a in 0..n {
                    for cj in 0..3 {
                        for b in 0..n {
                            let v = m[(ci * n + a) * 3 * n + cj * n + b];
                            if v != 0.0 {
                                st.rows.push(ci * nsc + idx[a]);
                                st.cols.push(cj * nsc + idx[b]);
                                st.vals.push(v);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Surface quadrature points of a set of triangles: `(point, weight, unit normal)`.
/// The normal follows the triangle orientation (outward for a consistently oriented closed mesh).
/// Triangles are split (midpoint refinement) until every edge is at most `max_edge`, so that
/// the rule resolves the piecewise-polynomial spline trace; pass a fraction of the grid size.
pub fn surface_quadrature(body: &CadBody, tris: &[usize], max_edge: f64) -> Vec<([f64; 3], f64, [f64; 3])> {
    let mut out = Vec::with_capacity(tris.len() * TRI_RULE.len());
    for &t in tris {
        let [i, j, k] = body.mesh.triangles[t];
        let mut stack = vec![(body.mesh.vertices[i], body.mesh.vertices[j], body.mesh.vertices[k])];
        while let Some((a, b, c)) = stack.pop() {
            let len = |p: [f64; 3], q: [f64; 3]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
            if len(a, b).max(len(b, c)).max(len(c, a)) > max_edge {
                let mid = |p: [f64; 3], q: [f64; 3]| [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])];
                let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
                stack.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]);
                continue;
            }
            push_triangle_rule(&mut out, a, b, c);
        }
    }
    out
}

fn push_triangle_rule(out: &mut Vec<([f64; 3], f64, [f64; 3])>, a: [f64; 3], b: [f64; 3], c: [f64; 3]) {
    {
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let nn = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if nn == 0.0 {
            return;
        }
        let area = 0.5 * nn;
        let unit = [n[0] / nn, n[1] / nn, n[2] / nn];
        for (l, w) in TRI_RULE {
            let x = [
                l[0] * a[0] + l[1] * b[0] + l[2] * c[0],
                l[0] * a[1] + l[1] * b[1] + l[2] * c[1],
                l[0] * a[2] + l[1] * b[2] + l[2] * c[2],
            ];
            out.push((x, w * area, unit));
        }
    }
}

/// Stiffness triplets and load produced by the surface boundary conditions.
#[derive(Default)]
pub struct SurfaceTerms {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<f64>,
    pub force: Vec<f64>,
    /// Names of faces that were requested but not found.
    pub missing_faces: Vec<String>,
}

/// Penalty Dirichlet `beta int_G (u - g).v` with a position-dependent value `g(x)` on the
/// given triangles of `mesh` (all components).
pub fn dirichlet_penalty_fn<G: Fn(&[f64; 3]) -> [f64; 3]>(sp: &SpaceBox, mesh: &crate::cut_cell::TriangleMesh3D, tris: &[usize], beta: f64, g: G) -> SurfaceTerms {
    let body = CadBody::new("surface", mesh.clone(), crate::cad_model::MaterialProperties::default());
    let h_min = sp.element_size().iter().cloned().fold(f64::MAX, f64::min);
    let nsc = sp.ndof_sc;
    let mut st = SurfaceTerms { force: vec![0.0; sp.ndof], ..Default::default() };
    let mut acc = SurfaceAccumulator::default();
    let ks = [[beta, 0.0, 0.0], [0.0, beta, 0.0], [0.0, 0.0, beta]];
    for (x, w, _) in surface_quadrature(&body, tris, 0.25 * h_min) {
        let nz = scalar_basis_at(sp, &x);
        let gx = g(&x);
        for c in 0..3 {
            for &(ga, na) in &nz {
                st.force[c * nsc + ga] += beta * w * na * gx[c];
            }
        }
        acc.add(&nz, ks, w);
    }
    acc.into_triplets(nsc, &mut st);
    st
}

/// Integrates all labelled boundary conditions of `assembly` that target faces of
/// `body_idx`. `penalty_factor` scales the Dirichlet penalty `beta = factor * E / h_min`.
pub fn surface_boundary_terms(sp: &SpaceBox, assembly: &CadAssembly, body_idx: usize, penalty_factor: f64) -> SurfaceTerms {
    let body = &assembly.bodies[body_idx];
    let nsc = sp.ndof_sc;
    let h_min = sp.element_size().iter().cloned().fold(f64::MAX, f64::min);
    let beta = penalty_factor * body.material.youngs_modulus / h_min;
    let mut st = SurfaceTerms { force: vec![0.0; sp.ndof], ..Default::default() };
    let mut acc = SurfaceAccumulator::default();
    for bc in &assembly.boundary_conditions {
        let face = match body.faces.get(&bc.target_face_name) {
            Some(f) => f,
            None => {
                if assembly.find_face(&bc.target_face_name).is_none() {
                    st.missing_faces.push(bc.target_face_name.clone());
                }
                continue;
            }
        };
        for (x, w, n) in surface_quadrature(body, &face.triangle_indices, 0.25 * h_min) {
            let nz = scalar_basis_at(sp, &x);
            match &bc.bc_type {
                BoundaryConditionType::Dirichlet { components, values } => {
                    let mut ks = [[0.0; 3]; 3];
                    for c in 0..3 {
                        if !components[c] {
                            continue;
                        }
                        ks[c][c] = beta;
                        for &(ga, na) in &nz {
                            st.force[c * nsc + ga] += beta * w * na * values[c];
                        }
                    }
                    acc.add(&nz, ks, w);
                }
                BoundaryConditionType::NeumannTraction { traction } => {
                    for &(ga, na) in &nz {
                        for c in 0..3 {
                            st.force[c * nsc + ga] += w * na * traction[c];
                        }
                    }
                }
                BoundaryConditionType::NeumannPressure { pressure } => {
                    for &(ga, na) in &nz {
                        for c in 0..3 {
                            st.force[c * nsc + ga] -= w * na * pressure * n[c];
                        }
                    }
                }
                BoundaryConditionType::RobinFoundation { normal_stiffness, tangential_stiffness } => {
                    let ks: [[f64; 3]; 3] = std::array::from_fn(|ci| {
                        std::array::from_fn(|cj| {
                            let delta = if ci == cj { 1.0 } else { 0.0 };
                            normal_stiffness * n[ci] * n[cj] + tangential_stiffness * (delta - n[ci] * n[cj])
                        })
                    });
                    acc.add(&nz, ks, w);
                }
            }
        }
    }
    acc.into_triplets(nsc, &mut st);
    st
}
