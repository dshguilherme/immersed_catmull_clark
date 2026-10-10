//! Trilinear hexahedral elasticity on a 2:1-balanced octree with hanging-node
//! multi-point constraints, a port of the MATLAB reference
//! `src/immersed/octree_structural_mesh.m`, `apply_boundary_conditions.m`,
//! `assemble_neumann_bc_3d.m`, `assemble_robin_bc_3d.m` and `solve_octree_gpu.m`
//! (validated in `rust/tests/octree_vs_matlab.rs`).
//!
//! Note: this path uses 8-node trilinear elements, not splines. DOFs are
//! interleaved per master node: `[u_x, u_y, u_z]` of master node m at `3m..3m+3`.
//! Differences from the MATLAB reference (deliberate bug fixes):
//! - boundary conditions take explicit facet lists; an empty list applies nothing
//!   (MATLAB applies a Neumann/Robin load to *all* facets when the filter is empty);
//! - the solve includes the penalty/Robin stiffness (MATLAB `solve_octree_gpu` drops it).

use crate::cut_cell::TriangleMesh3D;
use crate::immersed::ParityRayCaster;
use crate::octree::OctreeMesh3D;
use crate::solver::PcgSolver;
use crate::sparse::CsrMatrix;
use std::collections::HashMap;

const XI: [f64; 8] = [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0];
const ETA: [f64; 8] = [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0];
const ZT: [f64; 8] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];

pub struct OctreeFem {
    pub nodes: Vec<[f64; 3]>,
    pub elem_nodes: Vec<[usize; 8]>,
    pub leaf_bounds: Vec<[f64; 6]>,
    pub levels: Vec<usize>,
    pub is_hanging: Vec<bool>,
    pub master_ids: Vec<usize>,
    /// Constraint rows: node -> [(master index, weight)].
    pub t_rows: Vec<Vec<(usize, f64)>>,
    pub weights: Vec<f64>,
    pub young: f64,
    pub poisson: f64,
    pub ke_by_level: HashMap<usize, Vec<f64>>,
    pub k_master: CsrMatrix,
}

impl OctreeFem {
    pub fn n_master(&self) -> usize {
        self.master_ids.len()
    }

    /// Builds the structural mesh (MATLAB `octree_structural_mesh`). With a B-Rep,
    /// every leaf gets weight max(volume fraction, `fictitious_weight`) from
    /// `subcell_res` midpoint sampling.
    pub fn build(octree: &OctreeMesh3D, brep: Option<&TriangleMesh3D>, young: f64, poisson: f64, fictitious_weight: f64, subcell_res: [usize; 3]) -> Self {
        let leaves = octree.leaf_indices();
        let leaf_bounds: Vec<[f64; 6]> = leaves
            .iter()
            .map(|&i| {
                let b = &octree.cells[i].bounds;
                [b.min[0], b.max[0], b.min[1], b.max[1], b.min[2], b.max[2]]
            })
            .collect();
        let levels: Vec<usize> = leaves.iter().map(|&i| octree.cells[i].level).collect();
        let mut hmin = [f64::MAX; 3];
        for b in &leaf_bounds {
            for d in 0..3 {
                hmin[d] = hmin[d].min(b[2 * d + 1] - b[2 * d]);
            }
        }
        let tol = 1e-5 * hmin.iter().cloned().fold(f64::MAX, f64::min);
        let key = |p: [f64; 3]| [(p[0] / tol).round() as i64, (p[1] / tol).round() as i64, (p[2] / tol).round() as i64];

        // 1. unique nodes (stable order of first appearance, rounded coordinates)
        let mut map: HashMap<[i64; 3], usize> = HashMap::new();
        let mut nodes: Vec<[f64; 3]> = Vec::new();
        let mut elem_nodes = Vec::with_capacity(leaves.len());
        for b in &leaf_bounds {
            let corners = [
                [b[0], b[2], b[4]], [b[1], b[2], b[4]], [b[1], b[3], b[4]], [b[0], b[3], b[4]],
                [b[0], b[2], b[5]], [b[1], b[2], b[5]], [b[1], b[3], b[5]], [b[0], b[3], b[5]],
            ];
            let mut en = [0usize; 8];
            for (a, &c) in corners.iter().enumerate() {
                let k = key(c);
                let next = nodes.len();
                let id = *map.entry(k).or_insert(next);
                if id == next {
                    nodes.push([k[0] as f64 * tol, k[1] as f64 * tol, k[2] as f64 * tol]);
                }
                en[a] = id;
            }
            elem_nodes.push(en);
        }
        let n_nodes = nodes.len();

        // 2. hanging nodes: edge midpoints and face centres that coincide with a node
        let mut c_rows: Vec<Vec<(usize, f64)>> = (0..n_nodes).map(|i| vec![(i, 1.0)]).collect();
        let mut hanging = vec![false; n_nodes];
        const EDGES: [[usize; 2]; 12] = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
        const FACES: [[usize; 4]; 6] = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5]];
        // MATLAB loops over the edge index first, then the element (column-major order)
        for ed in EDGES {
            for en in &elem_nodes {
                let (na, nb) = (en[ed[0]], en[ed[1]]);
                let mid = [0.5 * (nodes[na][0] + nodes[nb][0]), 0.5 * (nodes[na][1] + nodes[nb][1]), 0.5 * (nodes[na][2] + nodes[nb][2])];
                if let Some(&m) = map.get(&key(mid)) {
                    if m != na && m != nb {
                        hanging[m] = true;
                        c_rows[m] = vec![(na, 0.5), (nb, 0.5)];
                    }
                }
            }
        }
        for fc in FACES {
            for en in &elem_nodes {
                let fnodes = [en[fc[0]], en[fc[1]], en[fc[2]], en[fc[3]]];
                let mut cen = [0.0; 3];
                for &f in &fnodes {
                    for d in 0..3 {
                        cen[d] += 0.25 * nodes[f][d];
                    }
                }
                if let Some(&m) = map.get(&key(cen)) {
                    if !fnodes.contains(&m) {
                        hanging[m] = true;
                        c_rows[m] = fnodes.iter().map(|&f| (f, 0.25)).collect();
                    }
                }
            }
        }
        // C <- C*C five times (resolves chains of hanging nodes)
        for _ in 0..5 {
            let next: Vec<Vec<(usize, f64)>> = c_rows
                .iter()
                .map(|row| {
                    let mut acc: HashMap<usize, f64> = HashMap::new();
                    for &(j, w) in row {
                        for &(k, w2) in &c_rows[j] {
                            *acc.entry(k).or_insert(0.0) += w * w2;
                        }
                    }
                    let mut v: Vec<(usize, f64)> = acc.into_iter().filter(|&(_, w)| w != 0.0).collect();
                    v.sort_unstable_by_key(|x| x.0);
                    v
                })
                .collect();
            c_rows = next;
        }
        let master_ids: Vec<usize> = (0..n_nodes).filter(|&i| !hanging[i]).collect();
        let mut master_index = vec![usize::MAX; n_nodes];
        for (m, &i) in master_ids.iter().enumerate() {
            master_index[i] = m;
        }
        let t_rows: Vec<Vec<(usize, f64)>> = c_rows
            .iter()
            .map(|row| row.iter().filter(|(j, _)| master_index[*j] != usize::MAX).map(|&(j, w)| (master_index[j], w)).collect())
            .collect();

        // 3. immersed weights
        let weights: Vec<f64> = match brep {
            None => vec![1.0; leaf_bounds.len()],
            Some(mesh) => {
                let rc = ParityRayCaster::new(mesh);
                use rayon::prelude::*;
                leaf_bounds
                    .par_iter()
                    .map(|b| {
                        let s = subcell_res;
                        let d = [(b[1] - b[0]) / s[0] as f64, (b[3] - b[2]) / s[1] as f64, (b[5] - b[4]) / s[2] as f64];
                        let mut n_in = 0usize;
                        for k in 0..s[2] {
                            for j in 0..s[1] {
                                for i in 0..s[0] {
                                    let q = [b[0] + (i as f64 + 0.5) * d[0], b[2] + (j as f64 + 0.5) * d[1], b[4] + (k as f64 + 0.5) * d[2]];
                                    if rc.is_inside(q) {
                                        n_in += 1;
                                    }
                                }
                            }
                        }
                        (n_in as f64 / (s[0] * s[1] * s[2]) as f64).max(fictitious_weight)
                    })
                    .collect()
            }
        };

        // 4. element stiffness per level (2x2x2 Gauss, trilinear)
        let root_scale = 2f64.powi(levels[0] as i32);
        let root_h = [
            (leaf_bounds[0][1] - leaf_bounds[0][0]) * root_scale,
            (leaf_bounds[0][3] - leaf_bounds[0][2]) * root_scale,
            (leaf_bounds[0][5] - leaf_bounds[0][4]) * root_scale,
        ];
        let mut ke_by_level = HashMap::new();
        for &l in &levels {
            ke_by_level.entry(l).or_insert_with(|| {
                let s = 2f64.powi(l as i32);
                hex8_stiffness([root_h[0] / s, root_h[1] / s, root_h[2] / s], young, poisson)
            });
        }

        // 5. K_master = sum_e T_e^T (w_e K_e) T_e
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (e, en) in elem_nodes.iter().enumerate() {
            let ke = &ke_by_level[&levels[e]];
            let w = weights[e];
            for a in 0..8 {
                for b in 0..8 {
                    for &(ma, ta) in &t_rows[en[a]] {
                        for &(mb, tb) in &t_rows[en[b]] {
                            let s = w * ta * tb;
                            for c in 0..3 {
                                for d in 0..3 {
                                    let v = ke[(3 * a + c) * 24 + 3 * b + d];
                                    if v != 0.0 {
                                        rows.push(3 * ma + c);
                                        cols.push(3 * mb + d);
                                        vals.push(s * v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let nm = master_ids.len();
        let k_master = CsrMatrix::from_triplets(3 * nm, 3 * nm, &rows, &cols, &vals);
        Self { nodes, elem_nodes, leaf_bounds, levels, is_hanging: hanging, master_ids, t_rows, weights, young, poisson, ke_by_level, k_master }
    }

    /// First leaf containing `p` (tolerance 1e-5, as in MATLAB), its trilinear shape values.
    fn locate(&self, p: [f64; 3]) -> Option<(usize, [f64; 8])> {
        let e = self.leaf_bounds.iter().position(|b| {
            p[0] >= b[0] - 1e-5 && p[0] <= b[1] + 1e-5 && p[1] >= b[2] - 1e-5 && p[1] <= b[3] + 1e-5 && p[2] >= b[4] - 1e-5 && p[2] <= b[5] + 1e-5
        })?;
        let b = self.leaf_bounds[e];
        let r = |v: f64, lo: f64, hi: f64| (2.0 * (v - 0.5 * (lo + hi)) / (hi - lo)).clamp(-1.0, 1.0);
        let (xi, eta, zt) = (r(p[0], b[0], b[1]), r(p[1], b[2], b[3]), r(p[2], b[4], b[5]));
        let n = std::array::from_fn(|a| 0.125 * (1.0 + XI[a] * xi) * (1.0 + ETA[a] * eta) * (1.0 + ZT[a] * zt));
        Some((e, n))
    }

    /// Master DOF expansion of node-level vector entries: returns (master dof, weight) for node `n`, component `c`.
    fn master_dofs(&self, node: usize, c: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.t_rows[node].iter().map(move |&(m, w)| (3 * m + c, w))
    }
}

/// 24x24 trilinear hexahedron stiffness for an element of size `h` (2x2x2 Gauss).
pub fn hex8_stiffness(h: [f64; 3], young: f64, poisson: f64) -> Vec<f64> {
    let lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu = young / (2.0 * (1.0 + poisson));
    let mut d = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] = lam;
        }
        d[i][i] = lam + 2.0 * mu;
        d[i + 3][i + 3] = mu;
    }
    let gp = [-1.0 / 3f64.sqrt(), 1.0 / 3f64.sqrt()];
    let det = h[0] * h[1] * h[2] / 8.0;
    let mut ke = vec![0.0; 576];
    for &xi in &gp {
        for &eta in &gp {
            for &zt in &gp {
                let mut bm = [[0.0; 24]; 6];
                for a in 0..8 {
                    let dx = 2.0 / h[0] * 0.125 * XI[a] * (1.0 + ETA[a] * eta) * (1.0 + ZT[a] * zt);
                    let dy = 2.0 / h[1] * 0.125 * ETA[a] * (1.0 + XI[a] * xi) * (1.0 + ZT[a] * zt);
                    let dz = 2.0 / h[2] * 0.125 * ZT[a] * (1.0 + XI[a] * xi) * (1.0 + ETA[a] * eta);
                    let c = 3 * a;
                    bm[0][c] = dx;
                    bm[1][c + 1] = dy;
                    bm[2][c + 2] = dz;
                    bm[3][c + 1] = dz;
                    bm[3][c + 2] = dy;
                    bm[4][c] = dz;
                    bm[4][c + 2] = dx;
                    bm[5][c] = dy;
                    bm[5][c + 1] = dx;
                }
                for i in 0..24 {
                    for j in 0..24 {
                        let mut s = 0.0;
                        for p in 0..6 {
                            for q in 0..6 {
                                s += bm[p][i] * d[p][q] * bm[q][j];
                            }
                        }
                        ke[i * 24 + j] += s * det;
                    }
                }
            }
        }
    }
    ke
}

/// Boundary condition on a set of B-Rep facets (explicit facet indices).
#[derive(Clone, Debug)]
pub enum OctreeBc {
    /// Strong Dirichlet on all master nodes influencing the leaves that contain the
    /// facet centroids (MATLAB 'strong').
    DirichletStrong { facets: Vec<usize>, components: [bool; 3], value: [f64; 3] },
    /// Penalty Dirichlet (MATLAB 'weak'/'penalty': Robin with k = penalty, t = penalty * value).
    DirichletPenalty { facets: Vec<usize>, components: [bool; 3], value: [f64; 3], penalty: f64 },
    Traction { facets: Vec<usize>, traction: [f64; 3] },
    Pressure { facets: Vec<usize>, pressure: f64 },
    /// Isotropic spring `k` (scalar) or normal/tangential springs.
    Robin { facets: Vec<usize>, kn: f64, kt: f64 },
}

pub struct OctreeBcSystem {
    pub stiffness: CsrMatrix,
    pub force: Vec<f64>,
    pub fixed: Vec<(usize, f64)>,
}

/// Applies boundary conditions (MATLAB `apply_boundary_conditions`, one-point facet
/// quadrature at the centroid, as in the reference).
pub fn apply_boundary_conditions(fem: &OctreeFem, brep: &TriangleMesh3D, bcs: &[OctreeBc]) -> OctreeBcSystem {
    let nd = 3 * fem.n_master();
    let mut force = vec![0.0; nd];
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let mut fixed: Vec<(usize, f64)> = Vec::new();
    let facet = |f: usize| {
        let t = brep.triangles[f];
        let (a, b, c) = (brep.vertices[t[0]], brep.vertices[t[1]], brep.vertices[t[2]]);
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let nn = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let area = 0.5 * nn;
        let unit = [n[0] / nn.max(1e-12), n[1] / nn.max(1e-12), n[2] / nn.max(1e-12)];
        let cen = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0];
        (cen, area, unit)
    };
    let add_load = |force: &mut Vec<f64>, cen: [f64; 3], area: f64, t: [f64; 3]| {
        if let Some((e, n)) = fem.locate(cen) {
            for a in 0..8 {
                for c in 0..3 {
                    for (dof, w) in fem.master_dofs(fem.elem_nodes[e][a], c) {
                        force[dof] += w * area * n[a] * t[c];
                    }
                }
            }
        }
    };
    let add_spring = |rr: &mut Vec<usize>, cc: &mut Vec<usize>, vv: &mut Vec<f64>, cen: [f64; 3], area: f64, ks: [[f64; 3]; 3]| {
        if let Some((e, n)) = fem.locate(cen) {
            let en = fem.elem_nodes[e];
            for a in 0..8 {
                for b in 0..8 {
                    for ci in 0..3 {
                        for cj in 0..3 {
                            let k = ks[ci][cj];
                            if k == 0.0 {
                                continue;
                            }
                            let v = area * n[a] * n[b] * k;
                            for (da, wa) in fem.master_dofs(en[a], ci) {
                                for (db, wb) in fem.master_dofs(en[b], cj) {
                                    rr.push(da);
                                    cc.push(db);
                                    vv.push(v * wa * wb);
                                }
                            }
                        }
                    }
                }
            }
        }
    };
    for bc in bcs {
        match bc {
            OctreeBc::Traction { facets, traction } => {
                for &f in facets {
                    let (cen, area, _) = facet(f);
                    add_load(&mut force, cen, area, *traction);
                }
            }
            OctreeBc::Pressure { facets, pressure } => {
                for &f in facets {
                    let (cen, area, n) = facet(f);
                    add_load(&mut force, cen, area, [-pressure * n[0], -pressure * n[1], -pressure * n[2]]);
                }
            }
            OctreeBc::Robin { facets, kn, kt } => {
                for &f in facets {
                    let (cen, area, n) = facet(f);
                    let ks = std::array::from_fn(|i| std::array::from_fn(|j| kn * n[i] * n[j] + kt * ((i == j) as u8 as f64 - n[i] * n[j])));
                    add_spring(&mut rr, &mut cc, &mut vv, cen, area, ks);
                }
            }
            OctreeBc::DirichletPenalty { facets, components, value, penalty } => {
                for &f in facets {
                    let (cen, area, _) = facet(f);
                    let ks = std::array::from_fn(|i| std::array::from_fn(|j| if i == j && components[i] { *penalty } else { 0.0 }));
                    add_spring(&mut rr, &mut cc, &mut vv, cen, area, ks);
                    let t = std::array::from_fn(|i| if components[i] { penalty * value[i] } else { 0.0 });
                    add_load(&mut force, cen, area, t);
                }
            }
            OctreeBc::DirichletStrong { facets, components, value } => {
                let mut elems: Vec<usize> = facets.iter().filter_map(|&f| fem.locate(facet(f).0).map(|x| x.0)).collect();
                elems.sort_unstable();
                elems.dedup();
                let mut masters: Vec<usize> = Vec::new();
                for e in elems {
                    for &n in &fem.elem_nodes[e] {
                        masters.extend(fem.t_rows[n].iter().filter(|&&(_, w)| w > 0.05).map(|&(m, _)| m));
                    }
                }
                masters.sort_unstable();
                masters.dedup();
                for m in masters {
                    for c in 0..3 {
                        if components[c] {
                            fixed.push((3 * m + c, value[c]));
                        }
                    }
                }
            }
        }
    }
    fixed.sort_unstable_by_key(|x| x.0);
    fixed.dedup_by_key(|x| x.0);
    let stiffness = if vv.is_empty() { fem.k_master.clone() } else { fem.k_master.plus_triplets(&rr, &cc, &vv) };
    OctreeBcSystem { stiffness, force, fixed }
}

/// Jacobi-PCG solve with strong elimination of the fixed DOFs. Returns (u, iterations, residual).
pub fn solve(sys: &OctreeBcSystem, tol: f64, max_iter: usize) -> (Vec<f64>, usize, f64) {
    let n = sys.force.len();
    let mut is_fixed = vec![false; n];
    let mut ub = vec![0.0; n];
    for &(d, v) in &sys.fixed {
        is_fixed[d] = true;
        ub[d] = v;
    }
    let mut kub = vec![0.0; n];
    sys.stiffness.matvec(&ub, &mut kub);
    let b: Vec<f64> = (0..n).map(|i| if is_fixed[i] { 0.0 } else { sys.force[i] - kub[i] }).collect();
    if b.iter().all(|&v| v == 0.0) {
        return (ub, 0, 0.0);
    }
    let diag: Vec<f64> = sys.stiffness.diagonal().iter().map(|&d| if d.abs() < 1e-12 { 1.0 } else { d }).collect();
    let mut pm = vec![0.0; n];
    let mut tmp = vec![0.0; n];
    let (x, it, res) = PcgSolver::new(max_iter, tol).solve(n, &b, &diag, |p, ap| {
        for i in 0..n {
            pm[i] = if is_fixed[i] { 0.0 } else { p[i] };
        }
        sys.stiffness.matvec(&pm, &mut tmp);
        for i in 0..n {
            ap[i] = if is_fixed[i] { 0.0 } else { tmp[i] };
        }
    });
    ((0..n).map(|i| if is_fixed[i] { ub[i] } else { x[i] }).collect(), it, res)
}
