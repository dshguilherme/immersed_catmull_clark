//! Multi-body assemblies on independent octrees with penalty contact coupling, a port
//! of the MATLAB reference `src/immersed/setup_assembly_3d.m` and
//! `solve_assembly_contact_3d.m` (validated in `rust/tests/contact_vs_matlab.rs`).
//!
//! Method (as in the reference, documented here because it is easy to over-read):
//! - contact points: one per facet of body B (its centroid), paired with the *first*
//!   facet of body A that opposes it (normals dot < -0.2), lies within `gap_tol` of the
//!   centroid along A's normal and contains the projection (barycentric tolerance 0.05);
//! - penalty enforcement with `k = gamma_c * E_eff / h`, where E_eff averages the
//!   diagonal lam + 2 mu of the two bodies' constitutive matrices (not Nitsche);
//! - unilateral: active set from the current gap (g <= 1e-6), re-solved until the set
//!   repeats or the displacement increment is below `tol`;
//! - bonded: all 3 components tied, and the initial gap vector is closed (the force
//!   term pulls the paired points together), i.e. bodies are glued after closing gaps.

use crate::cut_cell::TriangleMesh3D;
use crate::octree::{BoundingBox3D, OctreeMesh3D};
use crate::octree_fem::{apply_boundary_conditions, OctreeBc, OctreeFem};
use crate::solver::PcgSolver;
use crate::sparse::CsrMatrix;

pub struct BodySpec {
    pub name: String,
    pub brep: TriangleMesh3D,
    pub young: f64,
    pub poisson: f64,
    pub grid_res: [usize; 3],
    pub max_level: usize,
}

pub struct AssemblyBody {
    pub spec: BodySpec,
    pub fem: OctreeFem,
    pub bbox: [[f64; 2]; 3],
    pub dof_offset: usize,
    pub ndof: usize,
}

/// Contact points between bodies `a` and `b`, with interpolation from master DOFs:
/// `proj_a[p]` lists (master node, weight) such that u_A(x_p) = sum w * u_A[master].
pub struct Interface {
    pub body_a: usize,
    pub body_b: usize,
    pub pts_a: Vec<[f64; 3]>,
    pub pts_b: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub areas: Vec<f64>,
    pub facets_a: Vec<usize>,
    pub facets_b: Vec<usize>,
    pub proj_a: Vec<Vec<(usize, f64)>>,
    pub proj_b: Vec<Vec<(usize, f64)>>,
    pub valid: Vec<bool>,
}

pub struct Assembly {
    pub bodies: Vec<AssemblyBody>,
    pub interfaces: Vec<Interface>,
    pub total_dof: usize,
    pub gap_tol: f64,
}

fn tri_geom(m: &TriangleMesh3D, f: usize) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3], f64) {
    let t = m.triangles[f];
    let (a, b, c) = (m.vertices[t[0]], m.vertices[t[1]], m.vertices[t[2]]);
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    let nn = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let area = 0.5 * nn;
    let d = (2.0 * area).max(1e-12);
    (a, b, c, [n[0] / d, n[1] / d, n[2] / d], area)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Assembly {
    /// Builds each body's octree mesh and detects contact interfaces (MATLAB `setup_assembly_3d`).
    pub fn setup(specs: Vec<BodySpec>, gap_tol: Option<f64>, margin_ratio: f64) -> Self {
        let mut bodies = Vec::new();
        let mut offset = 0;
        for spec in specs {
            let mut lo = [f64::MAX; 3];
            let mut hi = [f64::MIN; 3];
            for v in &spec.brep.vertices {
                for d in 0..3 {
                    lo[d] = lo[d].min(v[d]);
                    hi[d] = hi[d].max(v[d]);
                }
            }
            let bbox: [[f64; 2]; 3] = std::array::from_fn(|d| {
                let m = margin_ratio * (hi[d] - lo[d]).max(1e-3);
                [lo[d] - m, hi[d] + m]
            });
            let oct = OctreeMesh3D::build(
                BoundingBox3D::new([bbox[0][0], bbox[1][0], bbox[2][0]], [bbox[0][1], bbox[1][1], bbox[2][1]]),
                spec.grid_res,
                spec.max_level,
                |_| false,
            );
            let fem = OctreeFem::build(&oct, Some(&spec.brep), spec.young, spec.poisson, 1e-4, [3, 3, 3]);
            let ndof = 3 * fem.n_master();
            bodies.push(AssemblyBody { spec, fem, bbox, dof_offset: offset, ndof });
            offset += ndof;
        }
        let gap_tol = gap_tol.unwrap_or_else(|| {
            0.05 * bodies
                .iter()
                .map(|b| (0..3).map(|d| (b.bbox[d][1] - b.bbox[d][0]).powi(2)).sum::<f64>().sqrt())
                .fold(f64::MAX, f64::min)
        });
        let mut interfaces = Vec::new();
        for i in 0..bodies.len() {
            for j in i + 1..bodies.len() {
                let (bi, bj) = (&bodies[i].bbox, &bodies[j].bbox);
                let overlap = (0..3).all(|d| bi[d][0] <= bj[d][1] + gap_tol && bi[d][1] >= bj[d][0] - gap_tol);
                if !overlap {
                    continue;
                }
                if let Some(inter) = detect_interface(&bodies, i, j, gap_tol) {
                    interfaces.push(inter);
                }
            }
        }
        Self { bodies, interfaces, total_dof: offset, gap_tol }
    }

    fn penalty(&self, inter: &Interface, gamma_c: f64) -> f64 {
        let h = |b: &AssemblyBody| {
            let s: f64 = b.fem.leaf_bounds.iter().map(|x| (x[1] - x[0]) + (x[3] - x[2]) + (x[5] - x[4])).sum();
            s / (3.0 * b.fem.leaf_bounds.len() as f64)
        };
        let p_mod = |b: &AssemblyBody| {
            let (e, nu) = (b.fem.young, b.fem.poisson);
            let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
            let mu = e / (2.0 * (1.0 + nu));
            (lam + 2.0 * mu).max(1.0)
        };
        let (a, b) = (&self.bodies[inter.body_a], &self.bodies[inter.body_b]);
        gamma_c * 0.5 * (p_mod(a) + p_mod(b)) / (0.5 * (h(a) + h(b)))
    }

    /// Normal gaps at all contact points for the assembly displacement `u`.
    pub fn gaps(&self, u: &[f64]) -> Vec<f64> {
        let mut out = Vec::new();
        for inter in &self.interfaces {
            let (oa, ob) = (self.bodies[inter.body_a].dof_offset, self.bodies[inter.body_b].dof_offset);
            for p in 0..inter.pts_a.len() {
                let disp = |proj: &Vec<(usize, f64)>, off: usize| -> [f64; 3] {
                    std::array::from_fn(|c| proj.iter().map(|&(m, w)| w * u[off + 3 * m + c]).sum())
                };
                let (da, db) = (disp(&inter.proj_a[p], oa), disp(&inter.proj_b[p], ob));
                let rel: [f64; 3] = std::array::from_fn(|c| (inter.pts_b[p][c] + db[c]) - (inter.pts_a[p][c] + da[c]));
                out.push(dot(rel, inter.normals[p]));
            }
        }
        out
    }

    /// Contact stiffness and force for the selected points (`active` over all points);
    /// `bonded` ties all components instead of the normal one.
    fn contact_terms(&self, active: &[bool], gamma_c: f64, bonded: bool) -> ((Vec<usize>, Vec<usize>, Vec<f64>), Vec<f64>) {
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let mut f = vec![0.0; self.total_dof];
        let mut k_pt = 0;
        for inter in &self.interfaces {
            let pen = self.penalty(inter, gamma_c);
            let (oa, ob) = (self.bodies[inter.body_a].dof_offset, self.bodies[inter.body_b].dof_offset);
            for p in 0..inter.pts_a.len() {
                let act = active[k_pt] && inter.valid[p];
                k_pt += 1;
                if !act {
                    continue;
                }
                let k = pen * inter.areas[p];
                let n = inter.normals[p];
                let pm: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| if bonded { (i == j) as u8 as f64 } else { n[i] * n[j] }));
                // jump operator J = [-M_A, M_B]: entries (dof, weight sign)
                let mut jterms: Vec<(usize, f64)> = Vec::new();
                for &(m, w) in &inter.proj_a[p] {
                    jterms.push((oa + 3 * m, -w));
                }
                for &(m, w) in &inter.proj_b[p] {
                    jterms.push((ob + 3 * m, w));
                }
                for &(da, wa) in &jterms {
                    for &(db, wb) in &jterms {
                        for ci in 0..3 {
                            for cj in 0..3 {
                                let v = k * wa * wb * pm[ci][cj];
                                if v != 0.0 {
                                    rr.push(da + ci);
                                    cc.push(db + cj);
                                    vv.push(v);
                                }
                            }
                        }
                    }
                }
                // initial-gap force: f = -k J^T P g0 (normal: g0 = (x_B - x_A).n)
                let g0: [f64; 3] = std::array::from_fn(|c| inter.pts_b[p][c] - inter.pts_a[p][c]);
                let pg: [f64; 3] = if bonded { g0 } else { let g = dot(g0, n); [g * n[0], g * n[1], g * n[2]] };
                for &(d, w) in &jterms {
                    for c in 0..3 {
                        f[d + c] -= k * w * pg[c];
                    }
                }
            }
        }
        ((rr, cc, vv), f)
    }

    fn solve_linear(&self, k: &CsrMatrix, f: &[f64], fixed: &[(usize, f64)], tol: f64) -> Vec<f64> {
        let n = self.total_dof;
        let mut is_fixed = vec![false; n];
        let mut ub = vec![0.0; n];
        for &(d, v) in fixed {
            is_fixed[d] = true;
            ub[d] = v;
        }
        let mut kub = vec![0.0; n];
        k.matvec(&ub, &mut kub);
        let b: Vec<f64> = (0..n).map(|i| if is_fixed[i] { 0.0 } else { f[i] - kub[i] }).collect();
        let diag: Vec<f64> = k.diagonal().iter().map(|&d| if d.abs() < 1e-12 { 1.0 } else { d }).collect();
        let mut pm = vec![0.0; n];
        let mut tmp = vec![0.0; n];
        let (x, _, _) = PcgSolver::new(50 * n, tol).solve(n, &b, &diag, |p, ap| {
            for i in 0..n {
                pm[i] = if is_fixed[i] { 0.0 } else { p[i] };
            }
            k.matvec(&pm, &mut tmp);
            for i in 0..n {
                ap[i] = if is_fixed[i] { 0.0 } else { tmp[i] };
            }
        });
        (0..n).map(|i| if is_fixed[i] { ub[i] } else { x[i] }).collect()
    }

    /// Solves the assembly with per-body boundary conditions (`bcs[i]` for body i).
    pub fn solve(&self, bcs: &[Vec<OctreeBc>], mode: ContactMode, gamma_c: f64, max_iter: usize, tol: f64, lin_tol: f64) -> ContactResult {
        let n = self.total_dof;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let mut force = vec![0.0; n];
        let mut fixed = Vec::new();
        for (i, body) in self.bodies.iter().enumerate() {
            let off = body.dof_offset;
            let k = &body.fem.k_master;
            for r in 0..k.nrows {
                for idx in k.indptr[r]..k.indptr[r + 1] {
                    rr.push(off + r);
                    cc.push(off + k.indices[idx]);
                    vv.push(k.data[idx]);
                }
            }
            if let Some(list) = bcs.get(i) {
                let sys = apply_boundary_conditions(&body.fem, &body.spec.brep, list);
                for (j, &f) in sys.force.iter().enumerate() {
                    force[off + j] += f;
                }
                fixed.extend(sys.fixed.iter().map(|&(d, v)| (off + d, v)));
                let (er, ec, ev) = sys.extra;
                rr.extend(er.iter().map(|&x| off + x));
                cc.extend(ec.iter().map(|&x| off + x));
                vv.extend(ev);
            }
        }
        fixed.sort_unstable_by_key(|x| x.0);
        fixed.dedup_by_key(|x| x.0);
        let k_base = CsrMatrix::from_triplets(n, n, &rr, &cc, &vv);
        let n_pts: usize = self.interfaces.iter().map(|i| i.pts_a.len()).sum();
        match mode {
            ContactMode::Bonded => {
                let ((r, c, v), fc) = self.contact_terms(&vec![true; n_pts], gamma_c, true);
                let k = k_base.plus_triplets(&r, &c, &v);
                let f: Vec<f64> = force.iter().zip(&fc).map(|(a, b)| a + b).collect();
                let u = self.solve_linear(&k, &f, &fixed, lin_tol);
                let gaps = self.gaps(&u);
                ContactResult { u, iterations: 1, converged: true, active: vec![true; n_pts], gaps }
            }
            ContactMode::Unilateral => {
                let mut u: Vec<f64> = vec![0.0; n];
                for &(d, v) in &fixed {
                    u[d] = v;
                }
                let mut prev: Option<Vec<bool>> = None;
                let mut converged = false;
                let mut iters = 0;
                for it in 1..=max_iter {
                    iters = it;
                    let valid: Vec<bool> = self.interfaces.iter().flat_map(|i| i.valid.clone()).collect();
                    let active: Vec<bool> = self.gaps(&u).iter().zip(&valid).map(|(&g, &ok)| g <= 1e-6 && ok).collect();
                    if it > 1 && prev.as_ref() == Some(&active) {
                        converged = true;
                        break;
                    }
                    prev = Some(active.clone());
                    let ((r, c, v), fc) = self.contact_terms(&active, gamma_c, false);
                    let k = k_base.plus_triplets(&r, &c, &v);
                    let f: Vec<f64> = force.iter().zip(&fc).map(|(a, b)| a + b).collect();
                    let un = self.solve_linear(&k, &f, &fixed, lin_tol);
                    let du: f64 = un.iter().zip(&u).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                    let nu: f64 = un.iter().map(|a| a * a).sum::<f64>().sqrt();
                    u = un;
                    if du / nu.max(1.0) < tol && it > 1 {
                        converged = true;
                        break;
                    }
                }
                let gaps = self.gaps(&u);
                let valid: Vec<bool> = self.interfaces.iter().flat_map(|i| i.valid.clone()).collect();
                let active = gaps.iter().zip(&valid).map(|(&g, &ok)| g <= 1e-6 && ok).collect();
                ContactResult { u, iterations: iters, converged, active, gaps }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContactMode {
    Unilateral,
    Bonded,
}

pub struct ContactResult {
    pub u: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub active: Vec<bool>,
    pub gaps: Vec<f64>,
}

fn detect_interface(bodies: &[AssemblyBody], i: usize, j: usize, gap_tol: f64) -> Option<Interface> {
    let (mi, mj) = (&bodies[i].spec.brep, &bodies[j].spec.brep);
    let geo_i: Vec<_> = (0..mi.triangles.len()).map(|f| tri_geom(mi, f)).collect();
    let mut inter = Interface {
        body_a: i, body_b: j, pts_a: vec![], pts_b: vec![], normals: vec![], areas: vec![],
        facets_a: vec![], facets_b: vec![], proj_a: vec![], proj_b: vec![], valid: vec![],
    };
    for fj in 0..mj.triangles.len() {
        let (a, b, c, nj, area_j) = tri_geom(mj, fj);
        let pj = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0];
        for (fi, &(av, bv, cv, ni, _)) in geo_i.iter().enumerate() {
            if dot(ni, nj) > -0.2 {
                continue;
            }
            let pa: [f64; 3] = std::array::from_fn(|k| pj[k] - av[k]);
            let dn = dot(pa, ni);
            if dn.abs() > gap_tol {
                continue;
            }
            let proj: [f64; 3] = std::array::from_fn(|k| pj[k] - dn * ni[k]);
            let u: [f64; 3] = std::array::from_fn(|k| bv[k] - av[k]);
            let v: [f64; 3] = std::array::from_fn(|k| cv[k] - av[k]);
            let w: [f64; 3] = std::array::from_fn(|k| proj[k] - av[k]);
            let (duu, duv, dvv, dwu, dwv) = (dot(u, u), dot(u, v), dot(v, v), dot(w, u), dot(w, v));
            let den = duu * dvv - duv * duv;
            if den.abs() < 1e-12 {
                continue;
            }
            let lb = (dvv * dwu - duv * dwv) / den;
            let lc = (duu * dwv - duv * dwu) / den;
            let la = 1.0 - lb - lc;
            if la >= -0.05 && lb >= -0.05 && lc >= -0.05 {
                inter.pts_a.push(proj);
                inter.pts_b.push(pj);
                inter.areas.push(area_j);
                inter.normals.push(ni);
                inter.facets_a.push(fi);
                inter.facets_b.push(fj);
                break;
            }
        }
    }
    if inter.pts_a.is_empty() {
        return None;
    }
    let projector = |fem: &OctreeFem, x: [f64; 3]| -> Option<Vec<(usize, f64)>> {
        let (e, n) = fem.locate(x)?;
        let mut out = Vec::new();
        for a in 0..8 {
            for &(m, w) in &fem.t_rows[fem.elem_nodes[e][a]] {
                if w > 1e-4 {
                    out.push((m, n[a] * w));
                }
            }
        }
        Some(out)
    };
    for p in 0..inter.pts_a.len() {
        match (projector(&bodies[i].fem, inter.pts_a[p]), projector(&bodies[j].fem, inter.pts_b[p])) {
            (Some(pa), Some(pb)) => {
                inter.proj_a.push(pa);
                inter.proj_b.push(pb);
                inter.valid.push(true);
            }
            _ => {
                inter.proj_a.push(Vec::new());
                inter.proj_b.push(Vec::new());
                inter.valid.push(false);
            }
        }
    }
    Some(inter)
}
