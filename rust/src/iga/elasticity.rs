//! Exact linear-elasticity element stiffness on a box, mirroring
//! `src/iga/iga_elasticity_element_matrices.m`.
//!
//! a(u, v) = int lambda div(u) div(v) + 2 mu eps(u):eps(v). On an affine box the
//! integrals factor into 1D element matrices, so with p+1 Gauss points per
//! direction the result is exact. Elements whose 1D matrices coincide in every
//! direction share one matrix ("type"); a uniform grid has (2p+1)^dim types at most.

use super::space::{unravel, SpaceBox};
use std::collections::HashMap;

/// Distinct element matrices and the type of every element.
#[derive(Clone, Debug)]
pub struct ElementMatrices {
    pub nsh: usize,
    /// `ntypes` row-major `[nsh x nsh]` matrices; row = test, column = trial,
    /// in the local ordering of `SpaceBox::connectivity`.
    pub types: Vec<Vec<f64>>,
    pub type_id: Vec<usize>,
}

impl ElementMatrices {
    #[inline]
    pub fn of_element(&self, e: usize) -> &[f64] {
        &self.types[self.type_id[e]]
    }
}

/// Elasticity element matrices for constant Lame parameters.
pub fn elasticity_element_matrices(sp: &SpaceBox, lambda: f64, mu: f64) -> ElementMatrices {
    let dim = sp.dim;
    let nsh_sc = sp.nsh_sc;

    // 1D element matrices per direction: m[d][t][k] with k = 00, 10, 01, 11
    // (first index = test; "1" = physical derivative on that side).
    let mut mats: Vec<Vec<[Vec<f64>; 4]>> = Vec::with_capacity(dim);
    let mut tid_dir: Vec<Vec<usize>> = Vec::with_capacity(dim);
    for d in 0..dim {
        let u = &sp.univ[d];
        let ld = sp.lengths[d];
        let nl = u.nsh();
        let mut groups: HashMap<Vec<i64>, usize> = HashMap::new();
        let mut distinct: Vec<[Vec<f64>; 4]> = Vec::new();
        let mut per_elem: Vec<[Vec<f64>; 4]> = Vec::with_capacity(u.nel);
        let mut scale: f64 = 0.0;
        for e in 0..u.nel {
            let mut m: [Vec<f64>; 4] = [vec![0.0; nl * nl], vec![0.0; nl * nl], vec![0.0; nl * nl], vec![0.0; nl * nl]];
            for q in 0..u.nquad {
                let w = u.qw[e * u.nquad + q] * ld;
                let base = (e * u.nquad + q) * nl;
                for a in 0..nl {
                    let na = u.shape[base + a];
                    let da = u.dshape[base + a] / ld;
                    for b in 0..nl {
                        let nb = u.shape[base + b];
                        let db = u.dshape[base + b] / ld;
                        m[0][a * nl + b] += w * na * nb;
                        m[1][a * nl + b] += w * da * nb;
                        m[2][a * nl + b] += w * na * db;
                        m[3][a * nl + b] += w * da * db;
                    }
                }
            }
            for k in 0..4 {
                for v in &m[k] {
                    scale = scale.max(v.abs());
                }
            }
            per_elem.push(m);
        }
        let mut tids = Vec::with_capacity(u.nel);
        for m in per_elem {
            let key: Vec<i64> = m.iter().flatten().map(|v| (v / scale * 1e12).round() as i64).collect();
            let next = distinct.len();
            let t = *groups.entry(key).or_insert(next);
            if t == next {
                distinct.push(m);
            }
            tids.push(t);
        }
        mats.push(distinct);
        tid_dir.push(tids);
    }

    // Element types = distinct combinations of 1D types, in element order.
    let mut combos: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut type_list: Vec<Vec<usize>> = Vec::new();
    let mut type_id = Vec::with_capacity(sp.nel);
    for e in 0..sp.nel {
        let esub = unravel(e, &sp.nel_dir);
        let key: Vec<usize> = (0..dim).map(|d| tid_dir[d][esub[d]]).collect();
        let next = type_list.len();
        let t = *combos.entry(key.clone()).or_insert(next);
        if t == next {
            type_list.push(key);
        }
        type_id.push(t);
    }

    let nsh = sp.nsh;
    let types = type_list
        .iter()
        .map(|key| {
            // g[i][j] = int d_i(phi_test) d_j(phi_trial), scalar functions.
            let mut g: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); dim]; dim];
            for i in 0..dim {
                for j in 0..dim {
                    let mut gij = vec![1.0];
                    let mut n = 1usize;
                    for d in 0..dim {
                        let k = match (d == i, d == j) {
                            (true, true) => 3,
                            (true, false) => 1,
                            (false, true) => 2,
                            (false, false) => 0,
                        };
                        let m = &mats[d][key[d]][k];
                        let nl = sp.nsh_dir[d];
                        gij = kron(m, nl, &gij, n);
                        n *= nl;
                    }
                    g[i][j] = gij;
                }
            }
            let mut lap = vec![0.0; nsh_sc * nsh_sc];
            for k in 0..dim {
                for (l, v) in lap.iter_mut().zip(&g[k][k]) {
                    *l += v;
                }
            }
            let mut ke = vec![0.0; nsh * nsh];
            for ci in 0..dim {
                for cj in 0..dim {
                    for a in 0..nsh_sc {
                        for b in 0..nsh_sc {
                            let mut v = lambda * g[ci][cj][a * nsh_sc + b] + mu * g[cj][ci][a * nsh_sc + b];
                            if ci == cj {
                                v += mu * lap[a * nsh_sc + b];
                            }
                            ke[(ci * nsh_sc + a) * nsh + cj * nsh_sc + b] = v;
                        }
                    }
                }
            }
            ke
        })
        .collect();
    ElementMatrices { nsh, types, type_id }
}

/// Kronecker product of square row-major matrices `a` (na x na) and `b` (nb x nb);
/// the index of `b` runs fastest.
fn kron(a: &[f64], na: usize, b: &[f64], nb: usize) -> Vec<f64> {
    let n = na * nb;
    let mut out = vec![0.0; n * n];
    for ia in 0..na {
        for ja in 0..na {
            let av = a[ia * na + ja];
            if av == 0.0 {
                continue;
            }
            for ib in 0..nb {
                for jb in 0..nb {
                    out[(ia * nb + ib) * n + ja * nb + jb] = av * b[ib * nb + jb];
                }
            }
        }
    }
    out
}

/// Lame parameters from Young's modulus and Poisson's ratio.
pub fn lame(young: f64, poisson: f64) -> (f64, f64) {
    let lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu = young / (2.0 * (1.0 + poisson));
    (lambda, mu)
}
