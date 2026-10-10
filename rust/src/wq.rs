//! Weighted-quadrature (WQ) stiffness formation with sum factorization, the
//! "FastFormation" kernel: a port of `src/iga/iga_wq_rules_1d.m`,
//! `iga_wq_elasticity_tensor.m` and the row loop of `src/fastformation/fast_stiffness_assembly.m`
//! (Calabro, Sangalli & Tani, CMAME 316, 2017). Validated in `rust/tests/wq_vs_matlab.rs`;
//! for uniform density it reproduces exact Gauss assembly.

use crate::iga::{basis_dense, SpaceBox, Space1d};
use crate::sparse::CsrMatrix;

/// Univariate WQ rules for one direction (parametric coordinates).
pub struct WqRules1d {
    pub points: Vec<f64>,
    pub ndof: usize,
    /// Points (indices into `points`) where test function i is nonzero.
    pub ind_points: Vec<Vec<usize>>,
    /// Trial functions of S_p whose support overlaps supp(B_i).
    pub neighbors: Vec<Vec<usize>>,
    /// Weights per test function: [W00, W10, W01, W11], each aligned with `ind_points[i]`.
    pub weights: [Vec<Vec<f64>>; 4],
    /// Values / parametric derivatives of S_p at `points`: row-major [npts x ndof].
    pub b: Vec<f64>,
    pub db: Vec<f64>,
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|i| a + i as f64 * (b - a) / (n - 1) as f64).collect();
    v[0] = a;
    v[n - 1] = b;
    v
}

/// Minimum-norm solution of A w = rhs with A [m x n] (row-major, full row rank).
fn min_norm(a: &[f64], m: usize, n: usize, rhs: &[f64]) -> Vec<f64> {
    // (A A^T) y = rhs, w = A^T y  (Gaussian elimination with partial pivoting)
    let mut g = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            g[i * m + j] = (0..n).map(|k| a[i * n + k] * a[j * n + k]).sum();
        }
    }
    let mut y = rhs.to_vec();
    for col in 0..m {
        let piv = (col..m).max_by(|&i, &j| g[i * m + col].abs().partial_cmp(&g[j * m + col].abs()).unwrap()).unwrap();
        if piv != col {
            for j in 0..m {
                g.swap(col * m + j, piv * m + j);
            }
            y.swap(col, piv);
        }
        for i in col + 1..m {
            let f = g[i * m + col] / g[col * m + col];
            for j in col..m {
                g[i * m + j] -= f * g[col * m + j];
            }
            y[i] -= f * y[col];
        }
    }
    for i in (0..m).rev() {
        let s: f64 = (i + 1..m).map(|j| g[i * m + j] * y[j]).sum();
        y[i] = (y[i] - s) / g[i * m + i];
    }
    (0..n).map(|k| (0..m).map(|i| a[i * n + k] * y[i]).sum()).collect()
}

/// Builds the four WQ rules for a uniform open knot vector of degree `p >= 1`.
pub fn wq_rules_1d(knots: &[f64], p: usize) -> WqRules1d {
    assert!(p >= 1);
    let mut brk: Vec<f64> = Vec::new();
    for &k in knots {
        if brk.last().map_or(true, |&b| b != k) {
            brk.push(k);
        }
    }
    let nb = brk.len();
    let mut pts = linspace(brk[0], brk[1], p + 2);
    for k in 2..nb.saturating_sub(2) {
        pts.push(brk[k]);
    }
    for k in 1..nb.saturating_sub(2) {
        pts.push(0.5 * (brk[k] + brk[k + 1]));
    }
    pts.extend(linspace(brk[nb - 2], brk[nb - 1], p + 2));
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pts.dedup();
    let npt = pts.len();
    let ndof = knots.len() - p - 1;
    let (b, db) = basis_dense(knots, p, &pts);
    let sp = Space1d::new(knots.to_vec(), p, p + 1);
    let kd: Vec<f64> = knots[1..knots.len() - 1].to_vec();
    let spd = Space1d::new(kd.clone(), p - 1, p + 1);
    let ndof_d = kd.len() - p;
    let (bd, _) = basis_dense(&kd, p - 1, &pts);

    let mut ind_points = Vec::with_capacity(ndof);
    let mut neighbors = Vec::with_capacity(ndof);
    let mut w: [Vec<Vec<f64>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for i in 0..ndof {
        let ind: Vec<usize> = (0..npt).filter(|&q| b[q * ndof + i] != 0.0).collect();
        let els = &sp.supp[i];
        let mut nb_p: Vec<usize> = els.iter().flat_map(|&e| sp.connectivity[e * (p + 1)..(e + 1) * (p + 1)].iter().copied()).collect();
        nb_p.sort_unstable();
        nb_p.dedup();
        let mut nb_d: Vec<usize> = els.iter().flat_map(|&e| spd.connectivity[e * p..(e + 1) * p].iter().copied()).collect();
        nb_d.sort_unstable();
        nb_d.dedup();
        // Gauss data on supp(B_i)
        let xg: Vec<f64> = els.iter().flat_map(|&e| sp.qn[e * sp.nquad..(e + 1) * sp.nquad].iter().copied()).collect();
        let wg: Vec<f64> = els.iter().flat_map(|&e| sp.qw[e * sp.nquad..(e + 1) * sp.nquad].iter().copied()).collect();
        let (bg, dbg) = basis_dense(knots, p, &xg);
        let (bdg, _) = basis_dense(&kd, p - 1, &xg);
        let nq = ind.len();
        let mat = |vals: &[f64], nd: usize, nbs: &[usize]| -> Vec<f64> {
            let mut a = vec![0.0; nbs.len() * nq];
            for (r, &j) in nbs.iter().enumerate() {
                for (c, &q) in ind.iter().enumerate() {
                    a[r * nq + c] = vals[q * nd + j];
                }
            }
            a
        };
        let ap = mat(&b, ndof, &nb_p);
        let ad = mat(&bd, ndof_d, &nb_d);
        let rhs = |trial: &[f64], nd: usize, nbs: &[usize], test: &[f64]| -> Vec<f64> {
            nbs.iter().map(|&j| (0..xg.len()).map(|g| trial[g * nd + j] * wg[g] * test[g * ndof + i]).sum()).collect()
        };
        w[0].push(min_norm(&ap, nb_p.len(), nq, &rhs(&bg, ndof, &nb_p, &bg)));
        w[1].push(min_norm(&ap, nb_p.len(), nq, &rhs(&bg, ndof, &nb_p, &dbg)));
        w[2].push(min_norm(&ad, nb_d.len(), nq, &rhs(&bdg, ndof_d, &nb_d, &bg)));
        w[3].push(min_norm(&ad, nb_d.len(), nq, &rhs(&bdg, ndof_d, &nb_d, &dbg)));
        ind_points.push(ind);
        neighbors.push(nb_p);
    }
    WqRules1d { points: pts, ndof, ind_points, neighbors, weights: w, b, db }
}

/// Density field for WQ assembly.
pub enum Density<'a> {
    /// Uniform (no SIMP scaling).
    None,
    /// Element-wise densities, lexicographic (direction 0 fastest).
    Element(&'a [f64]),
    /// Spline control-point densities on the displacement space (direction 0 fastest).
    Spline(&'a [f64]),
}

/// WQ elasticity stiffness (component-blocked, symmetrized) with SIMP factor
/// `emin + rho^penal (1 - emin)` evaluated at the WQ points.
pub fn wq_stiffness(sp: &SpaceBox, young: f64, poisson: f64, density: Density, penal: f64, emin: f64) -> CsrMatrix {
    let dim = sp.dim;
    let rules: Vec<WqRules1d> = (0..dim).map(|d| wq_rules_1d(&sp.univ[d].knots, sp.degree[d])).collect();
    let npts: Vec<usize> = rules.iter().map(|r| r.points.len()).collect();
    let ntot: usize = npts.iter().product();

    // SIMP factor on the WQ grid (direction 0 fastest)
    let simp: Vec<f64> = match density {
        Density::None => vec![1.0; ntot],
        Density::Element(x) => {
            let bins: Vec<Vec<usize>> = (0..dim)
                .map(|d| {
                    let br = &sp.univ[d].breaks;
                    let ne = br.len() - 1;
                    rules[d].points.iter().map(|&q| (br.partition_point(|&b| b <= q).saturating_sub(1)).min(ne - 1)).collect()
                })
                .collect();
            (0..ntot)
                .map(|g| {
                    let s = crate::iga::unravel(g, &npts);
                    let mut e = 0usize;
                    let mut stride = 1usize;
                    for d in 0..dim {
                        e += bins[d][s[d]] * stride;
                        stride *= sp.nel_dir[d];
                    }
                    emin + x[e].powf(penal) * (1.0 - emin)
                })
                .collect()
        }
        Density::Spline(x) => {
            // rho(q) = sum_a prod_d N_{a_d}(q_d) x_a, evaluated by mode products
            let mut cur = x.to_vec();
            let mut sizes = sp.ndof_dir.clone();
            for d in 0..dim {
                let r = &rules[d];
                let (nd, nq) = (r.ndof, r.points.len());
                let mut next_sizes = sizes.clone();
                next_sizes[d] = nq;
                let total: usize = next_sizes.iter().product();
                let mut next = vec![0.0; total];
                for idx in 0..total {
                    let s = crate::iga::unravel(idx, &next_sizes);
                    let mut acc = 0.0;
                    for a in 0..nd {
                        let mut src = s.clone();
                        src[d] = a;
                        let mut li = 0usize;
                        let mut stride = 1usize;
                        for dd in 0..dim {
                            li += src[dd] * stride;
                            stride *= sizes[dd];
                        }
                        acc += r.b[s[d] * nd + a] * cur[li];
                    }
                    next[idx] = acc;
                }
                cur = next;
                sizes = next_sizes;
            }
            cur.iter().map(|&rho| emin + rho.powf(penal) * (1.0 - emin)).collect()
        }
    };

    // constant parametric elasticity coefficients c[i][j][k1][k2]
    let lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu = young / (2.0 * (1.0 + poisson));
    let det: f64 = sp.lengths.iter().product();
    let coef = |i: usize, j: usize, k1: usize, k2: usize| {
        let dl = |a: usize, b: usize| (a == b) as u8 as f64;
        det / (sp.lengths[k1] * sp.lengths[k2]) * (lam * dl(i, k1) * dl(j, k2) + mu * (dl(i, j) * dl(k1, k2) + dl(i, k2) * dl(j, k1)))
    };

    let nsc = sp.ndof_sc;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for ii in 0..nsc {
        let ind = crate::iga::unravel(ii, &sp.ndof_dir);
        let pts: Vec<&Vec<usize>> = (0..dim).map(|d| &rules[d].ind_points[ind[d]]).collect();
        let nbs: Vec<&Vec<usize>> = (0..dim).map(|d| &rules[d].neighbors[ind[d]]).collect();
        let nqd: Vec<usize> = pts.iter().map(|p| p.len()).collect();
        let nnd: Vec<usize> = nbs.iter().map(|n| n.len()).collect();
        let nloc: usize = nnd.iter().product();
        // local SIMP factor on the point grid of this row
        let nql: usize = nqd.iter().product();
        let s_loc: Vec<f64> = (0..nql)
            .map(|g| {
                let s = crate::iga::unravel(g, &nqd);
                let mut gi = 0usize;
                let mut stride = 1usize;
                for d in 0..dim {
                    gi += pts[d][s[d]] * stride;
                    stride *= npts[d];
                }
                simp[gi]
            })
            .collect();
        let mut blocks = vec![vec![0.0; nloc]; dim * dim];
        for k1 in 0..dim {
            for k2 in 0..dim {
                // contraction matrices per direction: B[d] is [nn_d x nq_d]
                let bm: Vec<Vec<f64>> = (0..dim)
                    .map(|d| {
                        let r = &rules[d];
                        let (t, deriv) = if k1 == k2 && d == k1 {
                            (3, true)
                        } else if k1 != k2 && d == k1 {
                            (1, false)
                        } else if k1 != k2 && d == k2 {
                            (2, true)
                        } else {
                            (0, false)
                        };
                        let wts = &r.weights[t][ind[d]];
                        let vals = if deriv { &r.db } else { &r.b };
                        let mut m = vec![0.0; nnd[d] * nqd[d]];
                        for (jr, &j) in nbs[d].iter().enumerate() {
                            for (qc, &q) in pts[d].iter().enumerate() {
                                m[jr * nqd[d] + qc] = wts[qc] * vals[q * r.ndof + j];
                            }
                        }
                        m
                    })
                    .collect();
                // R = (B_0 x B_1 x B_2) applied to s_loc, mode by mode
                let mut cur = s_loc.clone();
                let mut sizes = nqd.clone();
                for d in 0..dim {
                    let mut next_sizes = sizes.clone();
                    next_sizes[d] = nnd[d];
                    let total: usize = next_sizes.iter().product();
                    let mut next = vec![0.0; total];
                    for idx in 0..total {
                        let s = crate::iga::unravel(idx, &next_sizes);
                        let mut acc = 0.0;
                        for q in 0..nqd[d] {
                            let mut src = s.clone();
                            src[d] = q;
                            let mut li = 0usize;
                            let mut stride = 1usize;
                            for dd in 0..dim {
                                li += src[dd] * stride;
                                stride *= sizes[dd];
                            }
                            acc += bm[d][s[d] * nqd[d] + q] * cur[li];
                        }
                        next[idx] = acc;
                    }
                    cur = next;
                    sizes = next_sizes;
                }
                for i in 0..dim {
                    for j in 0..dim {
                        let c = coef(i, j, k1, k2);
                        if c != 0.0 {
                            for (bv, rv) in blocks[i * dim + j].iter_mut().zip(&cur) {
                                *bv += c * rv;
                            }
                        }
                    }
                }
            }
        }
        for loc in 0..nloc {
            let s = crate::iga::unravel(loc, &nnd);
            let mut col = 0usize;
            let mut stride = 1usize;
            for d in 0..dim {
                col += nbs[d][s[d]] * stride;
                stride *= sp.ndof_dir[d];
            }
            for i in 0..dim {
                for j in 0..dim {
                    let v = blocks[i * dim + j][loc];
                    // symmetrize: 0.5 (K + K^T)
                    rr.push(i * nsc + ii);
                    cc.push(j * nsc + col);
                    vv.push(0.5 * v);
                    rr.push(j * nsc + col);
                    cc.push(i * nsc + ii);
                    vv.push(0.5 * v);
                }
            }
        }
    }
    CsrMatrix::from_triplets(sp.ndof, sp.ndof, &rr, &cc, &vv)
}
