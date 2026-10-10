//! Quadrature points, element shape functions, load vectors and field evaluation,
//! mirroring `src/iga/iga_quad_points.m`, `iga_shape_functions.m`,
//! `iga_load_vector.m` and `iga_eval_scalar_field.m`.

use super::space::{unravel, SpaceBox};

/// Physical Gauss points and weights of one element. Points are lexicographic
/// (direction 0 fastest). Returns `(x [nqn][dim], w [nqn])`.
pub fn element_quad_points(sp: &SpaceBox, e: usize) -> (Vec<[f64; 3]>, Vec<f64>) {
    let esub = unravel(e, &sp.nel_dir);
    let mut x = Vec::with_capacity(sp.nqn);
    let mut w = Vec::with_capacity(sp.nqn);
    for q in 0..sp.nqn {
        let qsub = unravel(q, &sp.nquad);
        let mut xq = [0.0; 3];
        let mut wq = 1.0;
        for d in 0..sp.dim {
            let u = &sp.univ[d];
            let k = esub[d] * u.nquad + qsub[d];
            xq[d] = sp.lo[d] + sp.lengths[d] * u.qn[k];
            wq *= sp.lengths[d] * u.qw[k];
        }
        x.push(xq);
        w.push(wq);
    }
    (x, w)
}

/// Scalar shape values `[q * nsh_sc + a]` and physical gradients
/// `[(q * nsh_sc + a) * dim + k]` of element `e` at its Gauss points.
pub fn element_shape_functions(sp: &SpaceBox, e: usize) -> (Vec<f64>, Vec<f64>) {
    let dim = sp.dim;
    let esub = unravel(e, &sp.nel_dir);
    let mut n = vec![1.0; sp.nqn * sp.nsh_sc];
    let mut dn = vec![1.0; sp.nqn * sp.nsh_sc * dim];
    for q in 0..sp.nqn {
        let qsub = unravel(q, &sp.nquad);
        for a in 0..sp.nsh_sc {
            let asub = unravel(a, &sp.nsh_dir);
            for d in 0..dim {
                let u = &sp.univ[d];
                let idx = (esub[d] * u.nquad + qsub[d]) * u.nsh() + asub[d];
                let v = u.shape[idx];
                let dv = u.dshape[idx] / sp.lengths[d];
                n[q * sp.nsh_sc + a] *= v;
                for k in 0..dim {
                    dn[(q * sp.nsh_sc + a) * dim + k] *= if k == d { dv } else { v };
                }
            }
        }
    }
    (n, dn)
}

/// Consistent body-force vector F_(A,c) = int f_c phi_A for `f(x) -> [f_0, f_1, f_2]`.
pub fn load_vector<F>(sp: &SpaceBox, f: F) -> Vec<f64>
where
    F: Fn(&[f64; 3]) -> [f64; 3],
{
    let mut out = vec![0.0; sp.ndof];
    for e in 0..sp.nel {
        let (x, w) = element_quad_points(sp, e);
        let (n, _) = element_shape_functions(sp, e);
        for q in 0..sp.nqn {
            let fq = f(&x[q]);
            for a in 0..sp.nsh_sc {
                let g = sp.conn_sc[e * sp.nsh_sc + a];
                let wn = w[q] * n[q * sp.nsh_sc + a];
                for c in 0..sp.dim {
                    out[c * sp.ndof_sc + g] += wn * fq[c];
                }
            }
        }
    }
    out
}

/// Values of a vector field (component-blocked DOF vector `u`) at the Gauss points
/// of element `e`: `[q][c]`.
pub fn element_vector_values(sp: &SpaceBox, u: &[f64], e: usize) -> Vec<[f64; 3]> {
    let (n, _) = element_shape_functions(sp, e);
    (0..sp.nqn)
        .map(|q| {
            let mut v = [0.0; 3];
            for a in 0..sp.nsh_sc {
                let g = sp.conn_sc[e * sp.nsh_sc + a];
                for c in 0..sp.dim {
                    v[c] += n[q * sp.nsh_sc + a] * u[c * sp.ndof_sc + g];
                }
            }
            v
        })
        .collect()
}

/// Displacement gradients at the Gauss points of element `e`: `[q][c][k]` = d u_c / d x_k.
pub fn element_vector_gradients(sp: &SpaceBox, u: &[f64], e: usize) -> Vec<[[f64; 3]; 3]> {
    let dim = sp.dim;
    let (_, dn) = element_shape_functions(sp, e);
    (0..sp.nqn)
        .map(|q| {
            let mut g = [[0.0; 3]; 3];
            for a in 0..sp.nsh_sc {
                let gi = sp.conn_sc[e * sp.nsh_sc + a];
                for c in 0..dim {
                    let uc = u[c * sp.ndof_sc + gi];
                    for k in 0..dim {
                        g[c][k] += dn[(q * sp.nsh_sc + a) * dim + k] * uc;
                    }
                }
            }
            g
        })
        .collect()
}

/// Evaluate a vector field at an arbitrary physical point inside the box.
pub fn eval_vector_at(sp: &SpaceBox, u: &[f64], x: &[f64; 3]) -> [f64; 3] {
    let dim = sp.dim;
    let mut first = vec![0usize; dim];
    let mut vals: Vec<Vec<f64>> = Vec::with_capacity(dim);
    for d in 0..dim {
        let u1 = &sp.univ[d];
        let xi = ((x[d] - sp.lo[d]) / sp.lengths[d]).clamp(0.0, 1.0);
        let (f, v, _) = super::basis::basis_funs(&u1.knots, u1.degree, xi);
        first[d] = f;
        vals.push(v);
    }
    let mut out = [0.0; 3];
    for a in 0..sp.nsh_sc {
        let asub = unravel(a, &sp.nsh_dir);
        let mut w = 1.0;
        let mut g = 0usize;
        let mut stride = 1usize;
        for d in 0..dim {
            w *= vals[d][asub[d]];
            g += (first[d] + asub[d]) * stride;
            stride *= sp.ndof_dir[d];
        }
        for c in 0..dim {
            out[c] += w * u[c * sp.ndof_sc + g];
        }
    }
    out
}
