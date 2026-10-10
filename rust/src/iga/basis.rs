//! B-spline basis values and first derivatives (Cox-de Boor), mirroring
//! `src/iga/iga_bspline_basis.m`.

/// Number of basis functions of a knot vector.
pub fn num_basis(knots: &[f64], degree: usize) -> usize {
    knots.len() - degree - 1
}

/// Span index `s` (0-based) with `knots[s] <= x < knots[s+1]`, clamped to the
/// valid range `[degree, ndof-1]`; the right end point maps to the last span.
pub fn find_span(knots: &[f64], degree: usize, x: f64) -> usize {
    let ndof = num_basis(knots, degree);
    let count = knots.partition_point(|&k| k <= x);
    count.saturating_sub(1).clamp(degree, ndof - 1)
}

/// Nonzero basis functions at `x`: returns `(first, values, derivatives)` where
/// `values[r]` is N_{first + r}(x) for r in 0..=degree. Derivatives are with
/// respect to the knot parameter.
pub fn basis_funs(knots: &[f64], degree: usize, x: f64) -> (usize, Vec<f64>, Vec<f64>) {
    let p = degree;
    let s = find_span(knots, p, x);
    let mut b = vec![1.0];
    let mut b_prev: Vec<f64> = Vec::new();
    for k in 1..=p {
        if k == p {
            b_prev = b.clone();
        }
        let mut bn = vec![0.0; k + 1];
        for r in 0..=k {
            let i = s + r - k; // global index of the degree-k function
            let mut val = 0.0;
            if r > 0 {
                let d = knots[i + k] - knots[i];
                if d > 0.0 {
                    val += (x - knots[i]) / d * b[r - 1];
                }
            }
            if r < k {
                let d = knots[i + k + 1] - knots[i + 1];
                if d > 0.0 {
                    val += (knots[i + k + 1] - x) / d * b[r];
                }
            }
            bn[r] = val;
        }
        b = bn;
    }
    let mut db = vec![0.0; p + 1];
    if p > 0 {
        for r in 0..=p {
            let i = s + r - p;
            let mut val = 0.0;
            if r > 0 {
                let d = knots[i + p] - knots[i];
                if d > 0.0 {
                    val += b_prev[r - 1] / d;
                }
            }
            if r < p {
                let d = knots[i + p + 1] - knots[i + 1];
                if d > 0.0 {
                    val -= b_prev[r] / d;
                }
            }
            db[r] = p as f64 * val;
        }
    }
    (s - p, b, db)
}

/// Dense values and derivatives of all basis functions at the points `xs`:
/// row-major `[xs.len() x ndof]`.
pub fn basis_dense(knots: &[f64], degree: usize, xs: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let ndof = num_basis(knots, degree);
    let mut n = vec![0.0; xs.len() * ndof];
    let mut dn = vec![0.0; xs.len() * ndof];
    for (q, &x) in xs.iter().enumerate() {
        let (first, v, d) = basis_funs(knots, degree, x);
        for r in 0..=degree {
            n[q * ndof + first + r] = v[r];
            dn[q * ndof + first + r] = d[r];
        }
    }
    (n, dn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iga::quadrature::open_knots;

    #[test]
    fn partition_of_unity_and_derivatives() {
        let xs: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        for p in 1..=4usize {
            for reg in [p as isize - 1, 0] {
                let knots = open_knots(7, p, reg);
                let ndof = num_basis(&knots, p);
                let (n, dn) = basis_dense(&knots, p, &xs);
                for q in 0..xs.len() {
                    let s: f64 = n[q * ndof..(q + 1) * ndof].iter().sum();
                    let ds: f64 = dn[q * ndof..(q + 1) * ndof].iter().sum();
                    assert!((s - 1.0).abs() < 1e-13);
                    assert!(ds.abs() < 1e-10);
                }
                // central finite differences away from knots
                let h = 1e-6;
                for j in 0..14 {
                    let x = 0.013 + 0.07 * j as f64;
                    let (_, dx) = basis_dense(&knots, p, &[x]);
                    let (np, _) = basis_dense(&knots, p, &[x + h]);
                    let (nm, _) = basis_dense(&knots, p, &[x - h]);
                    for i in 0..ndof {
                        let fd = (np[i] - nm[i]) / (2.0 * h);
                        assert!((dx[i] - fd).abs() < 1e-6, "p={} x={} i={}", p, x, i);
                    }
                }
            }
        }
    }
}
