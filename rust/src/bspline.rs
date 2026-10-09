//! In-house B-Spline Basis Evaluation (Cox-de Boor algorithm).
//! Zero external dependencies.

/// Evaluates 1D B-spline basis functions of given `degree` at parametric points `u`.
/// Returns a row-major 2D vector where `result[k][i]` is $N_{i, p}(u_k)$.
pub fn evaluate_bspline_basis_1d(degree: usize, knots: &[f64], u: &[f64]) -> Vec<Vec<f64>> {
    let nu = u.len();
    let nk = knots.len();
    let p = degree;
    assert!(nk > p + 1, "Knot vector length must exceed degree + 1");
    let ncp = nk - p - 1;

    // Degree 0 initialization
    let mut n_basis: Vec<Vec<f64>> = vec![vec![0.0; nk - 1]; nu];
    let last_knot = knots[nk - 1];
    for k in 0..nu {
        let uk = u[k];
        for i in 0..(nk - 1) {
            let left = knots[i];
            let right = knots[i + 1];
            if left >= right {
                continue; // Zero-length knot span
            }
            if (uk - last_knot).abs() < 1e-14 {
                // At right boundary, assign to last non-empty span
                if (right - last_knot).abs() < 1e-14 {
                    n_basis[k][i] = 1.0;
                    break;
                }
            } else if uk >= left && uk < right {
                n_basis[k][i] = 1.0;
                break;
            }
        }
    }

    // Cox-de Boor recursion
    for d in 1..=p {
        let mut n_next: Vec<Vec<f64>> = vec![vec![0.0; nk - d - 1]; nu];
        for k in 0..nu {
            let uk = u[k];
            for i in 0..(nk - d - 1) {
                let denom1 = knots[i + d] - knots[i];
                let denom2 = knots[i + d + 1] - knots[i + 1];

                let term1 = if denom1.abs() > 1e-15 {
                    ((uk - knots[i]) / denom1) * n_basis[k][i]
                } else {
                    0.0
                };

                let term2 = if denom2.abs() > 1e-15 {
                    ((knots[i + d + 1] - uk) / denom2) * n_basis[k][i + 1]
                } else {
                    0.0
                };

                n_next[k][i] = term1 + term2;
            }
        }
        n_basis = n_next;
    }

    // Slice to ncp basis functions
    let mut result = vec![vec![0.0; ncp]; nu];
    for k in 0..nu {
        for i in 0..ncp {
            result[k][i] = n_basis[k][i];
        }
    }
    result
}

/// Generates standard open knot vector for uniform B-splines on [0, 1].
pub fn open_knot_vector(num_elements: usize, degree: usize) -> Vec<f64> {
    let mut knots = Vec::with_capacity(num_elements + 2 * degree + 1);
    for _ in 0..=degree {
        knots.push(0.0);
    }
    let nel_f = num_elements as f64;
    for i in 1..num_elements {
        knots.push((i as f64) / nel_f);
    }
    for _ in 0..=degree {
        knots.push(1.0);
    }
    knots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_of_unity() {
        let degree = 3; // Cubic
        let nel = 4;
        let knots = open_knot_vector(nel, degree);
        let u = vec![0.0, 0.125, 0.25, 0.5, 0.75, 1.0];
        let basis = evaluate_bspline_basis_1d(degree, &knots, &u);

        for (k, row) in basis.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "Partition of unity failed at u={}", u[k]);
        }
    }
}
