//! Gauss-Legendre quadrature and uniform open knot vectors.

/// `n`-point Gauss-Legendre rule on [-1, 1], nodes in ascending order.
/// Exact for polynomials of degree <= 2n - 1.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 1, "gauss_legendre: n must be >= 1");
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        // Initial guess (Tricomi), then Newton on P_n.
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (p, d) = legendre_with_derivative(n, z);
            dp = d;
            let dz = p / d;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        let (_, d) = legendre_with_derivative(n, z);
        if d != 0.0 {
            dp = d;
        }
        x[i] = -z;
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
    (x, w)
}

/// Legendre polynomial P_n(z) and its derivative via the three-term recurrence.
fn legendre_with_derivative(n: usize, z: f64) -> (f64, f64) {
    let mut p0 = 1.0;
    let mut p1 = z;
    if n == 0 {
        return (1.0, 0.0);
    }
    for k in 2..=n {
        let kf = k as f64;
        let p2 = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf;
        p0 = p1;
        p1 = p2;
    }
    let dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
    (p1, dp)
}

/// Uniform open knot vector on [0, 1] with `nel` elements, end knots repeated
/// `degree + 1` times and interior knots repeated `degree - regularity` times.
pub fn open_knots(nel: usize, degree: usize, regularity: isize) -> Vec<f64> {
    assert!(nel >= 1, "open_knots: nel must be >= 1");
    assert!(regularity >= -1 && regularity < degree as isize, "open_knots: regularity must lie in [-1, degree-1]");
    let mult = (degree as isize - regularity) as usize;
    let mut knots = vec![0.0; degree + 1];
    for i in 1..nel {
        let b = i as f64 / nel as f64;
        for _ in 0..mult {
            knots.push(b);
        }
    }
    knots.extend(std::iter::repeat(1.0).take(degree + 1));
    knots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_rule_is_exact() {
        for n in 1..=10 {
            let (x, w) = gauss_legendre(n);
            for k in 0..(2 * n) {
                let exact = if k % 2 == 0 { 2.0 / (k as f64 + 1.0) } else { 0.0 };
                let approx: f64 = x.iter().zip(&w).map(|(xi, wi)| wi * xi.powi(k as i32)).sum();
                assert!((approx - exact).abs() < 1e-13, "n={} k={} err={}", n, k, approx - exact);
            }
            assert!(x.windows(2).all(|p| p[0] < p[1]));
        }
    }

    #[test]
    fn knot_vector_multiplicities() {
        let k = open_knots(3, 2, 0);
        assert_eq!(k, vec![0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0]);
        assert_eq!(open_knots(4, 3, 2).len(), 4 + 2 * 3 + 1);
    }
}
