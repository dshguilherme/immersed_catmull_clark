//! Univariate and tensor-product B-spline spaces on axis-aligned boxes, mirroring
//! `src/iga/iga_space_1d.m` and `src/iga/iga_space_box.m`.
//!
//! Numbering (identical to the MATLAB / GeoPDEs conventions, but 0-based):
//! - elements and scalar control points are lexicographic, direction 0 fastest;
//! - vector DOFs are blocked by component: `[u_0(:); u_1(:); u_2(:)]`;
//! - local element functions are lexicographic (direction 0 fastest), then
//!   blocked by component;
//! - boundary sides: `2d` is xi_d = 0 and `2d + 1` is xi_d = 1.

use super::basis::{basis_funs, num_basis};
use super::quadrature::{gauss_legendre, open_knots};

/// Univariate spline space with Gauss data on every element (parametric coordinates).
#[derive(Clone, Debug)]
pub struct Space1d {
    pub knots: Vec<f64>,
    pub degree: usize,
    pub ndof: usize,
    pub nel: usize,
    pub breaks: Vec<f64>,
    /// `connectivity[e * (p+1) + a]`: global index of local function `a` on element `e`.
    pub connectivity: Vec<usize>,
    /// Elements in the support of each function.
    pub supp: Vec<Vec<usize>>,
    pub nquad: usize,
    /// Gauss nodes and weights, `[nel x nquad]` row-major.
    pub qn: Vec<f64>,
    pub qw: Vec<f64>,
    /// Basis values / parametric derivatives, `[(e * nquad + q) * (p+1) + a]`.
    pub shape: Vec<f64>,
    pub dshape: Vec<f64>,
}

impl Space1d {
    pub fn new(knots: Vec<f64>, degree: usize, nquad: usize) -> Self {
        let p = degree;
        let mut breaks: Vec<f64> = Vec::new();
        for &k in &knots {
            if breaks.last().map_or(true, |&b| k != b) {
                breaks.push(k);
            }
        }
        let nel = breaks.len() - 1;
        let ndof = num_basis(&knots, p);
        let mut connectivity = Vec::with_capacity(nel * (p + 1));
        let mut supp = vec![Vec::new(); ndof];
        let (xg, wg) = gauss_legendre(nquad);
        let mut qn = Vec::with_capacity(nel * nquad);
        let mut qw = Vec::with_capacity(nel * nquad);
        let mut shape = Vec::with_capacity(nel * nquad * (p + 1));
        let mut dshape = Vec::with_capacity(nel * nquad * (p + 1));
        for e in 0..nel {
            let (a, b) = (breaks[e], breaks[e + 1]);
            let (first, _, _) = basis_funs(&knots, p, 0.5 * (a + b));
            for r in 0..=p {
                connectivity.push(first + r);
                supp[first + r].push(e);
            }
            let h = b - a;
            for q in 0..nquad {
                let x = a + 0.5 * (xg[q] + 1.0) * h;
                qn.push(x);
                qw.push(0.5 * wg[q] * h);
                let (f, v, d) = basis_funs(&knots, p, x);
                debug_assert_eq!(f, first);
                shape.extend_from_slice(&v);
                dshape.extend_from_slice(&d);
            }
        }
        Self { knots, degree, ndof, nel, breaks, connectivity, supp, nquad, qn, qw, shape, dshape }
    }

    #[inline]
    pub fn nsh(&self) -> usize {
        self.degree + 1
    }
}

/// Tensor-product B-spline vector space on an axis-aligned box (dim = 2 or 3).
/// The geometry map is affine: x_d = lo_d + L_d * xi_d with xi in [0, 1]^dim.
#[derive(Clone, Debug)]
pub struct SpaceBox {
    pub dim: usize,
    pub lo: Vec<f64>,
    pub lengths: Vec<f64>,
    pub nel_dir: Vec<usize>,
    pub nel: usize,
    pub degree: Vec<usize>,
    pub nquad: Vec<usize>,
    pub nqn: usize,
    pub univ: Vec<Space1d>,
    pub ndof_dir: Vec<usize>,
    pub ndof_sc: usize,
    pub ndof: usize,
    pub nsh_dir: Vec<usize>,
    pub nsh_sc: usize,
    pub nsh: usize,
    /// Scalar connectivity `[e * nsh_sc + a]`.
    pub conn_sc: Vec<usize>,
    /// Vector connectivity `[e * nsh + a]`, components blocked.
    pub connectivity: Vec<usize>,
}

impl SpaceBox {
    /// Maximal-regularity space (C^{p-1}) of uniform degree with p+1 Gauss points.
    pub fn new(bounds: &[[f64; 2]], nsub: &[usize], degree: usize) -> Self {
        let dim = bounds.len();
        let p = vec![degree; dim];
        let reg = vec![degree as isize - 1; dim];
        let nq = vec![degree + 1; dim];
        Self::with_params(bounds, nsub, &p, &reg, &nq)
    }

    pub fn with_params(bounds: &[[f64; 2]], nsub: &[usize], degree: &[usize], regularity: &[isize], nquad: &[usize]) -> Self {
        let dim = bounds.len();
        assert!(dim == 2 || dim == 3, "SpaceBox: dim must be 2 or 3");
        assert!(nsub.len() == dim && degree.len() == dim && regularity.len() == dim && nquad.len() == dim);
        let lo: Vec<f64> = bounds.iter().map(|b| b[0]).collect();
        let lengths: Vec<f64> = bounds.iter().map(|b| b[1] - b[0]).collect();
        let univ: Vec<Space1d> = (0..dim)
            .map(|d| Space1d::new(open_knots(nsub[d], degree[d], regularity[d]), degree[d], nquad[d]))
            .collect();
        let nel_dir: Vec<usize> = univ.iter().map(|u| u.nel).collect();
        let nel: usize = nel_dir.iter().product();
        let ndof_dir: Vec<usize> = univ.iter().map(|u| u.ndof).collect();
        let ndof_sc: usize = ndof_dir.iter().product();
        let nsh_dir: Vec<usize> = degree.iter().map(|p| p + 1).collect();
        let nsh_sc: usize = nsh_dir.iter().product();
        let nsh = dim * nsh_sc;
        let nqn: usize = nquad.iter().product();

        let mut conn_sc = Vec::with_capacity(nel * nsh_sc);
        for e in 0..nel {
            let esub = unravel(e, &nel_dir);
            for a in 0..nsh_sc {
                let asub = unravel(a, &nsh_dir);
                let mut g = 0usize;
                let mut stride = 1usize;
                for d in 0..dim {
                    let u = &univ[d];
                    g += u.connectivity[esub[d] * u.nsh() + asub[d]] * stride;
                    stride *= ndof_dir[d];
                }
                conn_sc.push(g);
            }
        }
        let mut connectivity = Vec::with_capacity(nel * nsh);
        for e in 0..nel {
            for c in 0..dim {
                for a in 0..nsh_sc {
                    connectivity.push(c * ndof_sc + conn_sc[e * nsh_sc + a]);
                }
            }
        }
        Self {
            dim, lo, lengths, nel_dir, nel, degree: degree.to_vec(), nquad: nquad.to_vec(), nqn, univ,
            ndof_dir, ndof_sc, ndof: dim * ndof_sc, nsh_dir, nsh_sc, nsh, conn_sc, connectivity,
        }
    }

    /// Element sizes per direction (physical).
    pub fn element_size(&self) -> Vec<f64> {
        (0..self.dim).map(|d| self.lengths[d] / self.nel_dir[d] as f64).collect()
    }

    /// Scalar control-point indices on boundary side `side` (0..2*dim).
    pub fn boundary_dofs_sc(&self, side: usize) -> Vec<usize> {
        let d = side / 2;
        let target = if side % 2 == 0 { 0 } else { self.ndof_dir[d] - 1 };
        (0..self.ndof_sc).filter(|&i| unravel(i, &self.ndof_dir)[d] == target).collect()
    }

    /// Vector DOFs (all components) on boundary side `side`.
    pub fn boundary_dofs(&self, side: usize) -> Vec<usize> {
        let sc = self.boundary_dofs_sc(side);
        (0..self.dim).flat_map(|c| sc.iter().map(move |&i| c * self.ndof_sc + i)).collect()
    }

    /// Greville abscissae of the scalar control points (physical), `[ndof_sc][dim]`.
    pub fn greville_points(&self) -> Vec<Vec<f64>> {
        let g1: Vec<Vec<f64>> = (0..self.dim)
            .map(|d| {
                let u = &self.univ[d];
                let p = u.degree;
                (0..u.ndof)
                    .map(|i| {
                        let m = if p == 0 { 0.5 * (u.knots[i] + u.knots[i + 1]) } else { u.knots[i + 1..=i + p].iter().sum::<f64>() / p as f64 };
                        self.lo[d] + self.lengths[d] * m
                    })
                    .collect()
            })
            .collect();
        (0..self.ndof_sc)
            .map(|i| {
                let s = unravel(i, &self.ndof_dir);
                (0..self.dim).map(|d| g1[d][s[d]]).collect()
            })
            .collect()
    }
}

/// Lexicographic (direction 0 fastest) multi-index of `idx` for sizes `dims`.
pub fn unravel(mut idx: usize, dims: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(dims.len());
    for &n in dims {
        out.push(idx % n);
        idx /= n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_connectivity() {
        let sp = SpaceBox::new(&[[0.0, 1.2], [0.0, 0.6], [0.0, 0.3]], &[4, 3, 2], 2);
        assert_eq!(sp.ndof_dir, vec![6, 5, 4]);
        assert_eq!(sp.ndof, 3 * 120);
        assert_eq!(sp.nsh, 81);
        // first element touches control point 0, last element the last one
        assert_eq!(sp.conn_sc[0], 0);
        assert_eq!(sp.conn_sc[sp.nel * sp.nsh_sc - 1], sp.ndof_sc - 1);
        assert_eq!(sp.boundary_dofs_sc(0).len(), 5 * 4);
        assert_eq!(sp.boundary_dofs(5).len(), 3 * 6 * 5);
    }
}
