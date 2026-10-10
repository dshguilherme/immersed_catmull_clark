//! Compressed sparse row matrices and element-wise assembly.

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
}

impl CsrMatrix {
    /// Builds a CSR matrix from (row, col, value) triplets, summing duplicates.
    pub fn from_triplets(nrows: usize, ncols: usize, rows: &[usize], cols: &[usize], vals: &[f64]) -> Self {
        assert!(rows.len() == cols.len() && rows.len() == vals.len());
        let mut count = vec![0usize; nrows + 1];
        for &r in rows {
            count[r + 1] += 1;
        }
        for i in 0..nrows {
            count[i + 1] += count[i];
        }
        let mut next = count.clone();
        let mut ci = vec![0usize; rows.len()];
        let mut cv = vec![0.0; rows.len()];
        for k in 0..rows.len() {
            let r = rows[k];
            ci[next[r]] = cols[k];
            cv[next[r]] = vals[k];
            next[r] += 1;
        }
        // sort each row by column and merge duplicates
        let mut indptr = vec![0usize; nrows + 1];
        let mut indices = Vec::with_capacity(rows.len());
        let mut data = Vec::with_capacity(rows.len());
        let mut perm: Vec<usize> = Vec::new();
        for r in 0..nrows {
            let (s, e) = (count[r], count[r + 1]);
            perm.clear();
            perm.extend(s..e);
            perm.sort_unstable_by_key(|&k| ci[k]);
            let mut last: Option<usize> = None;
            for &k in &perm {
                if last == Some(ci[k]) {
                    *data.last_mut().unwrap() += cv[k];
                } else {
                    indices.push(ci[k]);
                    data.push(cv[k]);
                    last = Some(ci[k]);
                }
            }
            indptr[r + 1] = indices.len();
        }
        Self { nrows, ncols, indptr, indices, data }
    }

    /// y = A x (parallel over rows).
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        y.par_iter_mut().enumerate().for_each(|(r, yr)| {
            let mut s = 0.0;
            for k in self.indptr[r]..self.indptr[r + 1] {
                s += self.data[k] * x[self.indices[k]];
            }
            *yr = s;
        });
    }

    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.nrows)
            .map(|r| {
                (self.indptr[r]..self.indptr[r + 1])
                    .find(|&k| self.indices[k] == r)
                    .map_or(0.0, |k| self.data[k])
            })
            .collect()
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        let row = &self.indices[self.indptr[r]..self.indptr[r + 1]];
        match row.binary_search(&c) {
            Ok(k) => self.data[self.indptr[r] + k],
            Err(_) => 0.0,
        }
    }

    /// Frobenius norm.
    pub fn frobenius(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

/// Assembles `sum_e scale[e] * K_e` into a CSR matrix, where `elem(e)` returns the
/// row-major `[nsh x nsh]` element matrix with local ordering `conn[e*nsh..]`.
pub fn assemble<'a, F>(ndof: usize, conn: &[usize], nsh: usize, nel: usize, elem: F, scale: Option<&[f64]>) -> CsrMatrix
where
    F: Fn(usize) -> &'a [f64],
{
    let nn = nsh * nsh * nel;
    let mut rows = Vec::with_capacity(nn);
    let mut cols = Vec::with_capacity(nn);
    let mut vals = Vec::with_capacity(nn);
    for e in 0..nel {
        let ke = elem(e);
        let s = scale.map_or(1.0, |w| w[e]);
        let ce = &conn[e * nsh..(e + 1) * nsh];
        for a in 0..nsh {
            for b in 0..nsh {
                rows.push(ce[a]);
                cols.push(ce[b]);
                vals.push(s * ke[a * nsh + b]);
            }
        }
    }
    CsrMatrix::from_triplets(ndof, ndof, &rows, &cols, &vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triplets_merge_duplicates() {
        let a = CsrMatrix::from_triplets(2, 2, &[0, 1, 0, 0], &[1, 0, 1, 0], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.get(0, 1), 4.0);
        assert_eq!(a.get(0, 0), 4.0);
        assert_eq!(a.get(1, 0), 2.0);
        assert_eq!(a.get(1, 1), 0.0);
        let mut y = vec![0.0; 2];
        a.matvec(&[1.0, 1.0], &mut y);
        assert_eq!(y, vec![8.0, 2.0]);
    }
}
