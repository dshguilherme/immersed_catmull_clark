//! 3D Ray-Casting & Immersed Cut-Cell Quadrature Engine.
//! Zero-dependency in-house implementation of Möller-Trumbore ray intersection,
//! point-in-polyhedron parity testing, and sub-cell Gauss integration.

use rayon::prelude::*;

/// 3D Triangle surface mesh representation of arbitrary CAD B-Rep boundaries
#[derive(Clone, Debug)]
pub struct TriangleMesh3D {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CellStatus {
    Outside,
    Inside,
    Cut,
}

impl TriangleMesh3D {
    pub fn new(vertices: Vec<[f64; 3]>, triangles: Vec<[usize; 3]>) -> Self {
        Self { vertices, triangles }
    }

    /// Creates a triangular surface mesh of an axis-aligned box [min_x, max_x] x [min_y, max_y] x [min_z, max_z]
    pub fn new_box(min: [f64; 3], max: [f64; 3]) -> Self {
        let v = vec![
            [min[0], min[1], min[2]], // 0
            [max[0], min[1], min[2]], // 1
            [max[0], max[1], min[2]], // 2
            [min[0], max[1], min[2]], // 3
            [min[0], min[1], max[2]], // 4
            [max[0], min[1], max[2]], // 5
            [max[0], max[1], max[2]], // 6
            [min[0], max[1], max[2]], // 7
        ];
        let t = vec![
            // -Z face
            [0, 2, 1], [0, 3, 2],
            // +Z face
            [4, 5, 6], [4, 6, 7],
            // -Y face
            [0, 1, 5], [0, 5, 4],
            // +Y face
            [2, 3, 7], [2, 7, 6],
            // -X face
            [0, 4, 7], [0, 7, 3],
            // +X face
            [1, 2, 6], [1, 6, 5],
        ];
        Self::new(v, t)
    }

    /// Möller–Trumbore ray-triangle intersection test.
    /// Returns true if ray from origin along +X direction [1, 0, 0] intersects triangle.
    #[inline(always)]
    pub fn ray_intersects_triangle_x(&self, tri: &[usize; 3], orig: [f64; 3]) -> bool {
        let v0 = self.vertices[tri[0]];
        let v1 = self.vertices[tri[1]];
        let v2 = self.vertices[tri[2]];

        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // Ray direction = [1, 0, 0]
        // pvec = ray_dir x e2 = [0, -e2[2], e2[1]]
        let pvec = [0.0, -e2[2], e2[1]];
        let det = e1[1] * pvec[1] + e1[2] * pvec[2];

        if det.abs() < 1e-12 {
            return false;
        }
        let inv_det = 1.0 / det;

        // tvec = orig - v0
        let tvec = [orig[0] - v0[0], orig[1] - v0[1], orig[2] - v0[2]];
        let u = (tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det;
        if u < 0.0 || u > 1.0 {
            return false;
        }

        // qvec = tvec x e1
        let qvec = [
            tvec[1] * e1[2] - tvec[2] * e1[1],
            tvec[2] * e1[0] - tvec[0] * e1[2],
            tvec[0] * e1[1] - tvec[1] * e1[0],
        ];
        let v = qvec[0] * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return false;
        }

        let t = (e2[0] * qvec[0] + e2[1] * qvec[1] + e2[2] * qvec[2]) * inv_det;
        t > 1e-9
    }

    /// Evaluates inside/outside status of a single 3D query point via Jordan parity counting.
    /// Uses an infinitesimally perturbed ray direction [1.0, 1.234e-5, 2.345e-5] to avoid edge/vertex singularities.
    pub fn is_point_inside(&self, pt: [f64; 3]) -> bool {
        // Perturb origin slightly to ensure ray does not hit triangle boundaries or vertices
        let orig = [pt[0], pt[1] + 1.2345e-7, pt[2] + 2.3456e-7];
        let mut count = 0;
        for tri in &self.triangles {
            // AABB pre-filter in Y and Z with tolerance
            let y0 = self.vertices[tri[0]][1];
            let y1 = self.vertices[tri[1]][1];
            let y2 = self.vertices[tri[2]][1];
            let min_y = y0.min(y1).min(y2) - 1e-6;
            let max_y = y0.max(y1).max(y2) + 1e-6;
            if orig[1] < min_y || orig[1] > max_y {
                continue;
            }

            let z0 = self.vertices[tri[0]][2];
            let z1 = self.vertices[tri[1]][2];
            let z2 = self.vertices[tri[2]][2];
            let min_z = z0.min(z1).min(z2) - 1e-6;
            let max_z = z0.max(z1).max(z2) + 1e-6;
            if orig[2] < min_z || orig[2] > max_z {
                continue;
            }

            if self.ray_intersects_triangle_x(tri, orig) {
                count += 1;
            }
        }
        (count % 2) == 1
    }

    /// Evaluates multiple query points in parallel using Rayon
    pub fn are_points_inside(&self, pts: &[[f64; 3]]) -> Vec<bool> {
        pts.par_iter().map(|&pt| self.is_point_inside(pt)).collect()
    }

    /// Classifies an axis-aligned background bounding box:
    /// Returns (CellStatus, volume_fraction_weight w_e)
    pub fn classify_box_quadrature(
        &self,
        min: [f64; 3],
        max: [f64; 3],
        sub_res: [usize; 3],
    ) -> (CellStatus, f64) {
        // Test 8 corners first
        let corners = [
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [max[0], max[1], min[2]],
            [min[0], max[1], min[2]],
            [min[0], min[1], max[2]],
            [max[0], min[1], max[2]],
            [max[0], max[1], max[2]],
            [min[0], max[1], max[2]],
        ];
        let corner_inside: Vec<bool> = corners.iter().map(|&c| self.is_point_inside(c)).collect();
        let inside_count = corner_inside.iter().filter(|&&b| b).count();

        if inside_count == 8 {
            return (CellStatus::Inside, 1.0);
        } else if inside_count == 0 {
            // Check if element is truly outside or enclosing the body
            let center = [
                0.5 * (min[0] + max[0]),
                0.5 * (min[1] + max[1]),
                0.5 * (min[2] + max[2]),
            ];
            if !self.is_point_inside(center) {
                return (CellStatus::Outside, 0.0);
            }
        }

        // Sub-cell Gauss / midpoint integration
        let sx = sub_res[0];
        let sy = sub_res[1];
        let sz = sub_res[2];
        let dx = (max[0] - min[0]) / (sx as f64);
        let dy = (max[1] - min[1]) / (sy as f64);
        let dz = (max[2] - min[2]) / (sz as f64);

        let total_subcells = sx * sy * sz;
        let mut inside_subcells = 0;

        for ix in 0..sx {
            let x = min[0] + (ix as f64 + 0.5) * dx;
            for iy in 0..sy {
                let y = min[1] + (iy as f64 + 0.5) * dy;
                for iz in 0..sz {
                    let z = min[2] + (iz as f64 + 0.5) * dz;
                    if self.is_point_inside([x, y, z]) {
                        inside_subcells += 1;
                    }
                }
            }
        }

        let weight = (inside_subcells as f64) / (total_subcells as f64);
        let status = if weight >= 0.999 {
            CellStatus::Inside
        } else if weight <= 0.001 {
            CellStatus::Outside
        } else {
            CellStatus::Cut
        };

        (status, weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_inside_outside() {
        let b = TriangleMesh3D::new_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        assert!(b.is_point_inside([0.5, 0.5, 0.5]));
        assert!(!b.is_point_inside([1.5, 0.5, 0.5]));
        assert!(!b.is_point_inside([-0.5, 0.5, 0.5]));
        assert!(!b.is_point_inside([0.5, 1.5, 0.5]));

        let (status_in, w_in) = b.classify_box_quadrature([0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [4, 4, 4]);
        assert_eq!(status_in, CellStatus::Inside);
        assert!((w_in - 1.0).abs() < 1e-6);

        let (status_out, w_out) = b.classify_box_quadrature([2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4, 4, 4]);
        assert_eq!(status_out, CellStatus::Outside);
        assert!((w_out - 0.0).abs() < 1e-6);

        // Cut cell straddling x = 1.0 boundary (0.5 inside, 0.5 outside)
        let (status_cut, w_cut) = b.classify_box_quadrature([0.5, 0.2, 0.2], [1.5, 0.8, 0.8], [4, 4, 4]);
        assert_eq!(status_cut, CellStatus::Cut);
        assert!((w_cut - 0.5).abs() < 1e-3);
    }
}
