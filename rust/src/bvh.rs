//! Dual-Layer Bounding Volume Hierarchy (BVH) & Spatial Ray Peeling.
//! Pillar 2: Broad-phase acceleration structure for instant visible pick, ray peeling through-pick, and contact queries.

use crate::cut_cell::TriangleMesh3D;

/// Axis-Aligned 3D Bounding Box
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Aabb3D {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb3D {
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    pub fn from_points(pts: &[[f64; 3]]) -> Self {
        if pts.is_empty() {
            return Self { min: [0.0; 3], max: [0.0; 3] };
        }
        let mut min = pts[0];
        let mut max = pts[0];
        for p in pts.iter().skip(1) {
            for k in 0..3 {
                if p[k] < min[k] { min[k] = p[k]; }
                if p[k] > max[k] { max[k] = p[k]; }
            }
        }
        Self { min, max }
    }

    pub fn expanded(&self, clearance: f64) -> Self {
        Self {
            min: [self.min[0] - clearance, self.min[1] - clearance, self.min[2] - clearance],
            max: [self.max[0] + clearance, self.max[1] + clearance, self.max[2] + clearance],
        }
    }

    pub fn intersects(&self, other: &Aabb3D) -> bool {
        for k in 0..3 {
            if self.max[k] < other.min[k] || self.min[k] > other.max[k] {
                return false;
            }
        }
        true
    }

    /// Fast slab test for ray-AABB intersection
    pub fn ray_intersect(&self, orig: [f64; 3], dir: [f64; 3]) -> Option<(f64, f64)> {
        let mut tmin = f64::NEG_INFINITY;
        let mut tmax = f64::INFINITY;

        for k in 0..3 {
            if dir[k].abs() < 1e-12 {
                if orig[k] < self.min[k] || orig[k] > self.max[k] {
                    return None;
                }
            } else {
                let inv_d = 1.0 / dir[k];
                let mut t0 = (self.min[k] - orig[k]) * inv_d;
                let mut t1 = (self.max[k] - orig[k]) * inv_d;
                if inv_d < 0.0 {
                    std::mem::swap(&mut t0, &mut t1);
                }
                tmin = tmin.max(t0);
                tmax = tmax.min(t1);
                if tmax < tmin {
                    return None;
                }
            }
        }

        if tmax >= 0.0 {
            Some((tmin.max(0.0), tmax))
        } else {
            None
        }
    }
}

/// Ray-intersection hit record along the ray parameter line
#[derive(Clone, Debug, PartialEq)]
pub struct RayHit {
    pub t: f64,
    pub point: [f64; 3],
    pub normal: [f64; 3],
    pub primitive_index: usize,
}

/// Node in the linear BVH tree
#[derive(Clone, Debug)]
pub struct BvhNode {
    pub bbox: Aabb3D,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub primitive_range: (usize, usize), // start..end in primitive indices
}

/// Linear Bounding Volume Hierarchy over a TriangleMesh3D
#[derive(Clone, Debug)]
pub struct LinearBvh {
    pub nodes: Vec<BvhNode>,
    pub prim_indices: Vec<usize>,
}

impl LinearBvh {
    /// Builds a BVH over triangles using median splits along the longest AABB axis
    pub fn build(mesh: &TriangleMesh3D) -> Self {
        let n = mesh.triangles.len();
        let mut prim_indices: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::new();

        if n > 0 {
            let mut centroids = Vec::with_capacity(n);
            let mut bboxes = Vec::with_capacity(n);

            for tri in &mesh.triangles {
                let p0 = mesh.vertices[tri[0]];
                let p1 = mesh.vertices[tri[1]];
                let p2 = mesh.vertices[tri[2]];
                let c = [(p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0, (p0[2] + p1[2] + p2[2]) / 3.0];
                let b = Aabb3D::from_points(&[p0, p1, p2]);
                centroids.push(c);
                bboxes.push(b);
            }

            Self::build_recursive(&mut nodes, &mut prim_indices, &centroids, &bboxes, 0, n);
        }

        Self { nodes, prim_indices }
    }

    fn build_recursive(
        nodes: &mut Vec<BvhNode>,
        prim_indices: &mut [usize],
        centroids: &[[f64; 3]],
        bboxes: &[Aabb3D],
        start: usize,
        end: usize,
    ) -> usize {
        let count = end - start;
        let mut global_min = [f64::MAX; 3];
        let mut global_max = [f64::MIN; 3];

        for &idx in &prim_indices[start..end] {
            let b = &bboxes[idx];
            for k in 0..3 {
                if b.min[k] < global_min[k] { global_min[k] = b.min[k]; }
                if b.max[k] > global_max[k] { global_max[k] = b.max[k]; }
            }
        }
        let node_bbox = Aabb3D::new(global_min, global_max);

        let node_idx = nodes.len();
        nodes.push(BvhNode {
            bbox: node_bbox,
            left_child: None,
            right_child: None,
            primitive_range: (start, end),
        });

        // Leaf threshold: 4 primitives
        if count <= 4 {
            return node_idx;
        }

        // Find longest axis of current bounding box
        let span = [
            global_max[0] - global_min[0],
            global_max[1] - global_min[1],
            global_max[2] - global_min[2],
        ];
        let axis = if span[0] >= span[1] && span[0] >= span[2] {
            0
        } else if span[1] >= span[0] && span[1] >= span[2] {
            1
        } else {
            2
        };

        // Median split
        let mid = start + count / 2;
        prim_indices[start..end].sort_by(|&a, &b| {
            centroids[a][axis].partial_cmp(&centroids[b][axis]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let left = Self::build_recursive(nodes, prim_indices, centroids, bboxes, start, mid);
        let right = Self::build_recursive(nodes, prim_indices, centroids, bboxes, mid, end);

        nodes[node_idx].left_child = Some(left);
        nodes[node_idx].right_child = Some(right);

        node_idx
    }

    /// Ray peeling / through-pick: collects ALL intersections along ray line sorted by t
    pub fn ray_peel(&self, orig: [f64; 3], dir: [f64; 3], mesh: &TriangleMesh3D) -> Vec<RayHit> {
        let mut hits = Vec::new();
        if self.nodes.is_empty() {
            return hits;
        }

        let mut stack = vec![0];
        while let Some(curr) = stack.pop() {
            let node = &self.nodes[curr];
            if node.bbox.ray_intersect(orig, dir).is_none() {
                continue;
            }

            if let (Some(left), Some(right)) = (node.left_child, node.right_child) {
                stack.push(right);
                stack.push(left);
            } else {
                // Leaf node: test primitives
                let (start, end) = node.primitive_range;
                for i in start..end {
                    let ti = self.prim_indices[i];
                    let tri = mesh.triangles[ti];
                    let v0 = mesh.vertices[tri[0]];
                    let v1 = mesh.vertices[tri[1]];
                    let v2 = mesh.vertices[tri[2]];

                    if let Some(t) = moller_trumbore(orig, dir, v0, v1, v2) {
                        let pt = [orig[0] + t * dir[0], orig[1] + t * dir[1], orig[2] + t * dir[2]];
                        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                        let norm = [
                            e1[1] * e2[2] - e1[2] * e2[1],
                            e1[2] * e2[0] - e1[0] * e2[2],
                            e1[0] * e2[1] - e1[1] * e2[0],
                        ];
                        let nlen = (norm[0].powi(2) + norm[1].powi(2) + norm[2].powi(2)).sqrt().max(1e-12);
                        hits.push(RayHit {
                            t,
                            point: pt,
                            normal: [norm[0] / nlen, norm[1] / nlen, norm[2] / nlen],
                            primitive_index: ti,
                        });
                    }
                }
            }
        }

        // Sort by ray parameter t (closest to furthest)
        hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }

    /// Queries the BVH for all primitives whose AABBs intersect a given expanded box (for contact candidate search)
    pub fn query_clearance_overlap(&self, query_box: &Aabb3D) -> Vec<usize> {
        let mut results = Vec::new();
        if self.nodes.is_empty() {
            return results;
        }

        let mut stack = vec![0];
        while let Some(curr) = stack.pop() {
            let node = &self.nodes[curr];
            if !node.bbox.intersects(query_box) {
                continue;
            }

            if let (Some(left), Some(right)) = (node.left_child, node.right_child) {
                stack.push(right);
                stack.push(left);
            } else {
                let (start, end) = node.primitive_range;
                for i in start..end {
                    results.push(self.prim_indices[i]);
                }
            }
        }
        results
    }
}

fn moller_trumbore(orig: [f64; 3], dir: [f64; 3], v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> Option<f64> {
    let edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    let pvec = [
        dir[1] * edge2[2] - dir[2] * edge2[1],
        dir[2] * edge2[0] - dir[0] * edge2[2],
        dir[0] * edge2[1] - dir[1] * edge2[0],
    ];

    let det = edge1[0] * pvec[0] + edge1[1] * pvec[1] + edge1[2] * pvec[2];
    if det.abs() < 1e-10 {
        return None;
    }

    let inv_det = 1.0 / det;
    let tvec = [orig[0] - v0[0], orig[1] - v0[1], orig[2] - v0[2]];
    let u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det;
    if u < -1e-4 || u > 1.0001 {
        return None;
    }

    let qvec = [
        tvec[1] * edge1[2] - tvec[2] * edge1[1],
        tvec[2] * edge1[0] - tvec[0] * edge1[2],
        tvec[0] * edge1[1] - tvec[1] * edge1[0],
    ];

    let v = (dir[0] * qvec[0] + dir[1] * qvec[1] + dir[2] * qvec[2]) * inv_det;
    if v < -1e-4 || u + v > 1.0001 {
        return None;
    }

    let t = (edge2[0] * qvec[0] + edge2[1] * qvec[1] + edge2[2] * qvec[2]) * inv_det;
    if t >= 0.0 {
        Some(t)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bvh_ray_peel_through_pick() {
        let mesh = TriangleMesh3D::new_box([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let bvh = LinearBvh::build(&mesh);

        // Ray passing straight through the box from (-1, 1, 1) to (+3, 1, 1)
        let hits = bvh.ray_peel([-1.0, 1.0, 1.0], [1.0, 0.0, 0.0], &mesh);
        assert!(!hits.is_empty());
        // Must hit front face (t ~ 1.0) and back face (t ~ 3.0)
        assert!(hits.len() >= 2);
        assert!(hits.first().unwrap().t <= hits.last().unwrap().t);
        assert!((hits.first().unwrap().t - 1.0).abs() < 1e-3);
        assert!((hits.last().unwrap().t - 3.0).abs() < 1e-3);
    }
}
