//! Automated Topology Pairing for Contact & Interfaces.
//! Pillar 5: Detects opposing contact interfaces and auto-designates Master/Slave surface pairs.

use crate::query::CadFaceInfo;
use crate::bvh::Aabb3D;

/// Detected contact interface pair between two CAD faces
#[derive(Clone, Debug, PartialEq)]
pub struct ContactInterfacePair {
    pub master_face_id: usize,
    pub slave_face_id: usize,
    pub clearance_distance: f64,
    pub normal_alignment: f64,
}

/// Automated contact pair search algorithm
pub struct ContactPairSearcher {
    pub clearance_tolerance: f64,
    pub opposing_angle_tolerance_deg: f64,
}

impl ContactPairSearcher {
    pub fn new(clearance_tolerance: f64, opposing_angle_tolerance_deg: f64) -> Self {
        Self {
            clearance_tolerance,
            opposing_angle_tolerance_deg,
        }
    }

    /// Searches candidate faces for contact interfaces
    pub fn find_contact_pairs(&self, faces: &[CadFaceInfo]) -> Vec<ContactInterfacePair> {
        let mut pairs = Vec::new();
        let cos_thresh = -(self.opposing_angle_tolerance_deg.to_radians()).cos();

        for i in 0..faces.len() {
            let f1 = &faces[i];
            let box1 = Aabb3D::new(f1.bbox_min, f1.bbox_max).expanded(self.clearance_tolerance);

            for j in (i + 1)..faces.len() {
                let f2 = &faces[j];
                let box2 = Aabb3D::new(f2.bbox_min, f2.bbox_max);

                // Broad phase: clearance AABB intersection
                if !box1.intersects(&box2) {
                    continue;
                }

                // Narrow phase 1: Opposing surface normals (n1 · n2 <= -cos(tol))
                let dot = f1.normal[0] * f2.normal[0]
                    + f1.normal[1] * f2.normal[1]
                    + f1.normal[2] * f2.normal[2];
                if dot > cos_thresh {
                    continue;
                }

                // Narrow phase 2: Centroid-to-plane clearance distance
                let dist_vec = [
                    f2.centroid[0] - f1.centroid[0],
                    f2.centroid[1] - f1.centroid[1],
                    f2.centroid[2] - f1.centroid[2],
                ];
                let dist_along_norm = (dist_vec[0] * f1.normal[0] + dist_vec[1] * f1.normal[1] + dist_vec[2] * f1.normal[2]).abs();

                if dist_along_norm <= self.clearance_tolerance {
                    // Split into Master (larger area) and Slave (smaller area)
                    let (master, slave) = if f1.area >= f2.area {
                        (f1.id, f2.id)
                    } else {
                        (f2.id, f1.id)
                    };

                    pairs.push(ContactInterfacePair {
                        master_face_id: master,
                        slave_face_id: slave,
                        clearance_distance: dist_along_norm,
                        normal_alignment: dot,
                    });
                }
            }
        }

        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contact_pair_search() {
        let faces = vec![
            // Face 0: Master plate at y=0.0 facing +Y (Normal [0, 1, 0], Area 100)
            CadFaceInfo {
                id: 0,
                centroid: [0.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                area: 100.0,
                bbox_min: [-5.0, 0.0, -5.0],
                bbox_max: [5.0, 0.0, 5.0],
                triangle_count: 2,
                is_planar: true,
            },
            // Face 1: Slave punch at y=0.5 facing -Y (Normal [0, -1, 0], Area 25)
            CadFaceInfo {
                id: 1,
                centroid: [0.0, 0.5, 0.0],
                normal: [0.0, -1.0, 0.0],
                area: 25.0,
                bbox_min: [-2.5, 0.5, -2.5],
                bbox_max: [2.5, 0.5, 2.5],
                triangle_count: 2,
                is_planar: true,
            },
            // Face 2: Distant face at y=50.0 facing -Y
            CadFaceInfo {
                id: 2,
                centroid: [0.0, 50.0, 0.0],
                normal: [0.0, -1.0, 0.0],
                area: 25.0,
                bbox_min: [-2.5, 50.0, -2.5],
                bbox_max: [2.5, 50.0, 2.5],
                triangle_count: 2,
                is_planar: true,
            },
        ];

        let searcher = ContactPairSearcher::new(1.0, 20.0);
        let pairs = searcher.find_contact_pairs(&faces);

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].master_face_id, 0); // Larger area is master
        assert_eq!(pairs[0].slave_face_id, 1);
        assert!((pairs[0].clearance_distance - 0.5).abs() < 1e-4);
    }
}
