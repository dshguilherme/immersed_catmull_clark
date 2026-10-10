//! Persistent Topology Naming (PTN) & Metric Invariant Signatures.
//! Pillar 6: Prevents topology ID invalidation across parametric CAD feature edits and mesh regeneration.

use crate::query::CadFaceInfo;
use std::collections::HashMap;

/// Geometric invariant metric signature of a boundary entity
#[derive(Clone, Debug, PartialEq)]
pub struct MetricSignature {
    pub area: f64,
    pub centroid: [f64; 3],
    pub normal: [f64; 3],
}

impl MetricSignature {
    pub fn from_face_info(face: &CadFaceInfo) -> Self {
        Self {
            area: face.area,
            centroid: face.centroid,
            normal: face.normal,
        }
    }

    /// Evaluates metric distance in feature space (centroid offset + normal deviation + area deviation)
    pub fn distance_to(&self, other: &MetricSignature) -> f64 {
        let pos_d2 = (self.centroid[0] - other.centroid[0]).powi(2)
            + (self.centroid[1] - other.centroid[1]).powi(2)
            + (self.centroid[2] - other.centroid[2]).powi(2);

        let norm_dot = (self.normal[0] * other.normal[0]
            + self.normal[1] * other.normal[1]
            + self.normal[2] * other.normal[2]).clamp(-1.0, 1.0);
        let ang_dev = (1.0 - norm_dot) * 10.0;

        let area_dev = if self.area > 1e-12 {
            ((self.area - other.area) / self.area).abs()
        } else {
            0.0
        };

        pos_d2.sqrt() + ang_dev + area_dev
    }
}

/// Persistent token identifying a topological boundary face
#[derive(Clone, Debug, PartialEq)]
pub struct PtnToken {
    pub tag_name: String,
    pub feature_history: String,
    pub signature: MetricSignature,
}

/// Registry managing persistent named topology bindings
#[derive(Clone, Debug, Default)]
pub struct PtnRegistry {
    pub tokens: HashMap<String, PtnToken>,
}

impl PtnRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds a tag name to a face's metric signature
    pub fn register(&mut self, tag_name: impl Into<String>, feature_history: impl Into<String>, face: &CadFaceInfo) {
        let tag = tag_name.into();
        self.tokens.insert(tag.clone(), PtnToken {
            tag_name: tag,
            feature_history: feature_history.into(),
            signature: MetricSignature::from_face_info(face),
        });
    }

    /// Resolves a registered persistent tag against regenerated or edited faces
    pub fn resolve(&self, tag_name: &str, current_faces: &[CadFaceInfo]) -> Option<usize> {
        let token = self.tokens.get(tag_name)?;
        let mut best_id = None;
        let mut min_dist = f64::MAX;

        for face in current_faces {
            let sig = MetricSignature::from_face_info(face);
            let dist = token.signature.distance_to(&sig);
            if dist < min_dist {
                min_dist = dist;
                best_id = Some(face.id);
            }
        }

        // Acceptance threshold
        if min_dist < 5.0 {
            best_id
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptn_resilience_across_re_meshing() {
        let original_face = CadFaceInfo {
            id: 3,
            centroid: [10.0, 2.5, 2.5],
            normal: [1.0, 0.0, 0.0],
            area: 12.5,
            bbox_min: [10.0, 0.0, 0.0],
            bbox_max: [10.0, 5.0, 5.0],
            triangle_count: 2,
            is_planar: true,
        };

        let mut ptn = PtnRegistry::new();
        ptn.register("RightClampedFace", "Extrude1:SideFaceX", &original_face);

        // Simulate re-meshing: Face ID changes from 3 to 17, vertex refinement slightly changes area from 12.5 to 12.48
        let remeshed_faces = vec![
            CadFaceInfo {
                id: 1,
                centroid: [0.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                area: 50.0,
                bbox_min: [-5.0, 0.0, -5.0],
                bbox_max: [5.0, 0.0, 5.0],
                triangle_count: 8,
                is_planar: true,
            },
            CadFaceInfo {
                id: 17,
                centroid: [10.001, 2.5, 2.5],
                normal: [0.999, 0.0, 0.0],
                area: 12.48,
                bbox_min: [10.0, 0.0, 0.0],
                bbox_max: [10.0, 5.0, 5.0],
                triangle_count: 32,
                is_planar: true,
            },
        ];

        let resolved_id = ptn.resolve("RightClampedFace", &remeshed_faces);
        assert_eq!(resolved_id, Some(17)); // Successfully tracked face across ID shifts!
    }
}
