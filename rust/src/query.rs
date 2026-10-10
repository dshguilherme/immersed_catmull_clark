//! Declarative Query Engine & Abstract Syntax Tree (AST) for Dynamic Named Selections.
//! Pillar 4: Rule-based topological queries that survive CAD parametric edits.
//! Supports spatial, orientation, dimensional, and morphological predicates with set-theoretic algebra (∪, ∩, \).

use std::collections::HashSet;

/// Morphological and geometric features extracted from a CAD face for query evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct CadFaceInfo {
    pub id: usize,
    pub centroid: [f64; 3],
    pub normal: [f64; 3],
    pub area: f64,
    pub bbox_min: [f64; 3],
    pub bbox_max: [f64; 3],
    pub triangle_count: usize,
    pub is_planar: bool,
}

impl CadFaceInfo {
    pub fn new(id: usize, centroid: [f64; 3], normal: [f64; 3], area: f64) -> Self {
        Self {
            id,
            centroid,
            normal,
            area,
            bbox_min: centroid,
            bbox_max: centroid,
            triangle_count: 1,
            is_planar: true,
        }
    }
}

/// Primitive predicates for filtering topological entities.
#[derive(Clone, Debug, PartialEq)]
pub enum FilterPredicate {
    /// Normal vector alignment: n · v >= cos(tolerance)
    NormalAlignment {
        direction: [f64; 3],
        tolerance_deg: f64,
    },
    /// Coordinate interval along axis (0=X, 1=Y, 2=Z)
    CoordinateBound {
        axis: usize,
        min: Option<f64>,
        max: Option<f64>,
    },
    /// Spatial inclusion inside an Axis-Aligned Bounding Box
    BoundingBoxInclusion {
        min_pt: [f64; 3],
        max_pt: [f64; 3],
    },
    /// Surface area bounds
    SurfaceArea {
        min_area: Option<f64>,
        max_area: Option<f64>,
    },
    /// Planar surface requirement
    PlanarOnly,
    /// Non-planar (curved) surface requirement
    CurvedOnly,
}

impl FilterPredicate {
    /// Evaluates if a face satisfies this individual predicate.
    pub fn matches(&self, face: &CadFaceInfo) -> bool {
        match self {
            FilterPredicate::NormalAlignment { direction, tolerance_deg } => {
                let dlen = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
                if dlen < 1e-12 {
                    return false;
                }
                let d = [direction[0] / dlen, direction[1] / dlen, direction[2] / dlen];
                let dot = face.normal[0] * d[0] + face.normal[1] * d[1] + face.normal[2] * d[2];
                let cos_tol = (tolerance_deg.to_radians()).cos();
                dot >= cos_tol - 1e-6
            }
            FilterPredicate::CoordinateBound { axis, min, max } => {
                let ax = *axis.min(&2);
                let val = face.centroid[ax];
                if let Some(min_val) = min {
                    if val < *min_val - 1e-6 {
                        return false;
                    }
                }
                if let Some(max_val) = max {
                    if val > *max_val + 1e-6 {
                        return false;
                    }
                }
                true
            }
            FilterPredicate::BoundingBoxInclusion { min_pt, max_pt } => {
                for k in 0..3 {
                    if face.centroid[k] < min_pt[k] - 1e-6 || face.centroid[k] > max_pt[k] + 1e-6 {
                        return false;
                    }
                }
                true
            }
            FilterPredicate::SurfaceArea { min_area, max_area } => {
                if let Some(min_a) = min_area {
                    if face.area < *min_a - 1e-6 {
                        return false;
                    }
                }
                if let Some(max_a) = max_area {
                    if face.area > *max_a + 1e-6 {
                        return false;
                    }
                }
                true
            }
            FilterPredicate::PlanarOnly => face.is_planar,
            FilterPredicate::CurvedOnly => !face.is_planar,
        }
    }
}

/// Abstract Syntax Tree (AST) representing a set-theoretic query recipe.
#[derive(Clone, Debug, PartialEq)]
pub enum QueryAst {
    /// Atomic predicate leaf
    Predicate(FilterPredicate),
    /// Set union: A ∪ B
    Union(Box<QueryAst>, Box<QueryAst>),
    /// Set intersection: A ∩ B
    Intersection(Box<QueryAst>, Box<QueryAst>),
    /// Set difference: A \ B (A and not B)
    Difference(Box<QueryAst>, Box<QueryAst>),
    /// Universal set (matches all faces)
    All,
    /// Empty set
    Empty,
}

impl QueryAst {
    /// Evaluates the AST against a slice of faces and returns matching face IDs.
    pub fn evaluate(&self, faces: &[CadFaceInfo]) -> HashSet<usize> {
        match self {
            QueryAst::All => faces.iter().map(|f| f.id).collect(),
            QueryAst::Empty => HashSet::new(),
            QueryAst::Predicate(pred) => faces
                .iter()
                .filter(|f| pred.matches(f))
                .map(|f| f.id)
                .collect(),
            QueryAst::Union(left, right) => {
                let l_res = left.evaluate(faces);
                let r_res = right.evaluate(faces);
                l_res.union(&r_res).copied().collect()
            }
            QueryAst::Intersection(left, right) => {
                let l_res = left.evaluate(faces);
                let r_res = right.evaluate(faces);
                l_res.intersection(&r_res).copied().collect()
            }
            QueryAst::Difference(left, right) => {
                let l_res = left.evaluate(faces);
                let r_res = right.evaluate(faces);
                l_res.difference(&r_res).copied().collect()
            }
        }
    }

    /// Fluent combinator: self ∪ other
    pub fn union(self, other: QueryAst) -> Self {
        QueryAst::Union(Box::new(self), Box::new(other))
    }

    /// Fluent combinator: self ∩ other
    pub fn intersect(self, other: QueryAst) -> Self {
        QueryAst::Intersection(Box::new(self), Box::new(other))
    }

    /// Fluent combinator: self \ other
    pub fn difference(self, other: QueryAst) -> Self {
        QueryAst::Difference(Box::new(self), Box::new(other))
    }

    // --- Shorthand Constructors ---
    pub fn normal_aligned(direction: [f64; 3], tolerance_deg: f64) -> Self {
        QueryAst::Predicate(FilterPredicate::NormalAlignment {
            direction,
            tolerance_deg,
        })
    }

    pub fn coord_range(axis: usize, min: Option<f64>, max: Option<f64>) -> Self {
        QueryAst::Predicate(FilterPredicate::CoordinateBound { axis, min, max })
    }

    pub fn area_range(min_area: Option<f64>, max_area: Option<f64>) -> Self {
        QueryAst::Predicate(FilterPredicate::SurfaceArea { min_area, max_area })
    }

    pub fn planar() -> Self {
        QueryAst::Predicate(FilterPredicate::PlanarOnly)
    }

    pub fn curved() -> Self {
        QueryAst::Predicate(FilterPredicate::CurvedOnly)
    }
}

/// A Dynamic Named Selection recipe stored as a lazy rule AST.
#[derive(Clone, Debug)]
pub struct NamedSelection {
    pub name: String,
    pub description: String,
    pub ast: QueryAst,
}

impl NamedSelection {
    pub fn new(name: impl Into<String>, ast: QueryAst) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            ast,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Lazily evaluates this Named Selection against candidate faces.
    pub fn resolve(&self, faces: &[CadFaceInfo]) -> HashSet<usize> {
        self.ast.evaluate(faces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_faces() -> Vec<CadFaceInfo> {
        vec![
            // Face 0: Bottom face (Normal -Y, y=0.0, area=20.0)
            CadFaceInfo {
                id: 0,
                centroid: [5.0, 0.0, 2.5],
                normal: [0.0, -1.0, 0.0],
                area: 20.0,
                bbox_min: [0.0, 0.0, 0.0],
                bbox_max: [10.0, 0.0, 5.0],
                triangle_count: 2,
                is_planar: true,
            },
            // Face 1: Top face (Normal +Y, y=5.0, area=20.0)
            CadFaceInfo {
                id: 1,
                centroid: [5.0, 5.0, 2.5],
                normal: [0.0, 1.0, 0.0],
                area: 20.0,
                bbox_min: [0.0, 5.0, 0.0],
                bbox_max: [10.0, 5.0, 5.0],
                triangle_count: 2,
                is_planar: true,
            },
            // Face 2: Left face (Normal -X, x=0.0, area=10.0)
            CadFaceInfo {
                id: 2,
                centroid: [0.0, 2.5, 2.5],
                normal: [-1.0, 0.0, 0.0],
                area: 10.0,
                bbox_min: [0.0, 0.0, 0.0],
                bbox_max: [0.0, 5.0, 5.0],
                triangle_count: 2,
                is_planar: true,
            },
            // Face 3: Right face (Normal +X, x=10.0, area=10.0)
            CadFaceInfo {
                id: 3,
                centroid: [10.0, 2.5, 2.5],
                normal: [1.0, 0.0, 0.0],
                area: 10.0,
                bbox_min: [10.0, 0.0, 0.0],
                bbox_max: [10.0, 5.0, 5.0],
                triangle_count: 2,
                is_planar: true,
            },
            // Face 4: Cylindrical hole (Curved, area=6.28)
            CadFaceInfo {
                id: 4,
                centroid: [5.0, 2.5, 2.5],
                normal: [0.707, 0.0, 0.707],
                area: 6.28,
                bbox_min: [4.0, 0.0, 2.0],
                bbox_max: [6.0, 5.0, 3.0],
                triangle_count: 16,
                is_planar: false,
            },
        ]
    }

    #[test]
    fn test_normal_alignment_query() {
        let faces = sample_faces();
        let q = QueryAst::normal_aligned([0.0, 1.0, 0.0], 15.0);
        let res = q.evaluate(&faces);
        assert_eq!(res.len(), 1);
        assert!(res.contains(&1));
    }

    #[test]
    fn test_coordinate_bound_query() {
        let faces = sample_faces();
        // Axis 1 (Y) <= 0.1 -> Bottom face
        let q = QueryAst::coord_range(1, None, Some(0.1));
        let res = q.evaluate(&faces);
        assert_eq!(res.len(), 1);
        assert!(res.contains(&0));
    }

    #[test]
    fn test_boolean_algebra_union_and_difference() {
        let faces = sample_faces();

        // All lateral faces: Normal +/- X (Faces 2 and 3)
        let q_left = QueryAst::normal_aligned([-1.0, 0.0, 0.0], 10.0);
        let q_right = QueryAst::normal_aligned([1.0, 0.0, 0.0], 10.0);
        let q_sides = q_left.union(q_right);
        let res_sides = q_sides.evaluate(&faces);
        assert_eq!(res_sides.len(), 2);
        assert!(res_sides.contains(&2));
        assert!(res_sides.contains(&3));

        // Planar faces except left: Planar \ Left
        let q_planar = QueryAst::planar();
        let q_not_left = q_planar.difference(QueryAst::normal_aligned([-1.0, 0.0, 0.0], 10.0));
        let res_not_left = q_not_left.evaluate(&faces);
        assert_eq!(res_not_left.len(), 3);
        assert!(res_not_left.contains(&0));
        assert!(res_not_left.contains(&1));
        assert!(res_not_left.contains(&3));
        assert!(!res_not_left.contains(&2));
        assert!(!res_not_left.contains(&4));
    }

    #[test]
    fn test_named_selection_recipe() {
        let faces = sample_faces();
        let sel = NamedSelection::new(
            "CylindricalHoles",
            QueryAst::curved().intersect(QueryAst::area_range(Some(1.0), Some(10.0))),
        );
        let matched = sel.resolve(&faces);
        assert_eq!(matched.len(), 1);
        assert!(matched.contains(&4));
    }
}
