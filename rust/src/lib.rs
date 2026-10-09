pub mod bspline;
pub mod octree;
pub mod structural_mesh;
pub mod solver;

pub use bspline::{evaluate_bspline_basis_1d, open_knot_vector};
pub use octree::{BoundingBox3D, OctreeCell, OctreeMesh3D};
pub use structural_mesh::StructuralMesh3D;
pub use solver::{PcgSolver, ActiveSetContactSolver};
