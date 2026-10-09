pub mod bspline;
pub mod octree;
pub mod structural_mesh;
pub mod solver;
pub mod cut_cell;
pub mod topopt;
pub mod cad_model;

pub use bspline::{evaluate_bspline_basis_1d, open_knot_vector};
pub use octree::{BoundingBox3D, OctreeCell, OctreeMesh3D};
pub use structural_mesh::StructuralMesh3D;
pub use solver::{PcgSolver, ActiveSetContactSolver};
pub use cut_cell::{TriangleMesh3D, CellStatus};
pub use topopt::{TopOpt3D, TopOpt3DConfig};
pub use cad_model::{CadBody, CadFace, CadAssembly, MaterialProperties, BoundaryConditionType, LabeledBoundaryCondition};
