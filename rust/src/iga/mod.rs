//! In-house isogeometric kernel: B-spline spaces on boxes, exact elasticity
//! element matrices, loads and field evaluation. A one-to-one port of the MATLAB
//! reference in `src/iga` (same DOF numbering, 0-based), validated against
//! matrices exported from MATLAB (`rust/tests/iga_vs_matlab.rs`).

pub mod basis;
pub mod elasticity;
pub mod fields;
pub mod quadrature;
pub mod space;

pub use basis::{basis_dense, basis_funs, find_span, num_basis};
pub use elasticity::{elasticity_element_matrices, lame, ElementMatrices};
pub use fields::{element_quad_points, element_shape_functions, element_vector_gradients, element_vector_values, eval_vector_at, load_vector};
pub use quadrature::{gauss_legendre, open_knots};
pub use space::{unravel, Space1d, SpaceBox};
