# MATLAB → Rust Port Status

Last audited: 2026-10-10 (against commit `eb4d55f` plus uncommitted work on `main`).

Status key:
- **Ported**: same algorithm, usable.
- **Partial**: some of the algorithm is there, with gaps listed.
- **Stub**: the API exists, but the numbers are placeholders.
- **Missing**: not ported yet.
- **Rust-only**: no MATLAB counterpart.

"Validated" means the result has been checked against MATLAB or an analytical reference on the same input. **No Rust module is validated yet.**

## Summary

The Rust crate has the *scaffolding* of the pipeline: an octree, a hanging-node map, inside/outside classification, a PCG solver, a SIMP update and a filter. It also has a large GUI and CAD-topology layer. **It has no finite-element discretization yet.** No module computes element stiffness matrices, quadrature points or basis-function gradients, and there's no assembly. As a result:

- **`solve_cad`** runs PCG on a made-up tridiagonal operator (`diag = 1.5E`, off-diagonal `-0.2·diag`). The displacements it prints are not elasticity results.
- **`topopt_3d`** feeds the optimizer synthetic strain energies from a closed-form "bending profile" and never does an FE solve.
- **`obstacle_course`**: none of the five checks exercises what the MATLAB obstacle with the same name tests (see below).
- **CUDA and WGPU benchmarks** use a synthetic tridiagonal 24×24 `Ke`. This is fine for timing a matvec kernel, but they don't compute elasticity.

**Impact on the README:** the "Component & Obstacle Course Timings" table (7,770× and 12,620× speedups) compares full MATLAB computations against these placeholders. Remove those figures or label them before anyone cites them. The element-matvec table compares kernels on comparable work, so it is fine with a caveat.

## Module matrix

| Capability | MATLAB (reference) | Rust | Status | Notes |
|---|---|---|---|---|
| 1D B-spline basis (Cox-de Boor) | `catmull_clark/evaluate_bspline_basis_1d.m` | `bspline.rs` | Ported | Values only. **No derivatives**, which stiffness needs. Has a partition-of-unity test. |
| Catmull-Clark subdivision / projection (2D, 3D, 1D octree transfer) | `catmull_clark/*` | — | Missing | |
| Tensor-product spline space / mesh (GeoPDEs `sp_bspline`, `msh_cartesian`) | external GeoPDEs | — | Missing | The MATLAB side depends on GeoPDEs here, despite what the README says. |
| Element stiffness (elasticity, Lamé) | `solve_immersed_iga_3d.m`, `octree_structural_mesh.m` (`ke_by_level`) | — | **Missing** | This is the critical blocker. |
| Weighted quadrature / FastFormation | `fastformation/wq_setup.m`, `wq_form.m`, `fast_stiffness_assembly*.m` | — | Missing | |
| CAD import STEP/MSH/VTU/STL | `brep/import*.m` + Gmsh Python | `cad_model.rs::parse_obj`, `from_json` | Partial | Rust reads OBJ and its own JSON format only. STEP still has to go through the Python/Gmsh path. |
| Inside/outside classification | `classify_background_cells.m` | `cut_cell.rs::is_point_inside` | Ported | Ray-parity test is O(#triangles) per point; `bvh.rs` exists but isn't wired in. |
| Cut-cell quadrature | `compute_cut_cell_quadrature.m`, `assemble_immersed_element_weights.m` | `cut_cell.rs::classify_box_quadrature` | Partial | Returns only a volume-fraction scalar from midpoint sub-cells. **No quadrature points or weights** for integrating the stiffness. |
| Ghost penalty | `assemble_ghost_penalty_stabilization.m` | — | Missing | |
| Octree + 2:1 balance | `octree_mesh_3d.m`, `balance_octree_3d.m`, `subdivide_octree_leaves.m` | `octree.rs` | Ported | Balancing hasn't been checked against MATLAB on the same tree. |
| Hanging-node MPC map T₃D | `octree_structural_mesh.m` | `structural_mesh.rs` | Partial | Builds constraint rows for the nodes of 8-node trilinear hex elements, but doesn't produce the T matrix or assemble a stiffness. Its "patch test" only checks that the constraints interpolate a linear field, so it would pass even if the solver were wrong. |
| Dirichlet BC (strong / Nitsche) | `apply_boundary_conditions.m`, `assemble_nitsche_dirichlet_3d.m` | `cad_model.rs::resolve_boundary_conditions_on_mesh` | Partial | Strong elimination only, and nodes are picked by a distance tolerance (0.15·span in `solve_cad`). No Nitsche. |
| Neumann traction / pressure | `assemble_neumann_bc_3d.m` | `cad_model.rs` (same) | Partial | Lumps the traction onto nearby nodes; it doesn't integrate it over the surface with basis functions. |
| Robin (elastic foundation) | `assemble_robin_bc_3d.m` | enum variant only | Stub | |
| PCG solver | `solve_octree_gpu.m` | `solver.rs::PcgSolver` | Ported | Generic Jacobi-preconditioned CG with a closure for the matrix-vector product. Fine. |
| Matrix-free octree operator Tᵀ K T p | `gpu_octree_matvec.m` | — | Missing | The CUDA/WGPU kernels could be reused for this once a real `Ke` exists. |
| Multi-body setup | `setup_assembly_3d.m` | `cad_model.rs::CadAssembly` | Partial | Holds multiple bodies, but `solve_cad` only uses `bodies[0]`. |
| Contact pairing (Nitsche, non-conforming) | `assemble_nitsche_contact_3d.m`, `assemble_octree_nitsche_contact.m`, `build_interface_projectors` | `pairing.rs` (face-level master/slave detection) | Partial | Rust pairs CAD faces, but has no gap computation at quadrature points and no projectors between the two bodies' meshes. |
| Active-set semi-smooth Newton contact | `solve_assembly_contact_3d.m` | `solver.rs::ActiveSetContactSolver` | Stub | Only checks gap signs and computes a pressure. No Newton loop and no coupling to the solve. |
| AMR indicators + Dörfler marking + loop | `compute_amr_stress_indicators.m`, `adaptive_mesh_refinement_loop.m` | — | Missing | Rust only refines geometrically around a point. |
| SIMP / OC / density filter | `topopt_iga_3d.m`, `build_cartesian_filter_3d.m`, `fast_sensitivities.m` | `topopt.rs` | Partial | The optimizer and filter are real. It needs element strain energies from an FE solve, which doesn't exist yet. |
| Immersed topopt | `topopt_immersed_iga_3d.m` | — | Missing | |
| GPU backends | MATLAB `gpuArray` | `bin/cuda_matrixfree_benchmark.rs`, `bin/wgpu_matrixfree_benchmark.rs`, `kernel.c/.ptx` | Partial | Kernels exist but run only on synthetic data; they aren't part of the library. |
| BC labeling GUI | `scripts/cad_bc_labeler_gui.py` (Python) | `bin/cad_gui.rs` | Rust-only (in progress) | Its in-process solve uses the same placeholder operator as `solve_cad`. |
| CAD topology layer: query AST, entity arena, BVH, auto-pairing, persistent naming ("Pillars" 1, 2, 4, 5, 6) | — | `query.rs`, `topology.rs`, `bvh.rs`, `pairing.rs`, `ptn.rs` | Rust-only (uncommitted) | `bvh` isn't used by `cut_cell` yet. No document defining the pillars was found. |

## Obstacle course: MATLAB vs Rust

| # | MATLAB `run_complete_obstacle_course.m` | Rust `bin/obstacle_course.rs` actually checks |
|---|---|---|
| 1 | 3D elasticity patch test (solve, reaction forces) | Hanging-node constraint rows reproduce a linear field. No solve. |
| 2 | Hertzian contact vs analytical pressure (target: within 5%) | 5 hard-coded gaps are classified by sign (2 active). |
| 3 | Re-entrant corner AMR, O(N⁻¹) convergence rate | The leaf count goes up after refining around a point. |
| 4 | NIST AP203 STEP assembly with Dirichlet, Neumann, Robin and bonded contact | B-spline partition of unity. |
| 5 | GPU PCG scalability on 10⁵+ DOFs | CG on a 1000-unknown 1D tridiagonal matrix. |

## Suggested porting order (critical path)

1. **B-spline derivatives**, then a tensor-product **cubic B-spline space** on a uniform grid with connectivity.
2. **Gauss quadrature and the elasticity element stiffness.** Validate one element and one small grid against MATLAB's `K` (Frobenius-norm difference).
3. **Cut-cell quadrature points and weights** (sub-cell Gauss using the in/out test, with the BVH), plus **ghost penalty**.
4. **Real `solve_cad`**: assembled or matrix-free K, then BCs, then PCG. Validate the cantilever tip deflection against MATLAB and against beam theory.
5. **Nitsche Dirichlet and surface-integrated Neumann/Robin.**
6. **Octree T₃D, then Tᵀ K T matrix-free**, and plug the CUDA kernel in on the real `Ke`.
7. **Contact** (projectors, active-set Newton), then **AMR**, then **immersed topopt**.
8. **Rewrite `obstacle_course.rs`** so each obstacle matches its MATLAB counterpart, then regenerate the README benchmark tables.
