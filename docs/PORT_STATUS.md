# MATLAB â†’ Rust Port Status

Last updated: 2026-10-10.

Status key:
- **Parity**: ported, with a test that compares against data exported from MATLAB (`tests/export_rust_*.m` writes the fixtures to `rust/tests/fixtures/`).
- **Verified**: also checked against an analytical answer.
- **Partial**: ported in part; the gaps are listed.
- **Missing**: not ported.
- **Rust-only**: no MATLAB counterpart.

Run everything with `.\run_rust.ps1 test --manifest-path rust/Cargo.toml --release`. The analytical checks are in `.\run_rust.ps1 run --release --manifest-path rust/Cargo.toml --bin obstacle_course`.

## Module matrix

| Capability | MATLAB (reference) | Rust | Status | Evidence |
|---|---|---|---|---|
| B-spline basis, derivatives of any order | `iga/iga_bspline_basis.m` | `iga/basis.rs` | Verified | finite differences, partition of unity |
| Box spline spaces (GeoPDEs numbering) | `iga/iga_space_box.m` | `iga/space.rs` | Parity | connectivity identical (`iga_vs_matlab`) |
| Exact elasticity element matrices | `iga/iga_elasticity_element_matrices.m` | `iga/elasticity.rs` | Parity + verified | KÂ·v < 1e-13; patch test; L2 rate p+1 (`iga_kernel`) |
| Load vectors, field evaluation | `iga/iga_load_vector.m`, â€¦ | `iga/fields.rs` | Parity | F < 1e-13 |
| Immersed solve (volume-fraction cut cells) | `immersed/solve_immersed_iga_3d.m` | `immersed.rs` | Parity | classification identical, u to 9e-12 (`immersed_vs_matlab`) |
| Legacy "ghost penalty" | `assemble_ghost_penalty_stabilization.m` | `immersed::legacy_stabilization` | Parity | kept only for parity (see findings) |
| Consistent ghost penalty (p-th normal-derivative jump) | â€” | `immersed::ghost_penalty` | Rust-only, verified | vanishes on polynomials of degree â‰¤ p |
| CAD-face BCs: penalty Dirichlet, traction, pressure, Robin | â€” | `immersed_bc.rs` | Rust-only, verified | traction integrates to tÂ·A; cantilever +1.2% vs Timoshenko |
| `solve_cad` and the GUI's in-app solve | (placeholder before) | `bin/solve_cad.rs`, `bin/cad_gui.rs` | Real solver | uses `immersed` + `immersed_bc` |
| Structured 3D SIMP topology optimization | `fastformation/topopt_iga_3d.m` (CPU path) | `topopt_iga.rs`, `bin/topopt_3d.rs` | Parity | compliance to 1e-7 over 6 iterations (`topopt_vs_matlab`) |
| Octree, 2:1 balance, hanging-node MPC, trilinear K_master | `octree_structural_mesh.m`, `balance_octree_3d.m`, â€¦ | `octree.rs`, `octree_fem.rs` | Parity | mesh identical, KÂ·v to 1e-12 (`octree_vs_matlab`) |
| Octree BC engine (strong/penalty Dirichlet, traction, pressure, Robin) | `apply_boundary_conditions.m` + `assemble_*_bc_3d.m` | `octree_fem::apply_boundary_conditions` | Parity | loads and K to 1e-12, u to 9e-11 |
| Multi-body assembly + penalty contact (unilateral, bonded) | `setup_assembly_3d.m`, `solve_assembly_contact_3d.m` | `contact.rs` | Parity | active set, gaps, base u to 1e-13 (`contact_vs_matlab`) |
| Stress-driven AMR loop | `compute_amr_stress_indicators.m`, `adaptive_mesh_refinement_loop.m` | `amr.rs` | Parity | 3 cycles identical (`amr_vs_matlab`) |
| PCG | `solve_octree_gpu.m` | `solver.rs` | Ported | absolute-threshold breakdown bug fixed |
| Immersed topology optimization | `immersed/topopt_immersed_iga_3d.m` | â€” | Missing | next |
| WQ / FastFormation assembly (2D, 3D) | `fastformation/fast_stiffness_assembly*.m`, `wq_setup.m`, `wq_form.m`, `iga/iga_wq_rules_1d.m` | â€” | Missing | |
| 2D topology optimization (element / spline densities) | `topopt_iga_fast.m`, `topopt_iga_2d_mf.m`, `fast_sensitivities.m` | â€” | Missing | |
| Catmull-Clark subdivision and projection | `catmull_clark/*` | â€” | Missing | |
| CAD import STEP/MSH/VTU/STL | `brep/*` + Gmsh (Python) | `cad_model.rs` (OBJ, JSON) | Partial | STEP still needs the Python/Gmsh bridge |
| GPU kernels | MATLAB `gpuArray` | `bin/cuda_matrixfree_benchmark.rs`, `bin/wgpu_matrixfree_benchmark.rs` | Partial | benchmarks only; not wired into the solvers |
| CAD topology layer (query, arena, BVH, pairing, naming) | â€” | `query.rs`, `topology.rs`, `bvh.rs`, `pairing.rs`, `ptn.rs` | Rust-only | the BVH isn't used by the ray casting yet |
| Legacy placeholders | â€” | `structural_mesh.rs`, `cad_model::resolve_boundary_conditions_on_mesh`, `topopt.rs` (old OC/filter), `solver::ActiveSetContactSolver` | Superseded | no longer used by the binaries |

## Findings from porting the MATLAB code

These are problems in the MATLAB reference. Where the port reproduces one, it's for parity only.

1. **"Ghost penalty" (`assemble_ghost_penalty_stabilization`)**: not a ghost penalty.
   - Its shared-control-point block is a no-op: because of the `repmat` orientation, it only touches diagonal entries, which sum to zero.
   - What remains are springs between control points paired by sorted index, with stiffness Î³h that isn't scaled by E.
   - At E = 1, which most scripts use, it makes an immersed cantilever about **1000Ã— too stiff**.
   - It affects `solve_immersed_iga_3d`, `topopt_immersed_iga_3d` and the NIST demos.
2. **`assemble_nitsche_dirichlet_3d`**: penalty only. It uses a trilinear trace at facet centroids and ignores `u_prescribed`.
3. **Cut cells**: volume-fraction scaling of the full-cell stiffness, not exact integration.
4. **3D WQ assembly**: the old `C_ijkl` Voigt map was wrong (33% error); fixed in `src/iga`.
5. **3Ã—3Ã—3 element template**: exact only for p â‰¤ 2; replaced.
6. **`solve_octree_gpu`**: ignores the penalty and Robin stiffness returned by `apply_boundary_conditions`, so weak Dirichlet (the default) and Robin BCs have no effect in that solver.
7. **`assemble_neumann_bc_3d` and `assemble_robin_bc_3d`**: an empty facet selection means all facets. Obstacle 5's load lands on the entire surface this way.
8. **Obstacle 2 ("Hertzian")**: a flat punch with only two contact points, so the system is singular in both unilateral and bonded mode; the punch's displacement is arbitrary. There is no comparison with Hertz theory.
9. **AMR indicator**: a stress heuristic, not an error estimator. Compliance isn't monotone under refinement, because the cut-cell weights change as cells split.
10. **Timing scripts in `src/fastformation/`**: several report extrapolated or modelled values; each header documents which.

## Next steps

1. Port the immersed topology optimization, with the consistent ghost penalty as default and legacy stabilization for parity.
2. Port the WQ / FastFormation assembly and the 2D topology optimization.
3. Build the verification suite (manufactured solutions on curved immersed domains, small-cut robustness, contact patch test, real Hertz problem, AMR rates). Then replace the MATLAB-side methods that fail it.
