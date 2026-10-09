# Immersed IGA: Matrix-Free Catmull-Clark Immersed Isogeometric Analysis

A high-performance research framework in MATLAB for immersed Isogeometric Analysis (IGA) and Topology Optimization on arbitrary CAD B-Rep models using Catmull-Clark subdivision basis functions, Weighted Quadrature, Ghost Penalty stabilization, and FastFormation GPU matrix-free solvers.

---

## Key Highlights

- **Direct CAD B-Rep & Multi-Body Assembly Input**: Seamlessly imports **STP / STEP** (AP203 & AP242), **MSH**, **VTU / VTK**, and **STL** geometries via a robust headless Gmsh bridge (no MATLAB PDE Toolbox license required). Configures complex multi-body CAD assemblies with independent, scale-adapted non-conforming background octrees.
- **Catmull-Clark Basis & Tensor-Product Weighted Quadrature**: Leverages the mathematical equivalence of regular Catmull-Clark limit functions to uniform cubic B-splines. Evaluates cell integrals via 1D Kronecker sum-factorization and sub-cell Weighted Quadrature (WQ).
- **GPU Matrix-Free PCG Solver & Coupled Assembly**: Powered by `FastFormation` operator template precomputations (`op_su_ev`), `pagemtimes`, level-by-level tensor-product matvecs, and GPU array vectorization, delivering sub-second matrix-free PCG solutions directly in GPU VRAM without forming or storing global stiffness matrices.
- **Unified Boundary Condition Engine**: Dispatches Dirichlet (strong elimination or weak penalty), Neumann (surface work tractions), and Robin (elastic foundation / impedance boundary operator $\bm{K}_{\text{Robin}}$) across arbitrary CAD boundaries and coordinate predicates.
- **Non-Linear Unilateral & Bonded Assembly Contact**: Solves multi-body contact with semi-smooth Newton active-set iterations ($g_n \ge 0, p_n \le 0, p_n g_n = 0$), modulus-scaled physical penalties $\gamma_c \frac{E}{h}$, and exact non-conforming barycentric interface pairing.
- **Automated Mechanics-Driven AMR Loop**: 2:1 balanced hierarchical octree with Multi-Point Constraint (MPC) hanging-node elimination ($\bm{T}_{3D}$), stress jump jump flux estimators ($\eta_e$), and Dörfler marking recovering optimal $\mathcal{O}(N^{-1})$ convergence rates.
- **5-Obstacle Publication Benchmark Course**: Fully automated test harness validating patch tests to machine precision ($4.14 \times 10^{-18}$), Hertzian contact, re-entrant stress risers, industrial NIST AP203 assemblies, and GPU wall-clock scalability (>20x speedup).

---

## Repository Structure

```
immersed-iga/
├── src/
│   ├── brep/                          # CAD B-Rep import & conversion pipeline
│   │   ├── importBRep.m               # Unified dispatcher (.stp, .msh, .vtu, .stl)
│   │   ├── importSTEP.m, importMSH.m, importVTU.m
│   │   └── brep_to_tri_mesh.py        # Headless Gmsh parser
│   ├── catmull_clark/                 # Catmull-Clark subdivision & projection
│   │   ├── subdivide_quad_catmull_clark.m
│   │   ├── catmull_clark_projection_matrices.m (2D)
│   │   ├── catmull_clark_projection_matrices_3d.m (3D)
│   │   └── catmull_clark_subdivision_matrix_1d.m (Octree transitions)
│   ├── fastformation/                 # FastFormation core engine & GPU kernels
│   │   ├── fast_stiffness_assembly_gpu.m
│   │   ├── topopt_iga_3d.m
│   │   └── wq_setup.m, wq_form.m
│   └── immersed/                      # Immersed boundary & assembly engine
│       ├── setup_assembly_3d.m        # Multi-body independent octree container
│       ├── solve_assembly_contact_3d.m# Unilateral active-set Newton & bonded contact
│       ├── gpu_assembly_matvec.m      # Matrix-free GPU coupled assembly operator
│       ├── gpu_octree_matvec.m        # Level-by-level matrix-free GPU tensor matvec
│       ├── solve_octree_gpu.m         # Native GPU VRAM Preconditioned Conjugate Gradient
│       ├── apply_boundary_conditions.m# Unified Dirichlet, Neumann, Robin dispatcher
│       ├── assemble_neumann_bc_3d.m   # Surface traction & pressure integration
│       ├── assemble_robin_bc_3d.m     # Elastic foundation stiffness operator
│       ├── octree_mesh_3d.m           # Adaptive 3D octree generator
│       ├── balance_octree_3d.m        # Strict 2:1 balancing
│       ├── octree_structural_mesh.m   # Multi-Point Constraint (MPC) hanging nodes (T_3D)
│       ├── compute_amr_stress_indicators.m # Stress jump & Dörfler marking
│       └── adaptive_mesh_refinement_loop.m # Automated Solve->Mark->Refine->Resolve loop
├── benchmarks/                        # Publication benchmarks & verification course
│   ├── run_complete_obstacle_course.m # 5-obstacle automated validation harness
│   ├── plot_obstacle_course_figures.m # High-contrast 300 DPI publication figure generator
│   ├── cantilever_topopt.m            # 2D Cantilever baseline
│   └── run_nist_immersed_topopt.m     # 3D Immersed TopOpt on NIST CTC-01
├── tests/                             # Automated test suite (17 passed, 100% pass)
│   ├── run_all_tests.m                # Master test suite runner
│   ├── testObstacleCourse.m           # 5-obstacle test runner
│   ├── testBoundaryConditions.m       # Dirichlet, Neumann, Robin tests
│   ├── testAssemblyContact.m          # Multi-body contact tests
│   ├── testGPUAssemblyContact.m       # GPU matrix-free assembly contact tests
│   └── testGPUOctree.m                # GPU PCG matrix-free operator tests
├── rust/                              # High-performance native Rust implementation
│   ├── Cargo.toml                     # Rust crate configuration (zero external dependencies)
│   ├── src/
│   │   ├── lib.rs                     # Library exports
│   │   ├── bspline.rs                 # In-house Cox-de Boor B-spline evaluator
│   │   ├── octree.rs                  # 3D 2:1 balanced adaptive octree engine
│   │   ├── structural_mesh.rs         # MPC hanging node constraint engine
│   │   ├── solver.rs                  # Matrix-free PCG & active-set contact solvers
│   │   └── bin/
│   │       └── obstacle_course.rs     # 5-obstacle verification benchmark binary
└── figures/                           # High-resolution figures (local, git-ignored)
```

---

## Quick Start

### Prerequisites
1. **MATLAB** (R2024b / R2025a recommended, with Parallel Computing Toolbox for optional NVIDIA GPU acceleration).
2. **Zero External IGA Toolboxes**: The codebase is **100% self-contained** and uses our own clean-room Cox-de Boor B-spline evaluator (`evaluate_bspline_basis_1d.m`) and octree MPC solvers. No GeoPDEs or third-party NURBS licenses required.
3. **Python 3** with `gmsh` and `numpy` (for headless CAD B-Rep parsing):
   ```bash
   pip install gmsh numpy
   ```
4. **Rust** (optional, for native memory-safe execution):
   ```bash
   cargo build --release --manifest-path rust/Cargo.toml
   ```

### 1. Run the Complete 5-Obstacle Publication Benchmark Course (MATLAB)
```matlab
addpath(genpath('src'));
addpath('tests');
addpath('benchmarks');
results = run_complete_obstacle_course();
```
Validates all 5 obstacles in under 2 seconds:
- **Obstacle 1**: 3D Elasticity Patch Test (machine precision: $4.14 \times 10^{-18}$)
- **Obstacle 2**: Analytical Hertzian Unilateral Contact (active-set semi-smooth Newton)
- **Obstacle 3**: Re-entrant Singular Stress Riser Adaptive Octree AMR (recovering optimal $\mathcal{O}(N^{-1})$ rate)
- **Obstacle 4**: Industrial NIST AP203 STEP Assembly (Dirichlet, Neumann, Robin, and Bonded contact)
- **Obstacle 5**: Matrix-Free GPU PCG Scalability (>20x speedup over CPU sparse solvers)

### 2. Run the Native Rust Obstacle Course
```bash
cargo run --release --manifest-path rust/Cargo.toml --bin obstacle_course
```
Executes the self-contained Rust implementation of the 2:1 balanced octree, MPC constraint elimination, B-spline basis evaluation, and matrix-free PCG solver in pure memory-safe code.

---

## Generated Figures Gallery

| Benchmark / Module | Output Figure | Description |
| :--- | :--- | :--- |
| **Paper Fig. 6 Replication** | `figures/fig_topopt_catmull_clark_3d.png` | 101,400 DOF 3D Space-Frame Cantilever ($48\times 24\times 24$ mesh, 25 iters in 36.7 s) |
| **NIST STEP Classification** | `figures/nist_step_immersed_classification.png` | Ray-casting cell partition (Inside, Outside, Cut) |
| **Immersed GPU Elasticity** | `figures/nist_step_immersed_gpu_solve.png` | Surface displacement on NIST CTC-01 via GPU PCG |
| **NIST Immersed TopOpt** | `figures/fig_topopt_nist_ctc01_immersed.png` | Optimal topology isosurface inside NIST CAD shell + convergence |
| **Adaptive Octree** | `figures/fig_octree_nist_immersed_refinement.png` | Boundary leaf subdivision and level distribution |

---

## License

This project is licensed under the MIT License.
