# Immersed IGA: Matrix-Free Catmull-Clark Immersed Isogeometric Analysis

A high-performance research framework in MATLAB for immersed Isogeometric Analysis (IGA) and Topology Optimization on arbitrary CAD B-Rep models using Catmull-Clark subdivision basis functions, Weighted Quadrature, Ghost Penalty stabilization, and FastFormation GPU matrix-free solvers.

---

## Key Highlights

- **Direct CAD B-Rep Input**: Seamlessly imports **STP / STEP** (AP203 & AP242), **MSH**, **VTU / VTK**, and **STL** geometries via a robust headless Gmsh bridge (no MATLAB PDE Toolbox license required). Tested on official NIST PMI benchmark models (`nist_ctc_01_asme1_rd.stp`, etc.).
- **Catmull-Clark Basis & Tensor-Product Weighted Quadrature**: Leverages the mathematical equivalence of regular Catmull-Clark limit functions to uniform cubic B-splines. Evaluates cell integrals via 1D Kronecker sum-factorization and sub-cell Weighted Quadrature (WQ).
- **GPU Matrix-Free Solver**: Powered by `FastFormation` operator template precomputations (`op_su_ev`), `pagemtimes`, and GPU array vectorization, delivering sub-second matrix-free PCG solutions without storing global stiffness matrices.
- **Ghost Penalty Stabilization**: Eliminates small-cut ill-conditioning across active cut-cell interior facets, bounding condition numbers as cut fractions $\eta \to 0$.
- **Weak Boundary Conditions (Nitsche)**: Enforces Dirichlet conditions directly on immersed CAD surfaces via Nitsche's method.
- **3D Immersed Topology Optimization**: SIMP optimization with adjoint sensitivities computed on GPU, filtering, and Optimality Criteria updates inside arbitrary CAD domain envelopes.
- **Adaptive 3D Octree Local Refinement**: Hierarchical 2:1 balanced octree mesh generator with Catmull-Clark dyadic subdivision transition operators across T-junctions.

---

## Repository Structure

```
immersed-iga/
├── Articles/                          # Reference papers (FastFormation_TopOpt.pdf)
├── Models/
│   └── NIST-PMI-STEP-Files/           # NIST STEP benchmark geometries (AP203/AP242)
├── src/
│   ├── brep/                          # CAD B-Rep import & conversion pipeline
│   │   ├── importBRep.m               # Unified dispatcher (.stp, .msh, .vtu, .stl)
│   │   ├── importSTEP.m, importMSH.m, importVTU.m
│   │   ├── importMeshViaPython.m      # Headless Gmsh bridge
│   │   └── brep_to_tri_mesh.py        # Python Gmsh & VTU parser
│   ├── catmull_clark/                 # Catmull-Clark subdivision & projection
│   │   ├── subdivide_quad_catmull_clark.m
│   │   ├── catmull_clark_projection_matrices.m (2D)
│   │   ├── catmull_clark_projection_matrices_3d.m (3D)
│   │   └── catmull_clark_subdivision_matrix_1d.m (Octree transitions)
│   ├── fastformation/                 # FastFormation core engine & GPU kernels
│   │   ├── fast_stiffness_assembly_gpu.m
│   │   ├── topopt_iga_3d.m
│   │   └── wq_setup.m, wq_form.m
│   └── immersed/                      # Immersed boundary engine
│       ├── classify_background_cells.m# 3D Ray-casting classifier
│       ├── compute_cut_cell_quadrature.m
│       ├── assemble_immersed_element_weights.m
│       ├── assemble_ghost_penalty_stabilization.m
│       ├── assemble_nitsche_dirichlet_3d.m
│       ├── solve_immersed_iga_3d.m    # Unified 3D immersed elasticity solver
│       ├── topopt_immersed_iga_3d.m   # 3D CAD immersed topology optimization
│       ├── build_cartesian_filter_3d.m# O(N*r^3) sensitivity filter
│       └── octree_mesh_3d.m           # Adaptive 2:1 balanced octree
├── benchmarks/                        # Reproduction benchmarks & drivers
│   ├── cantilever_topopt.m            # 2D Cantilever baseline
│   ├── topopt_iga_catmull_clark.m     # 2D Catmull-Clark benchmark
│   ├── run_3d_catmull_clark_topopt.m  # 3D 101,400 DOF Cantilever (Paper Fig. 6)
│   └── run_nist_immersed_topopt.m     # 3D Immersed TopOpt on NIST CTC-01
├── examples/
│   ├── demo_nist_step_immersed.m      # NIST STEP import & classification demo
│   ├── demo_immersed_gpu_solve.m      # Immersed GPU elasticity solve
│   └── demo_octree_immersed_refinement.m # Adaptive octree refinement demo
├── figures/                           # Exported publication figures & results
└── tests/                             # Automated test suite
```

---

## Quick Start

### Prerequisites
1. **MATLAB** (R2024b / R2025a recommended, with Parallel Computing Toolbox for NVIDIA GPU).
2. **GeoPDEs** (https://github.com/kbmag/GeoPDEs) added to MATLAB path.
3. **Python 3** with `gmsh` and `numpy`:
   ```bash
   pip install gmsh numpy
   ```

### 1. Immersed 3D GPU Elasticity Solve on NIST STEP Model
```matlab
addpath(genpath('src'));
step_file = 'Models/NIST-PMI-STEP-Files/NIST-PMI-STEP-Files/AP203 geometry only/nist_ctc_01_asme1_rd.stp';
brep = importBRep(step_file);

opts.grid_res = [24, 16, 12];
opts.gamma_gp = 0.05; % Ghost penalty
sol = solve_immersed_iga_3d(brep, opts);
fprintf('Solved in %.3f s | Compliance: %.4e\n', sol.time_solve, sol.compliance);
```

### 2. Immersed Topology Optimization inside NIST CAD Geometry
```matlab
run('benchmarks/run_nist_immersed_topopt.m');
```
Optimizes material distribution strictly within the CAD envelope in ~16 seconds on GPU (0.64 s/iteration), generating:
- [`figures/fig_topopt_nist_ctc01_immersed.png`](file:///C:/Users/dshgu/immersed-iga/figures/fig_topopt_nist_ctc01_immersed.png)

### 3. Adaptive Octree Local Refinement Demo
```matlab
run('examples/demo_octree_immersed_refinement.m');
```
Generates 2:1 balanced boundary refinement:
- [`figures/fig_octree_nist_immersed_refinement.png`](file:///C:/Users/dshgu/immersed-iga/figures/fig_octree_nist_immersed_refinement.png)

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
