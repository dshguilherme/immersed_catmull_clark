# Immersed IGA: Immersed Isogeometric Analysis on Background Spline Grids

Research code for immersed isogeometric analysis (IGA) and topology optimization of
CAD B-Rep models. The CAD surface is embedded in a background tensor-product B-spline
grid (uniform or 2:1-balanced octree), and the elasticity problem is solved on that
grid with matrix-free CPU/GPU solvers. On regular grids, cubic B-splines coincide with
regular Catmull-Clark limit functions, which is where the Catmull-Clark naming in the
code comes from. Most immersed solvers here default to quadratic (p = 2) B-splines.

The MATLAB code in `src/` is the reference implementation. `rust/` is an early port;
see [Rust port](#rust-port) and `docs/PORT_STATUS.md`.

---

## Status at a glance (audited 2026-10-10)

| Area | State | Notes |
| :--- | :--- | :--- |
| B-spline kernel (`src/iga`) | **Verified** | In-house replacement for GeoPDEs. Matches GeoPDEs to machine precision; patch test, rigid-body null space and optimal L2 convergence rates are covered by `tests/testIgaKernel.m` |
| FastFormation / weighted quadrature (2D and 3D) | **Verified** | Equals exact Gauss assembly for uniform density (machine precision) |
| Structured topology optimization (2D/3D, SIMP + OC) | Working | Sensitivities checked against element energies |
| CAD import (STEP/MSH/VTU/STL via Gmsh) | Working | |
| Immersed boundary treatment | **Simplified** | Cut cells use the full-cell stiffness scaled by the cell's volume fraction (fictitious-domain / density scaling). The geometry is not integrated exactly, so expect reduced accuracy near the boundary |
| Dirichlet BC on immersed surfaces (`assemble_nitsche_dirichlet_3d`) | **Penalty only** | Penalty term only (no Nitsche consistency/symmetry terms); trace evaluated by trilinear interpolation at facet centroids; `u_prescribed` is ignored |
| Multi-body contact, AMR, octree MPC, GPU PCG | Implemented, **not yet verified** against analytical benchmarks | See [Verification](#verification) |

---

## Requirements

- **MATLAB** R2024b or newer. The Parallel Computing Toolbox is optional and needed only for the GPU paths.
- **Python 3** with `gmsh` and `numpy`, for headless CAD import (`pip install gmsh numpy`).
- **No GeoPDEs or NURBS toolbox.** `src/iga` provides the B-spline spaces, quadrature,
  elasticity operators and weighted-quadrature rules. A few timing benchmarks can
  optionally time GeoPDEs as an external reference: set the `GEOPDES_PATH` environment
  variable to the folder that contains GeoPDEs (see `benchmarks/geopdes_baseline_available.m`).
  Without it, those baseline columns are reported as `NaN`.
- **Rust** (optional), with the LLVM-MinGW toolchain configured by `run_rust.ps1`. Crate
  dependencies: `rayon`, `wgpu`, `cudarc` (CUDA driver API), `eframe` (GUI).

---

## Repository structure

```
immersed-iga/
├── src/
│   ├── iga/             # In-house B-spline kernel (replaces GeoPDEs): knots, Cox-de Boor
│   │                    # values and derivatives, box spaces, exact elasticity element
│   │                    # matrices, load vectors, weighted-quadrature rules
│   ├── brep/            # CAD import (.stp/.msh/.vtu/.stl) via headless Gmsh
│   ├── catmull_clark/   # Catmull-Clark subdivision and projection matrices
│   ├── fastformation/   # Weighted-quadrature stiffness formation (CPU/GPU), sensitivities,
│   │                    # structured 2D/3D topology optimization, timing scripts
│   └── immersed/        # Cell classification, cut-cell weights, ghost penalty, BC engine,
│                        # octree + MPC, AMR loop, multi-body contact, GPU PCG, immersed topopt
├── tests/               # matlab.unittest suite (run_all_tests.m)
├── benchmarks/          # Obstacle course, topology-optimization benchmarks, optional
│                        # GeoPDEs baselines (external_geopdes/ needs GeoPDEs)
├── examples/            # Demos
├── rust/                # Rust port (library + binaries, CAD BC-labelling GUI)
├── scripts/             # Python plotting and the PyVista BC labeller
└── docs/PORT_STATUS.md  # MATLAB -> Rust port audit
```

`Models/`, `Articles/` and `figures/` are local only and git-ignored.

---

## Quick start (MATLAB)

```matlab
% Test suite (no GeoPDEs needed)
run('tests/run_all_tests.m');

% Obstacle course (see the caveats under Verification)
addpath(genpath('src')); addpath('tests', 'benchmarks');
results = run_complete_obstacle_course();

% Immersed elasticity solve on a STEP model
brep = importBRep('Models/.../nist_ctc_01_asme1_rd.stp');
sol  = solve_immersed_iga_3d(brep, struct('grid_res', [24 16 12]));
```

---

## Verification

**What is verified** (`tests/testIgaKernel.m`, part of `run_all_tests.m`):
- Gauss rules are exact; B-spline partition of unity holds; derivatives agree with finite differences.
- The assembled stiffness is symmetric, with exactly 3 (2D) or 6 (3D) rigid-body modes in its null space.
- Patch test: linear displacement fields are reproduced to round-off (2D and 3D, p = 2, 3).
- Manufactured solution (plane strain): optimal L2 convergence rate p + 1 for p = 2, 3.
- Weighted-quadrature formation equals exact Gauss assembly for uniform density (2D and 3D).
- Element-density sensitivities equal the element strain energies.
- With `GEOPDES_PATH` set, the stiffness matrix also agrees with GeoPDEs `op_su_ev` to around 1e-15.

**What the obstacle course actually checks** (`benchmarks/run_complete_obstacle_course.m`):

| Obstacle | Check performed |
| :--- | :--- |
| 1. Patch test | Rigid-body energy and linear-field reproduction on an axis-aligned octree (trilinear elements, no cut cells) |
| 2. "Hertzian" contact | Flat punch on a block on 2×2×2 grids. Passes if the active set converged with at least one active pair and min gap ≥ −0.01. **No comparison with the Hertz solution** |
| 3. AMR | Geometric refinement near the re-entrant corner. Passes if the DOF count grows by more than 1.5×. **No error or convergence-rate measurement** |
| 4. NIST assembly | 2-body bonded assembly on 2×2×2 grids; checks that the solution is finite. Marked as passed if the STEP file is missing |
| 5. GPU solver | Matrix-free vs assembled matvec agreement (around 1e-16) and PCG convergence on 267 DOFs. **No speedup measurement** |

These are smoke tests, not validation. A verification suite with manufactured solutions on
curved immersed domains, small-cut robustness, contact patch tests, a real Hertz problem and
AMR rates is planned.

---

## Performance

**Element-level matrix-free matvec** (`y_e = w_e K_e p_e`), measured on an NVIDIA GeForce RTX 2050:

| Elements | DOFs | MATLAB CPU | MATLAB GPU | Rust CPU (1T) | Rust CPU (Rayon) | Rust WGPU | Rust CUDA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 500 | 12,000 | 0.052 ms | 0.084 ms | 0.070 ms | 0.202 ms | 0.048 ms | 0.012 ms |
| 2,000 | 48,000 | 0.080 ms | 0.096 ms | 0.263 ms | 0.292 ms | 0.061 ms | 0.016 ms |
| 8,000 | 192,000 | 0.222 ms | 0.126 ms | 1.064 ms | 0.550 ms | 0.085 ms | 0.034 ms |
| 32,000 | 768,000 | 1.027 ms | 0.243 ms | 4.333 ms | 1.145 ms | 0.325 ms | 0.120 ms |

Caveats:
- These time a single batched element matvec with a synthetic 24×24 element matrix (trilinear-hex size). They are kernel throughput figures, not solver or assembly timings.
- The DOF column is `24 × elements` with no shared DOFs.
- The MATLAB columns have not been re-run since the audit.

Earlier versions of this README listed 7,770× and 12,620× Rust speedups for the octree and the
obstacle course. Those figures compared full MATLAB computations with Rust placeholder code
and have been withdrawn. Some timing scripts in `src/fastformation/` also contain
extrapolated or modelled (unmeasured) values; each script documents this in its header.

---

## Corrections from the 2026-10-10 audit

- **3D weighted quadrature:** the previous 3D FastFormation assembly (`C_ijkl` Voigt map) had an indexing
  error that put ∂₃u₃ into γ₁₃. The resulting stiffness was 33% off even at uniform density. The
  in-house version is exact. 2D was unaffected.
- **Degree p ≥ 3 in the 3D immersed solvers:** the old 3×3×3 "template" shortcut is exact only for p ≤ 2.
  For p = 3 it gave 38% error on interior elements. Element matrices are now computed exactly for any degree.
- **2D cantilever load:** the external `forceCantileverCentered` also loaded quadrature points outside the
  intended patch, through a vector-subscript cross product. The corrected load changes 2D compliance values by about 0.3%.

---

## Rust port

`rust/` currently provides:
- The octree with 2:1 balancing, hanging-node constraint maps, inside/outside classification, a generic PCG solver, and the SIMP/OC update with its density filter.
- CAD labelling and topology utilities, and an egui BC-labelling GUI.
- CUDA and WGPU matvec kernels.

It does **not** yet contain a finite-element discretization. `solve_cad` and `topopt_3d` run on
placeholder operators and synthetic strain energies, and the Rust `obstacle_course` does not
reproduce the MATLAB checks. `docs/PORT_STATUS.md` tracks each module and the porting order.

```powershell
.\run_rust.ps1 build --release --manifest-path rust/Cargo.toml
.\run_rust.ps1 test  --manifest-path rust/Cargo.toml
```

---

## License

MIT. GeoPDEs (GPL) is neither bundled nor required.
