# CLAUDE.md

Immersed isogeometric analysis (IGA) on CAD B-Reps: Catmull-Clark / cubic B-spline basis on a
background grid or 2:1 octree, cut-cell quadrature, ghost penalty, Nitsche BCs, multi-body contact,
matrix-free GPU PCG, and SIMP topology optimization. The MATLAB code is the reference implementation;
`rust/` is a port in progress. Port status per module: `docs/PORT_STATUS.md` — keep it current.

## Layout
- `src/iga/` — in-house B-spline kernel (box spaces, exact elasticity element matrices, loads, WQ rules). Replaced GeoPDEs; numbering matches GeoPDEs conventions. Verified by `tests/testIgaKernel.m`.
- `src/brep/` — CAD import (STEP/MSH/VTU/STL) via headless Gmsh (`*.py` called from MATLAB).
- `src/catmull_clark/` — subdivision, CC projection matrices (2D/3D), in-house Cox-de Boor.
- `src/fastformation/` — weighted quadrature (`wq_setup`/`wq_form`), fast/GPU assembly, structured topopt. Many `test_*`/`benchmark_*` scripts here are exploratory, not part of the suite.
- `src/immersed/` — classification, cut-cell quadrature, ghost penalty, BC engine, octree + MPC, AMR loop, contact, GPU PCG, immersed topopt.
- `tests/` — MATLAB unit tests (`matlab.unittest`); `benchmarks/` — obstacle course & paper benchmarks; `examples/` — demos.
- `rust/` — crate `immersed_iga` (lib in `src/*.rs`, executables in `src/bin/`).
- `scripts/` — Python: matplotlib publication figures, PyVista BC labeler (older than the Rust GUI).
- `Models/` (NIST STEP files), `Articles/` (paper .tex/.pdf), `figures/` exist locally but are **git-ignored by design** — never commit pictures, CAD, or manuscript files (see `.gitignore`).

## Commands
MATLAB (R2025a on PATH):
```
matlab -batch "run('tests/run_all_tests.m');"
matlab -batch "addpath(genpath('src')); addpath('tests','benchmarks'); run_complete_obstacle_course();"
```
No GeoPDEs dependency: never reintroduce GeoPDEs/NURBS-toolbox calls (GPL; the repo is MIT) or hard-coded `addpath` to folders outside the repo. GeoPDEs is allowed only as an optional, labelled timing/validation baseline via `benchmarks/geopdes_baseline_available.m` (env var `GEOPDES_PATH`); `benchmarks/external_geopdes/` holds scripts that test third-party code.

Rust — always go through `run_rust.ps1` (it sets up the LLVM-MinGW GNU toolchain; plain `cargo` fails to link):
```
.\run_rust.ps1 check --manifest-path rust/Cargo.toml --all-targets
.\run_rust.ps1 test  --manifest-path rust/Cargo.toml
.\run_rust.ps1 run --release --manifest-path rust/Cargo.toml --bin <obstacle_course|solve_cad|topopt_3d|cad_gui|cuda_matrixfree_benchmark|wgpu_matrixfree_benchmark>
```
The CUDA binary loads the prebuilt `rust/src/kernel.ptx` (sm_86, from `kernel.c`). Bins read and write `benchmarks/*.json`.

## Conventions & cautions
- Commit style: conventional commits (`feat(rust): …`, `feat(gui): …`, `perf(rust): …`).
- Known method limitations (keep in mind before claiming accuracy): cut cells use volume-fraction scaling of the full-cell stiffness, not exact cut-cell integration; `assemble_nitsche_dirichlet_3d` is a penalty-only trilinear-trace approximation; the obstacle course is a smoke test (see README "Verification").
- **Correctness over appearance.** Several existing Rust executables and benchmark tables print "PASSED" or report speedups from placeholder physics (synthetic operators and strain energies, see PORT_STATUS). Don't add more of that: a verification check must test the real quantity against an analytical or MATLAB reference, and benchmarks must compare like-for-like work. If something is a stub, label it as one in the code and in its output.
- When porting a module, validate it against the MATLAB output on the same input (export to JSON/CSV), not only against self-consistency.
- Rust: keep the core library free of GUI/GPU dependencies where possible; `rayon` is fine in the core.
- Gauntlet (`.gauntlet/`) is installed for PRD-driven build/review runs; its `test_command` is the MATLAB suite only.
