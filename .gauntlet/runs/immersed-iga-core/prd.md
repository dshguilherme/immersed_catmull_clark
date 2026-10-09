# PRD: Immersed Catmull-Clark IGA Production Stack & Multi-Body Assembly Engine

**Status:** Approved Draft v1.0  
**Author:** Guilherme H. da Silva & Antigravity  
**Date:** 2026-10-09  
**Working name:** immersed-iga-core  
**Target Repository:** https://github.com/dshguilherme/immersed_catmull_clark  

---

## §1 Overview

### 1.1 Problem Statement
Traditional Isogeometric Analysis (IGA) promised to eliminate mesh generation by computing directly on CAD splines. However, in practice, converting industrial boundary representation (B-Rep) CAD solids (trimmed NURBS surfaces) into analysis-suitable watertight volumetric parameterizations remains an intractable bottleneck. 
Immersed boundary methods circumvent volumetric meshing by embedding the CAD model into a background Cartesian grid. However, existing immersed spline formulations lack:
1. A unified boundary condition handler for all three fundamental types (Dirichlet, Neumann, Robin) on arbitrary immersed CAD surfaces.
2. Robust multi-body assembly kinematics supporting non-conforming background grids with disparate feature sizes.
3. Unilateral contact mechanics (non-penetration inequality $g_n \ge 0, p_n \le 0$) without saddle-point instabilities or Lagrange multipliers.
4. Matrix-free GPU scalability on adaptively refined octrees with multi-point constraints (MPC).
5. A rigorous obstacle course benchmark suite validating accuracy, stability, and speed against analytical and industrial benchmarks.

### 1.2 Solution Summary
We deliver a complete, analysis-ready software stack in MATLAB with FastFormation GPU acceleration:
- **Phase 1: Unified Boundary Condition Engine (`apply_boundary_conditions.m`)**: Evaluates Dirichlet (weak Nitsche), Neumann (distributed surface traction/pressure), and Robin (elastic foundation) boundary conditions from CAD face tags or coordinate filters.
- **Phase 2: Multi-Body Assembly & Semi-Smooth Newton Contact (`solve_assembly_contact_3d.m`)**: Automates $N$-body CAD assemblies with independent octree background meshes and solves unilateral contact via an active-set semi-smooth Newton iteration.
- **Phase 3: Matrix-Free GPU Octree Operator (`gpu_octree_matvec.m`)**: Evaluates the action $\bm{y} = \bm{T}_{3D}^T \bm{K}_{\text{uncon}} \bm{T}_{3D} \bm{p}$ entirely on GPU memory without assembling the global sparse matrix, scaling to millions of DOFs.
- **Phase 4: Publication Benchmark Obstacle Course (`run_complete_obstacle_course.m`)**: A 5-test gauntlet verifying Patch Tests, Hertzian contact, singular stress risers, industrial NIST STEP assemblies, and GPU wall-clock scaling.

### 1.3 The Assumption This Validates
That immersed Catmull-Clark IGA on 2:1 balanced octrees can simulate complex, multi-body CAD assemblies with unilateral contact, arbitrary boundary conditions, and millions of DOFs on a single desktop GPU in minutes—completely bypassing volumetric mesh generation.

---

## §2 Goals and Non-Goals

### 2.1 Goals
| ID | Goal |
| :--- | :--- |
| **G1** | Provide a single unified interface for Dirichlet, Neumann, and Robin boundary conditions on immersed CAD surfaces. |
| **G2** | Enable automated simulation of $N$-body assemblies with non-conforming background octrees and disparate feature scales. |
| **G3** | Implement robust unilateral contact ($g_n \ge 0$) via positive-definite Nitsche active-set semi-smooth Newton iterations. |
| **G4** | Accelerate the MPC-projected octree system via a matrix-free GPU Preconditioned Conjugate Gradient (PCG) solver. |
| **G5** | Pass an exhaustive 5-part verification obstacle course matching analytical solutions and condition number bounds $\kappa = \mathcal{O}(1)$. |

### 2.2 Non-Goals
- Large-strain geometric nonlinearity or finite plasticity (v1 targets linear elasticity with finite contact).
- Self-contact within a single deforming body (v1 targets multi-component assembly interfaces).
- Commercial GUI frontend (v1 is an analysis-ready MATLAB/Python API).

---

## §3 System Architecture & Modules

### 3.1 New Modules
1. `src/immersed/apply_boundary_conditions.m`: Unified BC dispatcher and assembler.
2. `src/immersed/assemble_robin_bc_3d.m`: Weak Robin / spring foundation operator.
3. `src/immersed/assemble_neumann_bc_3d.m`: Distributed traction and pressure work vector.
4. `src/immersed/solve_assembly_contact_3d.m`: Assembly manager & semi-smooth Newton contact solver.
5. `src/immersed/gpu_octree_matvec.m`: GPU matrix-free octree action evaluator.
6. `benchmarks/run_complete_obstacle_course.m`: End-to-end 5-test validation harness.

---

## §4 Functional Requirements

- **FR-1 (Boundary Conditions)**: Given any CAD surface and BC specification, `apply_boundary_conditions.m` must assemble the corresponding consistent stiffness matrices and load vectors.  
  *Acceptance*: Pass Patch Test with zero spurious strain and exact reaction forces.
- **FR-2 (Unilateral Contact)**: Given two interacting bodies, `solve_assembly_contact_3d.m` must prevent penetration ($g_n \ge -\text{tol}$) and produce non-negative contact pressures ($p_n \ge 0$).  
  *Acceptance*: Match Hertzian analytical contact pressure within 5% error.
- **FR-3 (GPU Octree Matrix-Free)**: Evaluate $\bm{y} = \bm{K} \bm{p}$ on GPU without assembling $\bm{K}_{\text{master}}$.  
  *Acceptance*: Residual $\| \bm{y}_{\text{GPU}} - \bm{y}_{\text{exact}} \| / \| \bm{y}_{\text{exact}} \| < 10^{-10}$ and $>5\times$ speedup over CPU sparse solve on $10^5+$ DOFs.
- **FR-4 (Obstacle Course)**: Execute the 5-test suite without manual intervention.  
  *Acceptance*: All 5 tests report PASS with automated verification metrics.
