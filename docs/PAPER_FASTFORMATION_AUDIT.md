# FastFormation paper vs. code (audit 2026-10-10)

Paper: *Fast Formation of Isogeometric Galerkin Matrices and Sensitivities via Weighted
Quadrature and GPU Acceleration in Topology Optimization* (da Silva & Lenzi). Source read:
`OneDrive/Documents/FastFormation/FastFormation_TopOpt.tex`, version of 2026-10-08 18:55. The
PDF in Downloads is password-protected. Each claim below was traced to the script that
produces it.

## What is solid

- **2D weighted quadrature is correct.** `fast_stiffness_assembly` / `wq_form` reproduce exact Gauss assembly to about 1e-15 at uniform density (Table 1). This is re-verified without GeoPDEs (`benchmark_frobenius_norm`) and independently in Rust (`rust/tests/wq_vs_matlab.rs`).
- **The 2D p- and k-refinement timing tables** compare real WQ assembly against GeoPDEs Gauss assembly measured directly. The exception is the p-refinement "GPU 5.0 ms", which times only an elementwise `vals0 .* scale` on the GPU, not an assembly.
- **The 3D WQ indexing bug** found in the old `C_ijkl` Voigt map (33% error) **does not affect any number in the paper**, because no paper result runs 3D WQ assembly (see below).

## Claims that don't match what the scripts compute

1. **Algorithm described vs implemented.** The paper (eq. `interior_points`, `min_norm_weights`, `weights_deriv_relation`) describes:
   - quadrature points strictly inside elements (2–3 per element, none on interfaces);
   - a weighted minimum-norm solve with Z = diag(M̂ₐ hq);
   - derivative weights by recurrence.

   The code (`quadrule_stiff_fast`, now `src/iga/iga_wq_rules_1d.m`) implements the original Calabrò–Sangalli–Tani layout instead:
   - knots plus midpoints, so points lie on element interfaces;
   - an unweighted minimum-norm solve;
   - separate solves in the derivative space.

   The Option A argument ("each quadrature point maps uniquely to a single element") holds for the paper's layout but not for the code's. There, interface points are assigned to the right-hand element by `discretize`.
2. **"Fast CPU (WQ + Sum-Fact)" in the 3D scaling figure** (0.4357 s, "387× speedup", "evaluates the identical operator") comes from `benchmark_wallclock_scaling.m`. That script times a loop copying element template matrices multiplied by a scale factor. It does no WQ formation and no assembly.
3. **FP16 numbers** (8.67 ms, "3.24× over FP64", "19,450×") are **not measured**. The script sets `t_fp16 = t_fp32 / clamp(mem_ratio, 1.8, 2.8)`.
4. **"Standard Gauss measured directly up to 101,184 DOFs (168.63 s)".** The script measures up to 10k DOFs, hard-codes 168.63 s at 101,184 DOFs ("measured in task-974"), and power-law extrapolates the cases in between.
5. **Topology-optimization results labelled "CPU WQ"** (Table `end_to_end_results`, the 2D Option A/B figures, the 3D cantilever and arch dome) come from `topopt_iga_fast`, `topopt_iga_2d_mf`, `topopt_iga_3d`, `run_3d_cantilever_truss` and `run_3d_arch_dome`. These use Gauss element matrices precomputed once (`op_su_ev` or the 3×3×3 template, exact for p = 2) and scaled per iteration. They don't use weighted quadrature. The results themselves are valid; the label is wrong.
6. **The 2D cantilever load** used the external `forceCantileverCentered`, which also loaded points outside the patch. The corrected load changes 2D compliances by about 0.3%. The values quoted in the paper (J₀ = 259.81, …) are therefore from the old load.
7. **Proposition 1(4)** asserts O(h^{p+1}) via ‖E_quad‖ = O(h^{p+1}) without proof for piecewise-constant ρ. No convergence study in the paper tests it.

## Suggested fixes before submission

- Either implement the interior-point / weighted-min-norm / recurrence variant the paper describes, or describe the implemented Calabrò layout.
- Re-run the 3D scaling with real WQ assembly, now possible because the 3D `C_ijkl` is fixed. Measure FP16 or drop it. Measure or clearly mark extrapolated Gauss times.
- Run the topology-optimization benchmarks through `wq_form` / `fast_stiffness_assembly` if they are to be called WQ results, or relabel them. Re-run them with the corrected load.
- Add a convergence study (manufactured solution, piecewise-constant ρ) to support Proposition 1(4).
