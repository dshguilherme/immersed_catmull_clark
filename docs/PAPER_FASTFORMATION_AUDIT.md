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
6. ~~**The 2D cantilever load** changes the 2D compliances by about 0.3%.~~ **Retracted.** The rerun with the corrected load (`iga_cantilever_tip_load`) gives J₀ = 259.8107, identical to the paper's J₀ = 259.81, so the quoted 2D compliances are consistent with the corrected load.
7. **Proposition 1(4)** asserts O(h^{p+1}) via ‖E_quad‖ = O(h^{p+1}) without proof for piecewise-constant ρ. No convergence study in the paper tests it. The new study (below) shows it does not hold.

## Status after the fixes (2026-10-10)

**WQ layout (item 1): implemented as described in the paper.** `iga_wq_rules_1d` (and `rust/src/wq.rs`) now default to the interior layout:
- x_{k,m} = x_{k-1} + (2m−1)/(2q_k)·h_k with q_k = 3, and q_k = max(3, p+1) in the two end elements (needed for full row rank);
- W⁽⁰⁾ by weighted minimum norm with Z = diag(M̂ₐ(x_q)·h_q);
- W⁽¹⁾ by the derivative recurrence;
- exactness imposed on S_p^{p−2}, which contains both S_p and its derivatives, so W⁽⁰⁾ and W⁽¹⁾ serve all four stiffness terms.

The old layout remains available as `'calabro'` / `WqLayout::Calabro`. The batched `wq_setup` / `wq_form` now cover 3D: one sum-factorized contraction per derivative pair, exact, about 600× faster than the per-row loop.

**Table 1, rerun (`benchmark_frobenius_norm`, interior layout, against GeoPDEs `op_su_ev_tp` and against exact Gauss):**

| p | 10×5 | 20×10 | 40×20 | ρ random, interior | ρ random, Calabrò |
|---|---|---|---|---|---|
| 2 | 6.5e-16 | 1.1e-15 | 1.9e-15 | 3–4 % | 26–30 % |
| 3 | 5.5e-16 | 6.2e-16 | 9.5e-16 | 2.4–3.4 % | 18–24 % |
| 4 | 2.1e-15 | 2.2e-15 | 1.3e-15 | 0.9–1.1 % | 19–23 % |
| 5 | 3.9e-14 | 2.8e-14 | 3.2e-14 | 0.9–1.4 % | 15–21 % |

- The mesh columns give E_F against exact Gauss at uniform density. The GeoPDEs values are the same to within 5e-15, and asymmetry is 0. The p = 5 row is about 3e-14, slightly above the paper's ≤ 1.3e-15.
- The last two columns are ‖K_WQ − K_G‖_F/‖K_G‖_F with the same random element-wise density. This supports the paper's claim that interior points are needed for discontinuous coefficients.

**2D topology optimization, rerun through real WQ (`run_paper_topopt_2d`):**
- Option A: J₀ = 259.81 → J₅₀ = 54.66. The paper reports 54.60.
- Option B: J₀ = 259.81 → J₅₀ = 55.32, with ρ evaluated at the WQ points. The paper reports 57.13, obtained with Gauss element matrices and ρ sampled at element centres (P₁xP₂ᵀ), so Section 6.4.2 must describe the WQ-point evaluation instead.
- Timings must be re-measured on an idle machine.

**Proposition 1(4) study (`benchmark_wq_consistency`).** Piecewise-constant ρ on a fixed 4×2 block pattern, meshes 8×4 to 128×64, comparing u_WQ with u_Gauss on the same mesh:

| | p = 2 | p = 3 |
|---|---|---|
| \|J_WQ − J_G\|/J_G, finest | 2.5e-3, rate ≈ 1 | 6.0e-3, rate ≈ 1 |
| energy-norm difference, finest | 2.4e-2, rate ≈ 0.35–0.6 | 4.0e-2, rate ≈ 0.5 |

The WQ consistency error converges, but like O(h) in compliance and O(h^{1/2}) in energy, not O(h^{p+1}). This is consistent with an O(1) quadrature error confined to the strip of rows whose support crosses a material interface. Proposition 1(4) must be weakened accordingly. The Calabrò layout is still at 20–35% energy error at 128×64.

**Timing benchmarks:** rewritten to measure every reported number.
- `benchmark_p_refinement`, `benchmark_k_refinement`, `benchmark_wallclock_scaling`:
  - GeoPDEs Gauss with SIMP coefficients;
  - WQ formation on CPU and GPU (FP64/FP32), with setup reported separately;
  - the GPU element matrix-free apply, labelled as such;
  - no FP16, because MATLAB `gpuArray` has no half arithmetic;
  - no extrapolation.
- `benchmark_to_full_algorithm` CPU rows now use `topopt_iga_wq`.
- **Not yet run:** the machine was saturated by orphaned `MATLABWindow` processes, which made the GeoPDEs baseline about 20× slower than in the paper.

## Suggested fixes before submission

- Either implement the interior-point / weighted-min-norm / recurrence variant the paper describes, or describe the implemented Calabrò layout.
- Re-run the 3D scaling with real WQ assembly, now possible because the 3D `C_ijkl` is fixed. Measure FP16 or drop it. Measure or clearly mark extrapolated Gauss times.
- Run the topology-optimization benchmarks through `wq_form` / `fast_stiffness_assembly` if they are to be called WQ results, or relabel them. Re-run them with the corrected load.
- Add a convergence study (manufactured solution, piecewise-constant ρ) to support Proposition 1(4).
