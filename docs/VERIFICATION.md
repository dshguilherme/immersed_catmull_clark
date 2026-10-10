# Verification studies of the immersed method (2026-10-10)

Driver: `.\run_rust.ps1 run --profile fast --manifest-path rust/Cargo.toml --bin verification_studies` (add `--quick` for a short run).
Results go to `benchmarks/verification_results.json`.

The studies use the Rust immersed solver. Its parity tests show it reproduces the MATLAB
`solve_immersed_iga_3d` discretization: volume-fraction cut cells, p = 2 B-splines and
Emin = 1e-4 in empty cells. Boundary values are imposed by a penalty Dirichlet term on the
immersed surface, β = 1e3·E/h. These are measurements, not pass/fail checks.

## S1: patch test on a cut domain

The domain is a tilted box with a linear displacement field u imposed on the whole immersed surface. A consistent method reproduces u to round-off.

| Stabilization | cells | rel. L2 error |
|---|---|---|
| none | 8 / 16 | 4.8e-3 / 5.2e-3 |
| legacy (MATLAB) | 8 / 16 | 6.2e-3 / 6.0e-3 |
| ghost penalty | 8 / 16 | 4.2e-3 / 5.0e-3 |

Diagnostic, on an axis-aligned box with no stabilization:
- The error is **independent of the penalty** (β/(E/h) = 1e2, 1e3 and 1e5 give the same result).
- It is present **away from the boundary**: interior (d > 2h) 8e-4, boundary layer 1.3e-3 at 16 cells.

**Conclusion:** the volume-fraction cut-cell treatment isn't patch-test consistent. Scaling the full-cell stiffness by w doesn't represent the cut geometry.

## S2: manufactured solution on an immersed sphere

R = 0.4, u = 0.01·(sin πx cos πy, sin πy cos πz, sin πz cos πx), with the matching body force. The body force is integrated as Σ w_e ∫_e f·v. For p = 2 the optimal L2 rate is 3.

| cells | total rel. L2 | interior (> 2h from surface) | boundary layer |
|---|---|---|---|
| 6 | 5.8e-2 | 2.9e-2 | 6.2e-2 |
| 9 | 5.9e-2 | 4.9e-3 | 6.2e-2 |
| 12 | 1.2e-1 | 4.2e-3 | 1.3e-1 |
| 18 | 2.6e-1 | 4.8e-3 | 3.0e-1 |

With the ghost penalty (γ = 1e-2) the 12-cell total is 9.6e-2.

**Conclusion:** the method doesn't converge.
- The interior error plateaus at about 5e-3, the same inconsistency as S1.
- The error near the boundary *grows* under refinement. This fits the small-cut problem: cut cells whose sampled volume fraction is 0 (see S3) are held only by Emin and the surface penalty.
- Exact cut-cell quadrature would confirm this. That changes the method, so it's left for a decision.

## S3: small cuts

Unit box on an 8³ grid, with the outer cell layer cut at fraction ε. The numbers are the Lanczos estimate of cond(D^-1/2 K D^-1/2) and the PCG iterations.

| Stabilization | ε = 0.3 | ε = 0.01 |
|---|---|---|
| none | 9.1e3 (717 it) | 2.7e4 (745 it) |
| legacy (MATLAB) | 1.6e3 (295 it) | 5.5e3 (333 it) |
| ghost penalty | 5.6e3 (530 it) | 1.2e4 (510 it) |

Notes:
- With 4×4×4 midpoint sampling, every cut cell with ε < 1/8 gets **weight 0**, so small cuts degenerate into Emin cells.
- The legacy stabilization improves conditioning only by adding stiffness that breaks accuracy, as the immersed cantilever showed: about 1000× too stiff at E = 1.

## Recommended next steps (pending decision)

1. Replace volume-fraction scaling with exact cut-cell quadrature, for example octree or moment-fitting sub-cell integration of the actual integrand.
2. Replace the penalty Dirichlet with symmetric Nitsche, and keep the consistent ghost penalty (Rust `immersed::ghost_penalty`).
3. Re-run S1–S3. Targets: S1 at round-off, S2 at rate ≈ 3, S3 with conditioning bounded independently of ε.
