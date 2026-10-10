# Immersed / finite-cell isogeometric analysis: what the literature prescribes

Written 2026-10-10 in response to the failures in `docs/VERIFICATION.md`. The current method
fails the cut-domain patch test, doesn't converge on a sphere, and degrades with small cuts.
The literature treats this as a well-documented problem with three standard ingredients:
geometry-resolving integration of cut cells, weak (Nitsche) boundary conditions, and a
remedy for small cuts.

## 1. Integrate the cut cells; don't scale them

The finite cell method (FCM) and immersed IGA keep the background basis but **integrate each
cut cell only over its physical part**:
- adaptive quadrature: octree/quadtree bisection of cut cells plus Gauss points on sub-cells;
- or tessellating the cut region;
- or moment-fitting rules.

The empty part gets a small α (about 1e-8 to 1e-10) only as a fictitious-domain regularization.

- **Review:** Schillinger & Ruess, *The finite cell method: a review in the context of higher-order structural analysis of CAD and image-based geometric models*, Arch. Comput. Methods Eng. 22 (2015) 391–455, doi:10.1007/s11831-014-9115-y.
- **Octree plus tessellation with error control:** Divi, Verhoosel, Reali, Auricchio, van Brummelen, *Error-estimate-based adaptive integration for immersed isogeometric analysis*, Comput. Math. Appl. (2020), arXiv:1911.11519. It uses recursive bisection of cut cells, drops sub-cells fully outside the domain, tessellates the lowest level, and places integration points from a Strang-lemma error estimate. Its examples are 2D and 3D elasticity.
- **Moment fitting** (few points per cut cell, works from a CAD triangulation or voxels): Joulaian, Hubrich, Düster, Comput. Mech. 57 (2016). Extensions add optimized point positions and a smart octree (Hubrich et al., Comput. Mech. 60, 2017) and non-negative weights for nonlinear problems (Garhuom & Düster, Comput. Mech. 70, 2022).

**Implication for us:** the MATLAB/Rust "volume fraction × full-cell stiffness" is a crude
approximation of this. It doesn't integrate the stiffness over the physical part, which is
exactly why S1 (the patch test) fails independently of the penalty. Replace it with octree
sub-cell Gauss integration, which reuses the existing ray-parity inside test at sub-cell
level. Moment fitting can come later as an optimization.

## 2. Nitsche's method for Dirichlet conditions on the immersed boundary

Penalty-only Dirichlet is variationally inconsistent. The standard is symmetric Nitsche:
consistency term, symmetry term, and a stabilization β with β > C·(trace-inequality constant).

- Ruess, Schillinger, Bazilevs, Varduhn, Rank, *Weakly enforced essential boundary conditions for NURBS-embedded and trimmed NURBS geometries on the basis of the finite cell method*, IJNME (2013).
- **Trimmed IGA:** Nitsche loses stability on small trims unless stabilized. A minimal stabilization that restores well-posedness with optimal a-priori estimates is given by Buffa, Puppi, Vázquez, *A minimal stabilization procedure for isogeometric methods on trimmed geometries*, SIAM J. Numer. Anal. 58(5) (2020) 2711–2735, doi:10.1137/19M1244718, arXiv:1902.04937.

## 3. Small cuts: ghost penalty, aggregation or extension

Small cut cells give basis functions with tiny support inside the domain. The consequences
are ill-conditioning and a Nitsche β that blows up. de Prenter, Verhoosel & van Brummelen
derive cond(K) as a function of the smallest volume fraction (*Condition number analysis and
preconditioning of the finite cell method*, CMAME 316 (2017) 297–327, arXiv:1601.05129).

The remedies are reviewed in de Prenter, Verhoosel, van Brummelen, Larson, Badia, *Stability
and conditioning of immersed finite element methods: analysis and remedies*, Arch. Comput.
Methods Eng. (2023), arXiv:2208.08538:
- **Ghost penalty** (Burman 2010). For C^{p-1} splines only the **p-th normal-derivative jump** is non-zero on ghost faces. The scaling is h^{2p-1}, with γ around 1e-3 to 1e-2 in the spline literature (e.g. arXiv:2208.14994, 2212.00882, 1807.07380). This is exactly what `rust/src/immersed.rs::ghost_penalty` implements. The MATLAB `assemble_ghost_penalty_stabilization` is not this.
- **Cell aggregation** (AgFEM): Badia, Verdugo, Martín, CMAME 336 (2018) 533–553, arXiv:1709.09122. Basis functions of badly cut cells are constrained to those of interior neighbours. This gives body-fitted conditioning and optimal convergence without stabilization terms, and it has parallel and adaptive-tree versions.
- **Extended B-splines** (web-splines): Höllig, Reif, Wipper (2001–2002). Outer B-splines are tied to inner ones through local polynomial extension, which gives a uniformly stable basis.
- **Preconditioning alone:** SIPIC or additive-Schwarz preconditioners (de Prenter et al. 2017; multigrid in Comput. Mech. 2019) restore iterative convergence but not Nitsche stability.

## Recommended path for this code

1. **Cut-cell quadrature:** recursive octree bisection of cut cells (depth 3–5, physical sub-cells only, tessellation optional), with p+1 Gauss points per sub-cell. This needs per-element stiffness for cut cells, which the type-based element matrices already support: interior cells keep the shared type matrix. α = 1e-8 for the empty part.
2. **Nitsche** (symmetric) on the immersed surface, using the subdivided surface quadrature we already have. β from a local eigenvalue estimate or a fixed γ·E/h together with the ghost penalty, which keeps the trace inequality uniform.
3. **Ghost penalty:** keep the p-th-derivative version, which is already verified to vanish on polynomials of degree ≤ p. Retire the MATLAB legacy routine.
4. **Re-run** `verification_studies`. Targets: S1 at round-off, S2 with L2 rate ≈ p+1 = 3, S3 with cond bounded independently of ε.
5. **Later:** moment fitting for speed; AgFEM as an alternative to the ghost penalty if parameter tuning proves fragile.
