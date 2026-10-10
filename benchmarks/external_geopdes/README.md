# External GeoPDEs / WQ-library scripts

These scripts exercise third-party code (GeoPDEs and the GPL weighted-quadrature
routines that ship next to it, e.g. `Elasticity_WQ`) rather than this
repository, so they cannot run without those packages. Nothing in `src/`
depends on them. Put GeoPDEs on the path (or set `GEOPDES_PATH` and call
`geopdes_baseline_available`) before running.

- `test_gpu_approaches.m`: compares the external `Elasticity_WQ` with GeoPDEs `op_su_ev_tp`.
- `test_gpu_vectorization.m`: **broken** before the 2026-10-10 migration. It calls
  `Stiff_fast`, which is not defined anywhere, and the script ends at line 52. It is
  kept for reference only.
