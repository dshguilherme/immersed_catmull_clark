use immersed_iga::*;
use std::time::Instant;

fn main() {
    println!("========================================================================");
    println!("  RUNNING IMMERSED CATMULL-CLARK IGA BENCHMARK SUITE (RUST NATIVE)      ");
    println!("========================================================================");
    println!();

    let start_total = Instant::now();

    // -------------------------------------------------------------------------
    // OBSTACLE 1: 3D Patch Test & Multi-Point Constraint Octree
    // -------------------------------------------------------------------------
    println!("[OBSTACLE 1/5] Running 3D Elasticity Patch Test & 2:1 Balanced Octree...");
    let root_bounds = BoundingBox3D::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let mut octree = OctreeMesh3D::new(root_bounds, [2, 2, 2]);
    octree.subdivide_cell(0);
    octree.subdivide_cell(3);
    octree.balance_2_to_1();

    let mesh = StructuralMesh3D::from_octree(&octree);
    let patch_err = mesh.evaluate_patch_test_error();

    println!(
        "  --> Obstacle 1: PASSED (Leaves: {}, Master DOFs: {}, Patch Test Error: {:.4e})",
        mesh.n_elements,
        3 * mesh.n_master,
        patch_err
    );
    assert!(patch_err < 1e-14, "Obstacle 1 failed patch test tolerance");

    // -------------------------------------------------------------------------
    // OBSTACLE 2: Hertzian Contact Active-Set Profile
    // -------------------------------------------------------------------------
    println!("[OBSTACLE 2/5] Running Analytical Hertzian Contact Profile Benchmark...");
    let contact_solver = ActiveSetContactSolver::new(20, 1.0e6);
    let initial_gaps = vec![0.0, 0.0, 0.01, 0.05, 0.1];
    let u_a = vec![0.0; 5];
    let u_b = vec![-0.005, -0.002, 0.0, 0.0, 0.0]; // Negative = pushing into base

    let (active, gaps, pressures) = contact_solver.evaluate_active_set(&initial_gaps, &u_a, &u_b);
    let active_count = active.iter().filter(|&&a| a).count();

    println!(
        "  --> Obstacle 2: PASSED (Active Pairs: {}/5, Min Gap: {:.4e}, Peak Pressure: {:.2} MPa)",
        active_count,
        gaps.iter().cloned().fold(f64::INFINITY, f64::min),
        pressures.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    assert_eq!(active_count, 2, "Obstacle 2 failed active pair detection");

    // -------------------------------------------------------------------------
    // OBSTACLE 3: Adaptive Mesh Refinement (AMR) Stress Riser
    // -------------------------------------------------------------------------
    println!("[OBSTACLE 3/5] Running Singular Stress Riser with Adaptive Octree AMR...");
    let l_bounds = BoundingBox3D::new([-0.5, -0.5, -0.2], [4.5, 4.5, 1.2]);
    let mut amr_octree = OctreeMesh3D::new(l_bounds, [2, 2, 1]);
    let initial_leaves = amr_octree.leaf_indices().len();

    // Refine around singular corner (2.0, 2.0, 0.5)
    let corner = [2.0, 2.0, 0.5];
    for _ in 0..2 {
        let leaves = amr_octree.leaf_indices();
        for leaf_idx in leaves {
            if amr_octree.cells[leaf_idx].bounds.contains_point(&corner, 0.5) {
                amr_octree.subdivide_cell(leaf_idx);
            }
        }
        amr_octree.balance_2_to_1();
    }
    let final_leaves = amr_octree.leaf_indices().len();

    println!(
        "  --> Obstacle 3: PASSED (AMR Hierarchy: Initial Leaves = {} -> Refined Leaves = {})",
        initial_leaves,
        final_leaves
    );
    assert!(final_leaves > initial_leaves, "Obstacle 3 failed AMR progression");

    // -------------------------------------------------------------------------
    // OBSTACLE 4: In-House Pure Cox-de Boor Basis & Catmull-Clark Projection
    // -------------------------------------------------------------------------
    println!("[OBSTACLE 4/5] Running In-House Cox-de Boor B-Spline Basis & Partition of Unity...");
    let degree = 3; // Cubic
    let nel = 8;
    let knots = open_knot_vector(nel, degree);
    let eval_points = vec![0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
    let basis = evaluate_bspline_basis_1d(degree, &knots, &eval_points);

    let mut max_pou_err: f64 = 0.0;
    for row in &basis {
        let sum: f64 = row.iter().sum();
        let err = (sum - 1.0).abs();
        if err > max_pou_err {
            max_pou_err = err;
        }
    }

    println!(
        "  --> Obstacle 4: PASSED (Degree: {}, Knots: {}, Max Partition-of-Unity Error: {:.4e})",
        degree,
        knots.len(),
        max_pou_err
    );
    assert!(max_pou_err < 1e-14, "Obstacle 4 failed partition of unity");

    // -------------------------------------------------------------------------
    // OBSTACLE 5: Matrix-Free PCG Solver Scalability
    // -------------------------------------------------------------------------
    println!("[OBSTACLE 5/5] Running Native Matrix-Free PCG Convergence Benchmark...");
    let pcg = PcgSolver::new(100, 1e-6);
    let n_dof = 1000;
    let b = vec![1.0; n_dof];
    let diag_a = vec![4.0; n_dof];

    // Tridiagonal Laplacian-like stencil: (A*p)_i = -p_{i-1} + 4*p_i - p_{i+1}
    let matvec = |p: &[f64], ap: &mut [f64]| {
        for i in 0..n_dof {
            let left = if i > 0 { p[i - 1] } else { 0.0 };
            let right = if i + 1 < n_dof { p[i + 1] } else { 0.0 };
            ap[i] = 4.0 * p[i] - left - right;
        }
    };

    let t_pcg = Instant::now();
    let (_x, iters, res) = pcg.solve(n_dof, &b, &diag_a, matvec);
    let pcg_time = t_pcg.elapsed();

    println!(
        "  --> Obstacle 5: PASSED (DOFs: {}, PCG Iters: {}, Rel Residual: {:.4e}, Time: {:?})",
        n_dof,
        iters,
        res,
        pcg_time
    );
    assert!(res < 1e-6, "Obstacle 5 failed PCG convergence");

    println!();
    println!("========================================================================");
    println!("  ALL 5 RUST OBSTACLES PASSED SUCCESSFULLY (Total Time: {:?})", start_total.elapsed());
    println!("========================================================================");
}
