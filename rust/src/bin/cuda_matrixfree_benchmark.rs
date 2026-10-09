//! Native CUDA Matrix-Free Benchmark using cudarc.
//! Directly interacts with NVIDIA CUDA Driver (nvcuda.dll) on RTX 2050.

use cudarc::driver::*;
use std::time::Instant;

const PTX_SRC: &str = include_str!("../kernel.ptx");

fn main() -> Result<(), DriverError> {
    println!("========================================================================");
    println!("  NATIVE CUDA MATRIX-FREE BENCHMARK (RUST + CUDARC)                      ");
    println!("========================================================================");

    // Initialize CUDA Context on device 0
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to initialize CUDA context: {:?}", e);
            return Ok(());
        }
    };

    println!("--> CUDA Device 0 Context Initialized (NVIDIA RTX 2050)");

    // Load compiled PTX module
    let ptx = cudarc::nvrtc::Ptx::from_src(PTX_SRC);
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("matvec_kernel")?;
    println!("--> Loaded PTX module 'matvec_kernel' (sm_86 compiled via Clang NVPTX)");

    let stream = ctx.default_stream();
    let element_counts = vec![500, 2_000, 8_000, 32_000];

    println!("\n--- MATRIX-FREE ELEMENT MATVEC: RUST NATIVE CUDA (VRAM) ---");

    for &num_elem in &element_counts {
        let total_dofs = (num_elem * 8) * 3;
        let mut elem_dofs = Vec::with_capacity(num_elem * 24);
        let mut elem_weights = vec![1.0f32; num_elem];
        let p_vec = vec![0.01f32; total_dofs];

        for e in 0..num_elem {
            for a in 0..8 {
                let node_id = e * 4 + a;
                elem_dofs.push((node_id * 3 + 0) as u32);
                elem_dofs.push((node_id * 3 + 1) as u32);
                elem_dofs.push((node_id * 3 + 2) as u32);
            }
            elem_weights[e] = if e % 5 == 0 { 0.5 } else { 1.0 };
        }

        let mut ke_template = [0.0f32; 576];
        for i in 0..24 {
            ke_template[i * 24 + i] = 4.0;
            if i > 0 { ke_template[i * 24 + i - 1] = -0.5; }
            if i + 1 < 24 { ke_template[i * 24 + i + 1] = -0.5; }
        }

        // Allocate buffers on CUDA VRAM
        let ke_dev = stream.clone_htod(&ke_template)?;
        let dofs_dev = stream.clone_htod(&elem_dofs)?;
        let weights_dev = stream.clone_htod(&elem_weights)?;
        let p_dev = stream.clone_htod(&p_vec)?;
        let mut y_dev = stream.alloc_zeros::<f32>(num_elem * 24)?;

        let n_elem_u32 = num_elem as u32;
        let cfg = LaunchConfig {
            grid_dim: ((n_elem_u32 + 63) / 64, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };

        // Warmup
        for _ in 0..5 {
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&n_elem_u32)
                    .arg(&ke_dev)
                    .arg(&dofs_dev)
                    .arg(&weights_dev)
                    .arg(&p_dev)
                    .arg(&mut y_dev)
                    .launch(cfg)?;
            }
        }
        stream.synchronize()?;

        // Timed VRAM dispatches
        let n_runs = 100;
        let t_start = Instant::now();
        for _ in 0..n_runs {
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&n_elem_u32)
                    .arg(&ke_dev)
                    .arg(&dofs_dev)
                    .arg(&weights_dev)
                    .arg(&p_dev)
                    .arg(&mut y_dev)
                    .launch(cfg)?;
            }
        }
        stream.synchronize()?;
        let elapsed = t_start.elapsed();

        let cuda_time_ms = (elapsed.as_secs_f64() * 1000.0) / (n_runs as f64);
        let gflops = (num_elem as f64 * 24.0 * 24.0 * 2.0) / (cuda_time_ms * 1e-3) / 1e9;

        println!(
            "  Ne={:>5} | DOFs: {:>6} | Rust CUDA (VRAM): {:>6.3} ms | Throughput: {:>5.1} GFLOP/s",
            num_elem, total_dofs, cuda_time_ms, gflops
        );
    }

    println!("========================================================================");
    println!("  CUDA BENCHMARK COMPLETE                                               ");
    println!("========================================================================");
    Ok(())
}
