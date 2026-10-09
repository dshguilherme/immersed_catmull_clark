//! High-Performance Matrix-Free GPU Matvec & PCG Solver using WGPU Compute Shaders.
//! Executes level-by-level tensor contractions directly on GPU VRAM (Vulkan / DX12 / Metal).

use bytemuck::{Pod, Zeroable};
use pollster::block_on;
use rayon::prelude::*;
use std::time::Instant;
use wgpu::util::DeviceExt;

const SHADER_SRC: &str = r#"
// Level-by-level matrix-free element matvec compute kernel
// Evaluates y_e = w_e * K_e * p_e in parallel across all active elements

struct Uniforms {
    num_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read> ke_template: array<f32, 576>; // 24x24 element stiffness
@group(0) @binding(2) var<storage, read> elem_dofs: array<u32>;        // [num_elements * 24]
@group(0) @binding(3) var<storage, read> elem_weights: array<f32>;     // [num_elements]
@group(0) @binding(4) var<storage, read> p_vector: array<f32>;         // [total_dofs]
@group(0) @binding(5) var<storage, read_write> y_elem: array<f32>;     // [num_elements * 24]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let elem_idx = global_id.x;
    if (elem_idx >= params.num_elements) {
        return;
    }

    let w = elem_weights[elem_idx];
    let offset = elem_idx * 24u;

    // Gather local p_e from global p_vector
    var p_local: array<f32, 24>;
    for (var i = 0u; i < 24u; i = i + 1u) {
        let dof = elem_dofs[offset + i];
        p_local[i] = p_vector[dof];
    }

    // Local matrix-vector contraction: y_local = w * K_template * p_local
    for (var i = 0u; i < 24u; i = i + 1u) {
        var sum: f32 = 0.0;
        let row_offset = i * 24u;
        for (var j = 0u; j < 24u; j = j + 1u) {
            sum = sum + ke_template[row_offset + j] * p_local[j];
        }
        y_elem[offset + i] = w * sum;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    num_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct WgpuMatrixFreeContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub adapter_name: String,
}

impl WgpuMatrixFreeContext {
    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let adapter_name = adapter.get_info().name;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("MatrixFree Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatrixFree Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatrixFree BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MatrixFree PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatrixFree ComputePipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            adapter_name,
        })
    }

    /// Executes matrix-free matvec on GPU device
    pub fn compute_matvec_gpu(
        &self,
        num_elements: usize,
        total_dofs: usize,
        ke_template: &[f32; 576],
        elem_dofs: &[u32],
        elem_weights: &[f32],
        p_vector: &[f32],
    ) -> Vec<f32> {
        let uniforms = Uniforms {
            num_elements: num_elements as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let u_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let ke_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KE Buffer"),
            contents: bytemuck::cast_slice(ke_template),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let dof_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ElemDOFs Buffer"),
            contents: bytemuck::cast_slice(elem_dofs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let w_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weights Buffer"),
            contents: bytemuck::cast_slice(elem_weights),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let p_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("P Vector Buffer"),
            contents: bytemuck::cast_slice(p_vector),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (num_elements * 24 * std::mem::size_of::<f32>()) as u64;
        let y_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Y Elem Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ke_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dof_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: p_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: y_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((num_elements as u32) + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&y_buf, 0, &readback_buf, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        // Readback
        let buffer_slice = readback_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let y_local: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();

        // Scatter local element vectors to global y
        let mut y_global = vec![0.0f32; total_dofs];
        for e in 0..num_elements {
            let offset = e * 24;
            for i in 0..24 {
                let dof = elem_dofs[offset + i] as usize;
                y_global[dof] += y_local[offset + i];
            }
        }

        y_global
    }
}

fn main() {
    println!("========================================================================");
    println!("  MATRIX-FREE GPU WGPU / COMPUTE SHADER BENCHMARK (RUST NATIVE)          ");
    println!("========================================================================");

    let gpu_ctx = block_on(WgpuMatrixFreeContext::new());
    if gpu_ctx.is_none() {
        eprintln!("Warning: No compatible WGPU adapter found. Exiting benchmark.");
        return;
    }
    let gpu = gpu_ctx.unwrap();
    println!("--> Active GPU Device: {}", gpu.adapter_name);

    // Benchmark across various problem sizes: 500 to 32,000 elements
    let element_counts = vec![500, 2_000, 8_000, 32_000];

    println!("\n--- [1] MATRIX-FREE MATVEC: RUST CPU vs WGPU COMPUTE SHADER ---");
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

        // Benchmark Rust CPU Matvec (Single-threaded)
        let n_cpu_runs = 100;
        let mut y_cpu = vec![0.0f32; total_dofs];
        let t_cpu_start = Instant::now();
        for _ in 0..n_cpu_runs {
            y_cpu.fill(0.0);
            for e in 0..num_elem {
                let w = elem_weights[e];
                let offset = e * 24;
                let mut p_loc = [0.0f32; 24];
                for i in 0..24 {
                    p_loc[i] = p_vec[elem_dofs[offset + i] as usize];
                }
                for i in 0..24 {
                    let mut sum = 0.0f32;
                    let row = i * 24;
                    for j in 0..24 {
                        sum += ke_template[row + j] * p_loc[j];
                    }
                    y_cpu[elem_dofs[offset + i] as usize] += w * sum;
                }
            }
        }
        let cpu_time_ms = (t_cpu_start.elapsed().as_secs_f64() * 1000.0) / (n_cpu_runs as f64);

        // Benchmark Rust CPU Rayon Multi-threaded Matvec
        let mut y_elem_rayon = vec![0.0f32; num_elem * 24];
        let t_rayon_start = Instant::now();
        for _ in 0..n_cpu_runs {
            y_elem_rayon.par_chunks_mut(24)
                .enumerate()
                .for_each(|(e, y_loc)| {
                    let w = elem_weights[e];
                    let offset = e * 24;
                    let mut p_loc = [0.0f32; 24];
                    for i in 0..24 {
                        p_loc[i] = p_vec[elem_dofs[offset + i] as usize];
                    }
                    for i in 0..24 {
                        let mut sum = 0.0f32;
                        let row = i * 24;
                        for j in 0..24 {
                            sum += ke_template[row + j] * p_loc[j];
                        }
                        y_loc[i] = w * sum;
                    }
                });
        }
        let rayon_time_ms = (t_rayon_start.elapsed().as_secs_f64() * 1000.0) / (n_cpu_runs as f64);

        // Persistent VRAM GPU benchmark (simulating in-solver CG iterations where buffers stay in VRAM)
        let uniforms = Uniforms {
            num_elements: num_elem as u32,
            _pad0: 0, _pad1: 0, _pad2: 0,
        };
        let uniform_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&uniforms), usage: wgpu::BufferUsages::UNIFORM,
        });
        let ke_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&ke_template), usage: wgpu::BufferUsages::STORAGE,
        });
        let dofs_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&elem_dofs), usage: wgpu::BufferUsages::STORAGE,
        });
        let weights_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&elem_weights), usage: wgpu::BufferUsages::STORAGE,
        });
        let p_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&p_vec), usage: wgpu::BufferUsages::STORAGE,
        });
        let output_size = (num_elem * 24 * std::mem::size_of::<f32>()) as u64;
        let y_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: output_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &gpu.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ke_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dofs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: weights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: p_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: y_buf.as_entire_binding() },
            ],
        });

        // Warmup GPU
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&gpu.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((num_elem as u32) + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        gpu.queue.submit(Some(encoder.finish()));
        gpu.device.poll(wgpu::Maintain::Wait);

        // Timed pure VRAM GPU dispatches
        let n_gpu_runs = 50;
        let t_gpu_start = Instant::now();
        for _ in 0..n_gpu_runs {
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&gpu.pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                let workgroups = ((num_elem as u32) + 63) / 64;
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
            gpu.queue.submit(Some(enc.finish()));
        }
        gpu.device.poll(wgpu::Maintain::Wait);
        let gpu_vram_time_ms = (t_gpu_start.elapsed().as_secs_f64() * 1000.0) / (n_gpu_runs as f64);
        let gpu_gflops = (num_elem as f64 * 24.0 * 24.0 * 2.0) / (gpu_vram_time_ms * 1e-3) / 1e9;

        println!(
            "  Ne={:>5} | DOFs: {:>6} | CPU (1T): {:>6.3} ms | CPU (Rayon): {:>6.3} ms | WGPU: {:>6.3} ms ({:>5.1} GFLOP/s)",
            num_elem, total_dofs, cpu_time_ms, rayon_time_ms, gpu_vram_time_ms, gpu_gflops
        );
    }

    // Benchmark Cox-de Boor 1M points in Rust (Single vs Multi-Thread)
    println!("\n--- [2] COX-DE BOOR 1D B-SPLINE EVALUATION (1,000,000 POINTS, p=3) ---");
    let knots = vec![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
    let n_pts = 1_000_000;
    let u_pts: Vec<f64> = (0..n_pts).map(|i| i as f64 / (n_pts - 1) as f64).collect();

    // Warmup
    let _ = immersed_iga::bspline::evaluate_bspline_basis_1d(3, &knots, &[0.5]);

    let t_bspline = Instant::now();
    let _ = immersed_iga::bspline::evaluate_bspline_basis_1d(3, &knots, &u_pts);
    let bspline_time_ms = t_bspline.elapsed().as_secs_f64() * 1000.0;

    let t_bspline_rayon = Instant::now();
    let chunk_size = 50_000;
    let _ : Vec<Vec<Vec<f64>>> = u_pts.par_chunks(chunk_size)
        .map(|chunk| immersed_iga::bspline::evaluate_bspline_basis_1d(3, &knots, chunk))
        .collect();
    let bspline_rayon_time_ms = t_bspline_rayon.elapsed().as_secs_f64() * 1000.0;

    println!("--> Cox-de Boor (1M points, p=3): Single-Thread = {:.2} ms | Rayon Multi-Thread = {:.2} ms", bspline_time_ms, bspline_rayon_time_ms);

    println!("========================================================================");
    println!("  RUST BENCHMARK SUITE COMPLETE");
    println!("========================================================================");
}
