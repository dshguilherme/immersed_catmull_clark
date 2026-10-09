#!/usr/bin/env python3
"""
Publication-Quality Figure Generator for Immersed Catmull-Clark IGA & Octree AMR.
Generates vector and 300 DPI multi-panel figures purely with Python Matplotlib.
Zero MATLAB or external proprietary toolbox dependencies required.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Configure publication typography and styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
})

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def plot_benchmark_scaling():
    """Generates Figure 1: Multi-backend MatVec Wall-Clock & Throughput Scaling."""
    print("[1/3] Generating Figure 1: Matrix-Free Scalability & GFLOP/s...")

    elements = np.array([500, 2000, 8000, 32000])
    dofs = elements * 24

    matlab_cpu = np.array([0.052, 0.080, 0.222, 1.027])
    matlab_gpu = np.array([0.084, 0.096, 0.126, 0.243])
    rust_cpu_1t = np.array([0.070, 0.263, 1.064, 4.333])
    rust_cpu_rayon = np.array([0.202, 0.292, 0.550, 1.145])
    rust_wgpu = np.array([0.048, 0.061, 0.085, 0.325])
    rust_cuda = np.array([0.012, 0.016, 0.034, 0.120])

    # GFLOP/s = (Ne * 24 * 24 * 2) / (time_ms * 1e-3) / 1e9
    flops_per_elem = 24 * 24 * 2
    cuda_gflops = (elements * flops_per_elem) / (rust_cuda * 1e-3) / 1e9
    wgpu_gflops = (elements * flops_per_elem) / (rust_wgpu * 1e-3) / 1e9
    matlab_gpu_gflops = (elements * flops_per_elem) / (matlab_gpu * 1e-3) / 1e9

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Wall-Clock Execution Time (Log-Log)
    ax1.loglog(dofs, matlab_cpu, 'o--', color='#7f7f7f', label='MATLAB CPU (Sparse BLAS)', linewidth=1.5, markersize=5)
    ax1.loglog(dofs, rust_cpu_1t, 's--', color='#d62728', label='Rust CPU (Single-Thread)', linewidth=1.5, markersize=5)
    ax1.loglog(dofs, rust_cpu_rayon, '^-.', color='#e377c2', label='Rust CPU (Rayon Multi-Thread)', linewidth=1.5, markersize=5)
    ax1.loglog(dofs, matlab_gpu, 'd-', color='#ff7f0e', label='MATLAB GPU (cuBLAS GEMM)', linewidth=1.8, markersize=6)
    ax1.loglog(dofs, rust_wgpu, 'v-', color='#1f77b4', label='Rust WGPU (DX12 Compute)', linewidth=1.8, markersize=6)
    ax1.loglog(dofs, rust_cuda, '*-', color='#2ca02c', label='Rust Native CUDA (Warp-Coalesced)', linewidth=2.2, markersize=8)

    ax1.set_xlabel('Total Active DOFs ($N_{\\mathrm{dof}}$)')
    ax1.set_ylabel('Execution Time per MatVec [ms]')
    ax1.set_title('(a) Wall-Clock Scaling per Tensor Contraction', fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    ax1.set_xticks(dofs)
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())

    # Panel B: Sustained GPU Throughput [GFLOP/s]
    x_indices = np.arange(len(elements))
    width = 0.25

    ax2.bar(x_indices - width, matlab_gpu_gflops, width, label='MATLAB GPU (cuBLAS)', color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.bar(x_indices, wgpu_gflops, width, label='Rust WGPU (DX12)', color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.bar(x_indices + width, cuda_gflops, width, label='Rust CUDA (Warp-Coalesced)', color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Problem Size [Active Hex Elements $N_e$]')
    ax2.set_ylabel('Sustained GPU Throughput [GFLOP/s]')
    ax2.set_title('(b) GPU Hardware Efficiency on RTX 2050', fontweight='bold')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f'{ne:,}\n({ne*24:,} DOFs)' for ne in elements])
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')

    # Annotate peak throughput
    ax2.annotate(f'Peak: {cuda_gflops[-1]:.1f} GFLOP/s\n(2.0x faster than cuBLAS)',
                 xy=(x_indices[-1] + width, cuda_gflops[-1]),
                 xytext=(x_indices[-1] - 0.2, cuda_gflops[-1] + 15),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.0),
                 fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_matrixfree_gpu_benchmark.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  --> Saved: {out_path}")


def plot_speedup_scorecard():
    """Generates Figure 2: Speedup Scorecard comparing Rust to MATLAB baseline."""
    print("[2/3] Generating Figure 2: Speedup Scorecard...")

    categories = [
        'Cox-de Boor Basis\n(1M points, p=3)',
        'Octree 2:1 Balancing\n+ MPC Assembly',
        'Complete 5-Obstacle\nCourse (Total)',
        'CUDA MatVec\n(32k Elements)'
    ]
    matlab_times = [139.1, 396.4, 3080.0, 0.243]
    rust_times   = [177.3, 0.051, 0.244,  0.120]

    speedups = [m / r for m, r in zip(matlab_times, rust_times)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(categories, speedups, color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'], alpha=0.85, edgecolor='black', linewidth=0.6)

    ax.set_xscale('log')
    ax.set_xlabel('Rust Speedup Factor vs. MATLAB Baseline (Log Scale)')
    ax.set_title('Rust Architecture Speedup Factor Across Core Modules', fontweight='bold')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='1.0x (Parity)')

    for bar, sp in zip(bars, speedups):
        val_str = f"{sp:.2f}x" if sp < 10 else f"{sp:,.0f}x"
        ax.text(bar.get_width() * 1.3, bar.get_y() + bar.get_height()/2.0, val_str,
                va='center', ha='left', fontsize=9.5, fontweight='bold')

    ax.set_xlim(0.1, 50000)
    ax.legend(loc='lower right')
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "fig_speedup_scorecard.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  --> Saved: {out_path}")


def plot_topopt_results():
    """Generates Figure 3: 3D Topology Optimization Convergence and Density Slice."""
    print("[3/3] Generating Figure 3: 3D Topology Optimization Results...")

    json_path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "topopt_result.json")
    if not os.path.exists(json_path):
        print("  Warning: topopt_result.json not found, skipping Fig 3.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    compliance = np.array(data['compliance'])
    densities = np.array(data['densities'])
    nelx = data['nelx']
    nely = data['nely']
    nelz = data['nelz']

    # Reshape densities into 3D grid [nelz, nely, nelx]
    dens_3d = densities.reshape((nelz, nely, nelx))

    fig = plt.figure(figsize=(11, 4.5))

    # Panel A: Convergence of Compliance
    ax1 = fig.add_subplot(1, 2, 1)
    iters = np.arange(1, len(compliance) + 1)
    ax1.plot(iters, compliance, 'o-', color='#1f77b4', linewidth=1.8, markersize=5)
    ax1.set_xlabel('Topology Optimization Iteration')
    ax1.set_ylabel(r'Objective Compliance $c(\mathbf{\rho}) = \mathbf{u}^T \mathbf{K} \mathbf{u}$')
    ax1.set_title('(a) Convergence History (Solved in 390 ms)', fontweight='bold')
    ax1.set_xlim(1, len(compliance))

    # Panel B: Mid-plane 2D Density Contour Slice (Z = nelz/2)
    ax2 = fig.add_subplot(1, 2, 2)
    mid_z = nelz // 2
    slice_2d = dens_3d[mid_z, :, :]

    im = ax2.imshow(slice_2d, origin='lower', cmap='viridis', extent=[0, nelx, 0, nely], vmin=0, vmax=1)
    ax2.set_xlabel('Length [Element X]')
    ax2.set_ylabel('Height [Element Y]')
    ax2.set_title(rf'(b) Density Field $\tilde{{\rho}}$ Mid-Plane Slice ($Z = {mid_z}$)', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Relative Material Density $\\rho$', rotation=270, labelpad=12)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_topopt_rust_convergence.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  --> Saved: {out_path}")


if __name__ == "__main__":
    print("========================================================================")
    print("  GENERATING PUBLICATION FIGURES VIA PYTHON MATPLOTLIB                 ")
    print("========================================================================")
    plot_benchmark_scaling()
    plot_speedup_scorecard()
    plot_topopt_results()
    print("========================================================================")
    print("  ALL FIGURES SUCCESSFULLY GENERATED IN figures/                        ")
    print("========================================================================")
