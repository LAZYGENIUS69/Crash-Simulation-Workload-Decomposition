"""
bench/run_vehicle_scaling.py — Multi-Core Scaling Sweep on 3D Vehicle Mesh
==========================================================================
Course: High Performance Computing (24ISE503) — Activity 01
Iteration 5 Automotive Domain Benchmark

Evaluates OpenMP multi-core scaling for the authentic 3D vehicle-front mesh:
  - Workloads: 100k nodes (103,764 nodes, 1,213,230 springs) and 500k nodes (~500k nodes, ~6.2M springs)
  - Assembly mode: Mode 2 (Node Gather CSR - zero atomics, zero reduction)
  - Threads: 1, 2, 4, 8, 12 threads with OMP_WAIT_POLICY=passive
  - Statistics: 1 warmup + 5 timed repeats (median, min, max, std)
  - Theoretical comparison: Speedup, Parallel Efficiency, Karp-Flatt serial fraction,
    and Achieved Memory Bandwidth vs STREAM Triad ceiling (35.32 GB/s).
  - Outputs: results/vehicle_scaling_results.csv and figures/vehicle_scaling_curves.png
"""

import sys, pathlib, subprocess, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.vehicle_mesh import build_vehicle_front_mesh
from src.mesh_io import export_binary_mesh
from tests.conftest import get_solver_cmd, to_solver_path

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
TMP_DIR = ROOT_DIR / "tmp_vehicle_scaling"
TMP_DIR.mkdir(exist_ok=True)

STREAM_TRIAD_CEILING_GB_S = 35.32  # Measured in Iteration 4b via stream_bench.cpp

SCALING_CONFIGS = [
    {
        "scale": "100k",
        "steps": 100,
        "dt": 2.5e-6,
        "snap_freq": 0
    },
    {
        "scale": "500k",
        "steps": 25,
        "dt": 2.5e-6,
        "snap_freq": 0
    }
]

THREADS = [1, 2, 4, 8, 12]
REPEATS = 5

def parse_solver_wall_time(stdout_text):
    for line in stdout_text.splitlines():
        if "EXECUTION COMPLETED IN" in line:
            parts = line.split()
            return float(parts[3])
    raise ValueError(f"Could not parse execution time from output:\n{stdout_text}")

def calculate_bytes_per_step(num_nodes, num_springs):
    """
    Theoretical minimum DRAM traffic per step in Mode 2 Gather CSR (FP64):
      - Node kinematics (pos, vel, acc): 3 * 8 * 3 = 72 bytes/node
      - CSR adjacency index reads: 8 bytes/node + 4 bytes/edge
      - Spring parameters & neighbor coordinates: (16 + 24) bytes/spring
      - Wall contact & integration force: ~32 bytes/node
    Total ~ 112 bytes/node + 44 bytes/spring.
    """
    return 112.0 * num_nodes + 44.0 * num_springs

def run_scaling():
    results = []

    for cfg in SCALING_CONFIGS:
        scale_label = cfg["scale"]
        steps = cfg["steps"]
        dt = cfg["dt"]

        print(f"\n=======================================================")
        print(f"  GENERATING VEHICLE MESH: scale={scale_label}")
        print(f"=======================================================")
        nodes, springs, k_arr, rest_L, zones, masses, vel0, rear_mask = build_vehicle_front_mesh(
            scale_label, reorder="morton"
        )
        N = len(nodes)
        M = len(springs)
        bytes_per_step = calculate_bytes_per_step(N, M)
        total_gb_transferred = (bytes_per_step * steps) / 1e9

        print(f"Mesh instantiated: {N:,} nodes, {M:,} springs.")
        print(f"DRAM Traffic per step: {bytes_per_step / 1e6:.2f} MB | Total: {total_gb_transferred:.3f} GB")

        mesh_bin = TMP_DIR / f"mesh_{scale_label}.bin"
        export_binary_mesh(
            mesh_bin, nodes, vel0, masses, rear_mask, springs, k_arr, rest_L,
            dt=dt, damping=20.0, buckle_ratio=0.40, k_wall=1.0e7,
            n_steps=steps, snap_freq=0, enable_contact=1, enable_clamp=0
        )

        baseline_t1_median = None

        for p in THREADS:
            out_csv = TMP_DIR / f"out_{scale_label}_p{p}.csv"
            out_pos = TMP_DIR / f"pos_{scale_label}_p{p}.bin"

            env_cmd_prefix = []
            env_vars = os.environ.copy()
            env_vars["OMP_WAIT_POLICY"] = "passive"
            env_vars["OMP_PROC_BIND"] = "spread"

            base_cmd = get_solver_cmd("crash_solver") + [
                to_solver_path(mesh_bin),
                to_solver_path(out_csv),
                to_solver_path(out_pos),
                str(p),
                "2",  # Mode 2: Node Gather CSR
                "0"   # static scheduling
            ]

            print(f"\n--> Running scale={scale_label}, threads={p} (Mode 2 Gather CSR)...")
            # Warmup run
            subprocess.run(base_cmd, capture_output=True, text=True, env=env_vars)

            times = []
            for r in range(REPEATS):
                res = subprocess.run(base_cmd, capture_output=True, text=True, env=env_vars)
                if res.returncode != 0:
                    print("ERROR STDOUT:\n", res.stdout)
                    print("ERROR STDERR:\n", res.stderr)
                    raise RuntimeError(f"Solver failed at p={p}, r={r}")
                wtime = parse_solver_wall_time(res.stdout)
                times.append(wtime)
                print(f"    Repeat {r+1}/{REPEATS}: {wtime:.4f} s")

            t_med = float(np.median(times))
            t_min = float(np.min(times))
            t_max = float(np.max(times))
            t_std = float(np.std(times))

            if p == 1:
                baseline_t1_median = t_med

            speedup = baseline_t1_median / t_med
            efficiency = speedup / p

            # Karp-Flatt Metric: e = (1/S - 1/p) / (1 - 1/p)
            if p > 1:
                karp_flatt = (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)
            else:
                karp_flatt = 0.0

            achieved_bw_gb_s = total_gb_transferred / t_med
            pct_stream_ceiling = (achieved_bw_gb_s / STREAM_TRIAD_CEILING_GB_S) * 100.0

            print(f"  [SUMMARY p={p}] Median: {t_med:.4f} s | Speedup: {speedup:.2f}x | Efficiency: {efficiency*100:.1f}% | BW: {achieved_bw_gb_s:.2f} GB/s ({pct_stream_ceiling:.1f}% STREAM)")

            results.append({
                "scale": scale_label,
                "num_nodes": N,
                "num_springs": M,
                "steps": steps,
                "threads": p,
                "time_median_s": t_med,
                "time_min_s": t_min,
                "time_max_s": t_max,
                "time_std_s": t_std,
                "speedup": speedup,
                "efficiency": efficiency,
                "karp_flatt": karp_flatt,
                "bandwidth_gb_s": achieved_bw_gb_s,
                "stream_pct": pct_stream_ceiling
            })

    # Save to CSV
    df = pd.DataFrame(results)
    out_csv_path = RESULTS_DIR / "vehicle_scaling_results.csv"
    df.to_csv(out_csv_path, index=False)
    print(f"\nAll benchmark scaling results saved to: {out_csv_path}")

    # Plot Publication-Quality Scaling Curves
    plot_scaling_curves(df)

def plot_scaling_curves(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Left: Speedup vs Threads
    threads = THREADS
    ax1.plot(threads, threads, "k--", label="Ideal Linear Speedup", linewidth=1.5, alpha=0.7)

    colors = {"100k": "#1f77b4", "500k": "#2ca02c"}
    markers = {"100k": "o", "500k": "s"}

    for scale, grp in df.groupby("scale"):
        label = f"Vehicle {scale} ({grp['num_nodes'].iloc[0]:,} nodes)"
        ax1.errorbar(
            grp["threads"], grp["speedup"],
            yerr=grp["time_std_s"] / grp["time_median_s"] * grp["speedup"],
            label=label, color=colors[scale], marker=markers[scale],
            linewidth=2.2, markersize=8, capsize=4
        )

    ax1.set_xlabel("OpenMP Worker Threads", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Measured Speedup S(P)", fontsize=11, fontweight="bold")
    ax1.set_title("Multi-Core Speedup vs Worker Threads\n(Mode 2 Node Gather CSR, OMP_WAIT_POLICY=passive)", fontsize=12, fontweight="bold")
    ax1.set_xticks(threads)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Right: Parallel Efficiency & Memory Bandwidth vs STREAM
    for scale, grp in df.groupby("scale"):
        label = f"Efficiency: {scale}"
        ax2.plot(grp["threads"], grp["efficiency"] * 100.0,
                 label=label, color=colors[scale], marker=markers[scale],
                 linewidth=2.2, markersize=8)

    ax2.axhline(100.0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("OpenMP Worker Threads", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Parallel Efficiency [%]", fontsize=11, fontweight="bold")
    ax2.set_title("Parallel Efficiency & Memory Bandwidth Saturation\n(STREAM Triad Hardware Ceiling = 35.32 GB/s)", fontsize=12, fontweight="bold")
    ax2.set_xticks(threads)
    ax2.set_ylim(0, 110)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Annotate Bandwidth on 8-thread point
    for scale, grp in df.groupby("scale"):
        row8 = grp[grp["threads"] == 8].iloc[0]
        ax2.annotate(
            f"{row8['bandwidth_gb_s']:.1f} GB/s\n({row8['stream_pct']:.0f}% STREAM)",
            xy=(8, row8['efficiency'] * 100.0),
            xytext=(8.3, row8['efficiency'] * 100.0 - 12),
            arrowprops=dict(arrowstyle="->", color=colors[scale], lw=1.2),
            fontsize=9, fontweight="bold", color=colors[scale]
        )

    ax2.legend(loc="lower left", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "vehicle_scaling_curves.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Scaling curves figure saved: {fig_path}")

if __name__ == "__main__":
    run_scaling()
