"""
run_assembly_benchmark.py — Comprehensive Assembly Mode & Scaling Benchmark
=============================================================================
Course: High Performance Computing (24ISE503) — Activity 01
Iteration 4b Benchmark Suite

Compares:
  - Modes: Mode 0 (Private Slabs), Mode 1 (Atomic Updates), Mode 2 (Node Gather CSR)
  - Workloads: 10k nodes (6,000 steps), 100k nodes (500 steps), 500k nodes (75 steps)
  - Thread Counts: 1, 2, 4, 8, 12 threads
  - Repetitions: 1 warmup + 5 timed repeats (reporting min, median, max, std)
  - Baseline: Unified single-thread baseline per mesh
  - FP32 vs FP64 comparison on Mode 2 (Memory Bandwidth sensitivity)
  - Output: results/assembly_sweep_raw.csv and results/hw.json
"""

import sys, pathlib, subprocess, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src.car_mesh import build_car_mesh
from src.mesh_io  import export_binary_mesh

BASE_DIR = pathlib.Path(__file__).parent.resolve()
WSL_BASE = "/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition"
BIN_SOLVER_FP64 = f"{WSL_BASE}/bin/crash_solver"
BIN_SOLVER_FP32 = f"{WSL_BASE}/bin/crash_solver_fp32"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TMP_DIR = BASE_DIR / "tmp_assembly_bench"
TMP_DIR.mkdir(exist_ok=True)

MESH_CONFIGS = [
    {
        "label": "10k",
        "nx": 50, "ny": 15, "nz": 13,
        "steps": 6000,
        "desc": "10k nodes (9,750 nodes, 45,393 springs)"
    },
    {
        "label": "100k",
        "nx": 125, "ny": 30, "nz": 27,
        "steps": 500,
        "desc": "100k nodes (101,250 nodes, 489,627 springs)"
    },
    {
        "label": "500k",
        "nx": 250, "ny": 50, "nz": 40,
        "steps": 75,
        "desc": "500k nodes (500,000 nodes, 2,449,090 springs)"
    }
]

THREADS = [1, 2, 4, 8, 12]
MODES = [
    (0, "Private Slabs"),
    (1, "Atomic Updates"),
    (2, "Node Gather CSR")
]
REPEATS = 5

def parse_solver_output(stdout_text):
    total_time = None
    s1_ms = s2_ms = s3_ms = s4_ms = s5_ms = 0.0
    for line in stdout_text.splitlines():
        if "EXECUTION COMPLETED IN" in line:
            parts = line.split()
            total_time = float(parts[3])
        elif "S1 Spring Forces" in line:
            s1_ms = float(line.split(":")[1].split("ms")[0].strip())
        elif "S2 Force Assembly" in line:
            s2_ms = float(line.split(":")[1].split("ms")[0].strip())
        elif "S3 Contact Stage" in line:
            s3_ms = float(line.split(":")[1].split("ms")[0].strip())
        elif "S4 Integration" in line:
            s4_ms = float(line.split(":")[1].split("ms")[0].strip())
        elif "S5 CFL Reduction" in line:
            s5_ms = float(line.split(":")[1].split("ms")[0].strip())
    return total_time, s1_ms, s2_ms, s3_ms, s4_ms, s5_ms

def main():
    print("=" * 80)
    print("  HPC CRASH SIMULATION: ITERATION 4b RIGOROUS BENCHMARK SUITE")
    print("=" * 80)

    # 1. Capture Hardware & Environment Metadata
    lscpu_res = subprocess.run(["wsl", "lscpu"], capture_output=True, text=True)
    git_hash_res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    git_commit = git_hash_res.stdout.strip() if git_hash_res.returncode == 0 else "uncommitted"

    hw_info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit,
        "os": "Windows 11 (WSL2 Ubuntu 24.04 LTS)",
        "compiler": "GCC 13.3.0 (-std=c++17 -O3 -fopenmp -march=native)",
        "cpu": "13th Gen Intel Core i5-1334U (10 Cores: 2P + 8E, 12 vCPUs)",
        "l3_cache": "12 MiB (confirmed via lscpu)",
        "ram": "16 GB DDR4/DDR5 Dual-Channel",
        "stream_triad_peak_gb_s": 35.32,
        "arithmetic_intensity_flop_per_byte": 0.38,
        "lscpu_raw": lscpu_res.stdout
    }
    with open(RESULTS_DIR / "hw.json", "w") as f:
        json.dump(hw_info, f, indent=2)
    print("[+] Saved hardware info to results/hw.json")

    # 2. Generate Binary Meshes
    print("\nGenerating binary meshes...")
    for cfg in MESH_CONFIGS:
        nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = \
            build_car_mesh(nx=cfg["nx"], ny=cfg["ny"], nz=cfg["nz"], seed=42)
        N = len(nodes)
        M = len(springs)
        rear_mask = np.zeros(N, dtype=np.int32)
        cfg["N"] = N
        cfg["M"] = M
        
        bin_path = TMP_DIR / f"mesh_{cfg['label']}.bin"
        export_binary_mesh(
            bin_path, nodes, vel0, masses, rear_mask, springs, k_arr, rest_L,
            dt=5e-6, damping=20.0, buckle_ratio=0.50, k_wall=1e7,
            n_steps=cfg["steps"], snap_freq=cfg["steps"] + 1, enable_contact=1, enable_clamp=1
        )
        print(f"  {cfg['label']}: N={N}, M={M}, Steps={cfg['steps']} -> {bin_path.name}")

    records = []

    # 3. Main Benchmark Sweep across Meshes, Modes, and Threads
    for cfg in MESH_CONFIGS:
        label = cfg["label"]
        N = cfg["N"]
        M = cfg["M"]
        steps = cfg["steps"]
        wsl_bin = f"{WSL_BASE}/tmp_assembly_bench/mesh_{label}.bin"
        wsl_csv = f"{WSL_BASE}/tmp_assembly_bench/out_{label}.csv"
        wsl_pos = f"{WSL_BASE}/tmp_assembly_bench/out_{label}.bin"

        print("\n" + "=" * 80)
        print(f"  BENCHMARKING MESH: {cfg['desc']} | Steps: {steps}")
        print("=" * 80)

        # Baseline: 1-thread Gather (Mode 2)
        baseline_t_median = None

        for mode_id, mode_name in MODES:
            print(f"\n--- Mode {mode_id}: {mode_name} ---")
            for th in THREADS:
                # Warm-up run
                subprocess.run([
                    "wsl", BIN_SOLVER_FP64,
                    wsl_bin, wsl_csv, wsl_pos, str(th), str(mode_id)
                ], capture_output=True, text=True)

                times = []
                s1_list, s2_list, s3_list, s4_list, s5_list = [], [], [], [], []

                for r in range(REPEATS):
                    res = subprocess.run([
                        "wsl", BIN_SOLVER_FP64,
                        wsl_bin, wsl_csv, wsl_pos, str(th), str(mode_id)
                    ], capture_output=True, text=True)
                    t_wall, s1, s2, s3, s4, s5 = parse_solver_output(res.stdout)
                    if t_wall is None:
                        print(f"[-] Run failed: {res.stderr}")
                        continue
                    times.append(t_wall)
                    s1_list.append(s1); s2_list.append(s2); s3_list.append(s3)
                    s4_list.append(s4); s5_list.append(s5)

                t_median = np.median(times)
                t_min = np.min(times)
                t_max = np.max(times)
                t_std = np.std(times)

                if mode_id == 2 and th == 1:
                    baseline_t_median = t_median

                records.append({
                    "mesh": label,
                    "num_nodes": N,
                    "num_springs": M,
                    "num_steps": steps,
                    "precision": "FP64",
                    "mode_id": mode_id,
                    "mode_name": mode_name,
                    "threads": th,
                    "time_median_s": t_median,
                    "time_min_s": t_min,
                    "time_max_s": t_max,
                    "time_std_s": t_std,
                    "s1_spring_ms": np.median(s1_list),
                    "s2_assembly_ms": np.median(s2_list),
                    "s3_contact_ms": np.median(s3_list),
                    "s4_integrate_ms": np.median(s4_list),
                    "s5_cfl_ms": np.median(s5_list)
                })

                print(f"  Threads={th:2d} | Median: {t_median:.4f}s (Min: {t_min:.4f}s, Max: {t_max:.4f}s) | S1: {np.median(s1_list):.1f}ms, S2: {np.median(s2_list):.1f}ms", flush=True)

        # 4. FP32 Precision Benchmark for Mode 2 (Gather) to test bandwidth effect
        print(f"\n--- FP32 Build (Node Gather CSR, Single Precision) ---", flush=True)
        for th in THREADS:
            subprocess.run([
                "wsl", BIN_SOLVER_FP32,
                wsl_bin, wsl_csv, wsl_pos, str(th), "2"
            ], capture_output=True, text=True)

            times_fp32 = []
            for r in range(REPEATS):
                res = subprocess.run([
                    "wsl", BIN_SOLVER_FP32,
                    wsl_bin, wsl_csv, wsl_pos, str(th), "2"
                ], capture_output=True, text=True)
                t_wall, s1, s2, s3, s4, s5 = parse_solver_output(res.stdout)
                times_fp32.append(t_wall)

            t_med_fp32 = np.median(times_fp32)
            records.append({
                "mesh": label,
                "num_nodes": N,
                "num_springs": M,
                "num_steps": steps,
                "precision": "FP32",
                "mode_id": 2,
                "mode_name": "Node Gather CSR (FP32)",
                "threads": th,
                "time_median_s": t_med_fp32,
                "time_min_s": np.min(times_fp32),
                "time_max_s": np.max(times_fp32),
                "time_std_s": np.std(times_fp32),
                "s1_spring_ms": 0.0, "s2_assembly_ms": 0.0, "s3_contact_ms": 0.0, "s4_integrate_ms": 0.0, "s5_cfl_ms": 0.0
            })
            print(f"  FP32 Threads={th:2d} | Median: {t_med_fp32:.4f}s", flush=True)

    # 5. Calculate Speedups and Karp-Flatt Metrics against unified 1-thread Gather baseline
    df = pd.DataFrame(records)
    
    # Add speedup, efficiency, and Karp-Flatt metric
    speedups = []
    efficiencies = []
    karp_flatts = []

    for idx, row in df.iterrows():
        # Baseline is 1-thread FP64 Node Gather CSR for the same mesh
        base_row = df[(df["mesh"] == row["mesh"]) & (df["precision"] == "FP64") & (df["mode_id"] == 2) & (df["threads"] == 1)]
        t1 = base_row["time_median_s"].values[0]
        p = row["threads"]
        sp = t1 / row["time_median_s"]
        eff = (sp / p) * 100.0
        if p > 1:
            kf = (1.0 / sp - 1.0 / p) / (1.0 - 1.0 / p)
        else:
            kf = 0.0
        speedups.append(sp)
        efficiencies.append(eff)
        karp_flatts.append(kf)

    df["speedup_vs_gather1"] = speedups
    df["efficiency_pct"] = efficiencies
    df["karp_flatt_e"] = karp_flatts

    # Save CSV
    out_csv = RESULTS_DIR / "assembly_sweep_raw.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved complete benchmark records to {out_csv}")

    # Print Summary Table
    print("\n" + "=" * 95)
    print("  SUMMARY: PERFORMANCE & SPEEDUP RELATIVE TO UNIFIED 1-THREAD GATHER BASELINE")
    print("=" * 95)
    print(f"{'Mesh':<6} | {'Precision':<6} | {'Mode':<22} | {'Th':>3} | {'Median(s)':>10} | {'Speedup':>9} | {'Eff(%)':>8} | {'Karp-Flatt':>11}")
    print("-" * 95)
    for idx, r in df.iterrows():
        print(f"{r['mesh']:<6} | {r['precision']:<6} | {r['mode_name']:<22} | {r['threads']:3d} | {r['time_median_s']:10.4f} | {r['speedup_vs_gather1']:8.2f}x | {r['efficiency_pct']:7.1f}% | {r['karp_flatt_e']:11.4f}")
    print("=" * 95)

if __name__ == "__main__":
    main()

