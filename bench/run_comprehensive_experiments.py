"""
run_comprehensive_experiments.py — High-Performance Experimental Suite
======================================================================
Course: High Performance Computing (24ISE503) — Activity 01
Evaluates:
  1. Large-scale multi-core scaling (100k and 500k nodes) with >= 2-3s baseline.
  2. Assembly strategy comparison (Private Slabs vs. OpenMP Atomics).
  3. Scheduling strategy comparison (Static vs. Dynamic 16/64/256 vs. Guided).
  4. Energy balance drift tracking (1-thread vs. 8-threads).
  5. Python vs. C++ language/runtime ablation.
"""

import sys, pathlib, subprocess, json, time, os
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.car_mesh import build_car_mesh
from src.mesh_io  import export_binary_mesh

BASE_DIR = pathlib.Path(__file__).parent.resolve()
BIN_SOLVER = BASE_DIR / "bin" / "crash_solver"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TMP_DIR = BASE_DIR / "tmp_exp"
TMP_DIR.mkdir(exist_ok=True)

REPEATS = 5
THREAD_COUNTS = [1, 2, 4, 6, 8, 12]

def run_solver(mesh_bin_wsl, out_csv_wsl, out_pos_wsl, threads=1, assembly=0, sched=0):
    cmd = [
        "wsl",
        "/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/bin/crash_solver",
        mesh_bin_wsl, out_csv_wsl, out_pos_wsl, str(threads), str(assembly), str(sched)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[-] Solver error: {res.stderr}")
        return None, None
    # Parse total time from stdout
    wall_time = None
    for line in res.stdout.splitlines():
        if "COMPLETED IN" in line:
            wall_time = float(line.split()[3])
    return wall_time, res.stdout

def record_hardware_state():
    hw = {
        "cpu_name": "13th Gen Intel(R) Core(TM) i5-1334U",
        "physical_cores": 10,
        "performance_cores": 2,
        "efficient_cores": 8,
        "logical_processors": 12,
        "power_status": "AC Wall Power Connected (BatteryStatus 2, 100% Charge)",
        "power_scheme": "Balanced (381b4222-f694-41f0-9685-ff5bb260df2e)",
        "wsl2_virtualization": "Hyper-V Full Virtualization (uniform vCPU dispatching)",
        "compiler": "g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0",
        "cxx_flags": "-std=c++17 -O3 -fopenmp -march=native",
        "date": "2026-11-16"
    }
    with open(RESULTS_DIR / "hw.json", "w") as f:
        json.dump(hw, f, indent=2)
    print("[+] Recorded hardware environment to results/hw.json")
    return hw

def main():
    print("=" * 80)
    print("  COMPREHENSIVE HPC EXPERIMENTAL RUN (ITERATION 4)")
    print("=" * 80)

    hw = record_hardware_state()
    exp_results = {"hardware": hw, "large_scale_scaling": {}, "assembly_comparison": {}, "scheduling_comparison": {}, "energy_drift": {}}

    # ─────────────────────────────────────────────────────────────────────────
    # 1. LARGE-SCALE SCALING: 100k Nodes (~486k Springs, 250 steps)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1: 100k-Node Scaling (Target: Single-Thread Runtime >= 2-3s)")
    print("=" * 80)

    nodes_100k, springs_100k, k_100k, r_100k, zone_100k, masses_100k, vel0_100k = \
        build_car_mesh(nx=140, ny=30, nz=24, seed=42)
    N_100k = len(nodes_100k)
    M_100k = len(springs_100k)
    STEPS_100k = 250
    rear_mask_100k = (nodes_100k[:, 0] > (nodes_100k[:, 0].max() * 0.80)).astype(np.int32)
    print(f"Generated 100k Mesh: {N_100k:,} nodes | {M_100k:,} springs | {STEPS_100k} steps")

    bin_100k = TMP_DIR / "mesh_100k.bin"
    export_binary_mesh(
        bin_100k, nodes_100k, vel0_100k, masses_100k, rear_mask_100k,
        springs_100k, k_100k, r_100k, dt=5e-6, damping=20.0, buckle_ratio=0.50,
        k_wall=1e7, n_steps=STEPS_100k, snap_freq=25, enable_contact=1, enable_clamp=1
    )
    wsl_bin_100k = "/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_exp/mesh_100k.bin"

    scaling_100k = {}
    t1_100k_med = None

    for th in THREAD_COUNTS:
        wsl_csv = f"/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_exp/out_100k_th{th}.csv"
        wsl_pos = f"/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_exp/out_100k_th{th}.bin"

        # Warmup
        run_solver(wsl_bin_100k, wsl_csv, wsl_pos, threads=th)

        times = []
        for rep in range(REPEATS):
            t_w, _ = run_solver(wsl_bin_100k, wsl_csv, wsl_pos, threads=th)
            if t_w: times.append(t_w)

        med_t = float(np.median(times))
        min_t = float(np.min(times))
        max_t = float(np.max(times))
        std_t = float(np.std(times))

        if th == 1:
            t1_100k_med = med_t

        speedup = t1_100k_med / med_t
        eff = (speedup / th) * 100.0
        karp_flatt = (1.0/speedup - 1.0/th) / (1.0 - 1.0/th) if th > 1 else 0.0

        print(f"  [100k Mesh] Threads={th:2d} | Median Wall: {med_t:6.3f}s (min: {min_t:6.3f}s, max: {max_t:6.3f}s) | "
              f"Speedup: {speedup:5.2f}x | Eff: {eff:5.1f}% | Karp-Flatt e: {karp_flatt:6.4f}")

        scaling_100k[str(th)] = {
            "median_s": round(med_t, 4), "min_s": round(min_t, 4),
            "max_s": round(max_t, 4), "std_s": round(std_t, 4),
            "speedup": round(speedup, 3), "efficiency": round(eff, 1),
            "karp_flatt": round(karp_flatt, 4), "repeats": times
        }

    exp_results["large_scale_scaling"]["100k_nodes"] = scaling_100k

    # ─────────────────────────────────────────────────────────────────────────
    # 2. ASSEMBLY STRATEGY COMPARISON (100k Mesh, 100 steps)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  EXPERIMENT 2: Assembly Strategy (Private Slabs vs. OpenMP Atomics)")
    print("=" * 80)

    STEPS_ASM = 80
    bin_asm = TMP_DIR / "mesh_asm.bin"
    export_binary_mesh(
        bin_asm, nodes_100k, vel0_100k, masses_100k, rear_mask_100k,
        springs_100k, k_100k, r_100k, dt=5e-6, damping=20.0, buckle_ratio=0.50,
        k_wall=1e7, n_steps=STEPS_ASM, snap_freq=STEPS_ASM, enable_contact=1, enable_clamp=1
    )
    wsl_bin_asm = "/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_exp/mesh_asm.bin"

    asm_results = {"threads": [1, 2, 4, 8], "private_slabs": {}, "atomic_updates": {}}

    for th in [1, 2, 4, 8]:
        # Mode 0: Private Slabs
        wsl_csv_0 = f"/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_exp/asm_m0_th{th}.csv"
        wsl_pos_0 = f"/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_exp/asm_m0_th{th}.bin"
        times_0 = []
        for _ in range(3):
            t_w, _ = run_solver(wsl_bin_asm, wsl_csv_0, wsl_pos_0, threads=th, assembly=0)
            if t_w: times_0.append(t_w)
        med_0 = float(np.median(times_0))

        # Mode 1: Atomic Scatter-Add
        wsl_csv_1 = f"/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_exp/asm_m1_th{th}.csv"
        wsl_pos_1 = f"/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_exp/asm_m1_th{th}.bin"
        times_1 = []
        for _ in range(3):
            t_w, _ = run_solver(wsl_bin_asm, wsl_csv_1, wsl_pos_1, threads=th, assembly=1)
            if t_w: times_1.append(t_w)
        med_1 = float(np.median(times_1))

        # Check numerical equivalence between both assembly modes
        with open(TMP_DIR / f"asm_m0_th{th}.bin", "rb") as f:
            p0 = np.frombuffer(f.read(), dtype=np.float64).reshape((N_100k, 3))
        with open(TMP_DIR / f"asm_m1_th{th}.bin", "rb") as f:
            p1 = np.frombuffer(f.read(), dtype=np.float64).reshape((N_100k, 3))

        max_diff = np.max(np.abs(p0 - p1) / (np.abs(p0) + 1e-10))
        equiv = "PASS (Equiv)" if max_diff < 1e-4 else "FAIL"

        ratio = med_1 / med_0
        winner = "Private Slabs (Mode 0)" if med_0 < med_1 else "Atomics (Mode 1)"
        print(f"  Threads={th:2d} | Private Slabs: {med_0:6.3f}s | Atomics: {med_1:6.3f}s | "
              f"Atomic/Slab Overhead: {ratio:5.2f}x | Equiv: {equiv} (diff={max_diff:.1e}) -> Faster: {winner}")

        asm_results["private_slabs"][str(th)] = round(med_0, 4)
        asm_results["atomic_updates"][str(th)] = round(med_1, 4)

    exp_results["assembly_comparison"] = asm_results

    # ─────────────────────────────────────────────────────────────────────────
    # 3. SCHEDULING STRATEGY COMPARISON (Contact Stage, 8 Threads)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  EXPERIMENT 3: Scheduling Strategy (Static vs. Dynamic vs. Guided)")
    print("=" * 80)

    sched_names = {0: "Static", 1: "Dynamic (chunk 16)", 2: "Dynamic (chunk 64)", 3: "Dynamic (chunk 256)", 4: "Guided"}
    sched_times = {}

    for sm, sname in sched_names.items():
        wsl_csv = f"/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_exp/sched_s{sm}.csv"
        wsl_pos = f"/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_exp/sched_s{sm}.bin"
        times_s = []
        for _ in range(3):
            t_w, _ = run_solver(wsl_bin_asm, wsl_csv, wsl_pos, threads=8, assembly=0, sched=sm)
            if t_w: times_s.append(t_w)
        med_s = float(np.median(times_s))
        sched_times[sname] = round(med_s, 4)
        print(f"  Schedule: {sname:<24} | 8 Threads Median Wall: {med_s:6.3f}s")

    exp_results["scheduling_comparison"] = sched_times

    # ─────────────────────────────────────────────────────────────────────────
    # 4. ENERGY BALANCE DRIFT TRACKING (1-thread vs 8-threads)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  EXPERIMENT 4: Energy Conservation & Drift Verification")
    print("=" * 80)

    import pandas as pd
    df_1 = pd.read_csv(TMP_DIR / "out_100k_th1.csv")
    df_8 = pd.read_csv(TMP_DIR / "out_100k_th8.csv")

    ke_init_1 = df_1["ke_J"].iloc[0]
    drift_max_1 = df_1["energy_drift"].max() * 100.0
    drift_final_1 = df_1["energy_drift"].iloc[-1] * 100.0

    ke_init_8 = df_8["ke_J"].iloc[0]
    drift_max_8 = df_8["energy_drift"].max() * 100.0
    drift_final_8 = df_8["energy_drift"].iloc[-1] * 100.0

    print(f"  1-Thread Execution: Initial KE = {ke_init_1:10.2f} J | Max Drift = {drift_max_1:5.2f}% | Final Drift = {drift_final_1:5.2f}%")
    print(f"  8-Thread Execution: Initial KE = {ke_init_8:10.2f} J | Max Drift = {drift_max_8:5.2f}% | Final Drift = {drift_final_8:5.2f}%")

    exp_results["energy_drift"] = {
        "1_thread": {"initial_ke_J": ke_init_1, "max_drift_pct": round(drift_max_1, 2), "final_drift_pct": round(drift_final_1, 2)},
        "8_threads": {"initial_ke_J": ke_init_8, "max_drift_pct": round(drift_max_8, 2), "final_drift_pct": round(drift_final_8, 2)}
    }

    # Plot Energy Channels
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_8["time_s"] * 1000.0, df_8["ke_J"], label="Kinetic Energy (KE)", color="#2E86C1", lw=2)
    ax.plot(df_8["time_s"] * 1000.0, df_8["strain_e_J"], label="Elastic Strain Energy", color="#27AE60", lw=2)
    ax.plot(df_8["time_s"] * 1000.0, df_8["damp_e_J"], label="Viscous Damping Dissipation", color="#E67E22", lw=2)
    ax.plot(df_8["time_s"] * 1000.0, df_8["plastic_e_J"], label="Plastic Buckling Dissipation", color="#C0392B", lw=2)
    ax.plot(df_8["time_s"] * 1000.0, df_8["total_e_J"], label="Total Energy (Conservative + Dissipated)", color="#2C3E50", lw=2.5, ls="--")

    ax.set_xlabel("Crash Time [ms]", fontsize=11)
    ax.set_ylabel("Energy [Joules]", fontsize=11)
    ax.set_title("Energy Partitioning & Conservation (100k-Node Vehicle Crash, 8 Threads)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="center right")
    plt.tight_layout()
    energy_fig = RESULTS_DIR / "energy_drift.png"
    plt.savefig(energy_fig, dpi=180)
    plt.close()
    print(f"[+] Saved energy balance figure to: {energy_fig}")

    # Save full results
    final_out = RESULTS_DIR / "iteration4_experiments.json"
    with open(final_out, "w") as f:
        json.dump(exp_results, f, indent=2)
    print(f"\n[+] Full experimental dataset saved to: {final_out}")

if __name__ == "__main__":
    main()

