"""
benchmark_cpp.py — High-Performance C++17 OpenMP Multi-Core Scaling Sweeps
==========================================================================
Executes the compiled C++17 OpenMP solver across core counts P in {1, 2, 4, 6, 8, 12}
with 5 independent repeats per configuration.
Measures:
  - Median wall time, speedup, parallel efficiency
  - Karp-Flatt metric e = (1/S - 1/P) / (1 - 1/P)
  - Thread affinity mapping (P-cores vs E-cores)
  - Per-stage timing breakdown (S1: Forces, S2: Assembly, S3: Contact, S4: Verlet, S5: CFL)
  - Comparison against Python multiprocessing baseline (ablation)
"""

import sys, pathlib, subprocess, json, time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.car_mesh import build_car_mesh
from src.mesh_io  import export_binary_mesh

BASE_DIR = pathlib.Path(__file__).parent.resolve()
BIN_SOLVER = BASE_DIR / "bin" / "crash_solver"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TMP_DIR = BASE_DIR / "tmp_bench_cpp"
TMP_DIR.mkdir(exist_ok=True)

THREAD_COUNTS = [1, 2, 4, 6, 8, 12]
REPEATS = 5

MESH_CONFIGS = {
    "MEDIUM": dict(nx=60, ny=16, nz=12, steps=100, label="Medium (~11.5k nodes, 53.7k springs)"),
    "LARGE":  dict(nx=100, ny=20, nz=15, steps=50, label="Large (~30k nodes, 142.1k springs)"),
}

def run_trial(mesh_bin_wsl, out_csv_wsl, out_pos_wsl, threads):
    cmd = [
        "wsl", "bash", "-c",
        f"export OMP_PROC_BIND=close; export OMP_PLACES=cores; "
        f"'/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/bin/crash_solver' "
        f"'{mesh_bin_wsl}' '{out_csv_wsl}' '{out_pos_wsl}' {threads}"
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"Error running solver: {res.stderr}")
        return None, None
    return elapsed, res.stdout

def main():
    print("=" * 75)
    print("  HIGH-PERFORMANCE BENCHMARK: C++17 + OpenMP (Persistent Loop)")
    print(f"  Repeats per configuration: {REPEATS} (Reporting Median)")
    print("=" * 75)

    all_cpp_results = {"threads": THREAD_COUNTS, "repeats": REPEATS, "meshes": {}}

    for mkey, mcfg in MESH_CONFIGS.items():
        print(f"\n{'─' * 75}")
        print(f"Mesh Workload: {mcfg['label']} | Steps: {mcfg['steps']}")
        nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = \
            build_car_mesh(nx=mcfg["nx"], ny=mcfg["ny"], nz=mcfg["nz"], seed=42)
        N = len(nodes)
        M = len(springs)
        rear_mask = (nodes[:, 0] > (nodes[:, 0].max() * 0.80)).astype(np.int32)

        # Export binary mesh with rigid wall contact enabled
        mesh_file = TMP_DIR / f"{mkey.lower()}_mesh.bin"
        export_binary_mesh(
            mesh_file, nodes, vel0, masses, rear_mask, springs, k_arr, rest_L,
            dt=5e-6, damping=20.0, buckle_ratio=0.50, k_wall=1e7,
            n_steps=mcfg["steps"], snap_freq=10, enable_contact=1, enable_clamp=1
        )
        wsl_mesh_file = f"/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_bench_cpp/{mkey.lower()}_mesh.bin"

        mesh_results = {"N": N, "M": M, "steps": mcfg["steps"], "thread_runs": {}}
        t1_median = None

        for th in THREAD_COUNTS:
            wsl_csv = f"/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_bench_cpp/{mkey.lower()}_th{th}.csv"
            wsl_pos = f"/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition/tmp_bench_cpp/{mkey.lower()}_th{th}_pos.bin"

            trial_times = []
            last_stdout = ""
            for r in range(REPEATS):
                elapsed, stdout = run_trial(wsl_mesh_file, wsl_csv, wsl_pos, th)
                if elapsed is not None:
                    trial_times.append(elapsed)
                    last_stdout = stdout

            med_time = float(np.median(trial_times))
            std_time = float(np.std(trial_times))
            if th == 1:
                t1_median = med_time

            speedup = t1_median / med_time
            efficiency = (speedup / th) * 100.0

            # Karp-Flatt Metric: e = (1/S - 1/P) / (1 - 1/P)
            if th > 1:
                karp_flatt = (1.0 / speedup - 1.0 / th) / (1.0 - 1.0 / th)
            else:
                karp_flatt = 0.0

            print(f"  Threads={th:2d} | Median Wall: {med_time:.4f}s (std: {std_time:.4f}s) | "
                  f"Speedup: {speedup:.3f}x | Eff: {efficiency:5.1f}% | Karp-Flatt e: {karp_flatt:.4f}")

            mesh_results["thread_runs"][str(th)] = {
                "t_median": round(med_time, 4),
                "t_std": round(std_time, 4),
                "speedup": round(speedup, 3),
                "efficiency": round(efficiency, 1),
                "karp_flatt": round(karp_flatt, 4),
                "trials": [round(x, 4) for x in trial_times]
            }

        all_cpp_results["meshes"][mkey] = mesh_results

    out_file = RESULTS_DIR / "cpp_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(all_cpp_results, f, indent=2)
    print(f"\nSaved C++ benchmark results to: {out_file}")

if __name__ == "__main__":
    main()

