"""
run_benchmark.py — Full benchmark: sequential + parallel, 2 mesh sizes, 6 worker counts
========================================================================================
Produces results/benchmark_results.json (all raw numbers, reproducible).
Hardware, versions, seed recorded.

Run: python run_benchmark.py

Expected runtime: ~10–20 min on 12-core laptop
"""
import sys, pathlib, json, time, platform, multiprocessing
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
from src.car_mesh import build_car_mesh, mesh_summary
from src.solver   import run_sequential, run_parallel

# ── Configuration ─────────────────────────────────────────────────────────────
SEED          = 42
WORKER_COUNTS = [1, 2, 4, 6, 8, 12]
N_REPEATS     = 3       # median over repeats for stability
SNAP_FREQ     = 999999  # no snapshots during benchmark (pure timing)

# Two mesh sizes (satisfy rubric: "at least two workload sizes")
MESHES = {
    "SMALL":  dict(nx=18, ny=6,  nz=5,  n_steps=200, label="Small (5 k nodes)"),
    "LARGE":  dict(nx=40, ny=12, nz=10, n_steps=100, label="Large (48 k nodes)"),
}

OUT_DIR = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

def median_time(fn, n_rep=N_REPEATS):
    """Run fn() n_rep times, return median wall time and last return value."""
    times = []
    last  = None
    for _ in range(n_rep):
        last = fn()
        times.append(last[1])   # t_wall always index 1
    return float(np.median(times)), last

def main():
    print("=" * 70)
    print("BENCHMARK: Crash-Simulation Workload Decomposition")
    print(f"CPU cores available: {multiprocessing.cpu_count()}")
    print(f"Python: {platform.python_version()}  NumPy: {np.__version__}")
    print(f"OS: {platform.platform()}")
    print(f"Seed: {SEED} | Repeats: {N_REPEATS}")
    print("=" * 70)

    hardware = dict(
        cpu       = platform.processor(),
        cores     = multiprocessing.cpu_count(),
        os        = platform.platform(),
        python    = platform.python_version(),
        numpy     = np.__version__,
        seed      = SEED,
        repeats   = N_REPEATS,
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    all_results = dict(hardware=hardware, meshes={})

    for mesh_key, mcfg in MESHES.items():
        nx, ny, nz   = mcfg["nx"], mcfg["ny"], mcfg["nz"]
        n_steps      = mcfg["n_steps"]
        label        = mcfg["label"]

        print(f"\n{'─'*70}")
        print(f"Mesh: {label}  (nx={nx}, ny={ny}, nz={nz}, steps={n_steps})")
        nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = \
            build_car_mesh(nx=nx, ny=ny, nz=nz, seed=SEED)
        mesh_summary(nodes, springs, zone_node)

        rear_mask = nodes[:, 0] > (nodes[:, 0].max() * 0.80)

        mesh_res = dict(
            n_nodes   = len(nodes),
            n_springs = len(springs),
            n_steps   = n_steps,
            label     = label,
            workers   = {}
        )

        # ── Sequential baseline (1 worker, no Pool overhead) ─────────
        print(f"\n  Sequential (1 worker, no Pool):")
        def seq_fn():
            return run_sequential(nodes, vel0, masses, springs, k_arr, rest_L,
                                  rear_mask, n_steps, snap_freq=SNAP_FREQ)
        t1_med, (_, _, stage1) = median_time(seq_fn)
        print(f"    median wall = {t1_med:.4f}s  "
              f"force={stage1['force_ms']:.1f}ms  "
              f"integrate={stage1['integrate_ms']:.1f}ms")
        mesh_res["workers"]["1_seq"] = dict(
            t_median  = t1_med,
            speedup   = 1.0,
            efficiency= 100.0,
            stage_ms  = stage1,
            mode      = "sequential"
        )

        # ── Parallel runs ─────────────────────────────────────────────
        for nw in WORKER_COUNTS:
            print(f"\n  Parallel, workers={nw}:")
            def par_fn(nw=nw):
                return run_parallel(nodes, vel0, masses, springs, k_arr, rest_L,
                                    rear_mask, n_steps, nw, snap_freq=SNAP_FREQ)
            t_med, (_, _, wt, stage_p) = median_time(par_fn)
            speedup    = t1_med / t_med
            efficiency = speedup / nw * 100.0

            # Amdahl serial fraction estimate from speedup
            # S = 1 / ((1-P) + P/N)  → P = (1/S - 1) / (1/N - 1)
            if speedup > 0 and nw > 1 and speedup < nw:
                try:
                    P = (1/speedup - 1) / (1/nw - 1)
                    P = max(0.0, min(1.0, P))
                except ZeroDivisionError:
                    P = None
            else:
                P = None

            print(f"    median wall = {t_med:.4f}s  "
                  f"speedup={speedup:.3f}x  eff={efficiency:.1f}%  "
                  f"P_amdahl={P:.3f}" if P else
                  f"    median wall = {t_med:.4f}s  speedup={speedup:.3f}x  eff={efficiency:.1f}%")

            # Per-worker load balance
            if wt:
                mean_times = {w: float(np.mean(t)) for w, t in wt.items()}
                std_times  = {w: float(np.std(t))  for w, t in wt.items()}
                lb_imbal   = max(mean_times.values()) / (sum(mean_times.values())/len(mean_times)) \
                             if mean_times else 1.0
                print(f"    Load imbalance ratio: {lb_imbal:.4f}")
            else:
                mean_times = {}; std_times = {}; lb_imbal = None

            mesh_res["workers"][str(nw)] = dict(
                t_median      = t_med,
                speedup       = round(speedup, 4),
                efficiency    = round(efficiency, 2),
                P_amdahl      = round(P, 4) if P else None,
                lb_imbalance  = round(lb_imbal, 4) if lb_imbal else None,
                worker_mean_s = {str(k): round(v, 6) for k, v in mean_times.items()},
                worker_std_s  = {str(k): round(v, 6) for k, v in std_times.items()},
                stage_ms      = {k: round(v, 3) for k, v in stage_p.items()},
                mode          = "parallel"
            )

        all_results["meshes"][mesh_key] = mesh_res

    # ── Save ───────────────────────────────────────────────────────────
    out_path = OUT_DIR / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to: {out_path}")

    # ── Summary table ─────────────────────────────────────────────────
    print("\nSUMMARY (speedup × efficiency):")
    print(f"{'Mesh':<12} {'Workers':<10} {'Time(s)':<10} {'Speedup':<10} {'Eff%':<8}")
    print("-" * 55)
    for mk, mr in all_results["meshes"].items():
        for wk, wr in mr["workers"].items():
            print(f"{mk:<12} {wk:<10} {wr['t_median']:<10.4f} "
                  f"{wr['speedup']:<10.3f} {wr['efficiency']:<8.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

