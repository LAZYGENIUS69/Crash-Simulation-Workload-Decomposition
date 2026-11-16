"""
run_benchmark2.py — Re-run with larger meshes for meaningful speedup curves
Two sizes:
  MEDIUM: nx=60, ny=16, nz=12  → ~11,520 nodes
  LARGE:  nx=100, ny=20, nz=15 → ~30,000 nodes
"""
import sys, pathlib, json, time, platform, multiprocessing
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
from src.car_mesh import build_car_mesh, mesh_summary
from src.solver   import run_sequential, run_parallel

SEED          = 42
WORKER_COUNTS = [1, 2, 4, 6, 8, 12]
N_REPEATS     = 3
SNAP_FREQ     = 999999

MESHES = {
    "MEDIUM": dict(nx=60,  ny=16, nz=12, n_steps=100, label="Medium (~11k nodes)"),
    "LARGE":  dict(nx=100, ny=20, nz=15, n_steps=50,  label="Large  (~30k nodes)"),
}

OUT_DIR = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

def median_time(fn, n_rep=N_REPEATS):
    times = []; last = None
    for _ in range(n_rep):
        last = fn()
        times.append(last[1])
    return float(np.median(times)), last

def main():
    print("=" * 70)
    print("BENCHMARK v2: larger meshes")
    print(f"CPU: {multiprocessing.cpu_count()} cores | Python {platform.python_version()}")
    print("=" * 70)

    hardware = dict(cpu=platform.processor(), cores=multiprocessing.cpu_count(),
                    os=platform.platform(), python=platform.python_version(),
                    numpy=np.__version__, seed=SEED, repeats=N_REPEATS,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    all_results = dict(hardware=hardware, meshes={})

    for mesh_key, mcfg in MESHES.items():
        nx, ny, nz = mcfg["nx"], mcfg["ny"], mcfg["nz"]
        n_steps    = mcfg["n_steps"]
        label      = mcfg["label"]

        print(f"\n{'─'*70}\nMesh: {label}")
        nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = \
            build_car_mesh(nx=nx, ny=ny, nz=nz, seed=SEED)
        mesh_summary(nodes, springs, zone_node)
        rear_mask = nodes[:,0] > nodes[:,0].max() * 0.80

        mesh_res = dict(n_nodes=len(nodes), n_springs=len(springs),
                        n_steps=n_steps, label=label, workers={})

        # Sequential
        def seq_fn():
            return run_sequential(nodes, vel0, masses, springs, k_arr, rest_L,
                                  rear_mask, n_steps, snap_freq=SNAP_FREQ)
        t1_med, (_, _, stage1) = median_time(seq_fn)
        print(f"\n  Sequential: {t1_med:.4f}s  "
              f"force={stage1['force_ms']:.1f}ms integrate={stage1['integrate_ms']:.1f}ms")
        mesh_res["workers"]["1_seq"] = dict(
            t_median=t1_med, speedup=1.0, efficiency=100.0,
            stage_ms=stage1, mode="sequential")

        for nw in WORKER_COUNTS:
            def par_fn(nw=nw):
                return run_parallel(nodes, vel0, masses, springs, k_arr, rest_L,
                                    rear_mask, n_steps, nw, snap_freq=SNAP_FREQ)
            t_med, (_, _, wt, stage_p) = median_time(par_fn)
            speedup    = t1_med / t_med
            efficiency = speedup / nw * 100.0
            if speedup > 0 and nw > 1 and 0 < speedup < nw:
                try:   P = max(0., min(1., (1/speedup - 1) / (1/nw - 1)))
                except: P = None
            else: P = None

            lb = None
            if wt:
                mt = {w: float(np.mean(t)) for w, t in wt.items()}
                lb = max(mt.values()) / (sum(mt.values())/len(mt)) if mt else 1.0

            print(f"  nw={nw:2d}: {t_med:.4f}s  s={speedup:.3f}x  e={efficiency:.1f}%"
                  + (f"  P={P:.3f}" if P else ""))

            mesh_res["workers"][str(nw)] = dict(
                t_median=t_med, speedup=round(speedup,4),
                efficiency=round(efficiency,2),
                P_amdahl=round(P,4) if P else None,
                lb_imbalance=round(lb,4) if lb else None,
                stage_ms={k:round(v,3) for k,v in stage_p.items()},
                mode="parallel")

        all_results["meshes"][mesh_key] = mesh_res

    out = OUT_DIR / "benchmark2_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    print("\nSUMMARY:")
    for mk, mr in all_results["meshes"].items():
        for wk, wr in mr["workers"].items():
            print(f"  {mk:<8} nw={wk:<6} t={wr['t_median']:.4f}s "
                  f"s={wr['speedup']:.3f}x e={wr['efficiency']:.1f}%")

if __name__ == "__main__":
    main()

