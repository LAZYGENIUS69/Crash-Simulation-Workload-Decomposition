"""
probe_vectorised.py — Iteration 1b: test NumPy-vectorised force kernel speedup
PURPOSE  : Confirm that replacing the Python for-loop with NumPy vector ops
           gives real speedup on this machine (the probe showed the loop was the bottleneck).
WHAT IT  : Same Pool architecture, but force kernel uses pure NumPy broadcasting.
EXPECTED : speedup(8) >= 4x for LARGE mesh
SEED     : 42
RUN CMD  : python probe_vectorised.py
"""

import time
import multiprocessing as mp
import numpy as np

SEED     = 42
STEPS    = 200
SMALL_N  = 2_000
LARGE_N  = 20_000
WORKER_COUNTS = [1, 2, 4, 6, 8, 10, 12]

rng = np.random.default_rng(SEED)


def build_mesh(n_nodes):
    pos     = rng.random((n_nodes, 3)).astype(np.float64)
    # linear chain springs
    i_arr   = np.arange(n_nodes - 1, dtype=np.int32)
    j_arr   = i_arr + 1
    springs = np.column_stack([i_arr, j_arr])
    # also add diagonal springs for a richer mesh
    if n_nodes > 2:
        i2 = np.arange(n_nodes - 2, dtype=np.int32)
        j2 = i2 + 2
        springs = np.vstack([springs, np.column_stack([i2, j2])])
    k      = 1500.0
    rest_L = 0.05
    return pos, springs, k, rest_L


# ---------------------------------------------------------------------------
# VECTORISED FORCE KERNEL
# ---------------------------------------------------------------------------
def compute_forces_vec(args):
    """
    Vectorised spring force for a subset of springs.
    springs_chunk: (M,2) int32
    pos          : (N,3) float64  read-only view
    Returns      : (N,3) float64 force contributions
    """
    pos, springs_chunk, k, rest_L, n_nodes = args

    i_idx = springs_chunk[:, 0]   # shape (M,)
    j_idx = springs_chunk[:, 1]

    delta = pos[j_idx] - pos[i_idx]          # (M,3)
    dist  = np.linalg.norm(delta, axis=1, keepdims=True)   # (M,1)
    # avoid div-by-zero
    safe  = np.where(dist < 1e-12, 1.0, dist)
    mag   = k * (dist - rest_L) / safe       # (M,1) scalar Hooke
    f_vec = mag * delta / safe               # (M,3) direction × magnitude

    # Accumulate via np.add.at (scatter-add) — avoids race condition
    force = np.zeros((n_nodes, 3), dtype=np.float64)
    np.add.at(force, i_idx,  f_vec)
    np.add.at(force, j_idx, -f_vec)
    return force


def run_trial(pos, vel, springs, k, rest_L, n_workers, n_steps):
    n_nodes = pos.shape[0]
    mass    = 1.0
    dt      = 1e-4

    chunks = np.array_split(springs, n_workers)
    pos  = pos.copy()
    vel  = vel.copy()

    t0 = time.perf_counter()

    if n_workers == 1:
        for _ in range(n_steps):
            force = compute_forces_vec((pos, springs, k, rest_L, n_nodes))
            vel  += (force / mass) * dt
            pos  += vel * dt
    else:
        with mp.Pool(processes=n_workers) as pool:
            for _ in range(n_steps):
                args  = [(pos, c, k, rest_L, n_nodes) for c in chunks]
                parts = pool.map(compute_forces_vec, args)
                force = np.sum(parts, axis=0)
                vel  += (force / mass) * dt
                pos  += vel * dt

    return time.perf_counter() - t0, pos


def main():
    print("=" * 65)
    print("PROBE 1b: NumPy-vectorised force kernel — Pool speedup test")
    print(f"Workers tested : {WORKER_COUNTS}")
    print(f"Steps per trial: {STEPS} | Seed: {SEED}")
    print("=" * 65)

    import json, pathlib
    all_results = {}

    for label, n_nodes in [("SMALL", SMALL_N), ("LARGE", LARGE_N)]:
        pos, springs, k, rest_L = build_mesh(n_nodes)
        vel = np.zeros_like(pos)
        n_springs = len(springs)
        print(f"\n--- Mesh: {label} | {n_nodes:,} nodes | {n_springs:,} springs ---")
        t1 = None
        all_results[label] = {}
        for nw in WORKER_COUNTS:
            elapsed, _ = run_trial(pos, vel, springs, k, rest_L, nw, STEPS)
            if t1 is None:
                t1 = elapsed
            speedup    = t1 / elapsed
            efficiency = speedup / nw * 100
            all_results[label][nw] = dict(t=round(elapsed,4), s=round(speedup,4), e=round(efficiency,2))
            print(f"  workers={nw:2d} | t={elapsed:7.3f}s | speedup={speedup:.3f}x | eff={efficiency:.1f}%")

    print("\n" + "=" * 65)
    large_s8 = all_results["LARGE"][8]["s"]
    print(f"VERDICT (large mesh, 8 workers): speedup = {large_s8:.2f}x")
    if large_s8 >= 3.0:
        print("  PASS — good speedup. A1 vectorised is the architecture.")
    elif large_s8 >= 1.5:
        print("  MARGINAL — usable, honest numbers, report IPC overhead as finding.")
    else:
        print("  FAIL — IPC overhead still dominates; switch to shared_memory.")
    print("=" * 65)

    out = pathlib.Path(__file__).parent / "probe_vec_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {out}")


if __name__ == "__main__":
    main()

