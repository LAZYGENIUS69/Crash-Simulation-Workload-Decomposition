"""
probe_speedup.py — Iteration 1 falsification test
PURPOSE : Determine if multiprocessing.Pool gives real speedup on this machine
          before committing to the architecture.
WHAT IT MEASURES : wall-time for force-compute kernel, 1 / 2 / 4 / 8 workers,
                   two mesh sizes (SMALL=2000 nodes, LARGE=20000 nodes).
EXPECTED RESULT : speedup(8) >= 2.0 for LARGE mesh  → stick with A1 (Pool)
                  speedup(8) <  1.5 for both meshes  → upgrade to shared_memory
HARDWARE : Intel 12-core Windows 11 laptop, Python 3.14, NumPy 2.5.5
SEED     : 42 (fixed for reproducibility)
RUN CMD  : python probe_speedup.py
"""

import time
import math
import multiprocessing as mp
import numpy as np

SEED = 42
STEPS = 100         # Verlet timesteps to simulate
SMALL_NODES = 2_000
LARGE_NODES = 20_000
WORKER_COUNTS = [1, 2, 4, 8]

rng = np.random.default_rng(SEED)


def build_mesh(n_nodes):
    """
    Build a simple 1D chain of nodes for timing purposes.
    nodes  : (n_nodes, 3) float64 positions
    springs: (n_springs, 2) int32 pairs  — adjacent pairs in chain
    k      : spring constant (N/m)
    rest_L : rest length (m)
    """
    pos = rng.random((n_nodes, 3)).astype(np.float64)
    springs = np.column_stack([
        np.arange(n_nodes - 1, dtype=np.int32),
        np.arange(1, n_nodes,  dtype=np.int32)
    ])
    k = 1500.0
    rest_L = 0.05
    return pos, springs, k, rest_L


# ---------------------------------------------------------------------------
# FORCE KERNEL — runs in each worker
# ---------------------------------------------------------------------------
def compute_forces_chunk(args):
    """
    Compute spring forces for a contiguous slice of the spring list.
    Returns ndarray of shape (n_nodes_in_slab, 3) — only relevant rows non-zero.
    """
    pos, springs_chunk, k, rest_L, n_nodes = args
    force = np.zeros((n_nodes, 3), dtype=np.float64)
    for sp in springs_chunk:
        i, j = int(sp[0]), int(sp[1])
        delta = pos[j] - pos[i]
        dist  = math.sqrt(delta[0]**2 + delta[1]**2 + delta[2]**2)
        if dist < 1e-12:
            continue
        mag = k * (dist - rest_L) / dist     # Hooke scalar / dist
        f   = mag * delta                     # vector force
        force[i] +=  f
        force[j] -= f
    return force


def run_trial(pos, vel, springs, k, rest_L, n_workers, n_steps):
    """
    Run n_steps of Verlet integration, parallelising force computation
    across n_workers Pool processes.
    Returns (elapsed_seconds, final_pos).
    """
    n_nodes = pos.shape[0]
    mass    = 1.0        # kg per node
    dt      = 1e-4       # timestep (s) — CFL-safe for k=1500, m=1

    # Split springs evenly across workers
    chunks = np.array_split(springs, n_workers)
    # Pre-build arg tuples (pos is re-packed each step — that's the serial fraction)

    pos  = pos.copy()
    vel  = vel.copy()
    acc  = np.zeros_like(pos)

    t0 = time.perf_counter()

    if n_workers == 1:
        # Single-process path (avoids Pool overhead entirely)
        for _ in range(n_steps):
            force = compute_forces_chunk((pos, springs, k, rest_L, n_nodes))
            acc   = force / mass
            vel  += acc * dt
            pos  += vel * dt
    else:
        with mp.Pool(processes=n_workers) as pool:
            for _ in range(n_steps):
                args  = [(pos, c, k, rest_L, n_nodes) for c in chunks]
                parts = pool.map(compute_forces_chunk, args)
                force = sum(parts)             # reduce
                acc   = force / mass
                vel  += acc * dt
                pos  += vel * dt

    elapsed = time.perf_counter() - t0
    return elapsed, pos


def main():
    print("=" * 60)
    print("PROBE: multiprocessing.Pool speedup test")
    print(f"Python workers tested : {WORKER_COUNTS}")
    print(f"Timesteps per trial   : {STEPS}")
    print(f"Seed                  : {SEED}")
    print("=" * 60)

    results = {}

    for label, n_nodes in [("SMALL", SMALL_NODES), ("LARGE", LARGE_NODES)]:
        pos, springs, k, rest_L = build_mesh(n_nodes)
        vel = np.zeros_like(pos)
        n_springs = len(springs)
        print(f"\n--- Mesh: {label} | {n_nodes:,} nodes | {n_springs:,} springs ---")
        t1 = None
        results[label] = {}
        for nw in WORKER_COUNTS:
            elapsed, _ = run_trial(pos, vel, springs, k, rest_L, nw, STEPS)
            if t1 is None:
                t1 = elapsed
            speedup = t1 / elapsed
            efficiency = speedup / nw * 100
            results[label][nw] = dict(t=elapsed, s=speedup, e=efficiency)
            print(f"  workers={nw:2d} | t={elapsed:7.3f}s | speedup={speedup:.3f}x | eff={efficiency:.1f}%")

    print("\n" + "=" * 60)
    print("VERDICT:")
    large_s8 = results["LARGE"][8]["s"]
    if large_s8 >= 2.0:
        print(f"  PASS — Large mesh speedup(8)={large_s8:.2f}x >= 2.0 → proceed with A1 (Pool)")
    elif large_s8 >= 1.5:
        print(f"  MARGINAL — Large mesh speedup(8)={large_s8:.2f}x in [1.5,2.0)")
        print("  Recommendation: add shared_memory layer for high-core runs")
    else:
        print(f"  FAIL — Large mesh speedup(8)={large_s8:.2f}x < 1.5")
        print("  Recommendation: switch to shared_memory (A2) or vectorised NumPy")
    print("=" * 60)

    # Save raw numbers for the ledger
    import json, pathlib
    out = pathlib.Path(__file__).parent / "probe_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {out}")


if __name__ == "__main__":
    main()

