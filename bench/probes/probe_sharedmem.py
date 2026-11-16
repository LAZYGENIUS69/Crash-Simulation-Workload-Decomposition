"""
probe_sharedmem.py — Iteration 1c: shared_memory zero-copy parallel test
PURPOSE  : Test multiprocessing.shared_memory as the parallelism primitive.
           Workers READ positions from shared buffer (zero-copy) and write
           forces to their own private slab. Coordinator reduces and integrates.
           This maps directly to SMP MPI shared-memory windows in real HPC.
EXPECTED : speedup(8) >= 3.0 for LARGE mesh
SEED     : 42
RUN CMD  : python probe_sharedmem.py
"""

import time
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import os

SEED          = 42
STEPS         = 200
SMALL_N       = 2_000
LARGE_N       = 20_000
WORKER_COUNTS = [1, 2, 4, 6, 8, 10, 12]

rng = np.random.default_rng(SEED)


def build_mesh(n_nodes):
    pos = rng.random((n_nodes, 3)).astype(np.float64)
    # linear chain + diagonal springs
    i1 = np.arange(n_nodes - 1, dtype=np.int32)
    j1 = i1 + 1
    i2 = np.arange(n_nodes - 2, dtype=np.int32)
    j2 = i2 + 2
    springs = np.vstack([
        np.column_stack([i1, j1]),
        np.column_stack([i2, j2])
    ])
    return pos, springs, 1500.0, 0.05


# ---------------------------------------------------------------------------
# WORKER: reads pos from shared mem, writes force contribution to its own slab
# ---------------------------------------------------------------------------
def worker_fn(shm_pos_name, shm_force_name, pos_shape,
              springs_chunk, k, rest_L,
              worker_id, n_workers, barrier, n_steps):
    """
    Each worker:
      1. Attaches to shared pos and force buffers
      2. At each step: computes its spring chunk forces,
         scatter-adds into force slab, then hits barrier
    """
    # Attach shared pos (read-only view)
    shm_pos = shared_memory.SharedMemory(name=shm_pos_name)
    pos = np.ndarray(pos_shape, dtype=np.float64, buffer=shm_pos.buf)

    n_nodes = pos_shape[0]
    # Each worker owns a force slab of the same (N,3) shape but we'll
    # write only to a private buffer and return via pipe — simpler + safe
    # (true shared force with atomic add requires extra sync; this is honest)
    
    # Open a pipe end passed at process creation is tricky; instead use
    # a dedicated shared memory slab per worker for forces
    shm_force = shared_memory.SharedMemory(name=shm_force_name)
    # force buffer layout: (n_workers, n_nodes, 3) → worker_id slice
    full_force = np.ndarray((n_workers, n_nodes, 3), dtype=np.float64,
                             buffer=shm_force.buf)
    my_force = full_force[worker_id]  # shape (n_nodes, 3)

    i_idx = springs_chunk[:, 0]
    j_idx = springs_chunk[:, 1]

    for step in range(n_steps):
        # --- COMPUTE PHASE (parallel across workers) ---
        # Read positions (no copy — shared mem)
        p_i = pos[i_idx]           # (M,3) view copy
        p_j = pos[j_idx]
        delta = p_j - p_i
        dist  = np.linalg.norm(delta, axis=1, keepdims=True)
        safe  = np.where(dist < 1e-12, 1.0, dist)
        f_vec = k * (dist - rest_L) / safe * delta / safe   # (M,3)
        my_force[:] = 0.0
        np.add.at(my_force, i_idx,  f_vec)
        np.add.at(my_force, j_idx, -f_vec)

        # --- SYNC BARRIER (models MPI_Barrier + halo exchange) ---
        barrier.wait()
        # Coordinator (outside) does reduce + Verlet; then hits barrier again
        barrier.wait()

    shm_pos.close()
    shm_force.close()


def run_trial_shm(pos_init, vel_init, springs, k, rest_L, n_workers, n_steps):
    n_nodes = pos_init.shape[0]
    mass    = 1.0
    dt      = 1e-4

    # Allocate shared memory
    pos_bytes   = pos_init.nbytes
    force_bytes = n_workers * n_nodes * 3 * 8   # float64

    shm_pos   = shared_memory.SharedMemory(create=True, size=pos_bytes)
    shm_force = shared_memory.SharedMemory(create=True, size=force_bytes)

    # Initialise shared pos
    shm_pos_arr = np.ndarray(pos_init.shape, dtype=np.float64, buffer=shm_pos.buf)
    shm_pos_arr[:] = pos_init

    full_force = np.ndarray((n_workers, n_nodes, 3), dtype=np.float64,
                             buffer=shm_force.buf)
    full_force[:] = 0.0

    vel = vel_init.copy()
    chunks = np.array_split(springs, n_workers)

    barrier = mp.Barrier(n_workers + 1)   # workers + coordinator

    if n_workers == 1:
        # Single-process: skip shared mem overhead, run directly
        shm_pos.unlink(); shm_pos.close()
        shm_force.unlink(); shm_force.close()

        pos = pos_init.copy()
        vel = vel_init.copy()
        i_idx = springs[:, 0]; j_idx = springs[:, 1]
        t0 = time.perf_counter()
        for _ in range(n_steps):
            delta = pos[j_idx] - pos[i_idx]
            dist  = np.linalg.norm(delta, axis=1, keepdims=True)
            safe  = np.where(dist < 1e-12, 1.0, dist)
            f_vec = k * (dist - rest_L) / safe * delta / safe
            force = np.zeros_like(pos)
            np.add.at(force, i_idx,  f_vec)
            np.add.at(force, j_idx, -f_vec)
            vel += (force / mass) * dt
            pos += vel * dt
        return time.perf_counter() - t0, pos

    # Launch workers
    procs = []
    for wid in range(n_workers):
        p = mp.Process(
            target=worker_fn,
            args=(shm_pos.name, shm_force.name, pos_init.shape,
                  chunks[wid], k, rest_L,
                  wid, n_workers, barrier, n_steps)
        )
        p.start()
        procs.append(p)

    t0 = time.perf_counter()
    for step in range(n_steps):
        barrier.wait()   # wait for all workers to finish force compute
        # Coordinator: reduce forces, Verlet integrate, write back pos
        total_force = full_force.sum(axis=0)   # (n_nodes,3)
        vel += (total_force / mass) * dt
        shm_pos_arr += vel * dt                # update shared pos in-place
        full_force[:] = 0.0
        barrier.wait()   # release workers for next step

    elapsed = time.perf_counter() - t0

    for p in procs:
        p.join()

    final_pos = shm_pos_arr.copy()
    shm_pos.unlink();   shm_pos.close()
    shm_force.unlink(); shm_force.close()
    return elapsed, final_pos


def main():
    print("=" * 65)
    print("PROBE 1c: shared_memory zero-copy parallel test")
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
            elapsed, _ = run_trial_shm(pos, vel, springs, k, rest_L, nw, STEPS)
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
        print("  PASS — shared_memory is the architecture.")
    elif large_s8 >= 1.5:
        print("  MARGINAL — usable, explain barrier overhead.")
    else:
        print("  FAIL — barrier cost dominates; need larger mesh or fewer sync points.")
    print("=" * 65)

    out = pathlib.Path(__file__).parent / "probe_shm_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {out}")


if __name__ == "__main__":
    main()

