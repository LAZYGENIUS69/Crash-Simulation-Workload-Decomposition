"""
probe_large_mesh.py — Iteration 1d: test at production mesh sizes
PURPOSE  : Determine what mesh size gives good speedup on this machine.
           The barrier cost is ~constant; find the crossover point.
APPROACH : Shared-memory workers, vary N = 10k, 50k, 100k, 200k
           Workers = 1 and 8 only (efficiency estimate)
SEED     : 42
RUN CMD  : python probe_large_mesh.py
"""
import time
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

SEED  = 42
STEPS = 100   # fewer steps — these meshes are large

rng = np.random.default_rng(SEED)


def build_mesh(n_nodes):
    pos     = rng.random((n_nodes, 3)).astype(np.float64)
    # 3D grid springs: nearest-neighbor on a linear chain + diagonal
    # For speed of probe, just use chain + 1-skip diagonal
    i1 = np.arange(n_nodes-1, dtype=np.int32)
    j1 = i1 + 1
    i2 = np.arange(n_nodes-2, dtype=np.int32)
    j2 = i2 + 2
    springs = np.vstack([np.column_stack([i1,j1]), np.column_stack([i2,j2])])
    return pos, springs, 1500.0, 0.05


def worker_fn(shm_pos_name, shm_force_name, pos_shape,
              springs_chunk, k, rest_L,
              worker_id, n_workers, barrier, n_steps):
    shm_pos   = shared_memory.SharedMemory(name=shm_pos_name)
    shm_force = shared_memory.SharedMemory(name=shm_force_name)
    pos       = np.ndarray(pos_shape, dtype=np.float64, buffer=shm_pos.buf)
    n_nodes   = pos_shape[0]
    full_force = np.ndarray((n_workers, n_nodes, 3), dtype=np.float64,
                             buffer=shm_force.buf)
    my_force   = full_force[worker_id]
    i_idx = springs_chunk[:, 0]
    j_idx = springs_chunk[:, 1]

    for _ in range(n_steps):
        delta = pos[j_idx] - pos[i_idx]
        dist  = np.linalg.norm(delta, axis=1, keepdims=True)
        safe  = np.where(dist < 1e-12, 1.0, dist)
        f_vec = k * (dist - rest_L) / safe * delta / safe
        my_force[:] = 0.0
        np.add.at(my_force, i_idx,  f_vec)
        np.add.at(my_force, j_idx, -f_vec)
        barrier.wait()
        barrier.wait()

    shm_pos.close(); shm_force.close()


def run_shm(pos_init, vel_init, springs, k, rest_L, n_workers, n_steps):
    n_nodes = pos_init.shape[0]
    mass = 1.0; dt = 1e-5

    if n_workers == 1:
        pos = pos_init.copy(); vel = vel_init.copy()
        i_idx = springs[:,0]; j_idx = springs[:,1]
        t0 = time.perf_counter()
        for _ in range(n_steps):
            delta = pos[j_idx] - pos[i_idx]
            dist  = np.linalg.norm(delta, axis=1, keepdims=True)
            safe  = np.where(dist < 1e-12, 1.0, dist)
            f_vec = k * (dist - rest_L) / safe * delta / safe
            force = np.zeros_like(pos)
            np.add.at(force, i_idx,  f_vec)
            np.add.at(force, j_idx, -f_vec)
            vel  += (force/mass)*dt; pos += vel*dt
        return time.perf_counter()-t0

    shm_pos   = shared_memory.SharedMemory(create=True, size=pos_init.nbytes)
    shm_force = shared_memory.SharedMemory(create=True,
                    size=n_workers*n_nodes*3*8)
    shm_pos_arr   = np.ndarray(pos_init.shape, dtype=np.float64, buffer=shm_pos.buf)
    shm_pos_arr[:] = pos_init
    full_force = np.ndarray((n_workers, n_nodes, 3), dtype=np.float64,
                             buffer=shm_force.buf)
    full_force[:] = 0.0

    vel = vel_init.copy()
    chunks  = np.array_split(springs, n_workers)
    barrier = mp.Barrier(n_workers+1)

    procs = [mp.Process(target=worker_fn,
                        args=(shm_pos.name, shm_force.name, pos_init.shape,
                              chunks[wid], k, rest_L, wid, n_workers,
                              barrier, n_steps))
             for wid in range(n_workers)]
    for p in procs: p.start()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        barrier.wait()
        force = full_force.sum(axis=0)
        vel  += (force/mass)*dt
        shm_pos_arr += vel*dt
        full_force[:] = 0.0
        barrier.wait()
    elapsed = time.perf_counter()-t0

    for p in procs: p.join()
    shm_pos.unlink(); shm_pos.close()
    shm_force.unlink(); shm_force.close()
    return elapsed


def main():
    print("="*65)
    print("PROBE 1d: shared_memory, production mesh sizes")
    print(f"Steps: {STEPS} | Seed: {SEED}")
    print("="*65)
    import json, pathlib

    sizes   = [10_000, 50_000, 100_000, 200_000]
    results = {}
    for n_nodes in sizes:
        pos, springs, k, rest_L = build_mesh(n_nodes)
        vel = np.zeros_like(pos)
        t1 = run_shm(pos, vel, springs, k, rest_L, 1,  STEPS)
        t8 = run_shm(pos, vel, springs, k, rest_L, 8,  STEPS)
        t12= run_shm(pos, vel, springs, k, rest_L, 12, STEPS)
        s8 = t1/t8; s12= t1/t12
        e8 = s8/8*100; e12= s12/12*100
        results[n_nodes] = dict(t1=round(t1,3), t8=round(t8,3), t12=round(t12,3),
                                s8=round(s8,3), s12=round(s12,3),
                                e8=round(e8,2), e12=round(e12,2))
        print(f"N={n_nodes:7,} | t1={t1:6.2f}s | "
              f"t8={t8:6.2f}s s8={s8:.2f}x e8={e8:.0f}% | "
              f"t12={t12:6.2f}s s12={s12:.2f}x e12={e12:.0f}%")

    out = pathlib.Path(__file__).parent / "probe_large_results.json"
    with open(out,"w") as f: json.dump(results,f,indent=2)
    print(f"\nSaved to {out}")

    # Find best mesh size (s8 >= 3.0 or highest s8)
    best_n  = max(results, key=lambda n: results[n]["s8"])
    best_s8 = results[best_n]["s8"]
    print(f"\nBest mesh for 8-worker speedup: N={best_n:,}, speedup={best_s8:.2f}x")
    if best_s8 >= 3.0:
        print("RECOMMENDATION: Use this as LARGE mesh in production solver.")
    else:
        print("RECOMMENDATION: Barrier cost dominates. Report honest Amdahl fit.")


if __name__ == "__main__":
    main()

