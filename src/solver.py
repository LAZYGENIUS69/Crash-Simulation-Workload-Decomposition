"""
solver.py — Sequential + Parallel Verlet crash-simulation solvers
=================================================================
DESIGN PHILOSOPHY
  Keep the parallel design as explainable as possible (viva requirement).
  Every piece of code maps directly to a real HPC concept.

PARALLEL STRATEGY: Coordinator + Shared-Memory Workers
  - Coordinator owns the full pos/vel/force arrays in shared memory
  - Workers are assigned a contiguous spring subset (not node subset)
  - Each worker: reads pos → computes spring forces → writes force contribution
  - Coordinator: waits at barrier → sums all force contributions → Verlet integrate
  - This maps 1-to-1 to the LS-DYNA MPP architecture:
      Worker  = MPI rank computing internal forces for its element partition
      Barrier = MPI_Barrier + allreduce for force assembly
      Coordinator = time integrator (global, serial)

  WHY NOT NODE-DOMAIN DECOMPOSITION?
  Domain decomposition by node requires ghost-node exchange at boundaries.
  For this 3D spring mesh the boundary:interior ratio is ~40%, meaning
  40% of nodes have cross-boundary springs → heavy ghost traffic.
  Spring decomposition is both simpler AND more load-balanced here.
  (Noted in report as an industrial tradeoff.)

SERIAL FRACTION (Amdahl):
  The coordinator's reduce + Verlet is O(N) serial work every timestep.
  We MEASURE it, fit Amdahl's law, and report the serial fraction honestly.
  This matches LS-DYNA's global timestep reduction bottleneck exactly.

CORRECTNESS:
  run_check() verifies par ≡ seq to rel tol 1e-5.

USAGE:
  from src.solver import run_sequential, run_parallel, run_check
"""

import time, multiprocessing as mp
from multiprocessing import shared_memory, Barrier, Queue
import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
DT           = 5e-6     # timestep [s]
DAMPING      = 20.0     # viscous damping coefficient [N·s/m]
BUCKLE_RATIO = 0.50     # spring buckles below this fraction of rest length
SNAP_FREQ    = 10       # save position snapshot every N steps


# ── Force kernel (shared between both solvers) ────────────────────────────────
def _spring_forces_full(pos, sp_i, sp_j, k_arr, rest_L, N):
    """
    Vectorised Hooke + buckling force kernel.
    Computes force contributions for every spring in sp_i/sp_j.

    Returns force array shape (N, 3).
    """
    delta = pos[sp_j] - pos[sp_i]                          # (M, 3)
    dist  = np.linalg.norm(delta, axis=1, keepdims=True)   # (M, 1)
    safe  = np.where(dist < 1e-14, 1.0, dist)

    # Zero out buckled springs (heavy compression)
    buckle = (dist < BUCKLE_RATIO * rest_L[:, None])

    hooke  = k_arr[:, None] * (dist - rest_L[:, None]) / safe  # (M, 1)
    hooke  = np.where(buckle, 0.0, hooke)
    f_vec  = hooke * delta                                     # (M, 3) Hooke force k*(dist-L0)*delta/dist

    force  = np.zeros((N, 3), dtype=np.float64)
    np.add.at(force, sp_i,  f_vec)
    np.add.at(force, sp_j, -f_vec)
    return force


# ── Sequential solver ─────────────────────────────────────────────────────────
def run_sequential(nodes_init, vel_init, masses,
                   springs, k_arr, rest_L,
                   rear_mask, n_steps, snap_freq=SNAP_FREQ):
    """
    Single-process Verlet integration (baseline).

    Returns
    -------
    snapshots : list[ndarray(N,3)]  — position snapshot at each snap_freq step + final
    t_wall    : float               — total wall-clock seconds
    stage_ms  : dict                — per-stage cumulative ms {force, integrate, io}
    """
    N    = nodes_init.shape[0]
    pos  = nodes_init.copy()
    vel  = vel_init.copy()
    invM = (1.0 / masses)[:, None]   # (N, 1)

    sp_i = springs[:, 0]
    sp_j = springs[:, 1]

    snapshots = [pos.copy()]
    t_f = t_i = t_io = 0.0
    t0  = time.perf_counter()

    # Initial force & acceleration pass for true Velocity-Verlet
    force = _spring_forces_full(pos, sp_i, sp_j, k_arr, rest_L, N)
    acc   = force * invM

    for step in range(n_steps):
        # ── VERLET STAGE A: KINEMATICS ──────────────────────────────
        ti0  = time.perf_counter()
        vel += acc * (0.5 * DT)
        pos += vel * DT
        vel[rear_mask] = 0.0
        pos[rear_mask] = nodes_init[rear_mask]
        t_i += time.perf_counter() - ti0

        # ── FORCE EVALUATION AT x(t + dt) ───────────────────────────
        tf0   = time.perf_counter()
        force = _spring_forces_full(pos, sp_i, sp_j, k_arr, rest_L, N)
        t_f  += time.perf_counter() - tf0

        # ── VERLET STAGE B: VELOCITY UPDATE ─────────────────────────
        ti0  = time.perf_counter()
        a_new = force * invM - DAMPING * invM * vel
        vel  += a_new * (0.5 * DT)
        acc   = a_new
        vel[rear_mask] = 0.0
        t_i += time.perf_counter() - ti0

        # ── SNAPSHOT ────────────────────────────────────────────────
        if (step + 1) % snap_freq == 0 or step == n_steps - 1:
            tio0 = time.perf_counter()
            snapshots.append(pos.copy())
            t_io += time.perf_counter() - tio0

    t_wall   = time.perf_counter() - t0
    stage_ms = dict(force_ms=t_f*1e3, integrate_ms=t_i*1e3, io_ms=t_io*1e3)
    return snapshots, t_wall, stage_ms


# ── Parallel worker: computes forces for its spring subset ────────────────────
def _force_worker(shm_pos_name, shm_fpart_name,
                  pos_shape, fpart_shape,
                  sp_i, sp_j, k_chunk, rest_chunk,
                  worker_id, barrier_compute, barrier_done,
                  n_steps, result_q):
    """
    Force-only worker.
    Reads pos from shared mem (read-only).
    Writes partial force to its own shared slab (shm_fpart).
    Two barriers per timestep:
      barrier_compute : all workers start computing at the same time
      barrier_done    : all workers done; coordinator can safely reduce
    """
    shm_pos   = shared_memory.SharedMemory(name=shm_pos_name)
    shm_fpart = shared_memory.SharedMemory(name=shm_fpart_name)
    pos       = np.ndarray(pos_shape,   dtype=np.float64, buffer=shm_pos.buf)
    f_part    = np.ndarray(fpart_shape, dtype=np.float64, buffer=shm_fpart.buf)

    N      = pos_shape[0]
    times  = []

    for step in range(n_steps):
        barrier_compute.wait()          # ─── wait for coordinator to write new pos
        t0 = time.perf_counter()

        f_local = _spring_forces_full(pos, sp_i, sp_j, k_chunk, rest_chunk, N)
        f_part[:] = f_local             # write to my slab of shared force array

        times.append(time.perf_counter() - t0)
        barrier_done.wait()             # ─── signal coordinator we are done

    shm_pos.close()
    shm_fpart.close()
    result_q.put((worker_id, np.array(times)))


# ── Parallel solver ───────────────────────────────────────────────────────────
def run_parallel(nodes_init, vel_init, masses,
                 springs, k_arr, rest_L,
                 rear_mask, n_steps, n_workers, snap_freq=SNAP_FREQ):
    """
    Parallel Verlet solver.
    Workers compute forces; coordinator reduces + integrates.

    Returns
    -------
    snapshots    : list[ndarray(N,3)]
    t_wall       : float [s]
    worker_times : dict {wid: ndarray(n_steps,)} — per-step force times [s]
    stage_ms     : dict — {reduce_ms, integrate_ms, io_ms, barrier_ms}
    """
    N = nodes_init.shape[0]

    # ── Shared memory: positions + per-worker force arrays ───────────
    shm_pos  = shared_memory.SharedMemory(create=True, size=nodes_init.nbytes)
    pos_shm  = np.ndarray(nodes_init.shape, dtype=np.float64, buffer=shm_pos.buf)
    pos_shm[:] = nodes_init

    # Each worker gets its own (N,3) force array — no locks needed
    # Layout: shm_forces[w] = (N,3) slice for worker w
    force_total_shape = (n_workers, N, 3)
    force_bytes       = int(np.prod(force_total_shape)) * 8
    shm_forces = shared_memory.SharedMemory(create=True, size=force_bytes)
    all_forces = np.ndarray(force_total_shape, dtype=np.float64,
                            buffer=shm_forces.buf)
    all_forces[:] = 0.0

    # ── Spring assignment (round-robin by spring index) ───────────────
    spring_chunks = np.array_split(np.arange(len(springs)), n_workers)

    # ── Barriers ─────────────────────────────────────────────────────
    barrier_compute = Barrier(n_workers + 1)   # workers + coordinator
    barrier_done    = Barrier(n_workers + 1)

    result_q = Queue()

    # ── Launch workers ────────────────────────────────────────────────
    procs = []
    for w, idx in enumerate(spring_chunks):
        # Each worker's force slab: row w of shm_forces
        fpart_name  = shm_forces.name
        fpart_shape = (N, 3)
        # We'll use a *view* trick: worker writes to shm_forces[w], which
        # occupies bytes [w*N*3*8 : (w+1)*N*3*8]
        # But SharedMemory is flat — pass offset as argument
        p = mp.Process(
            target=_force_worker_with_offset,
            args=(
                shm_pos.name, shm_forces.name,
                nodes_init.shape, force_total_shape,
                springs[idx, 0].copy(), springs[idx, 1].copy(),
                k_arr[idx].copy(), rest_L[idx].copy(),
                w, barrier_compute, barrier_done,
                n_steps, result_q
            ),
            daemon=True
        )
        p.start()
        procs.append(p)

    # ── Coordinator loop ──────────────────────────────────────────────
    vel   = vel_init.copy()
    invM  = (1.0 / masses)[:, None]
    snapshots = [nodes_init.copy()]
    t_red = t_int = t_io = t_bar = 0.0
    t0 = time.perf_counter()

    # Initial force & acceleration pass for Velocity-Verlet
    force_init = _spring_forces_full(pos_shm, springs[:, 0], springs[:, 1], k_arr, rest_L, N)
    acc = force_init * invM

    for step in range(n_steps):
        # ── VERLET STAGE A: KINEMATICS ──────────────────────────────
        ti0 = time.perf_counter()
        vel += acc * (0.5 * DT)
        pos_shm += vel * DT
        vel[rear_mask] = 0.0
        pos_shm[rear_mask] = nodes_init[rear_mask]
        t_int += time.perf_counter() - ti0

        # Signal workers to compute forces at pos_shm(t + dt)
        tb0 = time.perf_counter()
        barrier_compute.wait()
        barrier_done.wait()
        t_bar += time.perf_counter() - tb0

        # ── REDUCE: sum all workers' force arrays ─────────────────
        tr0   = time.perf_counter()
        force = all_forces.sum(axis=0)   # (N, 3) — serial reduction
        t_red += time.perf_counter() - tr0

        # ── VERLET STAGE B: VELOCITY UPDATE ─────────────────────────
        ti0  = time.perf_counter()
        a_new = force * invM - DAMPING * invM * vel
        vel  += a_new * (0.5 * DT)
        acc   = a_new
        vel[rear_mask] = 0.0
        t_int += time.perf_counter() - ti0

        # Snapshot
        if (step + 1) % snap_freq == 0 or step == n_steps - 1:
            tio0 = time.perf_counter()
            snapshots.append(pos_shm.copy())
            t_io += time.perf_counter() - tio0

    t_wall = time.perf_counter() - t0

    for p in procs:
        p.join()

    worker_times = {}
    while not result_q.empty():
        wid, times = result_q.get_nowait()
        worker_times[wid] = times

    final_pos = pos_shm.copy()
    if not any(np.allclose(s, final_pos) for s in snapshots[-1:]):
        snapshots.append(final_pos)

    shm_pos.unlink();    shm_pos.close()
    shm_forces.unlink(); shm_forces.close()

    stage_ms = dict(
        reduce_ms=t_red*1e3, integrate_ms=t_int*1e3,
        io_ms=t_io*1e3,      barrier_ms=t_bar*1e3
    )
    return snapshots, t_wall, worker_times, stage_ms


def _force_worker_with_offset(
        shm_pos_name, shm_forces_name,
        pos_shape, forces_total_shape,
        sp_i, sp_j, k_chunk, rest_chunk,
        worker_id, barrier_compute, barrier_done,
        n_steps, result_q):
    """
    Force worker that writes into its own row of the shared (P, N, 3) force tensor.
    """
    shm_pos    = shared_memory.SharedMemory(name=shm_pos_name)
    shm_forces = shared_memory.SharedMemory(name=shm_forces_name)
    pos        = np.ndarray(pos_shape,          dtype=np.float64, buffer=shm_pos.buf)
    all_forces = np.ndarray(forces_total_shape, dtype=np.float64, buffer=shm_forces.buf)
    my_force   = all_forces[worker_id]   # (N, 3) view — no copy

    N     = pos_shape[0]
    times = []

    for step in range(n_steps):
        barrier_compute.wait()          # wait for coordinator to update pos
        t0 = time.perf_counter()

        # Vectorised Hooke + buckling
        delta  = pos[sp_j] - pos[sp_i]
        dist   = np.linalg.norm(delta, axis=1, keepdims=True)
        safe   = np.where(dist < 1e-14, 1.0, dist)
        buckle = (dist < BUCKLE_RATIO * rest_chunk[:, None])
        hooke  = k_chunk[:, None] * (dist - rest_chunk[:, None]) / safe
        hooke  = np.where(buckle, 0.0, hooke)
        f_vec  = hooke * delta

        # Scatter-add into MY force row (no race condition — only I write this row)
        my_force[:] = 0.0
        np.add.at(my_force, sp_i, f_vec)
        np.add.at(my_force, sp_j, -f_vec)

        times.append(time.perf_counter() - t0)
        barrier_done.wait()             # signal coordinator

    shm_pos.close()
    shm_forces.close()
    result_q.put((worker_id, np.array(times)))


# ── Correctness check ─────────────────────────────────────────────────────────
def run_check(nodes, vel0, masses, springs, k_arr, rest_L, rear_mask,
              n_steps=30, n_workers=4, tol=1e-5):
    """Verify parallel ≡ sequential within relative tolerance."""
    print(f"\nCorrectness check: N={len(nodes)}, M={len(springs)}, "
          f"steps={n_steps}, workers={n_workers}")
    snaps_s, ts, _ = run_sequential(
        nodes, vel0, masses, springs, k_arr, rest_L,
        rear_mask, n_steps, snap_freq=n_steps+1)

    snaps_p, tp, _, _ = run_parallel(
        nodes, vel0, masses, springs, k_arr, rest_L,
        rear_mask, n_steps, n_workers, snap_freq=n_steps+1)

    fs = snaps_s[-1]; fp = snaps_p[-1]
    err   = np.abs(fp - fs)
    denom = np.abs(fs) + 1e-10
    rel   = err / denom
    max_r = rel.max(); mean_r = rel.mean()
    ok    = max_r < tol
    tag   = "PASS ✓" if ok else "FAIL ✗"
    print(f"  seq wall={ts:.3f}s  par wall={tp:.3f}s")
    print(f"  Max rel error={max_r:.2e}  Mean={mean_r:.2e}  → {tag}")
    if not ok:
        worst = np.unravel_index(rel.argmax(), rel.shape)
        print(f"  Worst node={worst[0]}, ax={worst[1]}: "
              f"seq={fs[worst]:.8f} par={fp[worst]:.8f}")
    return ok
