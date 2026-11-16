"""
test_assembly_equivalence.py — Numerical Equivalence of Assembly Modes
========================================================================
Assert that Mode 0 (Private Slabs), Mode 1 (Atomics), and Mode 2 (Gather)
yield identical trajectories to machine precision.
"""

import sys, pathlib, subprocess
import numpy as np
import pytest

from src.car_mesh import build_car_mesh
from src.mesh_io  import export_binary_mesh
from tests.conftest import get_solver_cmd, to_solver_path, ROOT_DIR

TMP_DIR = ROOT_DIR / "tmp_test_equiv"
TMP_DIR.mkdir(exist_ok=True)

def test_assembly_modes_equivalence():
    nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = \
        build_car_mesh(nx=16, ny=6, nz=5, seed=42)
    N = len(nodes)
    rear_mask = (nodes[:, 0] > (nodes[:, 0].max() * 0.80)).astype(np.int32)

    STEPS = 50
    mesh_bin = TMP_DIR / "equiv_mesh.bin"
    export_binary_mesh(
        mesh_bin, nodes, vel0, masses, rear_mask, springs, k_arr, rest_L,
        dt=5e-6, damping=20.0, buckle_ratio=0.50, k_wall=1e7,
        n_steps=STEPS, snap_freq=STEPS, enable_contact=1, enable_clamp=1
    )

    # Run 1-thread sequential baseline (Mode 2)
    out_csv_seq = TMP_DIR / "seq.csv"
    out_pos_seq = TMP_DIR / "seq.bin"
    cmd_seq = get_solver_cmd("crash_solver") + [
        to_solver_path(mesh_bin), to_solver_path(out_csv_seq), to_solver_path(out_pos_seq), "1", "2"
    ]
    subprocess.run(cmd_seq, check=True)

    with open(out_pos_seq, "rb") as f:
        pos_seq = np.frombuffer(f.read(), dtype=np.float64).reshape((N, 3))

    modes = [(0, "Private Slabs"), (1, "Atomic Updates"), (2, "Node Gather CSR")]
    threads_to_test = [1, 2, 4]

    for mode_id, mode_name in modes:
        for th in threads_to_test:
            out_csv = TMP_DIR / f"m{mode_id}_t{th}.csv"
            out_pos = TMP_DIR / f"m{mode_id}_t{th}.bin"
            cmd = get_solver_cmd("crash_solver") + [
                to_solver_path(mesh_bin), to_solver_path(out_csv), to_solver_path(out_pos), str(th), str(mode_id)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

            with open(out_pos, "rb") as f:
                pos_par = np.frombuffer(f.read(), dtype=np.float64).reshape((N, 3))

            diff = np.abs(pos_par - pos_seq)
            max_diff = np.max(diff)
            assert max_diff < 1e-10, f"Mode {mode_name} (Th={th}) exceeded diff: {max_diff:.2e}"

if __name__ == "__main__":
    test_assembly_modes_equivalence()
    print("All assembly modes are numerically identical.")
