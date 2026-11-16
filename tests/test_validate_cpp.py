"""
test_validate_cpp.py — Assert C++ OpenMP solver matches Python reference
========================================================================
With contact disabled and rear clamping enabled, C++ output must match
the Python reference solver across threads within stated tolerance.
"""

import sys, pathlib, subprocess
import numpy as np
import pytest

from src.car_mesh import build_car_mesh
from src.solver   import run_sequential
from src.mesh_io  import export_binary_mesh
from tests.conftest import get_solver_cmd, to_solver_path, ROOT_DIR

TMP_DIR = ROOT_DIR / "tmp_test_val"
TMP_DIR.mkdir(exist_ok=True)

def test_cpp_matches_python_reference():
    STEPS = 50
    nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = \
        build_car_mesh(nx=18, ny=6, nz=5, seed=42)
    N = len(nodes)
    rear_mask = (nodes[:, 0] > (nodes[:, 0].max() * 0.80)).astype(np.int32)

    snaps_py, _, _ = run_sequential(
        nodes, vel0, masses, springs, k_arr, rest_L,
        rear_mask.astype(bool), STEPS, snap_freq=STEPS+1
    )
    final_pos_py = snaps_py[-1]

    mesh_bin = TMP_DIR / "val_mesh.bin"
    export_binary_mesh(
        mesh_bin, nodes, vel0, masses, rear_mask, springs, k_arr, rest_L,
        dt=5e-6, damping=20.0, buckle_ratio=0.50, k_wall=1e7,
        n_steps=STEPS, snap_freq=STEPS, enable_contact=0, enable_clamp=1
    )

    threads_to_test = [1, 2, 4]
    tol = 1e-4

    for th in threads_to_test:
        out_csv = TMP_DIR / f"val_th{th}.csv"
        out_pos = TMP_DIR / f"val_th{th}.bin"

        cmd = get_solver_cmd("crash_solver") + [
            to_solver_path(mesh_bin),
            to_solver_path(out_csv),
            to_solver_path(out_pos),
            str(th)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Execution failed for {th} threads: {res.stderr}"

        with open(out_pos, "rb") as f:
            final_pos_cpp = np.frombuffer(f.read(), dtype=np.float64).reshape((N, 3))

        err = np.abs(final_pos_cpp - final_pos_py)
        rel_err = err / (np.abs(final_pos_py) + 1e-10)
        max_err = np.max(rel_err)
        assert max_err < tol, f"Threads={th}: max rel error {max_err:.2e} >= {tol}"

if __name__ == "__main__":
    test_cpp_matches_python_reference()
    print("All C++ vs Python validation checks PASSED.")
