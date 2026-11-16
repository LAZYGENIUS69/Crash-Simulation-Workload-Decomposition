"""
test_analytic_wall.py — Independent Verification of Rigid-Wall Penalty Contact
==============================================================================
Validates penalty wall contact mechanics against closed-form harmonic oscillator.
"""

import struct, subprocess, pathlib
import numpy as np
import pandas as pd
import pytest

from tests.conftest import get_solver_cmd, to_solver_path, ROOT_DIR

TMP_DIR = ROOT_DIR / "tmp_test_wall"
TMP_DIR.mkdir(exist_ok=True)

def test_analytic_wall_contact():
    M_MASS = 1.0
    V0 = -13.9
    K_WALL = 1.0e7
    DT = 1.0e-6       # 1 microsecond timestep
    STEPS = 2000      # 2 ms total time

    omega = np.sqrt(K_WALL / M_MASS)
    delta_max_analytic = np.abs(V0) / omega

    pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    vel = np.array([[V0, 0.0, 0.0]], dtype=np.float64)
    masses = np.array([M_MASS], dtype=np.float64)
    rear_mask = np.array([0], dtype=np.int32)
    springs = np.zeros((0, 2), dtype=np.int32)
    k_arr = np.zeros(0, dtype=np.float64)
    rest_L = np.zeros(0, dtype=np.float64)

    mesh_bin = TMP_DIR / "analytic_wall_mesh.bin"
    with open(mesh_bin, "wb") as f:
        header = struct.pack(
            "<IIIIIIII dddd",
            0x43525348, 1, 1, 0, STEPS, 10, 1, 0,
            float(DT), float(0.0), float(0.50), float(K_WALL)
        )
        f.write(header)
        f.write(pos.tobytes())
        f.write(vel.tobytes())
        f.write(masses.tobytes())
        f.write(rear_mask.tobytes())
        f.write(springs.tobytes())
        f.write(k_arr.tobytes())
        f.write(rest_L.tobytes())

    out_csv = TMP_DIR / "analytic_wall.csv"
    out_pos = TMP_DIR / "analytic_wall_pos.bin"

    cmd = get_solver_cmd("crash_solver") + [
        to_solver_path(mesh_bin),
        to_solver_path(out_csv),
        to_solver_path(out_pos),
        "1", "0", "0"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Solver failed: {res.stderr}"

    df = pd.read_csv(out_csv)
    max_intrusion_num = df["max_intrusion_m"].max()
    rel_err = np.abs(max_intrusion_num - delta_max_analytic) / delta_max_analytic

    tol = 0.01  # 1% tolerance due to discrete 1us timestep
    assert rel_err < tol, f"Analytic discrepancy too large: {rel_err:.4e} >= {tol}"

if __name__ == "__main__":
    test_analytic_wall_contact()
    print("Analytic wall contact test PASSED.")
