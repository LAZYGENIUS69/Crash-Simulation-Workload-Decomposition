"""
test_energy_closure.py — Rigorous Energy Conservation Audit
============================================================
Course: High Performance Computing (24ISE503) — Activity 01
Iteration 4b Verification Suite

Tests:
1. test_case_a_elastic: Purely elastic, undamped, no-contact case (drift < 0.1%).
2. test_case_b_full_crash (@pytest.mark.slow): Full crash with wall contact,
   damping, and plastic buckling (residual < 0.05%).
"""

import sys, pathlib, subprocess
import numpy as np
import pandas as pd
import pytest

from src.car_mesh import build_car_mesh
from src.mesh_io  import export_binary_mesh
from tests.conftest import get_solver_cmd, to_solver_path, ROOT_DIR

TMP_DIR = ROOT_DIR / "tmp_test_energy"
TMP_DIR.mkdir(exist_ok=True)

def test_case_a_elastic():
    """Fast unit test: elastic spring oscillation energy conservation."""
    nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = \
        build_car_mesh(nx=10, ny=4, nz=4, seed=42)
    N = len(nodes)
    rear_mask = np.zeros(N, dtype=np.int32)

    for i in range(N):
        vel0[i, 0] = -15.0 + 8.0 * np.sin(i * 0.5)
        vel0[i, 1] = 5.0 * np.cos(i * 0.3)
        vel0[i, 2] = 3.0 * np.sin(i * 0.7)

    n_steps = 500
    dt = 5e-6

    mesh_bin = TMP_DIR / "case_a_mesh.bin"
    export_binary_mesh(
        mesh_bin, nodes, vel0, masses, rear_mask, springs, k_arr, rest_L,
        dt=dt, damping=0.0, buckle_ratio=0.0, k_wall=0.0,
        n_steps=n_steps, snap_freq=10, enable_contact=0, enable_clamp=0
    )

    out_csv = TMP_DIR / "case_a_metrics.csv"
    out_pos = TMP_DIR / "case_a_pos.bin"

    cmd = get_solver_cmd("crash_solver") + [
        to_solver_path(mesh_bin),
        to_solver_path(out_csv),
        to_solver_path(out_pos),
        "2", "2"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Execution error: {res.stderr}"

    df = pd.read_csv(out_csv)
    max_drift = df["residual_pct"].max()
    tol = 0.10
    assert max_drift < tol, f"Elastic drift exceeded tolerance: {max_drift:.5f}% >= {tol}%"

@pytest.mark.slow
def test_case_b_full_crash():
    """Long test: complete crash event with contact, damping, and plastic buckling."""
    nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = \
        build_car_mesh(nx=16, ny=6, nz=5, seed=42)
    N = len(nodes)
    rear_mask = np.zeros(N, dtype=np.int32)

    x_min = nodes[:, 0].min()
    nodes[:, 0] -= (x_min - 0.05)
    vel0[:, 0] = -13.9
    vel0[:, 1] = 0.0
    vel0[:, 2] = 0.0

    n_steps = 1500
    dt = 5e-6

    mesh_bin = TMP_DIR / "case_b_mesh.bin"
    export_binary_mesh(
        mesh_bin, nodes, vel0, masses, rear_mask, springs, k_arr, rest_L,
        dt=dt, damping=20.0, buckle_ratio=0.50, k_wall=1e7,
        n_steps=n_steps, snap_freq=10, enable_contact=1, enable_clamp=0
    )

    out_csv = TMP_DIR / "case_b_metrics.csv"
    out_pos = TMP_DIR / "case_b_pos.bin"

    cmd = get_solver_cmd("crash_solver") + [
        to_solver_path(mesh_bin),
        to_solver_path(out_csv),
        to_solver_path(out_pos),
        "4", "2"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Execution error: {res.stderr}"

    df = pd.read_csv(out_csv)
    max_res = df["residual_pct"].max()
    tol = 0.05
    assert max_res < tol, f"Full crash energy residual exceeded tolerance: {max_res:.3f}% >= {tol}%"

if __name__ == "__main__":
    test_case_a_elastic()
    print("Case A PASSED.")
    test_case_b_full_crash()
    print("Case B PASSED.")
