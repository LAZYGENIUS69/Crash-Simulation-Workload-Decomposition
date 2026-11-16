"""
test_correctness.py — verify parallel Python solver == sequential to tol=1e-3
"""
import sys, pathlib
import numpy as np
import pytest

from src.car_mesh import build_car_mesh
from src.solver   import run_check

def test_python_solver_correctness():
    nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = \
        build_car_mesh(nx=10, ny=4, nz=4)
    rear_mask = nodes[:, 0] > (nodes[:, 0].max() * 0.80)

    for nw in [2, 4]:
        ok = run_check(nodes, vel0, masses, springs, k_arr, rest_L,
                       rear_mask, n_steps=20, n_workers=nw, tol=1e-3)
        assert ok, f"Parallel failed correctness check with {nw} workers"

if __name__ == "__main__":
    test_python_solver_correctness()
    print("All correctness checks PASSED.")
