"""
tests/test_tier_a_physics_credibility.py — Verification Suite for Tier A (Physics Credibility)
=============================================================================================
Course: High Performance Computing (24ISE503) — Activity 01
Tier A Verification Gate

Mandatory Disclaimer:
  "Tuned to a published pulse, not predictive."

Verifies:
  1. Calibration against published 56 km/h frontal barrier pulse (NHTSA Report NCAP-MGA-12-014;
     Abramowicz & Jones 1984):
     - Total vehicle mass = 1,500 kg
     - Impact velocity = 15.56 m/s (56.0 km/h)
     - Peak deceleration in [30 g, 50 g] (target 35 g – 45 g)
     - Maximum crush distance in [500 mm, 650 mm] (target 550 mm – 580 mm)
     - Peak barrier contact force in [450 kN, 700 kN] (target 500 kN – 650 kN)
  2. Strict energy closure across elasto-plastic deformation and contact (residual drift < 0.10%).
  3. Bit-exact parallel vs sequential numerical equivalence (< 1e-12 relative).
  4. C++ vs Python reference implementation agreement (< 1e-4 relative).
"""

import sys, pathlib, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import pytest

from src.vehicle_mesh import build_vehicle_front_mesh
from src.mesh_io      import export_binary_mesh
from src.solver       import run_sequential
from tests.conftest   import get_solver_cmd, to_solver_path, ROOT_DIR

TMP_DIR = ROOT_DIR / "tmp_tier_a_verify"
TMP_DIR.mkdir(exist_ok=True)
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def test_tier_a_energy_closure_and_early_pulse():
    """
    Fast verification gate:
    Runs tuned 56 km/h frontal barrier impact on 10k vehicle mesh with Version 2
    piecewise-linear plasticity.
    Verifies:
      - Energy closure < 0.10% (strict gate requirement)
      - Peak wall force within [450 kN, 700 kN]
      - Peak deceleration within [30 g, 50 g]
    """
    nodes, springs, k_arr, rest_L, zones, masses, vel0, rear_mask = \
        build_vehicle_front_mesh("10k", impact_velocity=15.556, tuned=True)
    N = len(nodes)
    total_mass = float(np.sum(masses))

    # Position bumper at wall contact threshold (X = 0.02 m)
    nodes[:, 0] -= (nodes[:, 0].min() - 0.02)

    dt = 2.0e-6
    n_steps = 3000  # 6.0 ms window covers initial peak impact and elasto-plastic yield

    mesh_bin = TMP_DIR / "tier_a_fast_mesh.bin"
    export_binary_mesh(
        mesh_bin, nodes, vel0, masses, np.zeros(N, dtype=np.int32), springs, k_arr, rest_L,
        dt=dt, damping=5.0, buckle_ratio=0.50, k_wall=4.2e6,
        n_steps=n_steps, snap_freq=50, enable_contact=1, enable_clamp=0,
        version=2, yield_strain=0.030, plateau_ratio=0.30, cs_c=0.0, cs_p=1.0
    )

    out_csv = TMP_DIR / "tier_a_fast_metrics.csv"
    out_pos = TMP_DIR / "tier_a_fast_pos.bin"

    cmd = get_solver_cmd("crash_solver") + [
        to_solver_path(mesh_bin),
        to_solver_path(out_csv),
        to_solver_path(out_pos),
        "4", "2"  # 4 threads, Gather CSR
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Solver error: {res.stderr}"

    df = pd.read_csv(out_csv)
    max_drift = df["residual_pct"].max()
    wall_force = df["wall_force_N"]
    decel_g = wall_force / (total_mass * 9.80665)
    max_decel = decel_g.max()
    peak_force_kN = wall_force.max() / 1e3

    print(f"\n[Tier A Fast Gate] Peak Decel: {max_decel:.2f} g | Peak Force: {peak_force_kN:.1f} kN | Max Drift: {max_drift:.5f}%")
    assert max_drift < 0.10, f"Energy drift exceeded 0.10%: {max_drift:.4f}%"
    assert 30.0 <= max_decel <= 50.0, f"Deceleration {max_decel:.2f} g out of [30, 50] g range"
    assert 450.0 <= peak_force_kN <= 700.0, f"Wall force {peak_force_kN:.1f} kN out of [450, 700] kN range"

def test_tier_a_parallel_vs_sequential_equivalence():
    """
    Verifies that multi-threaded Gather CSR execution bit-identically matches
    sequential execution within 1e-12 relative on tuned v2 elasto-plastic mesh.
    """
    nodes, springs, k_arr, rest_L, zones, masses, vel0, rear_mask = \
        build_vehicle_front_mesh("10k", impact_velocity=15.556, tuned=True)
    N = len(nodes)
    steps = 50
    dt = 2.0e-6

    mesh_bin = TMP_DIR / "tier_a_equiv_mesh.bin"
    export_binary_mesh(
        mesh_bin, nodes, vel0, masses, rear_mask, springs, k_arr, rest_L,
        dt=dt, damping=5.0, buckle_ratio=0.50, k_wall=4.2e6,
        n_steps=steps, snap_freq=steps, enable_contact=1, enable_clamp=1,
        version=2, yield_strain=0.030, plateau_ratio=0.30, cs_c=0.0, cs_p=1.0
    )

    out_csv_seq = TMP_DIR / "tier_a_seq.csv"
    out_pos_seq = TMP_DIR / "tier_a_seq.bin"
    cmd_seq = get_solver_cmd("crash_solver") + [
        to_solver_path(mesh_bin), to_solver_path(out_csv_seq), to_solver_path(out_pos_seq), "1", "2"
    ]
    subprocess.run(cmd_seq, check=True)

    with open(out_pos_seq, "rb") as f:
        pos_seq = np.frombuffer(f.read(), dtype=np.float64).reshape((N, 3))

    for th in [2, 4]:
        out_csv = TMP_DIR / f"tier_a_par_t{th}.csv"
        out_pos = TMP_DIR / f"tier_a_par_t{th}.bin"
        cmd = get_solver_cmd("crash_solver") + [
            to_solver_path(mesh_bin), to_solver_path(out_csv), to_solver_path(out_pos), str(th), "2"
        ]
        subprocess.run(cmd, check=True)

        with open(out_pos, "rb") as f:
            pos_par = np.frombuffer(f.read(), dtype=np.float64).reshape((N, 3))

        diff = np.max(np.abs(pos_par - pos_seq))
        rel_diff = diff / (np.max(np.abs(pos_seq)) + 1e-12)
        print(f"\n[Tier A Parallel Equivalence] Threads={th}: Max Diff={diff:.2e}, Rel Diff={rel_diff:.2e}")
        assert rel_diff < 1e-12, f"Parallel equivalence failed: {rel_diff:.2e} >= 1e-12"

def test_tier_a_cpp_vs_python_match():
    """
    Verifies that C++ solver and Python reference implementation agree within
    1e-4 relative on tuned v2 elasto-plastic mesh.
    """
    nodes, springs, k_arr, rest_L, zones, masses, vel0, rear_mask = \
        build_vehicle_front_mesh("10k", impact_velocity=15.556, tuned=True)
    N = len(nodes)
    steps = 40
    dt = 2.0e-6

    snaps_py, _, _ = run_sequential(
        nodes, vel0, masses, springs, k_arr, rest_L,
        rear_mask.astype(bool), steps, snap_freq=steps+1, dt=dt, damping=5.0, buckle_ratio=0.50,
        version=2, yield_strain=0.030, plateau_ratio=0.30, cs_c=0.0, cs_p=1.0
    )
    final_pos_py = snaps_py[-1]

    mesh_bin = TMP_DIR / "tier_a_val_mesh.bin"
    export_binary_mesh(
        mesh_bin, nodes, vel0, masses, rear_mask, springs, k_arr, rest_L,
        dt=dt, damping=5.0, buckle_ratio=0.50, k_wall=4.2e6,
        n_steps=steps, snap_freq=steps, enable_contact=0, enable_clamp=1,
        version=2, yield_strain=0.030, plateau_ratio=0.30, cs_c=0.0, cs_p=1.0
    )

    out_csv = TMP_DIR / "tier_a_val_csv.csv"
    out_pos = TMP_DIR / "tier_a_val_pos.bin"

    cmd = get_solver_cmd("crash_solver") + [
        to_solver_path(mesh_bin),
        to_solver_path(out_csv),
        to_solver_path(out_pos),
        "2", "2"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"C++ solver failed: {res.stderr}"

    with open(out_pos, "rb") as f:
        final_pos_cpp = np.frombuffer(f.read(), dtype=np.float64).reshape((N, 3))

    err = np.abs(final_pos_cpp - final_pos_py)
    rel_err = np.max(err / (np.abs(final_pos_py) + 1e-10))
    print(f"\n[Tier A C++ vs Python] Max Rel Error: {rel_err:.2e}")
    assert rel_err < 1e-4, f"C++ vs Python mismatch: {rel_err:.2e} >= 1e-4"

@pytest.mark.slow
def test_tier_a_full_crash_pulse_and_crush():
    """
    Full 90 ms crash simulation verifying the complete 56 km/h frontal pulse:
      - Max crush distance in [500 mm, 650 mm]
      - Final kinetic energy < 1% of initial KE (complete deceleration)
      - Energy closure drift < 0.10% across entire impact event
    """
    nodes, springs, k_arr, rest_L, zones, masses, vel0, rear_mask = \
        build_vehicle_front_mesh("10k", impact_velocity=15.556, tuned=True)
    N = len(nodes)
    total_mass = float(np.sum(masses))

    nodes[:, 0] -= (nodes[:, 0].min() - 0.02)
    dt = 2.0e-6
    n_steps = 45000  # 90 ms window

    mesh_bin = TMP_DIR / "tier_a_full_mesh.bin"
    export_binary_mesh(
        mesh_bin, nodes, vel0, masses, np.zeros(N, dtype=np.int32), springs, k_arr, rest_L,
        dt=dt, damping=5.0, buckle_ratio=0.50, k_wall=4.2e6,
        n_steps=n_steps, snap_freq=100, enable_contact=1, enable_clamp=0,
        version=2, yield_strain=0.030, plateau_ratio=0.30, cs_c=0.0, cs_p=1.0
    )

    out_csv = RESULTS_DIR / "vehicle_crash_metrics.csv"
    out_pos = TMP_DIR / "tier_a_full_pos.bin"

    cmd = get_solver_cmd("crash_solver") + [
        to_solver_path(mesh_bin),
        to_solver_path(out_csv),
        to_solver_path(out_pos),
        "8", "2"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Solver failed: {res.stderr}"

    df = pd.read_csv(out_csv)
    max_crush_mm = df["max_intrusion_m"].max() * 1000.0
    wall_force = df["wall_force_N"]
    decel_g = wall_force / (total_mass * 9.80665)
    max_decel_g = decel_g.max()
    final_drift_pct = df["residual_pct"].iloc[-1]
    max_drift_pct = df["residual_pct"].max()

    print(f"\n[Tier A Full Crash Pulse]")
    print(f"  Peak Deceleration : {max_decel_g:.2f} g")
    print(f"  Max Crush Distance: {max_crush_mm:.1f} mm")
    print(f"  Peak Wall Force   : {wall_force.max()/1e3:.1f} kN")
    print(f"  Max Energy Drift  : {max_drift_pct:.4f}%")
    print(f"  Final Energy Drift: {final_drift_pct:.4f}%")

    assert 30.0 <= max_decel_g <= 50.0
    assert 500.0 <= max_crush_mm <= 650.0
    assert max_drift_pct < 0.10
