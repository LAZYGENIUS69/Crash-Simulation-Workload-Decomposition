"""
quick_time_check.py — Verify step counts for 10k, 100k, 500k meshes so 1-thread takes >= 2-3s
"""
import sys, pathlib, subprocess, time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from src.car_mesh import build_car_mesh
from src.mesh_io  import export_binary_mesh

TMP_DIR = pathlib.Path("tmp_timing_check")
TMP_DIR.mkdir(exist_ok=True)
WSL_BASE = "/mnt/c/Users/abhay/Crash-Simulation Workload Decomposition"

configs = [
    ("10k",  50, 15, 13, 1500),
    ("100k", 125, 30, 27, 150),
    ("500k", 250, 50, 40, 30)
]

for label, nx, ny, nz, steps in configs:
    nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = \
        build_car_mesh(nx=nx, ny=ny, nz=nz, seed=42)
    N = len(nodes)
    M = len(springs)
    rear_mask = np.zeros(N, dtype=np.int32)
    
    bin_file = TMP_DIR / f"{label}.bin"
    export_binary_mesh(
        bin_file, nodes, vel0, masses, rear_mask, springs, k_arr, rest_L,
        dt=5e-6, damping=20.0, buckle_ratio=0.50, k_wall=1e7,
        n_steps=steps, snap_freq=steps, enable_contact=1, enable_clamp=1
    )
    
    wsl_bin = f"{WSL_BASE}/tmp_timing_check/{label}.bin"
    wsl_csv = f"{WSL_BASE}/tmp_timing_check/{label}.csv"
    wsl_pos = f"{WSL_BASE}/tmp_timing_check/{label}.pos"
    
    t0 = time.perf_counter()
    res = subprocess.run([
        "wsl", f"{WSL_BASE}/bin/crash_solver",
        wsl_bin, wsl_csv, wsl_pos, "1", "2"
    ], capture_output=True, text=True)
    dt = time.perf_counter() - t0
    print(f"Mesh {label:4s} | Nodes: {N:6d} | Springs: {M:7d} | Steps: {steps:4d} | 1-thread Time: {dt:.3f} s")

