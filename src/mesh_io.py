"""
mesh_io.py — Binary serialization between Python and C++ OpenMP solver
========================================================================
Binary Layout:
  Header (64 bytes):
    uint32 magic          : 0x43525348 ('CRSH')
    uint32 version        : 1
    uint32 num_nodes      : N
    uint32 num_springs    : M
    uint32 num_steps      : total Verlet timesteps
    uint32 snap_freq      : dump trajectory every K steps
    uint32 enable_contact : 0=disabled (ref mode), 1=wall contact, 2=wall+self contact
    uint32 enable_clamp   : 1=clamp rear nodes (ref mode), 0=free impact with wall
    float64 dt            : timestep [s]
    float64 damping       : viscous damping [N·s/m]
    float64 buckle_ratio  : buckling compression ratio (e.g. 0.50)
    float64 k_wall        : wall penalty stiffness [N/m]
  Data Arrays:
    float64 pos[N * 3]
    float64 vel[N * 3]
    float64 masses[N]
    int32   rear_mask[N]
    int32   springs[M * 2]
    float64 k_arr[M]
    float64 rest_L[M]
"""

import struct
import numpy as np
import pathlib

MAGIC = 0x43525348
VERSION = 1

def export_binary_mesh(filepath, nodes, vel, masses, rear_mask, springs, k_arr, rest_L,
                       dt=5e-6, damping=20.0, buckle_ratio=0.50, k_wall=1e7,
                       n_steps=100, snap_freq=10, enable_contact=0, enable_clamp=1):
    filepath = pathlib.Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    N = len(nodes)
    M = len(springs)

    with open(filepath, "wb") as f:
        # 64-byte Header
        header = struct.pack(
            "<IIIIIIII dddd",
            MAGIC, VERSION, N, M, n_steps, snap_freq, enable_contact, enable_clamp,
            float(dt), float(damping), float(buckle_ratio), float(k_wall)
        )
        f.write(header)

        # Arrays
        f.write(np.ascontiguousarray(nodes, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(vel, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(masses, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(rear_mask, dtype=np.int32).tobytes())
        f.write(np.ascontiguousarray(springs, dtype=np.int32).tobytes())
        f.write(np.ascontiguousarray(k_arr, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(rest_L, dtype=np.float64).tobytes())

    return filepath

