"""
car_mesh.py — Procedural 3D car-shaped spring-mass mesh
========================================================
Generates a recognizable 3D vehicle as a lattice of point masses
connected by springs with zone-dependent stiffness:

  Crumple Zone (front 20%) : k = k_soft   — deforms on impact
  Cabin / Safety Cell (60%): k = k_stiff  — stays rigid
  Rear Crumple Zone (20%)  : k = k_soft   — rear-impact zone

Coordinate system (all in metres):
  X : vehicle length direction (front = 0, rear = car_length)
  Y : vehicle width  direction (centred at 0)
  Z : vehicle height direction (bottom = 0)

Returns
-------
nodes   : (N, 3)  float64 — initial positions
springs : (M, 2)  int32   — spring endpoint indices
k_arr   : (M,)   float64 — spring constant per spring [N/m]
rest_L  : (M,)   float64 — rest length per spring [m]
zone    : (N,)   int32   — 0=crumple, 1=cabin, 2=rear-crumple
                           for colour-coding

Usage
-----
  from src.car_mesh import build_car_mesh
  nodes, springs, k_arr, rest_L, zone = build_car_mesh(nx=50, ny=10, nz=8)

License / Attribution
---------------------
  Fully procedural — no downloaded geometry. Original work.
  Spring-mass car model concept: standard HPC textbook approach
  (e.g., Zukas, "Introduction to Hydrocodes", 2004).
"""

import numpy as np

# ── Physical parameters ──────────────────────────────────────────────────────
CAR_LENGTH = 4.5    # m
CAR_WIDTH  = 1.8    # m
CAR_HEIGHT = 1.5    # m  (at cabin peak)
HOOD_H_FRAC = 0.60  # hood height as fraction of cabin height
TRUNK_H_FRAC = 0.65 # trunk height as fraction of cabin height

# Zone boundaries (fraction of car length from front)
FRONT_CRUMPLE_END = 0.20   # 0 – 20%  : front crumple zone
CABIN_END         = 0.80   # 20% – 80%: cabin / safety cell
# 80% – 100%: rear crumple zone

# Spring stiffness [N/m]
K_SOFT  =   500_000.0   # crumple zone
K_STIFF = 5_000_000.0   # cabin / safety cell
K_CABIN = 5_000_000.0

# Node mass [kg] — distributed vehicle mass / N (assigned externally)
NODE_MASS = 1.0   # placeholder; solver normalises

IMPACT_VELOCITY = -13.9   # m/s (~50 km/h, negative X = towards barrier)


# ── Roof profile helper ────────────────────────────────────────────────────
def _roof_height(x_norm):
    """
    Returns the roof height at normalised x position (0=front, 1=rear).
    Hood ramps up from front; cabin is flat; trunk ramps down.
    """
    if x_norm < FRONT_CRUMPLE_END:
        # Hood: linear ramp from HOOD_H_FRAC to 1.0
        t = x_norm / FRONT_CRUMPLE_END
        return HOOD_H_FRAC + t * (1.0 - HOOD_H_FRAC)
    elif x_norm < CABIN_END:
        return 1.0   # flat cabin roof
    else:
        # Trunk: linear ramp from 1.0 to TRUNK_H_FRAC
        t = (x_norm - CABIN_END) / (1.0 - CABIN_END)
        return 1.0 - t * (1.0 - TRUNK_H_FRAC)


def _zone(x_norm):
    if x_norm <= FRONT_CRUMPLE_END:
        return 0   # front crumple
    elif x_norm <= CABIN_END:
        return 1   # cabin
    else:
        return 2   # rear crumple


def build_car_mesh(nx=50, ny=10, nz=8, seed=42):
    """
    Parameters
    ----------
    nx : grid points along X (car length) — controls total node count
    ny : grid points along Y (car width)
    nz : grid points along Z (car height at cabin)
    seed : RNG seed (not used for deterministic grid; kept for API compat)

    Returns
    -------
    nodes   : (N,3) float64
    springs : (M,2) int32
    k_arr   : (M,) float64
    rest_L  : (M,) float64
    zone_node: (N,) int32  (0/1/2)
    masses  : (N,) float64 (uniform 1.0 kg placeholder)
    vel0    : (N,3) float64 — initial velocities (impact on front)
    """
    # ── Node positions ───────────────────────────────────────────────────
    nodes = []
    zone_node = []

    x_vals = np.linspace(0.0, CAR_LENGTH, nx)
    y_vals = np.linspace(-CAR_WIDTH/2, CAR_WIDTH/2, ny)

    for ix, xv in enumerate(x_vals):
        x_norm   = xv / CAR_LENGTH
        z_top    = CAR_HEIGHT * _roof_height(x_norm)
        z_vals   = np.linspace(0.0, z_top, nz)
        z_norms  = (nz - 1) if nz > 1 else 1
        for iy, yv in enumerate(y_vals):
            for iz, zv in enumerate(z_vals):
                nodes.append([xv, yv, zv])
                zone_node.append(_zone(x_norm))

    nodes     = np.array(nodes,     dtype=np.float64)
    zone_node = np.array(zone_node, dtype=np.int32)
    N         = len(nodes)

    # Node index helper: (ix, iy, iz) → flat index
    def nid(ix, iy, iz):
        return ix * (ny * nz) + iy * nz + iz

    # ── Springs ──────────────────────────────────────────────────────────
    sp_i, sp_j = [], []

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                n0 = nid(ix, iy, iz)
                # Along X
                if ix + 1 < nx:
                    sp_i.append(n0); sp_j.append(nid(ix+1, iy, iz))
                # Along Y
                if iy + 1 < ny:
                    sp_i.append(n0); sp_j.append(nid(ix, iy+1, iz))
                # Along Z
                if iz + 1 < nz:
                    sp_i.append(n0); sp_j.append(nid(ix, iy, iz+1))
                # Face diagonals (XY plane)
                if ix + 1 < nx and iy + 1 < ny:
                    sp_i.append(n0); sp_j.append(nid(ix+1, iy+1, iz))
                # Face diagonals (XZ plane)
                if ix + 1 < nx and iz + 1 < nz:
                    sp_i.append(n0); sp_j.append(nid(ix+1, iy, iz+1))

    sp_i = np.array(sp_i, dtype=np.int32)
    sp_j = np.array(sp_j, dtype=np.int32)
    springs = np.column_stack([sp_i, sp_j])
    M = len(springs)

    # ── Spring stiffness: average of endpoint zones ───────────────────
    zone_k = np.array([K_SOFT, K_STIFF, K_SOFT], dtype=np.float64)
    k_i    = zone_k[zone_node[sp_i]]
    k_j    = zone_k[zone_node[sp_j]]
    k_arr  = 0.5 * (k_i + k_j)   # harmonic-mean-like average

    # ── Rest lengths: Euclidean distance in initial config ───────────
    diff   = nodes[sp_j] - nodes[sp_i]
    rest_L = np.linalg.norm(diff, axis=1)

    # ── Masses (uniform placeholder, can be scaled to vehicle mass) ──
    masses = np.ones(N, dtype=np.float64)

    # ── Initial velocity: front crumple zone hits barrier ────────────
    # Front 20% of nodes get impact velocity in -X
    vel0 = np.zeros((N, 3), dtype=np.float64)
    front_mask = (zone_node == 0)
    vel0[front_mask, 0] = IMPACT_VELOCITY

    return nodes, springs, k_arr, rest_L, zone_node, masses, vel0


def mesh_summary(nodes, springs, zone_node):
    N, M = len(nodes), len(springs)
    zone_counts = np.bincount(zone_node, minlength=3)
    print(f"Car mesh: {N:,} nodes | {M:,} springs")
    print(f"  Zone 0 (front crumple): {zone_counts[0]:,} nodes")
    print(f"  Zone 1 (cabin):         {zone_counts[1]:,} nodes")
    print(f"  Zone 2 (rear crumple):  {zone_counts[2]:,} nodes")
    print(f"  Extent: X={nodes[:,0].min():.2f}–{nodes[:,0].max():.2f}m "
          f"Y={nodes[:,1].min():.2f}–{nodes[:,1].max():.2f}m "
          f"Z={nodes[:,2].min():.2f}–{nodes[:,2].max():.2f}m")


if __name__ == "__main__":
    # Quick smoke test
    nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = build_car_mesh(nx=20, ny=6, nz=5)
    mesh_summary(nodes, springs, zone_node)
    nodes2, springs2, k_arr2, rest_L2, zone_node2, masses2, vel02 = build_car_mesh(nx=50, ny=12, nz=10)
    mesh_summary(nodes2, springs2, zone_node2)

