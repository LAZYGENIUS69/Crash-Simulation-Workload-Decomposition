"""
src/vehicle_skin.py — Procedural 2D Triangular Vehicle Surface Skin
===================================================================
Course: High Performance Computing (24ISE503) — Activity 01
Iteration 5 Automotive Domain Visual Layer

Generates a procedural 2D triangular surface manifold wrapping the structural
vehicle-front mesh:
  1. Front Bumper Face & Lower Valance
  2. Radiator Grille & Fascia
  3. Crowned Hood / Bonnet
  4. Left & Right Fenders / Wheel Well Flanks
  5. Windshield Cowl & A-Pillar Bulkhead Face

Skin Binding & Kinematics:
  - Binds skin vertices to structural FE nodes via KD-Tree nearest-neighbor search (K=4).
  - Inverse-distance weighting (IDW):
      x_skin(t) = sum_{k=1}^K w_k * x_node,k(t),  where w_k = (1/d_k) / sum(1/d_j)
  - Purely kinematic deformation with ZERO physical overhead and ZERO energy drift.
"""

import numpy as np
import scipy.spatial as sp_spatial
import struct, pathlib

class VehicleSkin:
    def __init__(self, structural_nodes_init, k_neighbors=4):
        """
        Initializes and binds the vehicle exterior skin to the initial structural mesh.
        
        Parameters
        ----------
        structural_nodes_init : np.ndarray, shape (N, 3)
            Initial coordinates of the finite element structural nodes.
        k_neighbors : int
            Number of nearest structural nodes used for inverse-distance interpolation.
        """
        self.k = k_neighbors
        self.structural_nodes_init = np.array(structural_nodes_init, dtype=np.float64)
        self.vertices_init, self.faces = self._build_skin_manifold()
        
        # Build KD-tree on initial structural coordinates
        self.tree = sp_spatial.KDTree(self.structural_nodes_init)
        dists, self.neighbor_indices = self.tree.query(self.vertices_init, k=self.k)
        
        # Guard against zero distance (exact coincidence)
        safe_dists = np.maximum(dists, 1e-5)
        raw_weights = 1.0 / safe_dists
        self.weights = raw_weights / np.sum(raw_weights, axis=1, keepdims=True)

    def deform(self, current_nodes):
        """
        Deforms skin vertices according to the current structural node coordinates
        via displacement-transfer cage binding:
            x_skin(t) = x_skin(0) + sum_k w_k * [x_node,k(t) - x_node,k(0)]
            
        Parameters
        ----------
        current_nodes : np.ndarray, shape (N, 3)
            Current positions of the structural finite element nodes.
            
        Returns
        -------
        deformed_vertices : np.ndarray, shape (S, 3)
            Interpolated surface positions of the skin.
        """
        # Nodal displacements: (N, 3)
        node_disp = current_nodes - self.structural_nodes_init
        # Displacements of k nearest neighbors: (S, k, 3)
        neighbor_disp = node_disp[self.neighbor_indices]
        # Weighted displacement transfer: (S, 3)
        skin_disp = np.sum(neighbor_disp * self.weights[:, :, None], axis=1)
        return self.vertices_init + skin_disp

    def interpolate_nodal_scalar(self, scalar_field):
        """
        Interpolates a per-node scalar field (e.g. plastic strain, thread ID) to skin vertices.
        """
        # Shape: (S, k)
        vals = scalar_field[self.neighbor_indices]
        # Weighted average
        return np.sum(vals * self.weights, axis=1)

    def _build_skin_manifold(self):
        """
        Constructs the procedural triangular surface mesh representing the car's exterior panels.
        """
        all_verts = []
        all_faces = []

        def add_regular_grid(x_grid, y_grid, z_func):
            base_idx = len(all_verts)
            nx = len(x_grid)
            ny = len(y_grid)
            for i, x in enumerate(x_grid):
                for j, y in enumerate(y_grid):
                    z = z_func(x, y)
                    all_verts.append([x, y, z])
            
            for i in range(nx - 1):
                for j in range(ny - 1):
                    v0 = base_idx + i * ny + j
                    v1 = base_idx + (i + 1) * ny + j
                    v2 = base_idx + (i + 1) * ny + (j + 1)
                    v3 = base_idx + i * ny + (j + 1)
                    # Two triangles per quad
                    all_faces.append([v0, v1, v2])
                    all_faces.append([v0, v2, v3])

        # 1. Front Bumper Curved Face (X ~ 0.05..0.18, Y in [-0.75, 0.75], Z in [0.30, 0.55])
        by = np.linspace(-0.75, 0.75, 17)
        bz = np.linspace(0.30, 0.55, 6)
        def bumper_x(z, y):
            # Curved aerofoil front profile
            return 0.06 + 0.08 * (y / 0.75)**2 + 0.02 * ((z - 0.42) / 0.12)**2
        
        base_b = len(all_verts)
        for z in bz:
            for y in by:
                all_verts.append([bumper_x(z, y), y, z])
        ny_b = len(by)
        for iz in range(len(bz) - 1):
            for iy in range(len(by) - 1):
                v0 = base_b + iz * ny_b + iy
                v1 = base_b + (iz + 1) * ny_b + iy
                v2 = base_b + (iz + 1) * ny_b + (iy + 1)
                v3 = base_b + iz * ny_b + (iy + 1)
                all_faces.append([v0, v1, v2])
                all_faces.append([v0, v2, v3])

        # 2. Hood Surface (X in [0.20, 1.45], Y in [-0.78, 0.78])
        hx = np.linspace(0.20, 1.45, 22)
        hy = np.linspace(-0.78, 0.78, 17)
        def hood_z(x, y):
            # Slanted hood rising towards cowl with subtle central power bulge
            z_baseline = 0.58 + (x - 0.20) * (0.92 - 0.58) / (1.45 - 0.20)
            crown = 0.04 * (1.0 - (y / 0.78)**2)
            return z_baseline + crown
        add_regular_grid(hx, hy, hood_z)

        # 3. Windshield & Cowl (X in [1.45, 1.82], Y in [-0.78, 0.78])
        wx = np.linspace(1.45, 1.82, 10)
        wy = np.linspace(-0.78, 0.78, 17)
        def windshield_z(x, y):
            return 0.92 + (x - 1.45) * (1.18 - 0.92) / (1.82 - 1.45)
        add_regular_grid(wx, wy, windshield_z)

        # 4. Left Fender / Wing Flank (Y = +0.78, X in [0.20, 1.55], Z in [0.35, 0.90])
        fx = np.linspace(0.20, 1.55, 18)
        fz = np.linspace(0.35, 0.88, 8)
        base_lf = len(all_verts)
        for z in fz:
            for x in fx:
                all_verts.append([x, 0.78, z])
        nx_f = len(fx)
        for iz in range(len(fz) - 1):
            for ix in range(len(fx) - 1):
                v0 = base_lf + iz * nx_f + ix
                v1 = base_lf + (iz + 1) * nx_f + ix
                v2 = base_lf + (iz + 1) * nx_f + (ix + 1)
                v3 = base_lf + iz * nx_f + (ix + 1)
                all_faces.append([v0, v1, v2])
                all_faces.append([v0, v2, v3])

        # 5. Right Fender / Wing Flank (Y = -0.78, X in [0.20, 1.55], Z in [0.35, 0.90])
        base_rf = len(all_verts)
        for z in fz:
            for x in fx:
                all_verts.append([x, -0.78, z])
        for iz in range(len(fz) - 1):
            for ix in range(len(fx) - 1):
                v0 = base_rf + iz * nx_f + ix
                v1 = base_rf + (iz + 1) * nx_f + ix
                v2 = base_rf + (iz + 1) * nx_f + (ix + 1)
                v3 = base_rf + iz * nx_f + (ix + 1)
                all_faces.append([v0, v1, v2])
                all_faces.append([v0, v2, v3])

        # 6. Lower Air Dam & Valance (X in [0.10, 0.25], Y in [-0.75, 0.75], Z in [0.18, 0.30])
        vx = np.linspace(0.10, 0.25, 5)
        vy = np.linspace(-0.75, 0.75, 17)
        def valance_z(x, y):
            return 0.18 + (x - 0.10) * 0.8
        add_regular_grid(vx, vy, valance_z)

        verts_arr = np.array(all_verts, dtype=np.float64)
        faces_arr = np.array(all_faces, dtype=np.int32)
        return verts_arr, faces_arr


def load_simulation_frames(filepath):
    """
    Parses binary frame trajectory dumped by the C++ OpenMP solver.
    
    Layout:
      Header:
        uint32 magic      (0x46524D53 'FRMS')
        uint32 num_nodes  (N)
        uint32 num_frames (F)
      Per Frame:
        uint32 step
        float32 time_s
        float32 pos[N * 3]
        float32 plastic_strain[N]
        uint8   thread_id[N]
        
    Returns
    -------
    frames : list of dict
        List of frame snapshots containing step, time_s, pos, plastic_strain, thread_id.
    """
    filepath = pathlib.Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Frame file not found: {filepath}")

    frames = []
    with open(filepath, "rb") as f:
        magic, num_nodes, num_frames = struct.unpack("<III", f.read(12))
        assert magic == 0x46524D53, f"Invalid frame magic: 0x{magic:X}"

        record_pos_bytes = num_nodes * 3 * 4
        record_strain_bytes = num_nodes * 4
        record_tid_bytes = num_nodes * 1

        for _ in range(num_frames):
            step, time_s = struct.unpack("<If", f.read(8))
            pos_data = np.frombuffer(f.read(record_pos_bytes), dtype=np.float32).reshape(num_nodes, 3)
            strain_data = np.frombuffer(f.read(record_strain_bytes), dtype=np.float32)
            tid_data = np.frombuffer(f.read(record_tid_bytes), dtype=np.uint8)

            frames.append({
                "step": step,
                "time_s": float(time_s),
                "pos": pos_data,
                "plastic_strain": strain_data,
                "thread_id": tid_data
            })

    return frames
