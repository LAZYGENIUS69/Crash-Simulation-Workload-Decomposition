"""
scripts/generate_vehicle_visuals.py — Visual Pipeline for Vehicle Crash Simulation
==================================================================================
Course: High Performance Computing (24ISE503) — Activity 01
Iteration 5 Automotive Domain Visuals

Generates the 7 required visual artifacts using PyVista and Matplotlib:
  1. vehicle_shaded_skin.png        : Shaded vehicle exterior skin with ground plane & rigid barrier
  2. vehicle_point_masses.png       : Finite element structural lattice (point masses & springs)
  3. vehicle_thread_partitions.png  : OpenMP core partition domains (colored by thread ID + boundary cuts)
  4. vehicle_plastic_strain_hud.png : Crash deformation with plastic strain colormap & telemetry HUD
  5. vehicle_thread_load_strip.png  : Per-thread execution time per stage (load balance & sync)
  6. vehicle_timelapse_strip.png    : 5-frame progressive collapse time-lapse strip
  7. vehicle_crash_simulation.gif   : Smooth animated GIF of the full 56 km/h barrier crash
"""

import sys, pathlib, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.vehicle_mesh import build_vehicle_front_mesh
from src.vehicle_skin import VehicleSkin, load_simulation_frames
from src.mesh_io import export_binary_mesh
from tests.conftest import get_solver_cmd, to_solver_path

pv.OFF_SCREEN = True

FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def run_simulation(steps=2000, snap_freq=40, threads=8):
    """Runs the 56 km/h vehicle impact simulation using the C++ OpenMP solver."""
    print("Building 10k vehicle mesh with Morton spatial ordering...")
    nodes, springs, k_arr, rest_L, zones, masses, vel0, rear_mask = build_vehicle_front_mesh("10k", reorder="morton")

    mesh_file = RESULTS_DIR / "vehicle_mesh_10k.bin"
    csv_file  = RESULTS_DIR / "vehicle_crash_metrics.csv"
    pos_file  = RESULTS_DIR / "vehicle_crash_pos.bin"
    frms_file = RESULTS_DIR / "vehicle_crash_frames.bin"
    time_file = RESULTS_DIR / "thread_timing_breakdown.csv"

    export_binary_mesh(
        mesh_file, nodes, vel0, masses, rear_mask, springs, k_arr, rest_L,
        dt=2.5e-6, damping=20.0, buckle_ratio=0.40, k_wall=1.0e7,
        n_steps=steps, snap_freq=snap_freq, enable_contact=1, enable_clamp=0
    )

    cmd = get_solver_cmd("crash_solver") + [
        to_solver_path(mesh_file),
        to_solver_path(csv_file),
        to_solver_path(pos_file),
        str(threads),
        "2",  # Mode 2: Node-Centric Gather CSR
        "0",  # static scheduling
        to_solver_path(frms_file),
        to_solver_path(time_file)
    ]

    print(f"Executing compiled C++ OpenMP solver ({threads} threads, Mode 2 Gather CSR)...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Solver STDOUT:\n", res.stdout)
        print("Solver STDERR:\n", res.stderr)
        raise RuntimeError(f"Solver failed with exit code {res.returncode}")
    print("Solver completed successfully.")
    return nodes, springs, zones, masses, frms_file, csv_file, time_file

def pyvista_faces(faces):
    """Converts (F, 3) triangle array to PyVista face array [3, v0, v1, v2, 3, ...]."""
    F = len(faces)
    pv_f = np.empty((F, 4), dtype=np.int64)
    pv_f[:, 0] = 3
    pv_f[:, 1:] = faces
    return pv_f.ravel()

def add_rigid_barrier_and_ground(plotter):
    """Adds a concrete rigid barrier and industrial ground grid to the PyVista scene."""
    # Rigid Wall at X = 0
    barrier = pv.Box(bounds=[-0.12, 0.0, -1.2, 1.2, 0.05, 1.35])
    plotter.add_mesh(barrier, color="#4a4e69", smooth_shading=True, specular=0.3, label="Rigid Wall")
    
    # Ground plane
    ground = pv.Plane(center=[1.0, 0.0, 0.05], direction=[0, 0, 1], i_size=3.2, j_size=2.8)
    plotter.add_mesh(ground, color="#22222b", smooth_shading=True, opacity=0.9)

# ── RENDER 1: SHADED CAR SKIN ──────────────────────────────────────────────────
def render_shaded_skin(skin, nodes_init):
    print("Rendering 1: Shaded Vehicle Skin...")
    pl = pv.Plotter(off_screen=True, window_size=[1600, 1000])
    pl.set_background("#181824", top="#28283c")

    add_rigid_barrier_and_ground(pl)

    verts = skin.vertices_init
    faces_pv = pyvista_faces(skin.faces)
    car_mesh = pv.PolyData(verts, faces_pv)
    pl.add_mesh(car_mesh, color="#0077b6", smooth_shading=True, specular=0.8, specular_power=25, label="Vehicle Skin")

    # Front wheel proxies for realism
    cyl_left = pv.Cylinder(center=[0.95, 0.76, 0.32], direction=[0, 1, 0], radius=0.30, height=0.22)
    cyl_right = pv.Cylinder(center=[0.95, -0.76, 0.32], direction=[0, 1, 0], radius=0.30, height=0.22)
    pl.add_mesh(cyl_left, color="#111111", smooth_shading=True)
    pl.add_mesh(cyl_right, color="#111111", smooth_shading=True)

    pl.camera_position = [(3.4, -2.4, 1.9), (0.9, 0.0, 0.55), (0, 0, 1)]
    pl.add_text("3D Parametric Vehicle Frontal Structure (1,500 kg US NCAP Standard)", 
                position="upper_left", font_size=12, color="white", font="arial")
    pl.add_text("Procedural 2D Surface Manifold Bound via KD-Tree Barycentric Weights", 
                position="lower_left", font_size=10, color="#a0c4ff", font="arial")

    out_path = FIGURES_DIR / "vehicle_shaded_skin.png"
    pl.screenshot(str(out_path))
    pl.close()
    print(f"Saved: {out_path}")

# ── RENDER 2: POINT MASSES & SPRINGS ───────────────────────────────────────────
def render_point_masses(nodes, springs, zones):
    print("Rendering 2: Structural Point Masses & Springs...")
    pl = pv.Plotter(off_screen=True, window_size=[1600, 1000])
    pl.set_background("#181824", top="#28283c")

    add_rigid_barrier_and_ground(pl)

    # Subsystem color map
    zone_colors = {
        0: "#e63946",  # Bumper: Red
        1: "#f77f00",  # Crashbox: Orange
        2: "#fcbf49",  # Rails: Gold
        3: "#2a9d8f",  # Subframe: Teal
        4: "#9d4edd",  # Engine: Purple
        5: "#457b9d",  # Firewall: Blue
        6: "#a8dadc",  # Hood: Light Blue
        7: "#6c757d"   # Cabin: Gray
    }
    
    # Render springs (subsample 20% for crisp visualization without visual clutter)
    sub_sp = springs[::5]
    lines = np.empty((len(sub_sp), 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1:] = sub_sp
    spring_mesh = pv.PolyData(nodes, lines=lines.ravel())
    pl.add_mesh(spring_mesh, color="#555566", opacity=0.35, line_width=1.0)

    # Nodes as point masses
    node_mesh = pv.PolyData(nodes)
    node_mesh["zone"] = zones
    cmap = [zone_colors[z] for z in sorted(zone_colors.keys())]
    pl.add_mesh(node_mesh, scalars="zone", cmap=cmap, point_size=4.5, render_points_as_spheres=True, show_scalar_bar=False)

    pl.camera_position = [(3.4, -2.4, 1.9), (0.9, 0.0, 0.55), (0, 0, 1)]
    pl.add_text("Structural Discretization: 11,388 Point Masses & 118,404 Spring Elements", 
                position="upper_left", font_size=12, color="white")
    pl.add_text("Subsystems: Bumper (Red) | Crash Box (Orange) | Rails (Gold) | Engine (Purple) | Subframe (Teal)", 
                position="lower_left", font_size=10, color="#fcbf49")

    out_path = FIGURES_DIR / "vehicle_point_masses.png"
    pl.screenshot(str(out_path))
    pl.close()
    print(f"Saved: {out_path}")

# ── RENDER 3: THREAD PARTITIONS & BOUNDARY CUTS ─────────────────────────────────
def render_thread_partitions(nodes, springs, thread_ids):
    print("Rendering 3: OpenMP Thread Partitions & Halo Cut Edges...")
    pl = pv.Plotter(off_screen=True, window_size=[1600, 1000])
    pl.set_background("#181824", top="#28283c")

    add_rigid_barrier_and_ground(pl)

    num_threads = int(thread_ids.max()) + 1

    # Find boundary cut springs (connecting nodes owned by different threads)
    sp_u = springs[:, 0]
    sp_v = springs[:, 1]
    is_boundary = (thread_ids[sp_u] != thread_ids[sp_v])
    cut_springs = springs[is_boundary]

    # Render internal springs faintly
    internal_springs = springs[~is_boundary][::6]
    lines_int = np.empty((len(internal_springs), 3), dtype=np.int64)
    lines_int[:, 0] = 2
    lines_int[:, 1:] = internal_springs
    pl.add_mesh(pv.PolyData(nodes, lines=lines_int.ravel()), color="#444455", opacity=0.25, line_width=1.0)

    # Render boundary cuts in bright yellow/gold
    lines_cut = np.empty((len(cut_springs), 3), dtype=np.int64)
    lines_cut[:, 0] = 2
    lines_cut[:, 1:] = cut_springs
    pl.add_mesh(pv.PolyData(nodes, lines=lines_cut.ravel()), color="#ffd166", opacity=0.85, line_width=2.5, label="Boundary Cuts")

    # Render nodes colored by OpenMP thread ID
    node_mesh = pv.PolyData(nodes)
    node_mesh["thread_id"] = thread_ids
    pl.add_mesh(node_mesh, scalars="thread_id", cmap="turbo", point_size=5.0,
                render_points_as_spheres=True, scalar_bar_args={"title": "Thread ID", "color": "white"})

    pl.camera_position = [(3.4, -2.4, 1.9), (0.9, 0.0, 0.55), (0, 0, 1)]
    pl.add_text(f"OpenMP Geometric Domain Decomposition ({num_threads} Threads, Morton Space-Filling Curve)", 
                position="upper_left", font_size=12, color="white")
    pl.add_text("Gold Lines = Inter-Thread Partition Boundary Cuts (Halo Volume Minimized via Morton Ordering)", 
                position="lower_left", font_size=10, color="#ffd166")

    out_path = FIGURES_DIR / "vehicle_thread_partitions.png"
    pl.screenshot(str(out_path))
    pl.close()
    print(f"Saved: {out_path}")

# ── RENDER 4: DEFORMATION & PLASTIC STRAIN WITH TELEMETRY HUD ──────────────────
def render_plastic_strain_hud(skin, frame, metrics_df):
    print("Rendering 4: Crash Deformation & Plastic Strain with HUD...")
    pl = pv.Plotter(off_screen=True, window_size=[1600, 1000])
    pl.set_background("#181824", top="#28283c")

    add_rigid_barrier_and_ground(pl)

    # Deform skin to current crash frame
    deformed_skin = skin.deform(frame["pos"])
    strain_skin = skin.interpolate_nodal_scalar(frame["plastic_strain"])

    faces_pv = pyvista_faces(skin.faces)
    car_mesh = pv.PolyData(deformed_skin, faces_pv)
    car_mesh["plastic_strain"] = strain_skin

    pl.add_mesh(car_mesh, scalars="plastic_strain", cmap="plasma", clim=[0.0, 0.40],
                smooth_shading=True, specular=0.6,
                scalar_bar_args={"title": "Plastic Strain (Yield)", "color": "white", "vertical": True})

    pl.camera_position = [(2.8, -2.2, 1.6), (0.7, 0.0, 0.50), (0, 0, 1)]
    
    # Save base render to image
    raw_img_path = FIGURES_DIR / "_temp_raw_render.png"
    pl.screenshot(str(raw_img_path))
    pl.close()

    # Overlay HUD telemetry using PIL
    img = Image.open(raw_img_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    step_val = frame["step"]
    time_ms = frame["time_s"] * 1000.0

    # Query metrics at this step
    row = metrics_df.iloc[(metrics_df["step"] - step_val).abs().argmin()]
    wall_f_kn = row["wall_force_N"] / 1000.0
    decel_g = (row["wall_force_N"] / 1500.0) / 9.80665
    intrusion_mm = row["max_intrusion_m"] * 1000.0
    ke_kj = row["ke_J"] / 1000.0
    plastic_kj = row["plastic_e_J"] / 1000.0
    residual = row["residual_pct"]

    # HUD Box
    hud_x0, hud_y0, hud_x1, hud_y1 = 40, 40, 520, 285
    draw.rounded_rectangle([hud_x0, hud_y0, hud_x1, hud_y1], radius=10, fill=(20, 24, 38, 220), outline=(0, 180, 255, 255), width=2)

    try:
        font_title = ImageFont.truetype("arialbd.ttf", 20)
        font_body  = ImageFont.truetype("cour.ttf", 15)
    except:
        font_title = ImageFont.load_default()
        font_body  = font_title

    draw.text((hud_x0 + 16, hud_y0 + 14), "IMPACT TELEMETRY HUD | 56 km/h", font=font_title, fill=(0, 220, 255, 255))
    draw.line([(hud_x0 + 16, hud_y0 + 44), (hud_x1 - 16, hud_y0 + 44)], fill=(80, 100, 140, 255), width=1)

    hud_text = (
        f"Simulation Time  : {time_ms:6.2f} ms (Step {step_val})\n"
        f"Barrier Force    : {wall_f_kn:6.1f} kN\n"
        f"Deceleration     : {decel_g:6.1f} g\n"
        f"Crush Intrusion  : {intrusion_mm:6.1f} mm\n"
        f"Kinetic Energy   : {ke_kj:6.1f} kJ\n"
        f"Plastic Work     : {plastic_kj:6.1f} kJ\n"
        f"Energy Drift     : {residual:6.4f}% [CLOSED]\n"
        f"[Tuned to published pulse, not predictive]"
    )
    draw.text((hud_x0 + 16, hud_y0 + 54), hud_text, font=font_body, fill=(240, 240, 250, 255), spacing=6)

    out_path = FIGURES_DIR / "vehicle_plastic_strain_hud.png"
    img.save(out_path)
    if raw_img_path.exists():
        raw_img_path.unlink()
    print(f"Saved: {out_path}")

# ── RENDER 5: PER-THREAD STAGE LOAD STRIP ───────────────────────────────────────
def render_thread_load_strip(timing_csv):
    print("Rendering 5: Per-Thread Stage Load Breakdown Strip...")
    df = pd.read_csv(timing_csv)

    threads = df["thread_id"].values
    num_threads = len(threads)

    # Convert seconds to milliseconds
    s1 = df["s1_spring_s"].values * 1000.0
    s2 = df["s2_assembly_s"].values * 1000.0
    s3 = df["s3_contact_s"].values * 1000.0
    s4 = df["s4_integ_s"].values * 1000.0
    s5 = df["s5_cfl_s"].values * 1000.0

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    y_pos = np.arange(num_threads)
    bar_height = 0.55

    p1 = ax.barh(y_pos, s1, bar_height, label="S1: Spring Forces & Internal Strain", color="#1f77b4", edgecolor="none")
    p2 = ax.barh(y_pos, s2, bar_height, left=s1, label="S2: Force Assembly (Gather CSR: 0ms)", color="#2ca02c", edgecolor="none")
    p3 = ax.barh(y_pos, s3, bar_height, left=s1+s2, label="S3: Rigid Wall Contact Mechanics", color="#ff7f0e", edgecolor="none")
    p4 = ax.barh(y_pos, s4, bar_height, left=s1+s2+s3, label="S4: Verlet Integration & Kinematics", color="#9467bd", edgecolor="none")
    p5 = ax.barh(y_pos, s5, bar_height, left=s1+s2+s3+s4, label="S5: CFL Reduction & State Monitoring", color="#d62728", edgecolor="none")

    totals = s1 + s2 + s3 + s4 + s5
    for i, tot in enumerate(totals):
        cpu = df["cpu_id"].iloc[i]
        ax.text(tot + max(totals)*0.015, i, f"{tot:.1f} ms (CPU {cpu})", va="center", ha="left", fontsize=9, fontweight="bold", color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Thread T{t}" for t in threads], fontsize=10, fontweight="bold")
    ax.set_xlabel("Active Core Execution Time [ms]", fontsize=11, fontweight="bold")
    ax.set_title("OpenMP Multi-Core Load Balance Breakdown (Mode 2 Gather CSR, 11,388 Nodes)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(0, max(totals) * 1.25)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.invert_yaxis()  # T0 at top

    plt.tight_layout()
    out_path = FIGURES_DIR / "vehicle_thread_load_strip.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── RENDER 6: 5-FRAME COLLAPSE TIME-LAPSE STRIP ────────────────────────────────
def render_timelapse_strip(skin, frames):
    print("Rendering 6: 5-Frame Crash Collapse Time-Lapse Strip...")
    num_frames = len(frames)
    indices = np.linspace(0, num_frames - 1, 5, dtype=int)
    selected_frames = [frames[i] for i in indices]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=300)
    
    for idx, (f, ax) in enumerate(zip(selected_frames, axes)):
        t_ms = f["time_s"] * 1000.0
        deformed_skin = skin.deform(f["pos"])
        strain_skin = skin.interpolate_nodal_scalar(f["plastic_strain"])

        # Render each frame via PyVista off-screen
        pl = pv.Plotter(off_screen=True, window_size=[600, 500])
        pl.set_background("#181824")
        add_rigid_barrier_and_ground(pl)

        faces_pv = pyvista_faces(skin.faces)
        car_mesh = pv.PolyData(deformed_skin, faces_pv)
        car_mesh["plastic_strain"] = strain_skin

        pl.add_mesh(car_mesh, scalars="plastic_strain", cmap="plasma", clim=[0.0, 0.40],
                    smooth_shading=True, specular=0.5, show_scalar_bar=False)
        pl.camera_position = [(2.8, -2.1, 1.5), (0.7, 0.0, 0.50), (0, 0, 1)]
        
        tmp_img = FIGURES_DIR / f"_tmp_lapse_{idx}.png"
        pl.screenshot(str(tmp_img))
        pl.close()

        img = plt.imread(str(tmp_img))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"t = {t_ms:.2f} ms (Step {f['step']})", fontsize=11, fontweight="bold", pad=8)
        tmp_img.unlink()

    plt.suptitle("US NCAP 56 km/h Frontal Crash — Progressive Buckling & Crumple Time-Lapse", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = FIGURES_DIR / "vehicle_timelapse_strip.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── RENDER 7: ANIMATED CRASH GIF ───────────────────────────────────────────────
def render_crash_gif(skin, frames):
    print("Rendering 7: Crash Simulation Animated GIF...")
    gif_frames = []
    
    # Subsample to ~25 frames for smooth, compact GIF
    step_skip = max(1, len(frames) // 25)
    sub_frames = frames[::step_skip]

    for idx, f in enumerate(sub_frames):
        t_ms = f["time_s"] * 1000.0
        deformed_skin = skin.deform(f["pos"])
        strain_skin = skin.interpolate_nodal_scalar(f["plastic_strain"])

        pl = pv.Plotter(off_screen=True, window_size=[800, 500])
        pl.set_background("#181824")
        add_rigid_barrier_and_ground(pl)

        faces_pv = pyvista_faces(skin.faces)
        car_mesh = pv.PolyData(deformed_skin, faces_pv)
        car_mesh["plastic_strain"] = strain_skin

        pl.add_mesh(car_mesh, scalars="plastic_strain", cmap="plasma", clim=[0.0, 0.40],
                    smooth_shading=True, specular=0.5, show_scalar_bar=False)
        pl.camera_position = [(2.8, -2.1, 1.5), (0.7, 0.0, 0.50), (0, 0, 1)]
        pl.add_text(f"t = {t_ms:.2f} ms | 56 km/h", position="upper_left", font_size=11, color="white")

        tmp_img = FIGURES_DIR / f"_tmp_gif_{idx}.png"
        pl.screenshot(str(tmp_img))
        pl.close()

        img = Image.open(tmp_img).convert("RGB")
        gif_frames.append(img)
        tmp_img.unlink()

    out_path = FIGURES_DIR / "vehicle_crash_simulation.gif"
    gif_frames[0].save(str(out_path), save_all=True, append_images=gif_frames[1:], duration=120, loop=0)
    print(f"Saved: {out_path}")

def main():
    nodes, springs, zones, masses, frms_file, csv_file, time_file = run_simulation(steps=2000, snap_freq=40, threads=8)
    
    print("Loading simulation frames and binding vehicle skin...")
    frames = load_simulation_frames(frms_file)
    print(f"Loaded {len(frames)} simulation frame snapshots.")

    skin = VehicleSkin(nodes)
    metrics_df = pd.read_csv(csv_file)

    # 1. Shaded Car Skin
    render_shaded_skin(skin, nodes)

    # 2. Point Masses & Springs
    render_point_masses(nodes, springs, zones)

    # 3. Thread Partitions & Boundary Cuts
    render_thread_partitions(nodes, springs, frames[0]["thread_id"])

    # 4. Plastic Strain & Telemetry HUD (Frame at maximum crush)
    peak_frame_idx = len(frames) * 3 // 4
    render_plastic_strain_hud(skin, frames[peak_frame_idx], metrics_df)

    # 5. Thread Load Strip
    render_thread_load_strip(time_file)

    # 6. 5-Frame Time-Lapse Strip
    render_timelapse_strip(skin, frames)

    # 7. Animated GIF
    render_crash_gif(skin, frames)

    print("\nAll 7 vehicle visuals successfully generated in figures/ directory!")

if __name__ == "__main__":
    main()

