"""
plot_assembly_results.py — Generate high-resolution scaling & assembly comparison figures
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib

RESULTS_DIR = pathlib.Path("results")
df = pd.read_csv(RESULTS_DIR / "assembly_sweep_raw.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)

meshes = [("10k", "10k Nodes (6,000 steps)"),
          ("100k", "100k Nodes (500 steps)"),
          ("500k", "500k Nodes (75 steps)")]

colors = {
    "Private Slabs": "#d62728",
    "Atomic Updates": "#ff7f0e",
    "Node Gather CSR": "#1f77b4",
    "Node Gather CSR (FP32)": "#2ca02c"
}
markers = {
    "Private Slabs": "s",
    "Atomic Updates": "^",
    "Node Gather CSR": "o",
    "Node Gather CSR (FP32)": "D"
}

for ax, (m_id, m_title) in zip(axes, meshes):
    sub = df[df["mesh"] == m_id]
    
    # Ideal linear scaling reference
    ax.plot([1, 2, 4, 8, 12], [1, 2, 4, 8, 12], 'k--', alpha=0.35, label='Ideal Linear')
    
    for mode in ["Private Slabs", "Atomic Updates", "Node Gather CSR", "Node Gather CSR (FP32)"]:
        m_data = sub[sub["mode_name"] == mode].sort_values("threads")
        if len(m_data) > 0:
            ax.plot(
                m_data["threads"], m_data["speedup_vs_gather1"],
                color=colors[mode], marker=markers[mode], lw=2, ms=6,
                label=mode
            )
            
    ax.set_title(m_title, fontsize=12, fontweight='bold')
    ax.set_xlabel("OpenMP Threads", fontsize=10)
    ax.set_ylabel("Speedup vs. 1-Thread Gather", fontsize=10)
    ax.set_xticks([1, 2, 4, 8, 12])
    ax.grid(True, linestyle=':', alpha=0.6)
    if m_id == "10k":
        ax.legend(fontsize=8, loc='upper left')

plt.suptitle("HPC Crash Solver: Assembly Strategy Scaling vs. Mesh Size & Precision", fontsize=14, fontweight='bold')
plt.tight_layout()
plot_path = RESULTS_DIR / "assembly_scaling_comparison.png"
plt.savefig(plot_path)
plt.close()
print(f"[+] Saved comparison plot: {plot_path}")

