"""Plot comparison of roofline vs per-model linear baseline results."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
roof = pd.read_csv(os.path.join(SCRIPT_DIR, "roofline_baseline.csv"))
lin = pd.read_csv(os.path.join(SCRIPT_DIR, "per_model_linear_baseline.csv"))

roof = roof[roof["status"] == "ok"].copy()
lin = lin[lin["status"] == "ok"].copy()

# Build short labels: model-workload
def short_label(row):
    m = row["model"].replace("-v0-1", "").replace("-8x7b", "8x7b").replace("-hf", "")
    return f"{m}\n{row['workload']}"

roof["label"] = roof.apply(short_label, axis=1)
lin["label"] = lin.apply(short_label, axis=1)

# Merge on experiment for paired comparison
merged = roof.merge(lin, on="experiment", suffixes=("_roof", "_lin"))
merged["label"] = merged.apply(lambda r: short_label(r.rename({"model_roof": "model", "workload_roof": "workload"})), axis=1)
merged.sort_values("experiment", inplace=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Roofline vs Per-Model Linear Baseline Comparison", fontsize=16, fontweight="bold")

# ─── Plot 1: E2E Error (%) ───
ax = axes[0, 0]
x = np.arange(len(merged))
w = 0.35
bars1 = ax.bar(x - w/2, merged["e2e_error_roof"] * 100, w, label="Roofline", color="#e74c3c", alpha=0.85)
bars2 = ax.bar(x + w/2, merged["e2e_error_lin"] * 100, w, label="Per-Model Linear", color="#3498db", alpha=0.85)
ax.set_ylabel("E2E Error (%)")
ax.set_title("E2E Mean Error by Experiment")
ax.set_xticks(x)
ax.set_xticklabels(merged["label"], fontsize=7.5)
ax.set_yscale("log")
ax.axhline(y=10, color="green", linestyle="--", linewidth=1.5, label="10% target")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ─── Plot 2: GT vs BLIS Predicted E2E (scatter) ───
ax = axes[0, 1]
ax.scatter(merged["gt_e2e_ms_roof"], merged["blis_e2e_ms_roof"],
           marker="^", s=80, color="#e74c3c", alpha=0.8, label="Roofline", zorder=3)
ax.scatter(merged["gt_e2e_ms_lin"], merged["blis_e2e_ms_lin"],
           marker="o", s=80, color="#3498db", alpha=0.8, label="Per-Model Linear", zorder=3)
# Perfect prediction line
lims = [min(merged["gt_e2e_ms_roof"].min(), merged["gt_e2e_ms_lin"].min()) * 0.5,
        max(merged["blis_e2e_ms_roof"].max(), merged["blis_e2e_ms_lin"].max()) * 1.5]
ax.plot(lims, lims, "k--", alpha=0.4, label="Perfect prediction")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Ground Truth E2E (ms)")
ax.set_ylabel("BLIS Predicted E2E (ms)")
ax.set_title("Predicted vs Ground Truth E2E")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ─── Plot 3: ITL Error (%) ───
ax = axes[1, 0]
bars1 = ax.bar(x - w/2, merged["itl_error_roof"] * 100, w, label="Roofline", color="#e74c3c", alpha=0.85)
bars2 = ax.bar(x + w/2, merged["itl_error_lin"] * 100, w, label="Per-Model Linear", color="#3498db", alpha=0.85)
ax.set_ylabel("ITL Error (%)")
ax.set_title("ITL (Inter-Token Latency) Error by Experiment")
ax.set_xticks(x)
ax.set_xticklabels(merged["label"], fontsize=7.5)
ax.set_yscale("log")
ax.axhline(y=15, color="green", linestyle="--", linewidth=1.5, label="15% target")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ─── Plot 4: Summary comparison table as text ───
ax = axes[1, 1]
ax.axis("off")

roof_e2e = merged["e2e_error_roof"].values * 100
lin_e2e = merged["e2e_error_lin"].values * 100
roof_ttft = merged["ttft_error_roof"].values * 100
lin_ttft = merged["ttft_error_lin"].values * 100
roof_itl = merged["itl_error_roof"].values * 100
lin_itl = merged["itl_error_lin"].values * 100

table_data = [
    ["", "Roofline", "Per-Model Linear"],
    ["Experiments", str(len(roof)), str(len(lin))],
    ["Mean E2E Error", f"{roof_e2e.mean():.1f}%", f"{lin_e2e.mean():.1f}%"],
    ["Median E2E Error", f"{np.median(roof_e2e):.1f}%", f"{np.median(lin_e2e):.1f}%"],
    ["Mean TTFT Error", f"{roof_ttft.mean():.1f}%", f"{lin_ttft.mean():.1f}%"],
    ["Mean ITL Error", f"{roof_itl.mean():.1f}%", f"{lin_itl.mean():.1f}%"],
    ["E2E < 10%", f"{(roof_e2e < 10).sum()}/{len(roof_e2e)}", f"{(lin_e2e < 10).sum()}/{len(lin_e2e)}"],
    ["Best E2E", f"{roof_e2e.min():.1f}%", f"{lin_e2e.min():.1f}%"],
    ["Worst E2E", f"{roof_e2e.max():.1f}%", f"{lin_e2e.max():.1f}%"],
    ["Failure Mode", "FLOPs overestimate\n+ MFU=0 fallback", "NNLS degenerate\ncoefficients"],
]

table = ax.table(cellText=table_data, cellLoc="center", loc="center",
                 colWidths=[0.35, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)

# Style header row
for j in range(3):
    table[0, j].set_facecolor("#34495e")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, len(table_data)):
    color = "#ecf0f1" if i % 2 == 0 else "white"
    for j in range(3):
        table[i, j].set_facecolor(color)

ax.set_title("Summary Comparison", fontsize=12, fontweight="bold", pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = os.path.join(SCRIPT_DIR, "baseline_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()

# ─── Plot 5: Per-model breakdown (separate figure) ───
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("E2E Error by Model Group", fontsize=14, fontweight="bold")

for ax_idx, (df, title, color) in enumerate([
    (roof, "Roofline", "#e74c3c"),
    (merged, "Per-Model Linear", "#3498db"),
]):
    ax = axes2[ax_idx]
    if title == "Roofline":
        models = roof.groupby("model")["e2e_error"].apply(lambda s: s.values * 100)
        labels = []
        data = []
        for m, vals in sorted(models.items()):
            labels.append(m.replace("-v0-1", "").replace("-8x7b", "8x7b").replace("-hf", ""))
            data.append(vals)
    else:
        models = merged.groupby("model_lin")
        labels = []
        data = []
        for m, grp in sorted(models):
            labels.append(m.replace("-v0-1", "").replace("-8x7b", "8x7b").replace("-hf", ""))
            data.append(grp["e2e_error_lin"].values * 100)

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("E2E Error (%)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.axhline(y=10, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="10% target")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path2 = os.path.join(SCRIPT_DIR, "baseline_by_model.png")
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path2}")
plt.close()
