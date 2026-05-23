import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── 0. CONFIG ─────────────────────────────────────────────────────────────
CSV_FILE   = "agile_devops_simulation_dataset.csv"
OUTPUT_DIR = "."          # all outputs go here
N_BOOTSTRAP = 10_000      # Monte Carlo iterations
SEED        = 42

# Colours matching paper figures (blue = Agile-only, orange = Agile+DevOps)
C_AGILE  = "#5B9BD5"
C_DEVOPS = "#ED7D31"

# ── 1. LOAD DATASET ────────────────────────────────────────────────────────
print("=" * 60)
print("Loading dataset …")
df = pd.read_csv(CSV_FILE, index_col="Project_ID")
print(f"  Rows: {len(df)}   Columns: {len(df.columns)}")
print()

# ── 2. PAIRED METRIC DEFINITIONS ──────────────────────────────────────────
metrics = [
    {
        "name"    : "Release Frequency",
        "col_a"   : "Release_Frequency_Agile",
        "col_d"   : "Release_Frequency_DevOps",
        "unit"    : "releases / year",
        "fig_name": "fig_release_frequency.png",
        "lower_is_better": False,
    },
    {
        "name"    : "Defect Rate",
        "col_a"   : "Defect_Rate_Agile",
        "col_d"   : "Defect_Rate_DevOps",
        "unit"    : "defects / 1,000 LOC",
        "fig_name": "fig_defect_rate.png",
        "lower_is_better": True,
    },
    {
        "name"    : "Customer Satisfaction",
        "col_a"   : "Customer_Satisfaction_Agile",
        "col_d"   : "Customer_Satisfaction_DevOps",
        "unit"    : "%",
        "fig_name": "fig_customer_satisfaction.png",
        "lower_is_better": False,
    },
    {
        "name"    : "Deployment Success Rate",
        "col_a"   : "Deployment_Success_Agile",
        "col_d"   : "Deployment_Success_DevOps",
        "unit"    : "%",
        "fig_name": "fig_deployment_success.png",
        "lower_is_better": False,
    },
    {
        "name"    : "MTTR",
        "col_a"   : "MTTR_Agile",
        "col_d"   : "MTTR_DevOps",
        "unit"    : "hours",
        "fig_name": "fig_mttr.png",
        "lower_is_better": True,
    },
    {
        "name"    : "Development Cycle Time",
        "col_a"   : "Cycle_Time_Agile",
        "col_d"   : "Cycle_Time_DevOps",
        "unit"    : "days",
        "fig_name": "fig_cycle_time.png",
        "lower_is_better": True,
    },
]

# ── 3. MONTE CARLO BOOTSTRAP (N = 10,000) ─────────────────────────────────
print("Running Monte Carlo bootstrap (N = {:,}) …".format(N_BOOTSTRAP))
rng = np.random.default_rng(SEED)

results = []
stat_lines = []
stat_lines.append("STATISTICAL TEST RESULTS — Paired t-test on bootstrap means")
stat_lines.append("=" * 62)

for m in metrics:
    a_vals = df[m["col_a"]].values.astype(float)
    d_vals = m_d_vals = df[m["col_d"]].values.astype(float)

    # Bootstrap: resample with replacement N_BOOTSTRAP times
    boot_a = np.array([rng.choice(a_vals, size=len(a_vals), replace=True).mean()
                       for _ in range(N_BOOTSTRAP)])
    boot_d = np.array([rng.choice(d_vals, size=len(d_vals), replace=True).mean()
                       for _ in range(N_BOOTSTRAP)])

    mean_a = round(boot_a.mean())
    mean_d = round(boot_d.mean())

    # 95 % CI on the difference (DevOps − Agile)
    diff = boot_d - boot_a
    ci_lo, ci_hi = np.percentile(diff, [2.5, 97.5])

    # Paired t-test on bootstrap means
    t_stat, p_val = stats.ttest_rel(boot_a, boot_d)

    # Change label
    if m["name"] == "Release Frequency":
        change_val  = round(mean_d / mean_a, 1)
        change_str  = f"{change_val}x increase"
    elif m["lower_is_better"]:
        pct = round((mean_a - mean_d) / mean_a * 100, 1)
        change_str  = f"-{pct}% reduction"
        change_val  = pct
    else:
        pts = mean_d - mean_a
        change_str  = f"+{pts} percentage points"
        change_val  = pts

    # Format display values
    if m["unit"] == "releases / year":
        a_str = f"{mean_a} releases/year"
        d_str = f"{mean_d} releases/year"
    elif m["unit"] == "defects / 1,000 LOC":
        a_str = f"{mean_a} / 1,000 LOC"
        d_str = f"{mean_d} / 1,000 LOC"
    elif m["unit"] == "%":
        a_str = f"{mean_a}%"
        d_str = f"{mean_d}%"
    elif m["unit"] == "hours":
        a_str = f"{mean_a} hours"
        d_str = f"{mean_d} hours"
    else:
        a_str = f"{mean_a} days"
        d_str = f"{mean_d} days"

    results.append({
        "Performance Metric": m["name"],
        "Agile-Only"        : a_str,
        "Agile+DevOps"      : d_str,
        "Change"            : change_str,
        "95% CI (diff)"     : f"[{ci_lo:.2f}, {ci_hi:.2f}]",
        "p-value"           : f"< 0.001" if p_val < 0.001 else f"{p_val:.4f}",
        "_mean_a"           : mean_a,
        "_mean_d"           : mean_d,
    })

    sig = "*** (p < 0.001)" if p_val < 0.001 else f"p = {p_val:.4f}"
    stat_lines.append(
        f"\n{m['name']}\n"
        f"  Agile-Only mean : {boot_a.mean():.2f}  (SD {boot_a.std():.3f})\n"
        f"  Agile+DevOps mean: {boot_d.mean():.2f}  (SD {boot_d.std():.3f})\n"
        f"  t-statistic      : {t_stat:.3f}\n"
        f"  Significance     : {sig}\n"
        f"  Change           : {change_str}\n"
        f"  95% CI on diff   : [{ci_lo:.2f}, {ci_hi:.2f}]"
    )

print("  Done.\n")

# ── 4. SAVE STATS FILE ────────────────────────────────────────────────────
stats_path = os.path.join(OUTPUT_DIR, "simulation_stats.txt")
with open(stats_path, "w", encoding="utf-8") as f:
    f.write("\n".join(stat_lines))
print(f"Stats saved → {stats_path}")

# ── 5. QUANTITATIVE RESULTS TABLE ─────────────────────────────────────────
df_results = pd.DataFrame(results)[
    ["Performance Metric", "Agile-Only", "Agile+DevOps", "Change",
     "95% CI (diff)", "p-value"]
]
table_path = os.path.join(OUTPUT_DIR, "results_table.csv")
df_results.to_csv(table_path, index=False)
print(f"Quantitative table saved → {table_path}\n")

print("=" * 60)
print("TABLE 3 — Quantitative Simulation Results")
print("=" * 60)
print(df_results.to_string(index=False))
print()

# ── 6. QUALITATIVE COMPARISON TABLE ───────────────────────────────────────
qualitative = [
    ["Release Speed",         "High",               "Medium",   "Very High"],
    ["Quality Assurance",     "Medium",             "High",     "Very High"],
    ["Operational Reliability","Low",               "High",     "Very High"],
    ["Collaboration Scope",   "High (team-level)",  "Medium",   "High (cross-functional)"],
    ["Customer Satisfaction", "Medium",             "High",     "Very High"],
]
df_qual = pd.DataFrame(qualitative,
    columns=["Aspect", "Agile", "DevOps", "Agile+DevOps"])
qual_path = os.path.join(OUTPUT_DIR, "qualitative_table.csv")
df_qual.to_csv(qual_path, index=False)
print("=" * 60)
print("TABLE 4 — Qualitative Comparison")
print("=" * 60)
print(df_qual.to_string(index=False))
print(f"\nQualitative table saved → {qual_path}\n")

# ── 7. BAR CHARTS (one per metric) ────────────────────────────────────────
print("Generating bar charts …")

def save_bar(mean_a, mean_d, metric_name, unit, fig_name, lower_is_better):
    fig, ax = plt.subplots(figsize=(5, 4.2))

    bars = ax.bar(["Agile-Only", "Agile+DevOps"],
                  [mean_a, mean_d],
                  color=[C_AGILE, C_DEVOPS],
                  width=0.45, edgecolor="white", linewidth=1.2)

    # Value labels on bars
    for bar, val in zip(bars, [mean_a, mean_d]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(mean_a, mean_d) * 0.02,
                str(int(val)),
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title(metric_name, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(unit, fontsize=10)
    ax.set_ylim(0, max(mean_a, mean_d) * 1.22)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Improvement arrow annotation
    if lower_is_better:
        pct = round((mean_a - mean_d) / mean_a * 100, 1)
        label = f"DOWN {pct}% reduction"
        color = "#1a7a1a"
    elif metric_name == "Release Frequency":
        mult = round(mean_d / mean_a, 1)
        label = f"UP {mult}x increase"
        color = "#1a7a1a"
    else:
        pts = mean_d - mean_a
        label = f"UP +{pts} pts"
        color = "#1a7a1a"

    ax.text(0.98, 0.97, label, transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fig_name)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_name}")

for m, r in zip(metrics, results):
    save_bar(r["_mean_a"], r["_mean_d"],
             m["name"], m["unit"], m["fig_name"], m["lower_is_better"])

# ── 8. ADOPTION TRENDS LINE GRAPH ─────────────────────────────────────────
print("\nGenerating adoption trends graph …")
years  = list(range(2015, 2026))
agile  = [42, 52, 54, 58, 61, 65, 69, 72, 75, 78, 78]
devops = [12, 18, 24, 32, 39, 45, 53, 59, 67, 74, 74]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(years, agile,  marker="o", linewidth=2.2, label="Agile",
        color=C_AGILE,  markersize=6)
ax.plot(years, devops, marker="^", linewidth=2.2, label="DevOps",
        color=C_DEVOPS, linestyle="--", markersize=6)

ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Adoption (%)", fontsize=11)
ax.set_title("Global Adoption Trends: Agile vs DevOps (2015–2025)",
             fontsize=12, fontweight="bold")
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=35, fontsize=8)
ax.set_ylim(0, 95)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=11)

# Source note
ax.text(0.01, 0.02,
        "Sources: DORA State of DevOps Report [14]; Digital.ai State of Agile [15]",
        transform=ax.transAxes, fontsize=7, color="grey", va="bottom")

plt.tight_layout()
adop_path = os.path.join(OUTPUT_DIR, "adoption_trends.png")
plt.savefig(adop_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"  Saved: adoption_trends.png")

# ── 9. COMBINED SUMMARY BAR CHART (all 6 side-by-side) ───────────────────
print("\nGenerating combined overview chart …")
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()

for i, (m, r) in enumerate(zip(metrics, results)):
    ax = axes[i]
    bars = ax.bar(["Agile-Only", "Agile+DevOps"],
                  [r["_mean_a"], r["_mean_d"]],
                  color=[C_AGILE, C_DEVOPS], width=0.4,
                  edgecolor="white", linewidth=1)
    for bar, val in zip(bars, [r["_mean_a"], r["_mean_d"]]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(r["_mean_a"], r["_mean_d"]) * 0.02,
                str(int(val)), ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_title(m["name"], fontsize=10, fontweight="bold")
    ax.set_ylabel(m["unit"], fontsize=8)
    ax.set_ylim(0, max(r["_mean_a"], r["_mean_d"]) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

# Legend
patch_a = mpatches.Patch(color=C_AGILE,  label="Agile-Only")
patch_d = mpatches.Patch(color=C_DEVOPS, label="Agile+DevOps")
fig.legend(handles=[patch_a, patch_d], loc="lower center",
           ncol=2, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Agile-Only vs Agile+DevOps — All KPIs (N = 500 projects)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
combined_path = os.path.join(OUTPUT_DIR, "fig_combined_overview.png")
plt.savefig(combined_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"  Saved: fig_combined_overview.png")
