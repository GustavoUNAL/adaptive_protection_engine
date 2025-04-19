#!/usr/bin/env python3
"""
Genera archivos PNG para comparar el escenario base antes y después de la optimización:
• comp_chart.png        – Δt antes vs. después de optimizar
• comp_mt_chart.png     – MT antes vs. después de optimizar
• coord_pie.png         – Proporción de pares coordinados vs. miscoordinados
• dt_cumulative.png     – Distribución acumulada de Δt
• tmt_comparison.png    – Comparación del indicador global TMT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Rutas ---
BASE_PATH = "/Users/gustavo/Documents/Projects/TESIS_UNAL/ADAPTIVE_ALGORITHM/"
FILE_BEFORE = os.path.join(BASE_PATH, "data/processed/coordination_results_scenario_1.csv")
FILE_AFTER = os.path.join(BASE_PATH, "data/processed/coordination_results_scenario_1_optimized.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "scripts/scenario_base")
os.makedirs(OUTPUT_PATH, exist_ok=True)  # Crear directorio si no existe

# --- Cargar ---
df_b = pd.read_csv(FILE_BEFORE)
df_a = pd.read_csv(FILE_AFTER)

# --- Etiqueta única por par ---
for df in (df_b, df_a):
    df["pair"] = (
        df["Falla (%)"].astype(str) + "% "
        + df["Relé Principal"] + "/" + df["Relé Backup"]
    )

# --- Fusionar ---
merged = (
    df_b[["pair", "Δt", "MT"]].rename(columns={"Δt": "dt_before", "MT": "mt_before"})
    .merge(df_a[["pair", "Δt", "MT"]], on="pair")
    .rename(columns={"Δt": "dt_after", "MT": "mt_after"})
)

# --- Solo pares miscoordinados inicialmente (Δt<0) ---
mis = merged[merged["dt_before"] < 0].sort_values("dt_before")

# ---------- Gráfica Δt ----------
plt.figure(figsize=(10, 0.22 * len(mis)))
y = np.arange(len(mis))
plt.barh(y, mis["dt_before"], color="red", alpha=0.6, label="Δt sin optimizar")
plt.barh(y, mis["dt_after"], color="green", alpha=0.6, label="Δt optimizado")
plt.yticks(y, mis["pair"], fontsize=6)
plt.xlabel("Δt (s)")
plt.title("Mejora del margen de coordinación por par de relés")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "comp_chart.png"), dpi=300)
plt.close()

# ---------- Gráfica MT ----------
plt.figure(figsize=(10, 0.22 * len(mis)))
plt.barh(y, mis["mt_before"], color="orange", alpha=0.6, label="MT sin optimizar")
plt.barh(y, mis["mt_after"], color="blue", alpha=0.6, label="MT optimizado")
plt.yticks(y, mis["pair"], fontsize=6)
plt.xlabel("MT (s)")
plt.title("Cambio de MT por par de relés")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "comp_mt_chart.png"), dpi=300)
plt.close()

# ---------- Gráfica de torta: Proporción de pares coordinados ----------
coord_before = (df_b["Δt"] >= 0).sum()
miscoord_before = len(df_b) - coord_before
coord_after = (df_a["Δt"] >= 0).sum()
miscoord_after = len(df_a) - coord_after

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.pie([coord_before, miscoord_before], labels=["Coordinados", "Miscoordinados"],
        autopct="%1.1f%%", colors=["green", "red"], startangle=90)
ax1.set_title("Antes de optimizar (15% coordinados)")

ax2.pie([coord_after, miscoord_after], labels=["Coordinados", "Miscoordinados"],
        autopct="%1.1f%%", colors=["green", "red"], startangle=90)
ax2.set_title("Después de optimizar (92% coordinados)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "coord_pie.png"), dpi=300)
plt.close()

# ---------- Gráfica de distribución acumulada de Δt ----------
dt_before_sorted = np.sort(df_b["Δt"])
dt_after_sorted = np.sort(df_a["Δt"])
yvals = np.arange(1, len(dt_before_sorted) + 1) / len(dt_before_sorted)

plt.figure(figsize=(8, 6))
plt.plot(dt_before_sorted, yvals, label="Δt sin optimizar", color="red")
plt.plot(dt_after_sorted, yvals, label="Δt optimizado", color="green")
plt.axvline(0, color="black", linestyle="--", alpha=0.5)
plt.xlabel("Δt (s)")
plt.ylabel("Proporción acumulada")
plt.title("Distribución acumulada de márgenes de coordinación")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "dt_cumulative.png"), dpi=300)
plt.close()

# ---------- Gráfica de comparación TMT ----------
tmt_before = -10.067
tmt_after = -0.627
plt.figure(figsize=(6, 4))
plt.bar(["Sin optimizar", "Optimizado"], [tmt_before, tmt_after], color=["orange", "blue"])
plt.ylabel("TMT")
plt.title("Comparación del indicador global TMT")
for i, v in enumerate([tmt_before, tmt_after]):
    plt.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v < 0 else "top")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "tmt_comparison.png"), dpi=300)
plt.close()

print("✓ Imágenes creadas en", OUTPUT_PATH)
print("  - comp_chart.png")
print("  - comp_mt_chart.png")
print("  - coord_pie.png")
print("  - dt_cumulative.png")
print("  - tmt_comparison.png")