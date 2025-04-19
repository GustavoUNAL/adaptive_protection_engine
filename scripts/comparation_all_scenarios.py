#!/usr/bin/env python3
"""
make_all_figs.py
----------------
Crea las ocho figuras PNG necesarias para tu informe LaTeX.

Se apoya en dos grupos de archivos dentro de DATA_DIR:

(1) Globales (ya los tienes)
    • summary_scenarios.csv
    • summary_scenarios_optimized.csv
    • mt_distribution.csv
    • mt_distribution_optimized.csv

(2) Detallados por escenario (uno pre y uno post por cada n = 1…68)
    • coordination_results_scenario_<n>.csv
    • coordination_results_scenario_<n>_optimized.csv
      Deben incluir las columnas:
        "TDS (Main)", "TDS (Backup)",
        "Pickup (Main)", "Pickup (Backup)"

Genera siempre:
    descoord_vs_scenario.png
    tmt_vs_scenario.png
    mt_distribution_compare.png

Si los CSV detallados existen y contienen las columnas indicadas,
genera además:
    tds_distribution_all.png
    pickup_distribution_all.png
    tds_pickup_changes_68_scenarios.png
    max_tds_68_scenarios.png
    max_pickup_68_scenarios.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib, re, sys

# ------------------------------------------------------------------
DATA_DIR = pathlib.Path(
    "/Users/gustavo/Documents/Projects/TESIS_UNAL/ADAPTIVE_ALGORITHM/scripts/all_scenarios"
)
FILE_PAT = re.compile(r"coordination_results_scenario_(\d+)\.csv")
# ------------------------------------------------------------------


# ---------- I. FIGURAS GLOBALES (3 PNG) ----------
def make_global_figs():
    try:
        pre_sum = pd.read_csv(DATA_DIR / "summary_scenarios.csv")
        opt_sum = pd.read_csv(DATA_DIR / "summary_scenarios_optimized.csv")
        pre_mt  = pd.read_csv(DATA_DIR / "mt_distribution.csv",
                              header=None, names=["esc","par","abs_mt"])
        opt_mt  = pd.read_csv(DATA_DIR / "mt_distribution_optimized.csv",
                              header=None, names=["esc","par","abs_mt"])
    except FileNotFoundError as e:
        sys.exit(f"❌ Falta el archivo requerido: {e.filename}")

    # 1. Descoordinaciones por escenario
    plt.figure(figsize=(10,4))
    plt.plot(pre_sum["Escenario"], pre_sum["Descoordinados"], "o-", label="Pre‑opt.")
    plt.plot(opt_sum["Escenario"], opt_sum["Descoordinados"], "o-", label="Post‑opt.")
    plt.xticks(rotation=90)
    plt.xlabel("Escenario"); plt.ylabel("Pares descoordinados")
    plt.title("Descoordinaciones por escenario"); plt.legend(); plt.tight_layout()
    plt.savefig(DATA_DIR/"descoord_vs_scenario.png", dpi=300)
    plt.close()

    # 2. TMT por escenario
    plt.figure(figsize=(10,4))
    plt.plot(pre_sum["Escenario"], pre_sum["TMT (negativo)"], "o-", label="Pre‑opt.")
    plt.plot(opt_sum["Escenario"], opt_sum["TMT (negativo)"], "o-", label="Post‑opt.")
    plt.xticks(rotation=90)
    plt.xlabel("Escenario"); plt.ylabel("TMT")
    plt.title("TMT por escenario"); plt.legend(); plt.tight_layout()
    plt.savefig(DATA_DIR/"tmt_vs_scenario.png", dpi=300)
    plt.close()

    # 3. Distribución global de |mt|
    plt.figure(figsize=(8,4))
    plt.hist(pre_mt["abs_mt"], bins=60, alpha=.6, label="Pre‑opt.")
    plt.hist(opt_mt["abs_mt"], bins=60, alpha=.6, label="Post‑opt.")
    plt.xlabel("|mt| (s)"); plt.ylabel("Frecuencia")
    plt.title("Distribución global de |mt|"); plt.legend(); plt.tight_layout()
    plt.savefig(DATA_DIR/"mt_distribution_compare.png", dpi=300)
    plt.close()

    print("✓ Figuras globales generadas")


# ---------- II. FIGURAS TDS / PICKUP (5 PNG) ----------
def make_tds_pickup_figs():
    pre_list, post_list = [], []

    for csv_path in DATA_DIR.glob("coordination_results_scenario_*.csv"):
        if csv_path.name.endswith("_optimized.csv"):
            continue
        m = FILE_PAT.match(csv_path.name)
        if not m:
            continue
        idx = m.group(1)
        opt_path = csv_path.with_name(
            f"coordination_results_scenario_{idx}_optimized.csv"
        )
        if not opt_path.exists():
            print(f"⚠️ Falta optimizado del escenario {idx}; omitido.")
            continue

        pre_df  = pd.read_csv(csv_path)
        post_df = pd.read_csv(opt_path)

        req = {"TDS (Main)","TDS (Backup)","Pickup (Main)","Pickup (Backup)"}
        if not req.issubset(pre_df.columns) or not req.issubset(post_df.columns):
            print(f"⚠️ Escenario {idx} sin columnas TDS/Pickup; omitido.")
            continue

        pre_df["sc"]  = post_df["sc"] = int(idx)
        pre_list.append(pre_df);  post_list.append(post_df)

    if not pre_list:
        print("ℹ️ No se encontraron CSV detallados; "
              "no se generan figuras de TDS/Pickup.")
        return

    pre  = pd.concat(pre_list,  ignore_index=True)
    post = pd.concat(post_list, ignore_index=True)

    def stack(df, main, backup):
        return pd.concat([
            df[[main,"sc"]].rename(columns={main:"val"}),
            df[[backup,"sc"]].rename(columns={backup:"val"})
        ])

    # 4. Distribución TDS
    plt.figure(figsize=(8,4))
    plt.hist(stack(pre,"TDS (Main)","TDS (Backup)")["val"],  bins=60,
             alpha=.6, label="Pre‑opt.")
    plt.hist(stack(post,"TDS (Main)","TDS (Backup)")["val"], bins=60,
             alpha=.6, label="Post‑opt.")
    plt.xlabel("TDS"); plt.ylabel("Frecuencia")
    plt.title("Distribución global de TDS"); plt.legend(); plt.tight_layout()
    plt.savefig(DATA_DIR/"tds_distribution_all.png", dpi=300)
    plt.close()

    # 5. Distribución Pickup
    plt.figure(figsize=(8,4))
    plt.hist(stack(pre,"Pickup (Main)","Pickup (Backup)")["val"],  bins=60,
             alpha=.6, label="Pre‑opt.")
    plt.hist(stack(post,"Pickup (Main)","Pickup (Backup)")["val"], bins=60,
             alpha=.6, label="Post‑opt.")
    plt.xlabel("Pickup (p.u.)"); plt.ylabel("Frecuencia")
    plt.title("Distribución global de Pickup"); plt.legend(); plt.tight_layout()
    plt.savefig(DATA_DIR/"pickup_distribution_all.png", dpi=300)
    plt.close()

    # 6. Variación porcentual media
    tds_pct = 100 * (
        stack(post,"TDS (Main)","TDS (Backup)")["val"].mean() -
        stack(pre ,"TDS (Main)","TDS (Backup)")["val"].mean()
    ) / stack(pre ,"TDS (Main)","TDS (Backup)")["val"].mean()

    pkp_pct = 100 * (
        stack(post,"Pickup (Main)","Pickup (Backup)")["val"].mean() -
        stack(pre ,"Pickup (Main)","Pickup (Backup)")["val"].mean()
    ) / stack(pre ,"Pickup (Main)","Pickup (Backup)")["val"].mean()

    plt.figure(figsize=(5,4))
    plt.bar(["TDS","Pickup"], [tds_pct, pkp_pct],
            color=["steelblue","darkorange"])
    plt.ylabel("Cambio porcentual (%)")
    plt.title("Variación media tras la optimización")
    plt.tight_layout()
    plt.savefig(DATA_DIR/"tds_pickup_changes_68_scenarios.png", dpi=300)
    plt.close()

    # 7. TDS máximo por escenario
    def max_series(df, main, backup):
        return stack(df, main, backup).groupby("sc")["val"].max()

    plt.figure(figsize=(10,4))
    plt.plot(max_series(pre,"TDS (Main)","TDS (Backup)"),  "o-", label="Pre‑opt.")
    plt.plot(max_series(post,"TDS (Main)","TDS (Backup)"), "o-", label="Post‑opt.")
    plt.xlabel("Escenario"); plt.ylabel("TDS máx."); plt.legend()
    plt.title("TDS máximo por escenario"); plt.tight_layout()
    plt.savefig(DATA_DIR/"max_tds_68_scenarios.png", dpi=300)
    plt.close()

    # 8. Pickup máximo por escenario
    plt.figure(figsize=(10,4))
    plt.plot(max_series(pre,"Pickup (Main)","Pickup (Backup)"),  "o-", label="Pre‑opt.")
    plt.plot(max_series(post,"Pickup (Main)","Pickup (Backup)"), "o-", label="Post‑opt.")
    plt.xlabel("Escenario"); plt.ylabel("Pickup máx. (p.u.)"); plt.legend()
    plt.title("Pickup máximo por escenario"); plt.tight_layout()
    plt.savefig(DATA_DIR/"max_pickup_68_scenarios.png", dpi=300)
    plt.close()

    print("✓ Figuras de TDS/Pickup generadas")


# ------------------- MAIN -------------------
if __name__ == "__main__":
    make_global_figs()
    make_tds_pickup_figs()
    print("✓ Todas las figuras están en:", DATA_DIR)
