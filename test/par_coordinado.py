import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# --- Configuración General de Matplotlib (Opcional, para estilo) ---
plt.style.use('seaborn-v0_8-paper') # Estilo académico
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.family'] = 'serif'

# --- Cálculo Tiempo Operación (Función Auxiliar) ---
def calcular_tiempo_operacion_iec(I, Is, TDS, k=0.14, alpha=0.02, L=0):
    """
    Calcula el tiempo de operación usando la fórmula IEC Standard Inverse (SI).
    Retorna el tiempo en segundos o np.inf si I <= Is.
    """
    # Manejo de I como array o escalar
    if isinstance(I, (np.ndarray)):
        tiempo = np.full(I.shape, np.inf, dtype=float)
        if Is <= 0: return tiempo
        valid_I_mask = I > 0
        ratio = np.full(I.shape, -np.inf, dtype=float)
        ratio[valid_I_mask] = I[valid_I_mask] / Is
        calc_mask = (ratio > 1)
        if np.any(calc_mask):
            denominador = np.power(ratio[calc_mask], alpha) - 1
            # Evitar división por cero o valores muy pequeños que causen overflow en 1/denominador
            denominador_safe = np.where(denominador < 1e-9, 1e-9, denominador)
            # Calcular tiempo solo para la máscara válida
            tiempo[calc_mask] = TDS * ((k / denominador_safe) + L)
        return tiempo
    else: # Si I es un escalar
        if Is <= 0 or I <= Is:
            return np.inf
        ratio = I / Is
        denominador = ratio**alpha - 1
        if denominador < 1e-9:
            return np.inf # O un valor muy grande si se prefiere indicar operación casi instantánea
        return TDS * ((k / denominador) + L)

# --- Figura TCC: Gráfico con Diferencia de Tiempo (sin línea CTI) ---
# *** LA FUNCIÓN SE MANTIENE IGUAL, SOLO CAMBIAN LOS DATOS DE ENTRADA ***
# *** YA USA 'P' PARA PICKUP EN LAS ETIQUETAS ***
def graficar_tcc_con_diferencia(datos_par, archivo_salida='fig_tcc_con_diferencia.png'):
    """
    Genera un gráfico TCC completo (curvas, puntos op, Ishc) y muestra
    la diferencia de tiempo entre la curva de respaldo y el punto de
    operación principal en la Ishc principal. No muestra la línea CTI.
    Las etiquetas usan 'P' para el pickup.

    Args:
        datos_par (dict): Diccionario con info completa del par.
        archivo_salida (str): Nombre del archivo de salida.
    """
    # --- Extraer Parámetros Completos ---
    main = datos_par['main_relay']
    backup = datos_par['backup_relay']
    fault_desc = datos_par.get('fault', 'N/A')

    # Principal
    relay_m = main['relay']
    line_m = main['line']
    tds_m = main['TDS']
    pickup_m = main['pick_up'] # P (Pickup)
    ishc_m = main['Ishc']
    t_op_m = main['Time_out']

    # Respaldo
    relay_b = backup['relay']
    line_b = backup['line']
    tds_b = backup['TDS']
    pickup_b = backup['pick_up'] # P (Pickup)
    ishc_b = backup['Ishc']
    t_op_b = backup['Time_out']

    print(f"\n--- Generando TCC con Diferencia de Tiempo para {relay_m}/{relay_b} ---")
    print(f"  {relay_m} (Main):   P={pickup_m:.5f}, TDS={tds_m:.5f}, Ishc={ishc_m:.2f}, t_op={t_op_m:.4f}")
    print(f"  {relay_b} (Backup): P={pickup_b:.5f}, TDS={tds_b:.5f}, Ishc={ishc_b:.2f}, t_op={t_op_b:.4f}")

    # --- Generación de datos para las curvas ---
    corriente_min_pickup = min(pickup_m, pickup_b)
    if corriente_min_pickup <= 0:
      print(f"Error: Pickups deben ser positivos (P_main={pickup_m}, P_back={pickup_b}). No se puede generar gráfico.")
      return
    corriente_min_graf = corriente_min_pickup * 1.01
    corriente_max_graf = max(ishc_m, ishc_b, pickup_m, pickup_b) * 10
    corriente_max_graf = max(corriente_max_graf, ishc_m * 1.5)
    corriente_min_graf = max(corriente_min_graf, 0.01)
    # Aumentar número de puntos si las curvas son muy pronunciadas
    corrientes = np.logspace(np.log10(corriente_min_graf), np.log10(corriente_max_graf), 600)

    tiempos_m = calcular_tiempo_operacion_iec(corrientes, pickup_m, tds_m)
    tiempos_b = calcular_tiempo_operacion_iec(corrientes, pickup_b, tds_b)

    # --- Creación del Gráfico ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Graficar curvas TCC
    label_curva_m = f'Curva {relay_m} (Main, TDS={tds_m:.3f}, P={pickup_m:.3f}A)'
    label_curva_b = f'Curva {relay_b} (Backup, TDS={tds_b:.3f}, P={pickup_b:.3f}A)'
    ax.plot(corrientes, tiempos_m, label=label_curva_m, color='blue', linewidth=1.5)
    ax.plot(corrientes, tiempos_b, label=label_curva_b, color='red', linewidth=1.5)

    # 2. Graficar puntos de operación
    if np.isfinite(t_op_m) and np.isfinite(ishc_m):
        label_op_m = f'Op. {relay_m} ({t_op_m:.3f}s @ {ishc_m:.2f}A)'
        ax.plot(ishc_m, t_op_m, 'bo', markersize=7, label=label_op_m)
    if np.isfinite(t_op_b) and np.isfinite(ishc_b):
        label_op_b = f'Op. {relay_b} ({t_op_b:.3f}s @ {ishc_b:.2f}A)'
        ax.plot(ishc_b, t_op_b, 'ro', markersize=7, label=label_op_b)

    # 3. Graficar líneas de corriente de falla
    if np.isfinite(ishc_m):
        label_ishc_m = f'$I_{{shc}}$ {relay_m} = {ishc_m:.3f}A'
        ax.axvline(ishc_m, color='blue', linestyle='--', lw=1.5, label=label_ishc_m)
    if np.isfinite(ishc_b):
        label_ishc_b = f'$I_{{shc}}$ {relay_b} = {ishc_b:.3f}A'
        ax.axvline(ishc_b, color='red', linestyle='--', lw=1.5, label=label_ishc_b)

    # 4. Calcular y mostrar Diferencia de Tiempo en Ishc_m
    if np.isfinite(ishc_m) and np.isfinite(t_op_m):
        t_b_at_ishc_m = calcular_tiempo_operacion_iec(ishc_m, pickup_b, tds_b)

        if np.isfinite(t_b_at_ishc_m):
            time_diff = t_b_at_ishc_m - t_op_m
            print(f"  Tiempo Backup en Ishc_main ({ishc_m:.2f}A): {t_b_at_ishc_m:.4f}s")
            print(f"  Tiempo Main en Ishc_main ({ishc_m:.2f}A):   {t_op_m:.4f}s")
            print(f"  Diferencia de Tiempo (t_b - t_m): {time_diff:.4f}s")

            text_x = ishc_m * 1.05
            text_y = (t_op_m + t_b_at_ishc_m) / 2
            if ax.get_yscale() == 'log':
                 # Evitar log(0) o log(negativo)
                 log_y_m = np.log10(t_op_m) if t_op_m > 0 else -np.inf
                 log_y_b = np.log10(t_b_at_ishc_m) if t_b_at_ishc_m > 0 else -np.inf
                 # Solo ajustar si ambos son finitos y positivos
                 if np.isfinite(log_y_m) and np.isfinite(log_y_b) and abs(log_y_b - log_y_m) < 0.1:
                     factor_despl = 1.5 if t_b_at_ishc_m > t_op_m else 0.66
                     text_y = t_op_m * factor_despl

            diff_text = f"Dif = {time_diff:.3f}s"
            diff_color = 'green' if time_diff > 0 else 'magenta'
            ax.text(text_x, text_y, diff_text, color=diff_color, weight='bold',
                    fontsize=10, va='center', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec=diff_color))
        else:
            print(f"  No se pudo calcular t_backup en Ishc_main ({ishc_m:.2f}A). No se muestra diferencia.")

    # --- Configuración de ejes y formato ---
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Límites
    tiempos_visibles = []
    corrientes_visibles = []
    tiempos_m_validos = tiempos_m[np.isfinite(tiempos_m)]
    tiempos_b_validos = tiempos_b[np.isfinite(tiempos_b)]
    tiempos_visibles.extend(tiempos_m_validos)
    tiempos_visibles.extend(tiempos_b_validos)
    corrientes_visibles.extend(corrientes[np.isfinite(tiempos_m)])
    corrientes_visibles.extend(corrientes[np.isfinite(tiempos_b)])
    if np.isfinite(t_op_m): tiempos_visibles.append(t_op_m)
    if np.isfinite(ishc_m): corrientes_visibles.append(ishc_m)
    if np.isfinite(t_op_b): tiempos_visibles.append(t_op_b)
    if np.isfinite(ishc_b): corrientes_visibles.append(ishc_b)
    if 't_b_at_ishc_m' in locals() and np.isfinite(t_b_at_ishc_m):
         tiempos_visibles.append(t_b_at_ishc_m)

    corrientes_visibles = [c for c in corrientes_visibles if c > 0]
    tiempos_visibles = [t for t in tiempos_visibles if t > 0]

    if corrientes_visibles:
        min_c, max_c = min(corrientes_visibles), max(corrientes_visibles)
        # Añadir un poco más de margen si los valores son muy bajos
        margen_inf_c = 0.8 if min_c > 0.1 else 0.5
        margen_sup_c = 1.5 if max_c < 100 else 1.2
        ax.set_xlim(min_c * margen_inf_c, max_c * margen_sup_c)
    else:
        ax.set_xlim(0.01, 100)

    if tiempos_visibles:
        min_t, max_t = min(tiempos_visibles), max(tiempos_visibles)
        # Ajustar márgenes basados en el rango de tiempos
        margen_inf_t = 0.5 if min_t > 0.05 else 0.8
        margen_sup_t = 2.0 if max_t < 10 else 1.5
        y_min = min_t * margen_inf_t
        y_max = max_t * margen_sup_t
        y_min = max(y_min, 0.01)
        y_max = min(y_max, 1000)
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(0.01, 100)

    # Formato de los ejes
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)

    # Etiquetas y Título
    ax.set_xlabel('Corriente (A)')
    ax.set_ylabel('Tiempo (s)')
    titulo = f'Coordinación {relay_m} ({line_m}) vs {relay_b} ({line_b}) - Falla {fault_desc}%'
    ax.set_title(titulo)
    ax.legend(loc='best')
    ax.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
    print(f"Gráfico TCC con Diferencia de Tiempo guardado como: {archivo_salida}")
    plt.close(fig)


# === DEFINICIÓN DE DATOS NUEVOS ===
datos_par_R6_R62 = { # Nuevo par R6 / R62
    "scenario_id": "scenario_1",
    "fault": "10", # %
    "main_relay": {
      "relay": "R6",
      "pick_up": 0.06334, # P
      "Ishc": 2.64,
      "TDS": 0.05,
      "Time_out": 0.0904, # Tiempo principal muy bajo
      "line": "L6-7"
    },
    "backup_relay": {
      "relay": "R62",
      "line": "L6-26",
      "pick_up": 0.0485, # P
      "Ishc": 0.62,
      "TDS": 0.10844,
      "Time_out": 0.2904
    }
}


# === EJECUCIÓN PRINCIPAL ===
# Solo se genera la figura TCC solicitada con los nuevos datos
if __name__ == "__main__":
    print("=== Generando gráfico TCC específico con diferencia de tiempo (R6/R62) ===")

    # --- Generar Figura TCC para R6/R62 con Diferencia ---
    nombre_archivo_tcc_diff = f'fig_tcc_{datos_par_R6_R62["main_relay"]["relay"]}_{datos_par_R6_R62["backup_relay"]["relay"]}_f{datos_par_R6_R62["fault"]}_diff.png'
    graficar_tcc_con_diferencia(datos_par=datos_par_R6_R62,
                                archivo_salida=nombre_archivo_tcc_diff)

    print("\n=== ¡Gráfico TCC con diferencia de tiempo generado exitosamente! ===")
    print(f"Archivo guardado como: {nombre_archivo_tcc_diff}")