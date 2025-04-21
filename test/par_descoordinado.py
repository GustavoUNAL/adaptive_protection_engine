import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# --- Configuración General de Matplotlib (Opcional, para estilo) ---
# Se mantiene la configuración de estilo original
plt.style.use('seaborn-v0_8-paper') # Estilo académico
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.family'] = 'serif' # O 'sans-serif'

# --- Figura 1: Distribución Coordinación/Descoordinación ---
# (Se mantiene igual)
def graficar_distribucion_coordinacion(coordinados=15, descoordinados=85, archivo_salida='fig_distribucion_coordinacion.png'):
    """
    Genera un gráfico de pastel mostrando la distribución de pares
    coordinados vs. descoordinados.
    """
    etiquetas = ['Coordinados', 'Descoordinados']
    porcentajes = [coordinados, descoordinados]
    colores = ['#4CAF50', '#F44336'] # Verde para coordinados, Rojo para descoordinados
    explotar = (0, 0.1) # Resaltar la porción de descoordinados

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(porcentajes, explode=explotar, labels=etiquetas, colors=colores,
           autopct='%1.1f%%', shadow=True, startangle=90,
           pctdistance=0.85)

    # Círculo central para hacerlo un gráfico de dona (opcional)
    centro_circulo = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centro_circulo)

    ax.axis('equal') # Asegura que el gráfico sea circular
    plt.title('Distribución de Pares de Relés en Escenario Base', pad=20)
    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
    print(f"Gráfico de distribución guardado como: {archivo_salida}")
    plt.close(fig) # Cierra la figura para liberar memoria

# --- Cálculo Tiempo Operación (Función Auxiliar) ---
# (Se mantiene igual)
def calcular_tiempo_operacion_iec(I, Is, TDS, k=0.14, alpha=0.02, L=0):
    """
    Calcula el tiempo de operación usando la fórmula IEC Standard Inverse (SI).
    I: Corriente de falla
    Is: Corriente de arranque (Pickup)
    TDS: Time Dial Setting
    k, alpha, L: Constantes de la curva (SI por defecto)
    Retorna el tiempo en segundos o np.inf si I <= Is.
    """
    ratio = I / Is
    # Evita división por cero o logaritmo de no positivos y maneja I <= Is
    if isinstance(ratio, (np.ndarray)):
        tiempo = np.full_like(I, np.inf) # Inicializa con infinito
        mask = ratio > 1 # Solo calcula donde I > Is
        if np.any(mask):
            # Evita overflow en el denominador si ratio^alpha es muy cercano a 1
            denominador = np.power(ratio[mask], alpha) - 1
            # Evita división por cero si el denominador es extremadamente pequeño
            denominador_safe = np.where(denominador < 1e-9, 1e-9, denominador)
            tiempo[mask] = TDS * ((k / denominador_safe) + L)
        return tiempo
    else: # Si es un solo valor
        if ratio <= 1:
            return np.inf
        denominador = ratio**alpha - 1
        if denominador < 1e-9: # Evita división por cero
             return np.inf # O un valor muy grande si se prefiere
        return TDS * ((k / denominador) + L)

# --- Figura 2: Curvas Tiempo-Corriente (TCC) - Genérica ---
# (Se mantiene la función genérica adaptada previamente)
def graficar_tcc_relays(datos_par, cti=0.2, archivo_salida='fig_tcc_generada.png'):
    """
    Genera el gráfico TCC para un par de relés principal y de respaldo.
    Args:
        datos_par (dict): Diccionario con la información del par de relés.
                          Debe contener las claves 'main_relay' y 'backup_relay',
                          cada una con sub-diccionarios que incluyen 'relay',
                          'line', 'pick_up', 'Ishc', 'TDS', 'Time_out'.
                          También puede incluir 'fault' para el título.
        cti (float): Coordination Time Interval deseado.
        archivo_salida (str): Nombre del archivo donde se guardará la imagen.
    """
    # --- Extraer Parámetros del Diccionario ---
    main = datos_par['main_relay']
    backup = datos_par['backup_relay']
    fault_desc = datos_par.get('fault', 'N/A') # Obtener descripción de falla si existe

    # Relé Principal
    relay_m = main['relay']
    line_m = main['line']
    tds_m = main['TDS']
    pickup_m = main['pick_up']
    ishc_m = main['Ishc']
    t_op_m = main['Time_out']

    # Relé de Respaldo
    relay_b = backup['relay']
    line_b = backup['line']
    tds_b = backup['TDS']
    pickup_b = backup['pick_up']
    ishc_b = backup['Ishc']
    t_op_b = backup['Time_out']

    print(f"\n--- Graficando TCC para {relay_m}/{relay_b} (Falla: {fault_desc}%) ---")
    print(f"  {relay_m} (Main):   P={pickup_m:.5f}, TDS={tds_m:.5f}, Ishc={ishc_m:.2f}, t_op={t_op_m:.4f}")
    print(f"  {relay_b} (Backup): P={pickup_b:.5f}, TDS={tds_b:.5f}, Ishc={ishc_b:.2f}, t_op={t_op_b:.4f}")

    # --- Generación de datos para las curvas ---
    corriente_min_pickup = min(pickup_m, pickup_b)
    corriente_min_graf = corriente_min_pickup * 1.01
    corriente_max_graf = max(ishc_m, ishc_b) * 10
    corriente_min_graf = max(corriente_min_graf, 0.01)

    corrientes = np.logspace(np.log10(corriente_min_graf), np.log10(corriente_max_graf), 500)

    tiempos_m = calcular_tiempo_operacion_iec(corrientes, pickup_m, tds_m)
    tiempos_b = calcular_tiempo_operacion_iec(corrientes, pickup_b, tds_b)

    # --- Creación del Gráfico ---
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(corrientes, tiempos_m, label=f'Curva {relay_m} ({line_m}, TDS={tds_m:.3f}, P={pickup_m:.3f}A)', color='blue')
    ax.plot(corrientes, tiempos_b, label=f'Curva {relay_b} ({line_b}, TDS={tds_b:.3f}, P={pickup_b:.3f}A)', color='red')

    if np.isfinite(t_op_m) and np.isfinite(ishc_m):
        ax.plot(ishc_m, t_op_m, 'bo', markersize=8, label=f'Op. {relay_m} ({t_op_m:.3f}s @ {ishc_m:.2f}A)')
    if np.isfinite(t_op_b) and np.isfinite(ishc_b):
        ax.plot(ishc_b, t_op_b, 'ro', markersize=8, label=f'Op. {relay_b} ({t_op_b:.3f}s @ {ishc_b:.2f}A)')

    if np.isfinite(ishc_m):
        ax.axvline(ishc_m, color='blue', linestyle='--', lw=1.5, label=f'$I_{{shc}}$ {relay_m} = {ishc_m:.2f}A')
    if np.isfinite(ishc_b):
        ax.axvline(ishc_b, color='red', linestyle='--', lw=1.5, label=f'$I_{{shc}}$ {relay_b} = {ishc_b:.2f}A')

    # Graficar CTI en la corriente de falla del relé principal (ishc_m)
    if np.isfinite(ishc_m) and np.isfinite(t_op_m):
        t_b_at_ishc_m = calcular_tiempo_operacion_iec(ishc_m, pickup_b, tds_b)

        if t_b_at_ishc_m != np.inf:
            t_cti_end = t_op_m + cti
            ax.plot([ishc_m, ishc_m], [t_op_m, t_cti_end], 'g-', lw=2,
                    marker='_', markersize=8, markeredgewidth=2,
                    label=f'CTI ({cti:.2f}s)')

            delta_t = t_b_at_ishc_m - t_op_m - cti
            # Determinar el color del texto de Delta t según coordinación
            color_dt = 'green' if delta_t >= 0 else 'magenta'
            coord_status = 'OK' if delta_t >= 0 else 'FALLA'
            text_y_pos = (t_op_m + t_cti_end) / 2 if t_b_at_ishc_m > t_cti_end else (t_op_m + t_b_at_ishc_m) / 2
            ax.text(ishc_m * 1.1, text_y_pos, f'Δt = {delta_t:.3f}s\n({coord_status})',
                    color=color_dt, va='center', fontsize=9, weight='bold' if delta_t < 0 else 'normal')
            print(f"  CTI check at I={ishc_m:.2f}A: T_backup={t_b_at_ishc_m:.4f}s, T_main={t_op_m:.4f}s -> Δt={delta_t:.4f}s (Req: CTI={cti:.2f}s) -> {coord_status}")
        else:
            print(f"  Advertencia: No se pudo calcular el tiempo del relé de respaldo {relay_b} a I={ishc_m:.2f}A para verificar CTI.")

    # Configuración de ejes logarítmicos
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Límites de los ejes
    ax.set_xlim(corriente_min_graf * 0.9, corriente_max_graf)
    tiempos_m_validos = tiempos_m[np.isfinite(tiempos_m)]
    tiempos_b_validos = tiempos_b[np.isfinite(tiempos_b)]
    operacion_validos = [t for t in [t_op_m, t_op_b, t_b_at_ishc_m if 't_b_at_ishc_m' in locals() else np.inf] if np.isfinite(t)] # Añadir t_b_at_ishc_m si existe
    todos_tiempos_validos = np.concatenate([tiempos_m_validos, tiempos_b_validos, operacion_validos])

    if len(todos_tiempos_validos) > 0:
        y_min = min(todos_tiempos_validos) * 0.5
        y_max = max(todos_tiempos_validos) * 5
        y_min = max(y_min, 0.01)
        y_max = min(y_max, 100)
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
    titulo = f'Coordinación {relay_m} ({line_m}) vs {relay_b} ({line_b})'
    if fault_desc != 'N/A':
        titulo += f' - Falla {fault_desc}%'
    ax.set_title(titulo)
    ax.legend(loc='best') # Cambiado a 'best' para intentar mejor ubicación
    ax.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
    print(f"Gráfico TCC guardado como: {archivo_salida}")
    plt.close(fig)


# --- Figura 3: Distribución de Márgenes de Coordinación (Δt) ---
# (Se mantiene la función que acepta datos como argumento)
def graficar_distribucion_delta_t(delta_t_coordinados, archivo_salida='fig_distribucion_delta_t.png'):
    """
    Genera un histograma y un boxplot de los márgenes de coordinación (Δt)
    para los pares coordinados.
    Args:
        delta_t_coordinados (np.array): Array de numpy con los valores de Delta t.
        archivo_salida (str): Nombre base para los archivos de salida.
    """
    if delta_t_coordinados is None or len(delta_t_coordinados) == 0:
        print("\n--- Advertencia: No se proporcionaron datos de Delta t para graficar la distribución. ---")
        return

    n_datos = len(delta_t_coordinados)
    print(f"\n--- Graficando distribución de {n_datos} valores de Delta t ---")

    # --- Opción 1: Histograma ---
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    n_bins = min(max(5, n_datos // 3), 15)
    ax_hist.hist(delta_t_coordinados, bins=n_bins, color='skyblue', edgecolor='black')
    ax_hist.set_xlabel('Margen de Coordinación Δt (s)')
    ax_hist.set_ylabel('Frecuencia (Nº de Pares)')
    ax_hist.set_title(f'Distribución de Márgenes Δt para Pares Coordinados (N={n_datos})')
    ax_hist.grid(axis='y', alpha=0.7)

    media_dt = np.mean(delta_t_coordinados)
    mediana_dt = np.median(delta_t_coordinados)
    ax_hist.axvline(media_dt, color='red', linestyle='dashed', linewidth=1, label=f'Media = {media_dt:.4f}s')
    ax_hist.axvline(mediana_dt, color='green', linestyle='dashed', linewidth=1, label=f'Mediana = {mediana_dt:.4f}s')
    ax_hist.legend()

    plt.tight_layout()
    hist_filename = archivo_salida.replace('.png', '_hist.png')
    plt.savefig(hist_filename, dpi=300, bbox_inches='tight')
    print(f"Histograma Delta t guardado como: {hist_filename}")
    plt.close(fig_hist)

    # --- Opción 2: Box Plot ---
    fig_box, ax_box = plt.subplots(figsize=(6, 6))
    bp = ax_box.boxplot(delta_t_coordinados, patch_artist=True, showmeans=True,
                        medianprops={'color': 'black', 'linewidth': 1.5},
                        meanprops={'marker': 'D', 'markersize': 8, 'markerfacecolor': 'red', 'markeredgecolor': 'red'})
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax_box.set_ylabel('Margen de Coordinación Δt (s)')
    ax_box.set_xticks([])
    ax_box.set_xticklabels([])
    ax_box.set_title(f'Distribución de Márgenes Δt (Box Plot, N={n_datos})')
    ax_box.grid(axis='y', alpha=0.7)

    q1 = np.percentile(delta_t_coordinados, 25)
    q3 = np.percentile(delta_t_coordinados, 75)
    min_val = np.min(delta_t_coordinados)
    max_val = np.max(delta_t_coordinados)
    stats_text = (f"N = {n_datos}\n"
                  f"Min = {min_val:.4f}\n"
                  f"Q1 = {q1:.4f}\n"
                  f"Mediana = {mediana_dt:.4f}\n"
                  f"Media = {media_dt:.4f}\n"
                  f"Q3 = {q3:.4f}\n"
                  f"Max = {max_val:.4f}")
    ax_box.text(0.95, 0.5, stats_text, transform=ax_box.transAxes,
                va='center', ha='right', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout()
    boxplot_filename = archivo_salida.replace('.png', '_boxplot.png')
    plt.savefig(boxplot_filename, dpi=300, bbox_inches='tight')
    print(f"Box Plot Delta t guardado como: {boxplot_filename}")
    plt.close(fig_box)


# === DEFINICIÓN DE DATOS PARA LOS PARES ===

# --- Datos para el par R2/R55 ---
datos_par_R2_R55 = {
    "scenario_id": "scenario_1",
    "fault": "10", # %
    "main_relay": {
      "relay": "R2",
      "pick_up": 0.20622,
      "Ishc": 9.88,
      "TDS": 0.14301,
      "Time_out": 0.2488,
      "line": "L2-3"
    },
    "backup_relay": {
      "relay": "R55",
      "line": "L2-19",
      "pick_up": 0.12524,
      "Ishc": 0.32,
      "TDS": 0.08529,
      "Time_out": 0.6305
    }
}

# --- Datos para el par R15/R14 ---
datos_par_R15_R14 = {
    "scenario_id": "scenario_1",
    "fault": "10", # %
    "main_relay": {
      "relay": "R15",
      "pick_up": 0.07078,
      "Ishc": 1.86,
      "TDS": 0.13815,
      "Time_out": 0.2863,
      "line": "L15-16"
    },
    "backup_relay": {
      "relay": "R14",
      "line": "L14-15",
      "pick_up": 0.02616,
      "Ishc": 0.72,
      "TDS": 0.05, # ¡Este TDS es bajo!
      "Time_out": 0.1021 # ¡Este tiempo es muy bajo para ser respaldo!
    }
}

# --- Datos de ejemplo para la distribución Delta t ---
# !!! IMPORTANTE: Reemplaza estos valores con los 15 Δt REALES de tu análisis !!!
delta_t_reales_coordinados = np.array([
    0.0809, 0.0828, 0.0034, 0.0401, 0.1713, 0.0760, 0.0084, 0.0034,
    0.0373, 0.0129, 0.055, 0.110, 0.025, 0.095, 0.060
])
# delta_t_reales_coordinados = None # Descomenta si no tienes los datos


# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    print("=== Generando gráficos para la tesis ===")

    # --- Generar Figura 1 (Distribución General) ---
    graficar_distribucion_coordinacion(coordinados=15, descoordinados=85, # Ajusta si es necesario
                                      archivo_salida='fig1_distribucion_coordinacion.png')

    # --- Generar Figura TCC para R2/R55 ---
    nombre_archivo_tcc_R2_R55 = f'fig_tcc_{datos_par_R2_R55["main_relay"]["relay"]}_{datos_par_R2_R55["backup_relay"]["relay"]}_f{datos_par_R2_R55["fault"]}.png'
    graficar_tcc_relays(datos_par=datos_par_R2_R55,
                        cti=0.2, # CTI estándar
                        archivo_salida=nombre_archivo_tcc_R2_R55)

    # --- Generar Figura TCC para R15/R14 ---
    nombre_archivo_tcc_R15_R14 = f'fig_tcc_{datos_par_R15_R14["main_relay"]["relay"]}_{datos_par_R15_R14["backup_relay"]["relay"]}_f{datos_par_R15_R14["fault"]}.png'
    graficar_tcc_relays(datos_par=datos_par_R15_R14,
                        cti=0.2, # Usar el mismo CTI u otro si aplica
                        archivo_salida=nombre_archivo_tcc_R15_R14)
    # Nota: Los datos para R15/R14 parecen indicar descoordinación (backup opera en 0.1021s, main en 0.2863s).
    # La gráfica mostrará esto visualmente y calculará el Delta t negativo.

    # --- Generar Figura 3 (Distribución de Márgenes Δt) ---
    graficar_distribucion_delta_t(delta_t_coordinados=delta_t_reales_coordinados,
                                   archivo_salida='fig3_distribucion_delta_t.png')

    print("\n=== ¡Gráficos generados exitosamente! ===")
    if delta_t_reales_coordinados is not None and len(delta_t_reales_coordinados) < 15:
         print("!!! RECORDATORIO: Actualiza 'delta_t_reales_coordinados' con todos tus valores reales !!!")