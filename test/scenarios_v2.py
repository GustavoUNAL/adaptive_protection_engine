import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# --- Configuración General de Matplotlib (Opcional, para estilo) ---
plt.style.use('seaborn-v0_8-paper') # Estilo académico
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.family'] = 'serif' # O 'sans-serif'

# --- Función Auxiliar: Cálculo Tiempo IEC ---
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
    if isinstance(ratio, (np.ndarray)):
        tiempo = np.full_like(I, np.inf) # Inicializa con infinito
        mask = ratio > 1 # Solo calcula donde I > Is
        if np.any(mask):
            denominador = np.power(ratio[mask], alpha) - 1
            denominador_safe = np.where(denominador < 1e-9, 1e-9, denominador)
            tiempo[mask] = TDS * ((k / denominador_safe) + L)
        return tiempo
    else: # Si es un solo valor
        if ratio <= 1:
            return np.inf
        denominador = ratio**alpha - 1
        if denominador < 1e-9: # Evita división por cero
             return np.inf
        return TDS * ((k / denominador) + L)

# --- NUEVA Figura: Curvas TCC Ejemplo Descoordinado R11/R10 ---
def graficar_tcc_r11_r10(archivo_salida='fig_tcc_r11_r10.png'):
    """
    Genera el gráfico TCC para el par R11/R10 usando datos ilustrativos
    que resultan en miscoordinación.
    !!! REEMPLAZAR DATOS ILUSTRATIVOS CON VALORES REALES !!!
    """
    # --- Parámetros Ilustrativos (REEMPLAZAR) ---
    # Relé Principal (R11)
    tds_m = 0.150     # %%% ILUSTRATIVO %%%
    pickup_m = 0.050  # %%% ILUSTRATIVO %%%
    ishc_m = 1.500    # %%% ILUSTRATIVO %%%
    # t_op_m calculado a partir de los ilustrativos anteriores
    t_op_m = calcular_tiempo_operacion_iec(ishc_m, pickup_m, tds_m) # Aprox 0.350s con SI

    # Relé de Respaldo (R10)
    tds_b = 0.200     # %%% ILUSTRATIVO %%%
    pickup_b = 0.060  # %%% ILUSTRATIVO %%%
    ishc_b = 1.200    # %%% ILUSTRATIVO %%%
     # t_op_b calculado a partir de los ilustrativos anteriores
    t_op_b = calcular_tiempo_operacion_iec(ishc_b, pickup_b, tds_b) # Aprox 0.480s con SI

    # CTI
    cti = 0.2

    print(f"Datos ILUSTRATIVOS R11/R10: t_m={t_op_m:.3f}s, t_b={t_op_b:.3f}s")

    # --- Generación de datos para las curvas ---
    corriente_min = min(pickup_m, pickup_b) * 1.01
    corriente_max = max(ishc_m, ishc_b) * 10
    corrientes = np.logspace(np.log10(corriente_min), np.log10(corriente_max), 500)
    tiempos_m = calcular_tiempo_operacion_iec(corrientes, pickup_m, tds_m)
    tiempos_b = calcular_tiempo_operacion_iec(corrientes, pickup_b, tds_b)

    # --- Creación del Gráfico ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Graficar curvas TCC
    ax.plot(corrientes, tiempos_m, label=f'Curva R11 (Main, TDS={tds_m:.3f}, P={pickup_m:.3f}A)', color='blue')
    ax.plot(corrientes, tiempos_b, label=f'Curva R10 (Backup, TDS={tds_b:.3f}, P={pickup_b:.3f}A)', color='red')

    # Graficar puntos de operación
    ax.plot(ishc_m, t_op_m, 'bo', markersize=8, label=f'Op. R11 ({t_op_m:.3f}s @ {ishc_m:.3f}A)')
    ax.plot(ishc_b, t_op_b, 'ro', markersize=8, label=f'Op. R10 ({t_op_b:.3f}s @ {ishc_b:.3f}A)')

    # Graficar líneas de corriente de falla (verticales)
    ax.axvline(ishc_m, color='blue', linestyle='--', lw=1.5, label=f'$I_{{shc}}$ R11 = {ishc_m:.3f}A')
    ax.axvline(ishc_b, color='red', linestyle='--', lw=1.5, label=f'$I_{{shc}}$ R10 = {ishc_b:.3f}A')

    # Graficar CTI y mostrar miscoordinación
    # Tiempo de la curva de backup a la corriente de falla del principal
    t_b_at_ishc_m = calcular_tiempo_operacion_iec(ishc_m, pickup_b, tds_b)
    if t_b_at_ishc_m != np.inf and t_op_m != np.inf:
        t_cti_requerido = t_op_m + cti
        # Línea que muestra dónde debería estar el backup para coordinar
        ax.plot([ishc_m, ishc_m], [t_op_m, t_cti_requerido], 'g--', lw=2,
                marker='_', markersize=8, markeredgewidth=2,
                label=f'Tiempo Req. Backup ({t_cti_requerido:.3f}s)')
        # Línea que muestra dónde está realmente el backup a esa corriente
        ax.plot(ishc_m, t_b_at_ishc_m, 'rx', markersize=10, markeredgewidth=2,
                 label=f'Tiempo Real Backup ({t_b_at_ishc_m:.3f}s)')

        # Mostrar Delta t negativo
        delta_t = t_b_at_ishc_m - t_op_m - cti
        ax.text(ishc_m * 1.1, (t_op_m + t_b_at_ishc_m) / 2, f'Δt ≈ {delta_t:.3f}s (Descoordinado)',
                color='magenta', weight='bold', va='center')

    # Configuración de ejes logarítmicos y formato
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)

    # Límites (ajustar si es necesario)
    ax.set_xlim(corriente_min * 0.9, corriente_max)
    tiempos_validos = np.concatenate([tiempos_m[tiempos_m != np.inf], tiempos_b[tiempos_b != np.inf], [t_op_m, t_op_b, t_b_at_ishc_m]])
    tiempos_validos = tiempos_validos[tiempos_validos != np.inf] # Filtrar infinitos residuales
    if len(tiempos_validos) > 0:
      y_min = min(tiempos_validos) * 0.5
      y_max = max(tiempos_validos) * 5
      y_min = max(y_min, 0.01)
      y_max = min(y_max, 100)
      ax.set_ylim(y_min, y_max)
    else:
      ax.set_ylim(0.01, 100)

    # Etiquetas y Título
    ax.set_xlabel('Corriente (A)')
    ax.set_ylabel('Tiempo (s)')
    ax.set_title('Descoordinación R11 (L11-12) vs R10 (L10-11) - Falla 10% (Ilustrativo)')
    ax.legend(loc='best') # 'best' intenta encontrar la mejor ubicación
    ax.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
    print(f"Gráfico TCC R11/R10 (Ilustrativo) guardado como: {archivo_salida}")
    plt.close(fig)


# --- NUEVA Figura: Índice de Coordinación Global (100 pares) ---
def graficar_indice_coordinacion(n_pares=100, n_coordinados=15, archivo_salida='fig_indice_coordinacion_100pares.png'):
    """
    Genera un gráfico de barras mostrando el índice de coordinación (Delta t)
    para todos los pares, usando datos DUMMY.
    !!! REEMPLAZAR DATOS DUMMY CON VALORES REALES !!!
    """
    n_descoordinados = n_pares - n_coordinados

    # --- Generación de Datos DUMMY (REEMPLAZAR) ---
    # 15 valores positivos (coordinados) - usando los 10 conocidos + 5 dummy
    delta_t_pos = np.array([
        0.0809, 0.0828, 0.0034, 0.0401, 0.1713, 0.0760, 0.0084, 0.0034,
        0.0373, 0.0129, 0.055, 0.110, 0.025, 0.095, 0.060
    ])
    # 85 valores negativos (descoordinados) - generados aleatoriamente
    # Asegurarse que sean negativos, con distinta magnitud
    np.random.seed(42) # Para reproducibilidad del ejemplo
    delta_t_neg = -np.abs(np.random.normal(loc=0.15, scale=0.1, size=n_descoordinados))
    # Ajustar para que no sean exactamente cero si eso ocurre
    delta_t_neg[delta_t_neg == 0] = -0.001

    # Combinar y ordenar
    delta_t_todos = np.concatenate((delta_t_pos, delta_t_neg))
    # Crear etiquetas dummy para los pares
    par_labels = [f'Par_{i+1}' for i in range(n_pares)]

    # Ordenar por Delta t (ascendente: peor a mejor)
    indices_ordenados = np.argsort(delta_t_todos)
    delta_t_ordenado = delta_t_todos[indices_ordenados]
    par_labels_ordenado = [par_labels[i] for i in indices_ordenados]

    # Identificar el peor par (el primero en la lista ordenada)
    peor_par_label = par_labels_ordenado[0] # %%% DATO REQUERIDO: Usar ID real %%%
    peor_par_dt = delta_t_ordenado[0]      # %%% DATO REQUERIDO: Usar Delta t real %%%

    print(f"Peor par (DUMMY): {peor_par_label} con Delta t = {peor_par_dt:.3f}s")

    # --- Creación del Gráfico de Barras ---
    fig, ax = plt.subplots(figsize=(14, 7))

    colores = ['red' if dt < 0 else 'green' for dt in delta_t_ordenado]
    barras = ax.bar(range(n_pares), delta_t_ordenado, color=colores)

    # Resaltar el peor par (opcional)
    barras[0].set_edgecolor('black')
    barras[0].set_linewidth(1.5)
    # Anotar el peor par
    ax.annotate(f'Peor Par ({peor_par_label})\nΔt={peor_par_dt:.3f}s',
                xy=(0, peor_par_dt), xytext=(5, peor_par_dt - 0.1), # Ajustar posición texto
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=9, color='black', ha='left')

    # Línea en CTI=0
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

    # Etiquetas y Título
    ax.set_xlabel('Pares de Relés (Ordenados por Δt)')
    ax.set_ylabel('Margen de Coordinación Δt (s)')
    ax.set_title('Índice de Coordinación (Δt) para los 100 Pares - Escenario Base')

    # Ajustar etiquetas del eje X (mostrar solo algunas para claridad)
    tick_interval = 10 # Mostrar etiqueta cada 10 pares
    ax.set_xticks(np.arange(0, n_pares, tick_interval))
    ax.set_xticklabels([par_labels_ordenado[i] for i in np.arange(0, n_pares, tick_interval)], rotation=45, ha='right', fontsize=8)
    # O simplemente ocultar etiquetas si son demasiadas
    # ax.set_xticks([])

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
    print(f"Gráfico Índice Global (Datos Dummy) guardado como: {archivo_salida}")
    plt.close(fig)


# --- Ejecución ---
if __name__ == "__main__":
    print("Generando gráficos adicionales para la tesis...")

    # Generar Figura TCC R11/R10 (con datos ilustrativos)
    graficar_tcc_r11_r10(archivo_salida='fig_tcc_r11_r10.png')

    # Generar Figura Índice Global (con datos dummy)
    graficar_indice_coordinacion(n_pares=100, n_coordinados=15,
                                archivo_salida='fig_indice_coordinacion_100pares.png')

    # Opcional: Regenerar las figuras anteriores si es necesario
    # from previous_code import graficar_distribucion_coordinacion, graficar_tcc_r38_r39, graficar_distribucion_delta_t
    # graficar_distribucion_coordinacion(...)
    # graficar_tcc_r38_r39(...)
    # graficar_distribucion_delta_t(...)


    print("¡Gráficos adicionales generados exitosamente!")
    print("!!! RECORDATORIO: Reemplaza los datos ilustrativos y dummy con tus valores reales !!!")

