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

# --- Figura 1: Distribución Coordinación/Descoordinación ---
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

# --- Figura 2: Curvas Tiempo-Corriente (TCC) Ejemplo R38/R39 ---
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


def graficar_tcc_r38_r39(archivo_salida='fig_tcc_r38_r39.png'):
    """
    Genera el gráfico TCC para el par R38/R39 con los datos del ejemplo.
    """
    # --- Parámetros del Ejemplo ---
    # Relé Principal (R38)
    tds_m = 0.025
    pickup_m = 0.08573 # Is (Main)
    ishc_m = 0.980
    t_op_m = 0.070

    # Relé de Respaldo (R39)
    tds_b = 0.025
    pickup_b = 0.35316 # Is (Backup)
    ishc_b = 0.580
    t_op_b = 0.351

    # CTI
    cti = 0.2

    # --- Generación de datos para las curvas ---
    # Rango de corrientes para graficar (logarítmico)
    # Asegurarse que empieza por encima de ambos pickups y llega más allá de Ishc
    corriente_min = min(pickup_m, pickup_b) * 1.01
    corriente_max = max(ishc_m, ishc_b) * 10
    corrientes = np.logspace(np.log10(corriente_min), np.log10(corriente_max), 500)

    # Calcular tiempos para cada relé
    tiempos_m = calcular_tiempo_operacion_iec(corrientes, pickup_m, tds_m)
    tiempos_b = calcular_tiempo_operacion_iec(corrientes, pickup_b, tds_b)

    # --- Creación del Gráfico ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Graficar curvas TCC
    ax.plot(corrientes, tiempos_m, label=f'Curva R38 (Main, TDS={tds_m}, P={pickup_m:.3f}A)', color='blue')
    ax.plot(corrientes, tiempos_b, label=f'Curva R39 (Backup, TDS={tds_b}, P={pickup_b:.3f}A)', color='red')

    # Graficar puntos de operación
    ax.plot(ishc_m, t_op_m, 'bo', markersize=8, label=f'Op. R38 ({t_op_m:.3f}s @ {ishc_m:.3f}A)')
    ax.plot(ishc_b, t_op_b, 'ro', markersize=8, label=f'Op. R39 ({t_op_b:.3f}s @ {ishc_b:.3f}A)')

    # Graficar líneas de corriente de falla (verticales)
    ax.axvline(ishc_m, color='blue', linestyle='--', lw=1.5, label=f'$I_{{shc}}$ R38 = {ishc_m:.3f}A')
    ax.axvline(ishc_b, color='red', linestyle='--', lw=1.5, label=f'$I_{{shc}}$ R39 = {ishc_b:.3f}A')

    # Graficar CTI
    # Necesitamos el tiempo de la curva de backup a la corriente de falla del principal
    t_b_at_ishc_m = calcular_tiempo_operacion_iec(ishc_m, pickup_b, tds_b)
    if t_b_at_ishc_m != np.inf and t_op_m != np.inf:
        # Dibuja la línea CTI desde t_op_m hasta t_op_m + CTI (o hasta t_b_at_ishc_m si es menor)
        t_cti_end = t_op_m + cti
        # Asegurarse de no dibujar por encima del tiempo de backup a esa corriente
        # t_cti_end = min(t_cti_end, t_b_at_ishc_m) # Opcional: limitar al tiempo de backup
        ax.plot([ishc_m, ishc_m], [t_op_m, t_cti_end], 'g-', lw=2,
                marker='_', markersize=8, markeredgewidth=2,
                label=f'CTI ({cti:.2f}s)')

        # Mostrar Delta t (opcional, puede saturar el gráfico)
        delta_t = t_b_at_ishc_m - t_op_m - cti
        # ax.text(ishc_m * 1.1, (t_op_m + t_cti_end) / 2, f'Δt ≈ {delta_t:.3f}s', color='green', va='center')

    # Configuración de ejes logarítmicos
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Límites de los ejes (ajustar según sea necesario)
    ax.set_xlim(corriente_min * 0.9, corriente_max)
    # Encontrar límites Y razonables basados en los tiempos calculados y de operación
    tiempos_validos = np.concatenate([tiempos_m[tiempos_m != np.inf], tiempos_b[tiempos_b != np.inf], [t_op_m, t_op_b]])
    if len(tiempos_validos) > 0:
      y_min = min(tiempos_validos) * 0.5
      y_max = max(tiempos_validos) * 5
      # Asegurar límites mínimos/máximos razonables para log scale
      y_min = max(y_min, 0.01) # Límite inferior típico para TCC
      y_max = min(y_max, 100)  # Límite superior típico
      ax.set_ylim(y_min, y_max)
    else:
      ax.set_ylim(0.01, 100) # Valores por defecto si no hay datos válidos

    # Formato de los ejes para que muestren números normales en log
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    # Opcional: Forzar ticks específicos si es necesario
    # ax.xaxis.set_major_locator(mticker.LogLocator(numticks=10))
    # ax.yaxis.set_major_locator(mticker.LogLocator(numticks=10))


    # Etiquetas y Título
    ax.set_xlabel('Corriente (A)')
    ax.set_ylabel('Tiempo (s)')
    ax.set_title('Coordinación R38 (L1-2) vs R39 (L2-3) - Falla 90%')
    ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="-", alpha=0.5) # Rejilla logarítmica

    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
    print(f"Gráfico TCC guardado como: {archivo_salida}")
    plt.close(fig)

# --- Figura 3: Distribución de Márgenes de Coordinación (Δt) ---
def graficar_distribucion_delta_t(archivo_salida='fig_distribucion_delta_t.png'):
    """
    Genera un histograma o boxplot de los márgenes de coordinación (Δt)
    para los pares coordinados.
    """
    # !!! IMPORTANTE: Reemplaza estos valores con los 15 Δt REALES de tu análisis !!!
    # Estos son los 10 de la tabla + 5 valores inventados para el ejemplo
    delta_t_coordinados = np.array([
        0.0809, 0.0828, 0.0034, 0.0401, 0.1713, 0.0760, 0.0084, 0.0034,
        0.0373, 0.0129,
        # --- Valores inventados (REEMPLAZAR) ---
        0.055, 0.110, 0.025, 0.095, 0.060
    ])

    if len(delta_t_coordinados) != 15:
        print("Advertencia: Se esperaban 15 valores de Delta t, pero se proporcionaron", len(delta_t_coordinados))

    # --- Opción 1: Histograma ---
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    ax_hist.hist(delta_t_coordinados, bins=5, color='skyblue', edgecolor='black') # Ajustar 'bins' según necesidad
    ax_hist.set_xlabel('Margen de Coordinación Δt (s)')
    ax_hist.set_ylabel('Frecuencia (Nº de Pares)')
    ax_hist.set_title('Distribución de Márgenes Δt para Pares Coordinados')
    ax_hist.grid(axis='y', alpha=0.7)

    # Añadir líneas de media y mediana (opcional)
    media_dt = np.mean(delta_t_coordinados)
    mediana_dt = np.median(delta_t_coordinados)
    ax_hist.axvline(media_dt, color='red', linestyle='dashed', linewidth=1, label=f'Media = {media_dt:.3f}s')
    ax_hist.axvline(mediana_dt, color='green', linestyle='dashed', linewidth=1, label=f'Mediana = {mediana_dt:.3f}s')
    ax_hist.legend()

    plt.tight_layout()
    plt.savefig(archivo_salida.replace('.png', '_hist.png'), dpi=300, bbox_inches='tight')
    print(f"Histograma Delta t guardado como: {archivo_salida.replace('.png', '_hist.png')}")
    plt.close(fig_hist)

    # --- Opción 2: Box Plot ---
    fig_box, ax_box = plt.subplots(figsize=(6, 6))
    bp = ax_box.boxplot(delta_t_coordinados, patch_artist=True, showmeans=True,
                        medianprops={'color': 'black'},
                        meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markeredgecolor': 'red'})
    # Colorear la caja
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax_box.set_ylabel('Margen de Coordinación Δt (s)')
    # Ocultar etiqueta del eje X ya que es una sola variable
    ax_box.set_xticks([])
    ax_box.set_xticklabels([])
    ax_box.set_title('Distribución de Márgenes Δt (Box Plot)')
    ax_box.grid(axis='y', alpha=0.7)

    # Añadir texto con estadísticas clave (opcional)
    stats_text = (f"N = {len(delta_t_coordinados)}\n"
                  f"Min = {np.min(delta_t_coordinados):.3f}\n"
                  f"Q1 = {np.percentile(delta_t_coordinados, 25):.3f}\n"
                  f"Mediana = {mediana_dt:.3f}\n"
                  f"Media = {media_dt:.3f}\n"
                  f"Q3 = {np.percentile(delta_t_coordinados, 75):.3f}\n"
                  f"Max = {np.max(delta_t_coordinados):.3f}")
    # Posicionar el texto (ajustar coordenadas x, y)
    ax_box.text(1.1, mediana_dt, stats_text, va='center', ha='left', fontsize=9)


    plt.tight_layout()
    plt.savefig(archivo_salida.replace('.png', '_boxplot.png'), dpi=300, bbox_inches='tight')
    print(f"Box Plot Delta t guardado como: {archivo_salida.replace('.png', '_boxplot.png')}")
    plt.close(fig_box)


# --- Ejecución ---
if __name__ == "__main__":
    print("Generando gráficos para la tesis...")

    # Generar Figura 1
    graficar_distribucion_coordinacion(coordinados=15, descoordinados=85,
                                      archivo_salida='fig_distribucion_coordinacion.png')

    # Generar Figura 2
    graficar_tcc_r38_r39(archivo_salida='fig_tcc_r38_r39.png')

    # Generar Figura 3 (Histograma y BoxPlot)
    graficar_distribucion_delta_t(archivo_salida='fig_distribucion_delta_t.png')

    print("¡Gráficos generados exitosamente!")

