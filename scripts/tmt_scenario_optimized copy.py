import json
import os
import math
import copy
import traceback
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# --- Constantes (Coincidentes con tu primer script + Curva) ---
CTI = 0.2
TARGET_SCENARIO_ID = "scenario_1"  # Analizando scenario_1

# --- Ruta del archivo (Coincidente con tu primer script) ---
input_file = "/Users/gustavo/Documents/Projects/TESIS_UNAL/ADAPTIVE_ALGORITHM/data/processed/independent_relay_pairs.json"
output_dir = os.path.dirname(input_file)  # Directorio del archivo de entrada
output_csv = os.path.join(output_dir, f"coordination_results_{TARGET_SCENARIO_ID}_optimized.csv")  # Ruta del archivo CSV

# --- Constantes para la Curva de Tiempo Inverso (EJEMPLO: IEC Standard Inverse) ---
CURVE_A = 0.14
CURVE_B = 0.02
MIN_CURRENT_MULTIPLIER = 1.05
MAX_CURRENT_MULTIPLIER = 50
MIN_TIME_PLOT = 0.01
MAX_TIME_PLOT = 20

print(f"--- Iniciando Análisis (Script Original) y Visualización (CON Curvas Dinámicas + Validación Mejorada) ---")
print(f"Archivo de entrada: {input_file}")
print(f"Resultados se guardarán en: {output_csv}")
print(f"Analizando SOLAMENTE para: '{TARGET_SCENARIO_ID}'")
print(f"Usando curva IEC SI (A={CURVE_A}, B={CURVE_B}) para graficar.")

# --- Función para calcular la curva de tiempo inverso (Sin cambios) ---
def calculate_inverse_time_curve(tds, pickup, i_range):
    times = []
    if pickup <= 1e-6:
        return np.full_like(i_range, np.nan)
    for i in i_range:
        multiple = i / pickup
        if multiple <= 1.0:
            time = np.inf
        else:
            try:
                denominator = (multiple ** CURVE_B) - 1
                if denominator <= 1e-9:
                    time = np.inf
                else:
                    time = tds * (CURVE_A / denominator)
                if time < 0:
                    time = np.inf
            except (OverflowError, ValueError):
                time = np.inf
        times.append(time)
    return np.nan_to_num(np.array(times), nan=np.nan, posinf=np.nan, neginf=np.nan)

# --- Definición de summary_columns_map (Movido aquí para evitar NameError) ---
summary_columns_map = {
    "Falla (%)": "fault",
    "Línea Principal": "main_relay.line",
    "Relé Principal": "main_relay.relay",
    "TDS (Main)": "main_relay.TDS",
    "Pickup (Main)": "main_relay.pick_up",
    "I_shc (Main)": "main_relay.Ishc",
    "t_m (Main)": "main_relay.Time_out",
    "Línea Backup": "backup_relay.line",
    "Relé Backup": "backup_relay.relay",
    "TDS (Backup)": "backup_relay.TDS",
    "Pickup (Backup)": "backup_relay.pick_up",
    "I_shc (Backup)": "backup_relay.Ishc",
    "t_b (Backup)": "backup_relay.Time_out",
    "Δt": "delta_t",
    "MT": "mt"
}

# --- Nueva Función: Exportar resultados a CSV ---
def export_results_to_csv(coordinated_pairs, uncoordinated_pairs, summary_columns_map, output_file):
    print(f"Exportando resultados a CSV: {output_file}")
    try:
        # Combinar pares coordinados y descoordinados
        all_pairs = coordinated_pairs + uncoordinated_pairs
        if not all_pairs:
            print("No hay datos para exportar a CSV.")
            return

        # Crear lista de filas para el CSV
        csv_data = []
        for pair in all_pairs:
            row = {}
            # Obtener valores para cada columna definida en summary_columns_map
            for display_name, json_key in summary_columns_map.items():
                row[display_name] = get_nested_value(pair, json_key, default='N/A')
            # Agregar campo para indicar si es coordinado o descoordinado
            row['Estado'] = 'Coordinado' if pair in coordinated_pairs else 'Descoordinado'
            csv_data.append(row)

        # Crear DataFrame y guardar como CSV
        df = pd.DataFrame(csv_data)
        # Reordenar columnas para que 'Estado' esté al inicio
        cols = ['Estado'] + [col for col in df.columns if col != 'Estado']
        df = df[cols]
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Resultados exportados correctamente a: {output_file}")
    except Exception as e:
        print(f"Error al exportar a CSV: {e}")
        traceback.print_exc()

# --- Función auxiliar para get_nested_value (Necesaria para export_results_to_csv) ---
def get_nested_value(d, key_path, default='N/A'):
    keys = key_path.split('.')
    val = d
    try:
        for key in keys:
            val = val[key]
        if isinstance(val, (int, float)):
            if 'TDS' in key_path or 'Pickup' in key_path:
                return f"{val:.5f}"
            elif any(k in key_path for k in ['I_shc', 'Time_out', 'delta_t', 'mt']):
                return f"{val:.3f}"
            else:
                return val
        return val
    except (KeyError, TypeError):
        return default

# --- Fase 1: Análisis de Datos ---
coordinated_pairs = []
uncoordinated_pairs = []
tmt_total_scenario = 0.0
total_valid_pairs_scenario = 0
scenario_pairs_found = 0
skipped_pairs_count = 0
total_pairs_read = 0
try:
    print("Cargando datos (lista de pares)...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"El archivo no existe: {input_file}")
    with open(input_file, 'r') as f:
        relay_pairs_data = json.load(f)
    print("Datos cargados correctamente.")
    if not isinstance(relay_pairs_data, list):
        raise TypeError(f"Error: {input_file} no contiene lista JSON.")
    print(f"Calculando delta_t, mt y clasificando pares para '{TARGET_SCENARIO_ID}'...")
    for pair_entry in relay_pairs_data:
        total_pairs_read += 1
        if not isinstance(pair_entry, dict):
            continue
        current_scenario_id = pair_entry.get("scenario_id")
        if current_scenario_id != TARGET_SCENARIO_ID:
            continue
        scenario_pairs_found += 1
        main_relay_info = pair_entry.get('main_relay')
        backup_relay_info = pair_entry.get('backup_relay')
        fault_id = pair_entry.get('fault', 'N/A')
        line_id = main_relay_info.get('line', 'N/A') if isinstance(main_relay_info, dict) else 'N/A'
        if not isinstance(main_relay_info, dict) or not isinstance(backup_relay_info, dict):
            print(f"Advertencia ({TARGET_SCENARIO_ID}): Falta info relé en: {line_id}-Falla:{fault_id}. Par omitido.")
            skipped_pairs_count += 1
            continue
        main_time = main_relay_info.get('Time_out')
        backup_time = backup_relay_info.get('Time_out')
        if not isinstance(main_time, (int, float)) or not isinstance(backup_time, (int, float)):
            print(f"Advertencia ({TARGET_SCENARIO_ID}): Tiempo(s) no numéricos en: {line_id}-Falla:{fault_id} (Main: {main_time}, Backup: {backup_time}). Par omitido.")
            skipped_pairs_count += 1
            continue
        delta_t = backup_time - main_time - CTI
        mt = (delta_t - abs(delta_t)) / 2
        pair_info = copy.deepcopy(pair_entry)
        pair_info['delta_t'] = delta_t
        pair_info['mt'] = mt
        if delta_t >= 0:
            coordinated_pairs.append(pair_info)
        else:
            uncoordinated_pairs.append(pair_info)
    print("Procesamiento de pares completado.")
    if scenario_pairs_found == 0:
        print(f"No se encontraron pares válidos para '{TARGET_SCENARIO_ID}' en el archivo.")
        total_valid_pairs_scenario = 0
        miscoordination_count_scenario = 0
        tmt_total_scenario = 0.0
    else:
        total_valid_pairs_scenario = len(coordinated_pairs) + len(uncoordinated_pairs)
        miscoordination_count_scenario = len(uncoordinated_pairs)
        tmt_total_scenario = sum(pair.get("mt", 0) for pair in coordinated_pairs + uncoordinated_pairs if isinstance(pair.get("mt"), (int, float)))
    print(f"\n--- Resultados del Análisis de Coordinación para '{TARGET_SCENARIO_ID}' ---")
    print(f"Total de pares leídos: {total_pairs_read}")
    print(f"Total encontrados para '{TARGET_SCENARIO_ID}': {scenario_pairs_found}")
    if skipped_pairs_count > 0:
        print(f"Pares omitidos DENTRO de '{TARGET_SCENARIO_ID}': {skipped_pairs_count}")
    print(f"Total válidos analizados: {total_valid_pairs_scenario}")
    print(f"Coordinados (delta_t >= 0): {len(coordinated_pairs)}")
    print(f"Descoordinados (delta_t < 0): {miscoordination_count_scenario}")
    print(f"Suma total de penalización (MT Total = Suma(mt)) para '{TARGET_SCENARIO_ID}': {tmt_total_scenario:.5f}")

    # --- Exportar resultados a CSV ---
    export_results_to_csv(coordinated_pairs, uncoordinated_pairs, summary_columns_map, output_csv)

except FileNotFoundError:
    print(f"Error CRÍTICO: No se pudo encontrar: {input_file}")
    exit()
except TypeError as e:
    print(f"Error CRÍTICO: Tipo de datos JSON inválido: {e}")
    exit()
except json.JSONDecodeError as e:
    print(f"Error CRÍTICO: JSON mal formado: {e}")
    exit()
except Exception as e:
    print(f"Error inesperado durante el procesamiento: {e}")
    print("--- Traceback ---")
    traceback.print_exc()
    print("-----------------")
    print("ADVERTENCIA: Intentando continuar...")

# --- Fase 2: Preparación de Datos para Dash ---
print("\nPreparando datos para Dash...")
def create_dropdown_options(pair_list):
    options = []
    for idx, pair in enumerate(pair_list):
        main_relay_info = pair.get('main_relay', {})
        backup_relay_info = pair.get('backup_relay', {})
        label = (f"L:{pair.get('fault', 'N/A')}% - "
                 f"M:{main_relay_info.get('relay', 'N/A')} ({main_relay_info.get('line', 'N/A')}) / "
                 f"B:{backup_relay_info.get('relay', 'N/A')} ({backup_relay_info.get('line', 'N/A')})")
        options.append({"label": label, "value": idx})
    return options
coordinated_options = create_dropdown_options(coordinated_pairs)
uncoordinated_options = create_dropdown_options(uncoordinated_pairs)
def format_summary(pair_list, column_map):
    return [{disp_name: get_nested_value(pair, json_key) for disp_name, json_key in column_map.items()} for pair in pair_list]
coordinated_summary = format_summary(coordinated_pairs, summary_columns_map)
uncoordinated_summary = format_summary(uncoordinated_pairs, summary_columns_map)
print("Datos preparados.")

# --- Fase 3: Creación de la Aplicación Dash ---
print("Configurando aplicación Dash...")
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1(f"Análisis de Coordinación de Protecciones", style={'textAlign': 'center'}),
    html.H1(f" Escenario Base Optimizado", style={'textAlign': 'center'}),
    html.H3(f"TMT: {tmt_total_scenario:.3f}, Pares de Relés: {total_valid_pairs_scenario}", style={'textAlign': 'center', 'marginTop': '-10px', 'marginBottom': '20px'}),
    dcc.Tabs([
        dcc.Tab(label=f"Coordinados ({len(coordinated_pairs)})", children=[
            html.Div([
                dcc.Dropdown(id='coordinated-dropdown', options=coordinated_options, value=coordinated_options[0]['value'] if coordinated_options else None, placeholder="Selecciona par coordinado...", style={'width': '70%', 'margin': '20px auto'}),
                dcc.Graph(id='coordinated-graph', style={'height': '600px', 'width': '90%', 'margin': '0 auto'}),
                html.H4("Detalles del Par Seleccionado", style={'textAlign': 'center', 'marginTop': '20px'}),
                dash_table.DataTable(id='coordinated-pair-table', columns=[{"name": "Parámetro", "id": "parameter"}, {"name": "Valor", "id": "value"}], style_table={'width': '60%', 'margin': '20px auto'}, style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace': 'normal', 'height': 'auto'}, style_header={'fontWeight': 'bold'}),
                html.H3("Resumen Pares Coordinados", style={'textAlign': 'center', 'marginTop': '40px'}),
                dash_table.DataTable(id='coordinated-summary-table', columns=[{"name": i, "id": i} for i in summary_columns_map.keys()], data=coordinated_summary, style_table={'overflowX': 'auto', 'width': '95%', 'margin': '20px auto'}, style_cell={'textAlign': 'center', 'minWidth': '80px', 'maxWidth': '150px', 'padding': '5px', 'whiteSpace': 'normal'}, style_header={'fontWeight': 'bold', 'textAlign': 'center'}, page_size=10, sort_action="native", filter_action="native", tooltip_data=[{column: {'value': str(value), 'type': 'markdown'} for column, value in row.items()} for row in coordinated_summary] if coordinated_summary else None, tooltip_duration=None)
            ])
        ]),
        dcc.Tab(label=f"Descoordinados ({len(uncoordinated_pairs)})", children=[
            html.Div([
                dcc.Dropdown(id='uncoordinated-dropdown', options=uncoordinated_options, value=uncoordinated_options[0]['value'] if uncoordinated_options else None, placeholder="Selecciona par descoordinado...", style={'width': '70%', 'margin': '20px auto'}),
                dcc.Graph(id='uncoordinated-graph', style={'height': '600px', 'width': '90%', 'margin': '0 auto'}),
                html.H4("Detalles del Par Seleccionado", style={'textAlign': 'center', 'marginTop': '20px'}),
                dash_table.DataTable(id='uncoordinated-pair-table', columns=[{"name": "Parámetro", "id": "parameter"}, {"name": "Valor", "id": "value"}], style_table={'width': '60%', 'margin': '20px auto'}, style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace': 'normal', 'height': 'auto'}, style_header={'fontWeight': 'bold'}),
                html.H3("Magnitud Penalización |mt| (Descoordinados)", style={'textAlign': 'center', 'marginTop': '40px'}),
                dcc.Graph(id='mt-graph', style={'height': '400px', 'width': '90%', 'margin': '0 auto'}),
                html.H3("Resumen Pares Descoordinados", style={'textAlign': 'center', 'marginTop': '40px'}),
                dash_table.DataTable(id='uncoordinated-summary-table', columns=[{"name": i, "id": i} for i in summary_columns_map.keys()], data=uncoordinated_summary, style_table={'overflowX': 'auto', 'width': '95%', 'margin': '20px auto'}, style_cell={'textAlign': 'center', 'minWidth': '80px', 'maxWidth': '150px', 'padding': '5px', 'whiteSpace': 'normal'}, style_header={'fontWeight': 'bold', 'textAlign': 'center'}, page_size=10, sort_action="native", filter_action="native", tooltip_data=[{column: {'value': str(value), 'type': 'markdown'} for column, value in row.items()} for row in uncoordinated_summary] if uncoordinated_summary else None, tooltip_duration=None)
            ])
        ])
    ])
])

# --- Fase 4: Callback Actualizado con Validación Detallada ---
print("Definiendo callbacks (Validación Mejorada)...")

@app.callback(
    [Output('coordinated-graph', 'figure'), Output('coordinated-pair-table', 'data'),
     Output('uncoordinated-graph', 'figure'), Output('uncoordinated-pair-table', 'data'),
     Output('mt-graph', 'figure')],
    [Input('coordinated-dropdown', 'value'), Input('uncoordinated-dropdown', 'value')]
)
def update_dashboard(coordinated_idx, uncoordinated_idx):
    default_fig_layout = {'title': {'text': "Selecciona un par para ver la gráfica", 'x': 0.5}, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'plot_bgcolor': '#f9f9f9', 'paper_bgcolor': '#f9f9f9'}
    coordinated_fig = go.Figure(layout=default_fig_layout)
    coordinated_table_data = [{"parameter": "Info", "value": "Selecciona un par coordinado"}]
    uncoordinated_fig = go.Figure(layout=default_fig_layout)
    uncoordinated_table_data = [{"parameter": "Info", "value": "Selecciona un par descoordinado"}]
    mt_fig = go.Figure(layout={'title': {'text': "Gráfico |mt| (solo para descoordinados)", 'x': 0.5}})
    error_messages = []

    if coordinated_idx is not None and coordinated_pairs:
        try:
            pair = coordinated_pairs[coordinated_idx]
            main_relay = pair.get('main_relay', {})
            backup_relay = pair.get('backup_relay', {})
            main_tds = main_relay.get('TDS')
            main_pickup = main_relay.get('pick_up')
            main_ishc = main_relay.get('Ishc')
            main_time_out = main_relay.get('Time_out')
            main_name = main_relay.get('relay', 'N/A')
            main_line = main_relay.get('line', 'N/A')
            backup_tds = backup_relay.get('TDS')
            backup_pickup = backup_relay.get('pick_up')
            backup_ishc = backup_relay.get('Ishc')
            backup_time_out = backup_relay.get('Time_out')
            backup_name = backup_relay.get('relay', 'N/A')
            backup_line = backup_relay.get('line', 'N/A')
            fault_perc = pair.get('fault', 'N/A')
            pair_delta_t = pair.get('delta_t')
            pair_mt = pair.get('mt')

            can_plot_points = isinstance(main_ishc, (int, float)) and isinstance(main_time_out, (int, float)) and \
                              isinstance(backup_ishc, (int, float)) and isinstance(backup_time_out, (int, float))
            can_plot_main_curve = isinstance(main_tds, (int, float)) and isinstance(main_pickup, (int, float)) and main_pickup > 1e-6
            can_plot_backup_curve = isinstance(backup_tds, (int, float)) and isinstance(backup_pickup, (int, float)) and backup_pickup > 1e-6

            if can_plot_points:
                coordinated_fig = go.Figure()
                i_range = None
                if can_plot_main_curve or can_plot_backup_curve:
                    currents_for_max = [c for c in [main_ishc, backup_ishc] if isinstance(c, (int, float))]
                    if can_plot_main_curve:
                        currents_for_max.append(main_pickup * MAX_CURRENT_MULTIPLIER)
                    if can_plot_backup_curve:
                        currents_for_max.append(backup_pickup * MAX_CURRENT_MULTIPLIER)
                    max_current_val = max(currents_for_max) if currents_for_max else 10
                    min_current_val_main = max(main_pickup * MIN_CURRENT_MULTIPLIER, 1e-6) if can_plot_main_curve else 1e-6
                    min_current_val_backup = max(backup_pickup * MIN_CURRENT_MULTIPLIER, 1e-6) if can_plot_backup_curve else 1e-6
                    i_range_min_val = min(min_current_val_main, min_current_val_backup)
                    if i_range_min_val < max_current_val:
                        i_range = np.logspace(np.log10(i_range_min_val), np.log10(max_current_val), num=200)
                    else:
                        print(f"Advertencia: Rango de corriente inválido (min >= max) para par coord. {coordinated_idx}. No se generará i_range.")
                        i_range = None

                if can_plot_main_curve and i_range is not None:
                    main_curve_times = calculate_inverse_time_curve(main_tds, main_pickup, i_range)
                    coordinated_fig.add_trace(go.Scatter(x=i_range, y=main_curve_times, mode="lines", name=f"Curva {main_name} (Main)", hovertemplate=f"Relé: {main_name}<br>I: %{{x:.3f}} A<br>t: %{{y:.3f}} s<br>TDS: {main_tds:.5f}, Pk: {main_pickup:.5f} A", line=dict(color="blue", width=2)))
                elif not can_plot_main_curve:
                    error_messages.append(f"No se pudo graficar curva Main ({main_name}): TDS o Pickup inválido/faltante.")

                if can_plot_backup_curve and i_range is not None:
                    backup_curve_times = calculate_inverse_time_curve(backup_tds, backup_pickup, i_range)
                    coordinated_fig.add_trace(go.Scatter(x=i_range, y=backup_curve_times, mode="lines", name=f"Curva {backup_name} (Backup)", hovertemplate=f"Relé: {backup_name}<br>I: %{{x:.3f}} A<br>t: %{{y:.3f}} s<br>TDS: {backup_tds:.5f}, Pk: {backup_pickup:.5f} A", line=dict(color="red", width=2)))
                elif not can_plot_backup_curve:
                    error_messages.append(f"No se pudo graficar curva Backup ({backup_name}): TDS o Pickup inválido/faltante.")

                coordinated_fig.add_trace(go.Scatter(x=[main_ishc], y=[main_time_out], mode="markers", name=f"Op. {main_name} ({main_ishc:.3f}A)", hovertemplate=f"<b>Op. {main_name}</b><br>I_shc: {main_ishc:.3f} A<br>Tiempo: {main_time_out:.3f} s", marker=dict(color="blue", size=10, symbol='circle')))
                coordinated_fig.add_trace(go.Scatter(x=[backup_ishc], y=[backup_time_out], mode="markers", name=f"Op. {backup_name} ({backup_ishc:.3f}A)", hovertemplate=f"<b>Op. {backup_name}</b><br>I_shc: {backup_ishc:.3f} A<br>Tiempo: {backup_time_out:.3f} s", marker=dict(color="red", size=10, symbol='circle')))
                coordinated_fig.add_trace(go.Scatter(x=[main_ishc, main_ishc], y=[MIN_TIME_PLOT, main_time_out], mode="lines", name=f"I_shc {main_name}", line=dict(color="blue", dash="dash"), hoverinfo='skip'))
                coordinated_fig.add_trace(go.Scatter(x=[backup_ishc, backup_ishc], y=[MIN_TIME_PLOT, backup_time_out], mode="lines", name=f"I_shc {backup_name}", line=dict(color="red", dash="dash"), hoverinfo='skip'))
                coordinated_fig.add_trace(go.Scatter(x=[backup_ishc, backup_ishc], y=[main_time_out, main_time_out + CTI], mode="lines+markers", name=f"CTI ({CTI}s)", line=dict(color="green", width=2), marker=dict(symbol='line-ns-open', size=10, color='green'), hovertemplate=f"CTI = {CTI} s<br>t_m + CTI = {main_time_out+CTI:.3f} s"))

                title = f"Coordinación: {main_name} ({main_line}) vs {backup_name} ({backup_line})<br>Falla: {fault_perc}% - Escenario: {TARGET_SCENARIO_ID}"
                xaxis_config = {'title': "Corriente (A)", 'type': "log"}
                yaxis_config = {'title': "Tiempo (s)", 'type': "log", 'range': [np.log10(MIN_TIME_PLOT), np.log10(MAX_TIME_PLOT)]}
                if i_range is not None:
                    xaxis_config['range'] = [np.log10(i_range_min_val*0.9), np.log10(max_current_val*1.1)]
                else:
                    xaxis_config['range'] = [np.log10(min(main_ishc, backup_ishc)*0.8), np.log10(max(main_ishc, backup_ishc)*1.2)]

                coordinated_fig.update_layout(
                    title={'text': title, 'x': 0.5}, xaxis=xaxis_config, yaxis=yaxis_config,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=60, r=30, t=100, b=60), hovermode='closest'
                )
            else:
                error_messages.append("No se pudieron graficar puntos: Ishc o Time_out inválido/faltante.")
                coordinated_fig = go.Figure(layout={'title': {'text': f"Error en par coord. {coordinated_idx}:<br>{'<br>'.join(error_messages)}", 'x': 0.5}})

            coordinated_table_data = [
                {"parameter": "Escenario", "value": TARGET_SCENARIO_ID},
                {"parameter": "Falla (%)", "value": fault_perc},
                {"parameter": "Relé Principal", "value": f"{main_name} (Línea: {main_line})"},
                {"parameter": "TDS (Main)", "value": f"{main_tds:.5f}" if isinstance(main_tds, (int, float)) else 'N/A'},
                {"parameter": "Pickup (Main)", "value": f"{main_pickup:.5f} A" if isinstance(main_pickup, (int, float)) else 'N/A'},
                {"parameter": "I_shc (Main)", "value": f"{main_ishc:.3f} A" if isinstance(main_ishc, (int, float)) else 'N/A'},
                {"parameter": "t_m (Op. Main)", "value": f"{main_time_out:.3f} s" if isinstance(main_time_out, (int, float)) else 'N/A'},
                {"parameter": "Relé Backup", "value": f"{backup_name} (Línea: {backup_line})"},
                {"parameter": "TDS (Backup)", "value": f"{backup_tds:.5f}" if isinstance(backup_tds, (int, float)) else 'N/A'},
                {"parameter": "Pickup (Backup)", "value": f"{backup_pickup:.5f} A" if isinstance(backup_pickup, (int, float)) else 'N/A'},
                {"parameter": "I_shc (Backup)", "value": f"{backup_ishc:.3f} A" if isinstance(backup_ishc, (int, float)) else 'N/A'},
                {"parameter": "t_b (Op. Backup)", "value": f"{backup_time_out:.3f} s" if isinstance(backup_time_out, (int, float)) else 'N/A'},
                {"parameter": "Δt (t_b - t_m - CTI)", "value": f"{pair_delta_t:.3f} s" if isinstance(pair_delta_t, (int, float)) else 'N/A'},
                {"parameter": "MT (Penalización)", "value": f"{pair_mt:.3f} s" if isinstance(pair_mt, (int, float)) else 'N/A'}
            ]
            if error_messages:
                coordinated_table_data.append({"parameter": "Advertencia Gráfico", "value": "; ".join(error_messages)})

        except Exception as e:
            print(f"Error EXCEPCIÓN al procesar par coordinado {coordinated_idx}: {e}")
            traceback.print_exc()
            coordinated_fig = go.Figure(layout={'title': {'text': f"Error Inesperado al graficar par coord. {coordinated_idx}", 'x': 0.5}})
            coordinated_table_data = [{"parameter": "Error", "value": f"Excepción: {e}"}]

    if uncoordinated_idx is not None and uncoordinated_pairs:
        error_messages_uncoord = []
        try:
            pair = uncoordinated_pairs[uncoordinated_idx]
            main_relay = pair.get('main_relay', {})
            backup_relay = pair.get('backup_relay', {})
            main_tds = main_relay.get('TDS')
            main_pickup = main_relay.get('pick_up')
            main_ishc = main_relay.get('Ishc')
            main_time_out = main_relay.get('Time_out')
            main_name = main_relay.get('relay', 'N/A')
            main_line = main_relay.get('line', 'N/A')
            backup_tds = backup_relay.get('TDS')
            backup_pickup = backup_relay.get('pick_up')
            backup_ishc = backup_relay.get('Ishc')
            backup_time_out = backup_relay.get('Time_out')
            backup_name = backup_relay.get('relay', 'N/A')
            backup_line = backup_relay.get('line', 'N/A')
            fault_perc = pair.get('fault', 'N/A')
            pair_delta_t = pair.get('delta_t')
            pair_mt = pair.get('mt')

            can_plot_points = isinstance(main_ishc, (int, float)) and isinstance(main_time_out, (int, float)) and \
                              isinstance(backup_ishc, (int, float)) and isinstance(backup_time_out, (int, float))
            can_plot_main_curve = isinstance(main_tds, (int, float)) and isinstance(main_pickup, (int, float)) and main_pickup > 1e-6
            can_plot_backup_curve = isinstance(backup_tds, (int, float)) and isinstance(backup_pickup, (int, float)) and backup_pickup > 1e-6

            if can_plot_points:
                uncoordinated_fig = go.Figure()
                i_range = None
                if can_plot_main_curve or can_plot_backup_curve:
                    currents_for_max = [c for c in [main_ishc, backup_ishc] if isinstance(c, (int, float))]
                    if can_plot_main_curve:
                        currents_for_max.append(main_pickup * MAX_CURRENT_MULTIPLIER)
                    if can_plot_backup_curve:
                        currents_for_max.append(backup_pickup * MAX_CURRENT_MULTIPLIER)
                    max_current_val = max(currents_for_max) if currents_for_max else 10
                    min_current_val_main = max(main_pickup * MIN_CURRENT_MULTIPLIER, 1e-6) if can_plot_main_curve else 1e-6
                    min_current_val_backup = max(backup_pickup * MIN_CURRENT_MULTIPLIER, 1e-6) if can_plot_backup_curve else 1e-6
                    i_range_min_val = min(min_current_val_main, min_current_val_backup)
                    if i_range_min_val < max_current_val:
                        i_range = np.logspace(np.log10(i_range_min_val), np.log10(max_current_val), num=200)
                    else:
                        print(f"Advertencia: Rango de corriente inválido (min >= max) para par desc. {uncoordinated_idx}. No se generará i_range.")
                        i_range = None

                if can_plot_main_curve and i_range is not None:
                    main_curve_times = calculate_inverse_time_curve(main_tds, main_pickup, i_range)
                    uncoordinated_fig.add_trace(go.Scatter(x=i_range, y=main_curve_times, mode="lines", name=f"Curva {main_name} (Main)", hovertemplate=f"Relé: {main_name}<br>I: %{{x:.3f}} A<br>t: %{{y:.3f}} s<br>TDS: {main_tds:.5f}, Pk: {main_pickup:.5f} A", line=dict(color="blue", width=2)))
                elif not can_plot_main_curve:
                    error_messages_uncoord.append(f"No se pudo graficar curva Main ({main_name}): TDS o Pickup inválido/faltante.")

                if can_plot_backup_curve and i_range is not None:
                    backup_curve_times = calculate_inverse_time_curve(backup_tds, backup_pickup, i_range)
                    uncoordinated_fig.add_trace(go.Scatter(x=i_range, y=backup_curve_times, mode="lines", name=f"Curva {backup_name} (Backup)", hovertemplate=f"Relé: {backup_name}<br>I: %{{x:.3f}} A<br>t: %{{y:.3f}} s<br>TDS: {backup_tds:.5f}, Pk: {backup_pickup:.5f} A", line=dict(color="red", width=2)))
                elif not can_plot_backup_curve:
                    error_messages_uncoord.append(f"No se pudo graficar curva Backup ({backup_name}): TDS o Pickup inválido/faltante.")

                uncoordinated_fig.add_trace(go.Scatter(x=[main_ishc], y=[main_time_out], mode="markers", name=f"Op. {main_name} ({main_ishc:.3f}A)", hovertemplate=f"<b>Op. {main_name}</b><br>I_shc: {main_ishc:.3f} A<br>Tiempo: {main_time_out:.3f} s", marker=dict(color="blue", size=10, symbol='circle')))
                uncoordinated_fig.add_trace(go.Scatter(x=[backup_ishc], y=[backup_time_out], mode="markers", name=f"Op. {backup_name} ({backup_ishc:.3f}A)", hovertemplate=f"<b>Op. {backup_name}</b><br>I_shc: {backup_ishc:.3f} A<br>Tiempo: {backup_time_out:.3f} s", marker=dict(color="red", size=10, symbol='circle')))
                uncoordinated_fig.add_trace(go.Scatter(x=[main_ishc, main_ishc], y=[MIN_TIME_PLOT, main_time_out], mode="lines", name=f"I_shc {main_name}", line=dict(color="blue", dash="dash"), hoverinfo='skip'))
                uncoordinated_fig.add_trace(go.Scatter(x=[backup_ishc, backup_ishc], y=[MIN_TIME_PLOT, backup_time_out], mode="lines", name=f"I_shc {backup_name}", line=dict(color="red", dash="dash"), hoverinfo='skip'))
                uncoordinated_fig.add_trace(go.Scatter(x=[backup_ishc, backup_ishc], y=[main_time_out, main_time_out + CTI], mode="lines+markers", name=f"CTI ({CTI}s)", line=dict(color="green", width=2), marker=dict(symbol='line-ns-open', size=10, color='green'), hovertemplate=f"CTI = {CTI} s<br>t_m + CTI = {main_time_out+CTI:.3f} s"))

                title = f"Descoordinación: {main_name} ({main_line}) vs {backup_name} ({backup_line})<br>Falla: {fault_perc}% - Escenario: {TARGET_SCENARIO_ID}"
                xaxis_config = {'title': "Corriente (A)", 'type': "log"}
                yaxis_config = {'title': "Tiempo (s)", 'type': "log", 'range': [np.log10(MIN_TIME_PLOT), np.log10(MAX_TIME_PLOT)]}
                if i_range is not None:
                    xaxis_config['range'] = [np.log10(i_range_min_val*0.9), np.log10(max_current_val*1.1)]
                else:
                    xaxis_config['range'] = [np.log10(min(main_ishc, backup_ishc)*0.8), np.log10(max(main_ishc, backup_ishc)*1.2)]
                uncoordinated_fig.update_layout(title={'text': title, 'x': 0.5}, xaxis=xaxis_config, yaxis=yaxis_config, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=60, r=30, t=100, b=60), hovermode='closest')
            else:
                error_messages_uncoord.append("No se pudieron graficar puntos: Ishc o Time_out inválido/faltante.")
                uncoordinated_fig = go.Figure(layout={'title': {'text': f"Error en par desc. {uncoordinated_idx}:<br>{'<br>'.join(error_messages_uncoord)}", 'x': 0.5}})

            uncoordinated_table_data = [
                {"parameter": "Escenario", "value": TARGET_SCENARIO_ID},
                {"parameter": "Falla (%)", "value": fault_perc},
                {"parameter": "Relé Principal", "value": f"{main_name} (Línea: {main_line})"},
                {"parameter": "TDS (Main)", "value": f"{main_tds:.5f}" if isinstance(main_tds, (int, float)) else 'N/A'},
                {"parameter": "Pickup (Main)", "value": f"{main_pickup:.5f} A" if isinstance(main_pickup, (int, float)) else 'N/A'},
                {"parameter": "I_shc (Main)", "value": f"{main_ishc:.3f} A" if isinstance(main_ishc, (int, float)) else 'N/A'},
                {"parameter": "t_m (Op. Main)", "value": f"{main_time_out:.3f} s" if isinstance(main_time_out, (int, float)) else 'N/A'},
                {"parameter": "Relé Backup", "value": f"{backup_name} (Línea: {backup_line})"},
                {"parameter": "TDS (Backup)", "value": f"{backup_tds:.5f}" if isinstance(backup_tds, (int, float)) else 'N/A'},
                {"parameter": "Pickup (Backup)", "value": f"{backup_pickup:.5f} A" if isinstance(backup_pickup, (int, float)) else 'N/A'},
                {"parameter": "I_shc (Backup)", "value": f"{backup_ishc:.3f} A" if isinstance(backup_ishc, (int, float)) else 'N/A'},
                {"parameter": "t_b (Op. Backup)", "value": f"{backup_time_out:.3f} s" if isinstance(backup_time_out, (int, float)) else 'N/A'},
                {"parameter": "Δt (t_b - t_m - CTI)", "value": f"{pair_delta_t:.3f} s (<0!)" if isinstance(pair_delta_t, (int, float)) else 'N/A'},
                {"parameter": "MT (Penalización)", "value": f"{pair_mt:.3f} s" if isinstance(pair_mt, (int, float)) else 'N/A'}
            ]
            if error_messages_uncoord:
                uncoordinated_table_data.append({"parameter": "Advertencia Gráfico", "value": "; ".join(error_messages_uncoord)})

        except Exception as e:
            print(f"Error EXCEPCIÓN al procesar par descoordinado {uncoordinated_idx}: {e}")
            traceback.print_exc()
            uncoordinated_fig = go.Figure(layout={'title': {'text': f"Error Inesperado al graficar par desc. {uncoordinated_idx}", 'x': 0.5}})
            uncoordinated_table_data = [{"parameter": "Error", "value": f"Excepción: {e}"}]

    if uncoordinated_pairs:
        try:
            mt_values = [abs(pair.get("mt", 0)) for pair in uncoordinated_pairs if isinstance(pair.get("mt"), (int, float)) and pair.get("mt") < 0]
            mt_labels = []
            for pair in uncoordinated_pairs:
                if isinstance(pair.get("mt"), (int, float)) and pair.get("mt") < 0:
                    main_relay_info = pair.get('main_relay', {})
                    backup_relay_info = pair.get('backup_relay', {})
                    label = (f"F:{pair.get('fault', 'N/A')}% M:{main_relay_info.get('relay', 'N/A')}/B:{backup_relay_info.get('relay', 'N/A')}")
                    mt_labels.append(label)
            if mt_values:
                sorted_indices = np.argsort(mt_values)[::-1]
                mt_values_sorted = [mt_values[i] for i in sorted_indices]
                mt_labels_sorted = [mt_labels[i] for i in sorted_indices]
                mt_fig = go.Figure()
                mt_fig.add_trace(go.Bar(x=mt_labels_sorted, y=mt_values_sorted, name="|mt|", text=[f"{v:.3f}s" for v in mt_values_sorted], textposition='auto', marker_color='purple', hovertemplate="<b>Par</b>: %{x}<br><b>|mt|</b>: %{y:.3f} s<extra></extra>"))
                mt_fig.update_layout(title_text=f"Magnitud Penalización |mt| (Descoordinados, {TARGET_SCENARIO_ID})", title_x=0.5, xaxis_title="Pares Descoordinados", yaxis_title="|mt| (s)", xaxis={'tickangle': -45}, margin=dict(l=50, r=50, t=80, b=150), yaxis_range=[0, max(mt_values_sorted) * 1.1])
            else:
                mt_fig = go.Figure(layout={'title': {'text': f"No hay pares descoordinados con MT < 0 ({TARGET_SCENARIO_ID})", 'x': 0.5}})
        except Exception as e:
            print(f"Error generando gráfico MT: {e}")
            traceback.print_exc()
            mt_fig = go.Figure(layout={'title': {'text': "Error al generar gráfico MT", 'x': 0.5}})
    else:
        mt_fig = go.Figure(layout={'title': {'text': f"No hay pares descoordinados en {TARGET_SCENARIO_ID}", 'x': 0.5}})

    return coordinated_fig, coordinated_table_data, uncoordinated_fig, uncoordinated_table_data, mt_fig

# --- Fase 5: Ejecutar la Aplicación ---
if __name__ == '__main__':
    print("\nIniciando servidor Dash...")
    print(f"Accede a la aplicación en: http://127.0.0.1:8050/")
    app.run_server(debug=True)