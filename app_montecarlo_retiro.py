import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from montecarlo_engine import (
    monte_carlo_accumulation_withdrawal_mm,
    tabla_monte_carlo_por_edad,
)


# ============================================================
# Gráficos
# ============================================================

def plot_percentile_fan(tabla: pd.DataFrame, edad_inicio_retiro: int, target_mm: Optional[float]) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=tabla["edad"], y=tabla["p95_mm"], name="p95", mode="lines"))
    fig.add_trace(go.Scatter(x=tabla["edad"], y=tabla["p75_mm"], name="p75", mode="lines"))
    fig.add_trace(go.Scatter(x=tabla["edad"], y=tabla["p50_mediana_mm"], name="p50 / mediana", mode="lines"))
    fig.add_trace(go.Scatter(x=tabla["edad"], y=tabla["p25_mm"], name="p25", mode="lines"))
    fig.add_trace(go.Scatter(x=tabla["edad"], y=tabla["p5_mm"], name="p5", mode="lines"))
    fig.add_trace(go.Scatter(x=tabla["edad"], y=tabla["media_mm"], name="media", mode="lines"))

    fig.add_vline(
        x=edad_inicio_retiro,
        line_dash="dash",
        annotation_text="inicio retiro",
        annotation_position="top left",
    )

    if target_mm is not None:
        fig.add_hline(
            y=target_mm,
            line_dash="dot",
            annotation_text=f"objetivo {target_mm:,.0f} MM",
            annotation_position="top left",
        )

    fig.update_layout(
        title="Evolución del patrimonio por edad",
        xaxis_title="Edad",
        yaxis_title="Patrimonio (MM CLP)",
        hovermode="x unified",
        legend_title="Serie",
    )
    return fig


def plot_sample_paths(result: dict, n_sample: int = 300) -> go.Figure:
    paths = result["paths_mm"]
    inputs = result["inputs"]
    edad_inicial = inputs["edad_inicial"]
    edad_inicio_retiro = inputs["edad_inicio_retiro"]
    months = inputs["months"]
    target_mm = inputs["target_mm"]

    rng = np.random.default_rng(2026)
    n_sample = min(n_sample, paths.shape[0])
    idx = rng.choice(paths.shape[0], size=n_sample, replace=False)
    sample = paths[idx]
    x = edad_inicial + np.arange(months + 1) / 12

    fig = go.Figure()
    for row in sample:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=row,
                mode="lines",
                line={"width": 0.8},
                opacity=0.18,
                showlegend=False,
            )
        )

    fig.add_vline(x=edad_inicio_retiro, line_dash="dash", annotation_text="inicio retiro")
    if target_mm is not None:
        fig.add_hline(y=target_mm, line_dash="dot", annotation_text=f"objetivo {target_mm:,.0f} MM")

    fig.update_layout(
        title=f"Paths Monte Carlo simulados ({n_sample:,} paths mostrados)",
        xaxis_title="Edad",
        yaxis_title="Patrimonio (MM CLP)",
        hovermode="x unified",
    )
    return fig


def plot_final_distribution(result: dict) -> go.Figure:
    final_wealth = result["final_wealth_mm"]
    edad_final = result["inputs"]["edad_final"]
    fig = px.histogram(
        x=final_wealth,
        nbins=80,
        labels={"x": "Patrimonio final (MM CLP)", "y": "Frecuencia"},
        title=f"Distribución del patrimonio final a los {edad_final} años",
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_ruin_distribution(result: dict) -> go.Figure:
    ruin_age = result["ruin_age"]
    data = ruin_age[~np.isnan(ruin_age)]
    if len(data) == 0:
        fig = go.Figure()
        fig.update_layout(title="No hubo agotamiento de patrimonio en las simulaciones")
        return fig
    fig = px.histogram(
        x=data,
        nbins=40,
        labels={"x": "Edad de agotamiento", "y": "Frecuencia"},
        title="Distribución de edad de agotamiento del patrimonio",
    )
    fig.update_layout(showlegend=False)
    return fig


# ============================================================
# App Streamlit
# ============================================================

st.set_page_config(
    page_title="Monte Carlo Retiro Fijo",
    layout="wide",
)

st.title("Monte Carlo patrimonial: acumulación + retiro fijo")
st.caption("Unidad de trabajo: MM CLP. Ejemplo: 3.0 equivale a $3.000.000.")

with st.sidebar:
    st.header("Supuestos principales")

    col_age1, col_age2 = st.columns(2)
    with col_age1:
        edad_inicial = st.number_input("Edad inicial", min_value=18, max_value=80, value=28, step=1)
    with col_age2:
        edad_final = st.number_input("Edad final", min_value=int(edad_inicial) + 1, max_value=100, value=55, step=1)

    edad_inicio_retiro = st.number_input(
        "Edad desde que dejas de ahorrar y comienzas a retirar",
        min_value=int(edad_inicial),
        max_value=int(edad_final),
        value=min(40, int(edad_final)),
        step=1,
    )

    initial_capital_mm = st.number_input("Capital inicial (MM CLP)", min_value=0.0, value=50.0, step=1.0)
    target_mm = st.number_input("Objetivo patrimonial (MM CLP)", min_value=0.0, value=1_000.0, step=50.0)

    st.divider()
    st.subheader("Ahorro antes del retiro")
    monthly_saving_min_mm = st.number_input("Ahorro mensual mínimo", min_value=0.0, value=2.5, step=0.1)
    monthly_saving_mode_mm = st.number_input("Ahorro mensual más probable", min_value=0.0, value=3.0, step=0.1)
    monthly_saving_max_mm = st.number_input("Ahorro mensual máximo", min_value=0.0, value=3.5, step=0.1)
    contribution_timing_es = st.selectbox("Timing del ahorro", ["Fin de mes", "Inicio de mes"], index=0)
    contribution_timing = "end" if contribution_timing_es == "Fin de mes" else "begin"

    st.divider()
    st.subheader("Retiro fijo")
    withdrawal_monthly_mm = st.number_input("Retiro mensual fijo (MM CLP)", min_value=0.0, value=3.0, step=0.1)
    withdrawal_timing_es = st.selectbox("Timing del retiro", ["Fin de mes", "Inicio de mes"], index=0)
    withdrawal_timing = "end" if withdrawal_timing_es == "Fin de mes" else "begin"
    withdrawal_indexed_to_inflation = st.checkbox("Indexar retiro por inflación", value=False)
    inflation_annual_pct = st.number_input("Inflación anual para indexación (%)", min_value=0.0, value=3.0, step=0.25)

    st.divider()
    st.subheader("Retorno y riesgo")
    return_model_es = st.selectbox(
        "Modelo de retorno",
        ["Anual suavizado como tu código original", "Mensual IID más realista para retiro"],
        index=0,
    )
    return_model = "annual_smooth" if return_model_es.startswith("Anual") else "monthly_iid"

    annual_return_mean_pct = st.number_input("Retorno anual esperado (%)", value=10.0, step=0.5)
    annual_return_std_pct = st.number_input("Volatilidad anual (%)", min_value=0.1, value=5.0, step=0.5)
    annual_return_low_pct = st.number_input("Retorno anual mínimo truncado (%)", value=-55.0, step=1.0)
    annual_return_high_pct = st.number_input("Retorno anual máximo truncado (%)", value=25.0, step=1.0)
    mean_is_effective = st.checkbox("Calibrar media efectiva luego del truncamiento", value=True)

    st.divider()
    st.subheader("Simulación")
    n_paths = st.number_input("Número de simulaciones", min_value=1_000, max_value=200_000, value=50_000, step=5_000)
    seed = st.number_input("Seed", min_value=0, max_value=999_999, value=123, step=1)
    floor_zero = st.checkbox("Patrimonio no puede quedar negativo", value=True)

st.subheader("Aportes extraordinarios")
st.write("Opcional. Usa mes de simulación 1 para el primer mes. Deja monto 0 si no aplica.")

default_lump_df = pd.DataFrame(
    {
        "mes_simulacion": [24],
        "monto_mm": [0.0],
        "comentario": ["ejemplo: venta activo / bono / aporte"],
    }
)

lump_df = st.data_editor(
    default_lump_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "mes_simulacion": st.column_config.NumberColumn("Mes simulación", min_value=1, step=1),
        "monto_mm": st.column_config.NumberColumn("Monto MM CLP", step=1.0),
        "comentario": st.column_config.TextColumn("Comentario"),
    },
)

lump_events = []
for _, row in lump_df.dropna(subset=["mes_simulacion", "monto_mm"]).iterrows():
    month_idx = int(row["mes_simulacion"])
    amount = float(row["monto_mm"])
    if amount != 0:
        lump_events.append((month_idx, amount))
lump_events = tuple(lump_events)

run = st.button("Simular", type="primary")

if not run:
    st.info("Ajusta los supuestos y presiona Simular.")
    st.stop()

try:
    with st.spinner("Simulando escenarios..."):
        result = monte_carlo_accumulation_withdrawal_mm(
            edad_inicial=int(edad_inicial),
            edad_final=int(edad_final),
            edad_inicio_retiro=int(edad_inicio_retiro),
            n_paths=int(n_paths),
            initial_capital_mm=float(initial_capital_mm),
            annual_return_mean=float(annual_return_mean_pct) / 100,
            annual_return_std=float(annual_return_std_pct) / 100,
            annual_return_low=float(annual_return_low_pct) / 100,
            annual_return_high=float(annual_return_high_pct) / 100,
            monthly_saving_min_mm=float(monthly_saving_min_mm),
            monthly_saving_mode_mm=float(monthly_saving_mode_mm),
            monthly_saving_max_mm=float(monthly_saving_max_mm),
            withdrawal_monthly_mm=float(withdrawal_monthly_mm),
            contribution_timing=contribution_timing,
            withdrawal_timing=withdrawal_timing,
            target_mm=float(target_mm),
            seed=int(seed),
            mean_is_effective=bool(mean_is_effective),
            lump_sum_events=lump_events,
            floor_zero=bool(floor_zero),
            return_model=return_model,
            withdrawal_indexed_to_inflation=bool(withdrawal_indexed_to_inflation),
            inflation_annual=float(inflation_annual_pct) / 100,
        )
        tabla = tabla_monte_carlo_por_edad(result)

except Exception as e:
    st.error(f"Error en la simulación: {e}")
    st.stop()

# ============================================================
# KPIs
# ============================================================

summary = result["summary"].set_index("metric")
final_p50 = summary.loc["p50", "final_wealth_mm"]
ret_p50 = summary.loc["p50", "wealth_at_retirement_mm"]
prob_no_ruin = result["prob_no_ruin"] * 100
prob_target_ret = result["prob_reach_target_at_retirement"] * 100
prob_target_final = result["prob_reach_target_final"] * 100
prob_grow = result["prob_final_above_retirement_wealth"] * 100
median_ruin_age = result["median_ruin_age"]

st.subheader("Resumen del escenario")

k1, k2, k3, k4 = st.columns(4)
k1.metric("P50 al inicio del retiro", f"{ret_p50:,.0f} MM")
k2.metric("P50 final", f"{final_p50:,.0f} MM")
k3.metric("Prob. no agotarse", f"{prob_no_ruin:,.1f}%")
k4.metric("Prob. final > inicio retiro", f"{prob_grow:,.1f}%" if not np.isnan(prob_grow) else "N/A")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Prob. objetivo al retiro", f"{prob_target_ret:,.1f}%")
k6.metric("Prob. objetivo al final", f"{prob_target_final:,.1f}%")
k7.metric("Retiro total pedido", f"{result['total_withdrawal_requested_mm']:,.0f} MM")
k8.metric("Edad mediana agotamiento", "No se agota" if np.isnan(median_ruin_age) else f"{median_ruin_age:,.1f}")

st.caption(
    f"Media efectiva anual simulada: {result['inputs']['effective_truncated_mean_annualized'] * 100:,.2f}%. "
    f"Modelo de retorno: {return_model_es}."
)

# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Percentiles", "Paths", "Distribución final", "Riesgo de agotamiento", "Tablas"]
)

with tab1:
    st.plotly_chart(plot_percentile_fan(tabla, int(edad_inicio_retiro), float(target_mm)), use_container_width=True)

with tab2:
    n_sample = st.slider("Paths a mostrar", min_value=50, max_value=1_000, value=300, step=50)
    st.plotly_chart(plot_sample_paths(result, n_sample=n_sample), use_container_width=True)

with tab3:
    st.plotly_chart(plot_final_distribution(result), use_container_width=True)

with tab4:
    st.plotly_chart(plot_ruin_distribution(result), use_container_width=True)
    if result["prob_no_ruin"] == 1:
        st.success("En este escenario, ningún path agotó patrimonio dentro del horizonte simulado.")
    else:
        st.warning(
            f"En este escenario, {(1 - result['prob_no_ruin']) * 100:,.1f}% de los paths agotó patrimonio "
            "dentro del horizonte simulado."
        )

with tab5:
    st.write("Tabla por edad")
    st.dataframe(tabla, use_container_width=True)

    st.write("Resumen final")
    st.dataframe(result["summary"].round(2), use_container_width=True)

    csv_tabla = tabla.to_csv(index=False).encode("utf-8")
    csv_summary = result["summary"].to_csv(index=False).encode("utf-8")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Descargar tabla por edad CSV",
            data=csv_tabla,
            file_name="tabla_montecarlo_por_edad.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Descargar resumen CSV",
            data=csv_summary,
            file_name="resumen_montecarlo.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Nota: esto es una herramienta de simulación, no una recomendación financiera. "
    "El modo mensual IID suele mostrar más riesgo de secuencia que el modo anual suavizado."
)
