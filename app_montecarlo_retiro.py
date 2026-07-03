from __future__ import annotations

from typing import Optional

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
# Identidad visual / estilo dashboard
# ============================================================

COLOR_BG = "#041F5F"
COLOR_BG_2 = "#061844"
COLOR_CARD = "#0B1F4A"
COLOR_CARD_2 = "#102B66"
COLOR_PRIMARY = "#8B3DFF"
COLOR_PRIMARY_2 = "#B78CFF"
COLOR_CYAN = "#00D1FF"
COLOR_TEXT = "#F5F7FA"
COLOR_MUTED = "#B8C4D8"
COLOR_GOOD = "#30D158"
COLOR_WARN = "#FFD166"
COLOR_BAD = "#FF5C7A"

PLOTLY_TEMPLATE = "plotly_dark"
PERCENTILE_COLORS = {
    "p95": "#6EE7FF",
    "p75": "#36A3FF",
    "p50 / mediana": "#B78CFF",
    "p25": "#8B3DFF",
    "p5": "#FF5C7A",
    "media": "#FFFFFF",
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {COLOR_BG};
            --bg2: {COLOR_BG_2};
            --card: {COLOR_CARD};
            --card2: {COLOR_CARD_2};
            --primary: {COLOR_PRIMARY};
            --primary2: {COLOR_PRIMARY_2};
            --cyan: {COLOR_CYAN};
            --text: {COLOR_TEXT};
            --muted: {COLOR_MUTED};
            --good: {COLOR_GOOD};
            --warn: {COLOR_WARN};
            --bad: {COLOR_BAD};
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(139, 61, 255, 0.22), transparent 32%),
                radial-gradient(circle at top right, rgba(0, 209, 255, 0.14), transparent 30%),
                linear-gradient(135deg, var(--bg) 0%, #031135 50%, #02081F 100%);
            color: var(--text);
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(11, 31, 74, 0.98) 0%, rgba(4, 31, 95, 0.96) 100%);
            border-right: 1px solid rgba(139, 61, 255, 0.30);
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p {{
            color: var(--text) !important;
        }}

        .main .block-container {{
            padding-top: 1.35rem;
            padding-bottom: 2.0rem;
            max-width: 1500px;
        }}

        .quant-hero {{
            border: 1px solid rgba(139, 61, 255, 0.35);
            background: linear-gradient(135deg, rgba(11, 31, 74, 0.88), rgba(16, 43, 102, 0.72));
            border-radius: 24px;
            padding: 24px 28px;
            box-shadow: 0 18px 52px rgba(0, 0, 0, 0.30);
            margin-bottom: 18px;
        }}

        .quant-title {{
            font-size: 2.05rem;
            line-height: 1.08;
            font-weight: 800;
            letter-spacing: -0.035em;
            color: var(--text);
            margin-bottom: 8px;
        }}

        .quant-subtitle {{
            color: var(--muted);
            font-size: 1.02rem;
            max-width: 1050px;
            margin-bottom: 0px;
        }}

        .quant-pill-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 9px;
            margin-top: 16px;
        }}

        .quant-pill {{
            border: 1px solid rgba(0, 209, 255, 0.28);
            background: rgba(0, 209, 255, 0.08);
            color: var(--text);
            border-radius: 999px;
            padding: 6px 11px;
            font-size: 0.82rem;
            font-weight: 650;
        }}

        .metric-card {{
            min-height: 128px;
            border: 1px solid rgba(139, 61, 255, 0.26);
            background: linear-gradient(145deg, rgba(11, 31, 74, 0.94), rgba(6, 24, 68, 0.92));
            border-radius: 20px;
            padding: 18px 18px 15px 18px;
            box-shadow: 0 16px 35px rgba(0, 0, 0, 0.24);
            position: relative;
            overflow: hidden;
        }}

        .metric-card::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(139, 61, 255, 0.22), transparent 38%);
            pointer-events: none;
        }}

        .metric-label {{
            position: relative;
            z-index: 1;
            color: var(--muted);
            font-size: 0.79rem;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            font-weight: 750;
            margin-bottom: 8px;
        }}

        .metric-value {{
            position: relative;
            z-index: 1;
            color: var(--text);
            font-size: 1.70rem;
            line-height: 1.10;
            font-weight: 850;
            letter-spacing: -0.025em;
        }}

        .metric-note {{
            position: relative;
            z-index: 1;
            color: var(--muted);
            font-size: 0.80rem;
            margin-top: 8px;
        }}

        .metric-good .metric-value {{ color: var(--good); }}
        .metric-warn .metric-value {{ color: var(--warn); }}
        .metric-bad .metric-value {{ color: var(--bad); }}
        .metric-primary .metric-value {{ color: var(--primary2); }}
        .metric-cyan .metric-value {{ color: var(--cyan); }}

        .section-card {{
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(11, 31, 74, 0.60);
            border-radius: 18px;
            padding: 16px 18px;
            margin: 8px 0 18px 0;
        }}

        div[data-testid="stAlert"] {{
            border-radius: 16px;
            border: 1px solid rgba(139, 61, 255, 0.25);
            background: rgba(11, 31, 74, 0.85);
        }}

        .stButton > button {{
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(183, 140, 255, 0.65);
            background: linear-gradient(90deg, var(--primary), #5F7CFF);
            color: white;
            font-weight: 800;
            box-shadow: 0 12px 28px rgba(139, 61, 255, 0.28);
        }}

        .stButton > button:hover {{
            border-color: var(--cyan);
            filter: brightness(1.08);
        }}

        div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(139, 61, 255, 0.18);
        }}

        div[data-baseweb="tab-list"] {{
            gap: 8px;
        }}

        button[data-baseweb="tab"] {{
            background: rgba(11, 31, 74, 0.70);
            border-radius: 999px;
            padding-left: 14px;
            padding-right: 14px;
            border: 1px solid rgba(139, 61, 255, 0.18);
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            background: rgba(139, 61, 255, 0.28);
            border: 1px solid rgba(183, 140, 255, 0.55);
        }}

        hr {{
            border-color: rgba(139, 61, 255, 0.20) !important;
        }}

        .small-muted {{
            color: var(--muted);
            font-size: 0.88rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plot_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4, 31, 95, 0.24)",
        font={"color": COLOR_TEXT, "family": "Inter, Segoe UI, Arial"},
        title={"font": {"size": 20, "color": COLOR_TEXT}},
        legend={
            "bgcolor": "rgba(11, 31, 74, 0.35)",
            "bordercolor": "rgba(139, 61, 255, 0.25)",
            "borderwidth": 1,
        },
        margin={"l": 50, "r": 28, "t": 62, "b": 48},
    )
    fig.update_xaxes(gridcolor="rgba(184, 196, 216, 0.12)", zerolinecolor="rgba(184, 196, 216, 0.18)")
    fig.update_yaxes(gridcolor="rgba(184, 196, 216, 0.12)", zerolinecolor="rgba(184, 196, 216, 0.18)")
    return fig


def metric_card(label: str, value: str, note: str = "", tone: str = "primary") -> None:
    safe_note = f'<div class="metric-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="metric-card metric-{tone}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {safe_note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_mm(x: float, decimals: int = 0) -> str:
    if np.isnan(x):
        return "N/A"
    return f"{x:,.{decimals}f} MM"


def fmt_pct(x: float, decimals: int = 1) -> str:
    if np.isnan(x):
        return "N/A"
    return f"{x:,.{decimals}f}%"


# ============================================================
# Gráficos
# ============================================================


def plot_percentile_fan(tabla: pd.DataFrame, edad_inicio_retiro: int, target_mm: Optional[float]) -> go.Figure:
    fig = go.Figure()

    for col, name in [
        ("p95_mm", "p95"),
        ("p75_mm", "p75"),
        ("p50_mediana_mm", "p50 / mediana"),
        ("p25_mm", "p25"),
        ("p5_mm", "p5"),
        ("media_mm", "media"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=tabla["edad"],
                y=tabla[col],
                name=name,
                mode="lines",
                line={"width": 3 if name in {"p50 / mediana", "media"} else 2, "color": PERCENTILE_COLORS[name]},
            )
        )

    fig.add_vline(
        x=edad_inicio_retiro,
        line_dash="dash",
        line_color=COLOR_CYAN,
        annotation_text="inicio retiro",
        annotation_position="top left",
    )

    if target_mm is not None:
        fig.add_hline(
            y=target_mm,
            line_dash="dot",
            line_color=COLOR_WARN,
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
    return apply_plot_theme(fig)


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
                line={"width": 0.75, "color": COLOR_PRIMARY_2},
                opacity=0.16,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    median_path = np.percentile(paths, 50, axis=0)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=median_path,
            mode="lines",
            name="mediana",
            line={"width": 3.2, "color": COLOR_CYAN},
        )
    )

    fig.add_vline(x=edad_inicio_retiro, line_dash="dash", line_color=COLOR_CYAN, annotation_text="inicio retiro")
    if target_mm is not None:
        fig.add_hline(y=target_mm, line_dash="dot", line_color=COLOR_WARN, annotation_text=f"objetivo {target_mm:,.0f} MM")

    fig.update_layout(
        title=f"Paths Monte Carlo simulados ({n_sample:,} paths mostrados)",
        xaxis_title="Edad",
        yaxis_title="Patrimonio (MM CLP)",
        hovermode="x unified",
    )
    return apply_plot_theme(fig)


def plot_final_distribution(result: dict) -> go.Figure:
    final_wealth = result["final_wealth_mm"]
    edad_final = result["inputs"]["edad_final"]
    fig = px.histogram(
        x=final_wealth,
        nbins=80,
        labels={"x": "Patrimonio final (MM CLP)", "y": "Frecuencia"},
        title=f"Distribución del patrimonio final a los {edad_final} años",
        color_discrete_sequence=[COLOR_PRIMARY],
    )
    fig.update_traces(marker_line_width=0.3, marker_line_color="rgba(255,255,255,0.20)")
    fig.update_layout(showlegend=False)
    return apply_plot_theme(fig)


def plot_ruin_distribution(result: dict) -> go.Figure:
    ruin_age = result["ruin_age"]
    data = ruin_age[~np.isnan(ruin_age)]
    if len(data) == 0:
        fig = go.Figure()
        fig.update_layout(title="No hubo agotamiento de patrimonio en las simulaciones")
        return apply_plot_theme(fig)

    fig = px.histogram(
        x=data,
        nbins=40,
        labels={"x": "Edad de agotamiento", "y": "Frecuencia"},
        title="Distribución de edad de agotamiento del patrimonio",
        color_discrete_sequence=[COLOR_BAD],
    )
    fig.update_traces(marker_line_width=0.3, marker_line_color="rgba(255,255,255,0.20)")
    fig.update_layout(showlegend=False)
    return apply_plot_theme(fig)


# ============================================================
# App Streamlit
# ============================================================

st.set_page_config(
    page_title="Monte Carlo Retiro Fijo",
    page_icon="📈",
    layout="wide",
)
inject_css()

st.markdown(
    """
    <div class="quant-hero">
        <div class="quant-title">Monte Carlo patrimonial: acumulación + retiro fijo</div>
        <div class="quant-subtitle">
            Simulador en MM CLP para probar desde qué edad dejas de ahorrar, cuánto retiras fijo al mes
            y si el patrimonio sigue creciendo, se estabiliza o comienza a caer.
        </div>
        <div class="quant-pill-row">
            <div class="quant-pill">MM CLP</div>
            <div class="quant-pill">Acumulación</div>
            <div class="quant-pill">Retiro fijo mensual</div>
            <div class="quant-pill">Riesgo de agotamiento</div>
            <div class="quant-pill">Percentiles Monte Carlo</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

st.markdown(
    """
    <div class="section-card">
        <b>Aportes extraordinarios</b><br>
        <span class="small-muted">Opcional. Usa mes de simulación 1 para el primer mes. Deja monto 0 si no aplica.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

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

def survival_tone(pct: float) -> str:
    if pct >= 90:
        return "good"
    if pct >= 70:
        return "warn"
    return "bad"

st.markdown("### Resumen del escenario")

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("P50 inicio retiro", fmt_mm(ret_p50), "Patrimonio mediano al cortar el ahorro", "primary")
with k2:
    metric_card("P50 final", fmt_mm(final_p50), f"Edad final: {int(edad_final)} años", "cyan")
with k3:
    metric_card("Prob. no agotarse", fmt_pct(prob_no_ruin), "Paths que nunca llegan a cero", survival_tone(prob_no_ruin))
with k4:
    metric_card(
        "Final > inicio retiro",
        fmt_pct(prob_grow),
        "Evalúa si el capital crece pese al retiro",
        survival_tone(prob_grow) if not np.isnan(prob_grow) else "primary",
    )

st.write("")
k5, k6, k7, k8 = st.columns(4)
with k5:
    metric_card("Objetivo al retiro", fmt_pct(prob_target_ret), f"Target: {float(target_mm):,.0f} MM", "primary")
with k6:
    metric_card("Objetivo al final", fmt_pct(prob_target_final), "Probabilidad de cerrar sobre target", "cyan")
with k7:
    metric_card("Retiro total pedido", fmt_mm(result["total_withdrawal_requested_mm"]), "Suma nominal del calendario de retiros", "primary")
with k8:
    metric_card(
        "Edad mediana agotamiento",
        "No se agota" if np.isnan(median_ruin_age) else f"{median_ruin_age:,.1f}",
        "Solo considera paths que llegan a cero",
        "good" if np.isnan(median_ruin_age) else "bad",
    )

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
