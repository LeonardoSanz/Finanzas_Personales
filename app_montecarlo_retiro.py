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
COLOR_ORANGE = "#FFB86B"

PLOTLY_TEMPLATE = "plotly_dark"
PERCENTILE_COLORS = {
    "p95": "#6EE7FF",
    "p75": "#36A3FF",
    "p50 / mediana": "#B78CFF",
    "p25": "#8B3DFF",
    "p5": "#FF5C7A",
    "media": "#FFFFFF",
}

EDAD_FINAL_FIJA = 90


# ============================================================
# Formato CLP
# ============================================================

def clp_to_mm(value_clp: float | int) -> float:
    return float(value_clp) / 1_000_000


def mm_to_clp(value_mm: float | int) -> float:
    return float(value_mm) * 1_000_000


def fmt_int_dot(value: float | int) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    sign = "-" if float(value) < 0 else ""
    value_abs = abs(int(round(float(value))))
    return sign + f"{value_abs:,}".replace(",", ".")


def fmt_clp(value_clp: float | int) -> str:
    if value_clp is None or (isinstance(value_clp, float) and np.isnan(value_clp)):
        return "N/A"
    return f"${fmt_int_dot(value_clp)}"


def fmt_clp_from_mm(value_mm: float | int) -> str:
    if value_mm is None or (isinstance(value_mm, float) and np.isnan(value_mm)):
        return "N/A"
    return fmt_clp(mm_to_clp(value_mm))


def fmt_pct(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:,.{decimals}f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def mm_series_to_clp(series: pd.Series) -> pd.Series:
    return (series.astype(float) * 1_000_000).round(0).astype("Int64")


# ============================================================
# CSS
# ============================================================

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
            --orange: {COLOR_ORANGE};
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(139, 61, 255, 0.24), transparent 31%),
                radial-gradient(circle at top right, rgba(0, 209, 255, 0.16), transparent 30%),
                linear-gradient(135deg, var(--bg) 0%, #031135 50%, #02081F 100%);
            color: var(--text);
        }}

        .main .block-container {{
            padding-top: 1.15rem;
            padding-bottom: 2.0rem;
            max-width: 1580px;
        }}

        .quant-hero {{
            border: 1px solid rgba(139, 61, 255, 0.38);
            background: linear-gradient(135deg, rgba(11, 31, 74, 0.90), rgba(16, 43, 102, 0.70));
            border-radius: 26px;
            padding: 24px 28px;
            box-shadow: 0 18px 52px rgba(0, 0, 0, 0.30);
            margin-bottom: 18px;
        }}

        .quant-title {{
            font-size: 2.12rem;
            line-height: 1.08;
            font-weight: 850;
            letter-spacing: -0.035em;
            color: var(--text);
            margin-bottom: 8px;
        }}

        .quant-subtitle {{
            color: var(--muted);
            font-size: 1.02rem;
            max-width: 1120px;
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
            font-weight: 700;
        }}

        .input-panel {{
            border: 1px solid rgba(139, 61, 255, 0.26);
            background: linear-gradient(145deg, rgba(11, 31, 74, 0.78), rgba(6, 24, 68, 0.62));
            border-radius: 22px;
            padding: 18px 18px 8px 18px;
            box-shadow: 0 15px 38px rgba(0, 0, 0, 0.22);
            margin: 4px 0 20px 0;
        }}

        .section-title {{
            font-size: 1.06rem;
            letter-spacing: 0.03em;
            font-weight: 820;
            color: var(--text);
            margin-bottom: 2px;
        }}

        .section-caption {{
            color: var(--muted);
            font-size: 0.90rem;
            margin-bottom: 12px;
        }}

        .metric-card {{
            min-height: 142px;
            border: 1px solid rgba(139, 61, 255, 0.28);
            background: linear-gradient(145deg, rgba(11, 31, 74, 0.94), rgba(6, 24, 68, 0.92));
            border-radius: 22px;
            padding: 18px 18px 15px 18px;
            box-shadow: 0 16px 35px rgba(0, 0, 0, 0.24);
            position: relative;
            overflow: hidden;
        }}

        .metric-card::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(139, 61, 255, 0.22), transparent 42%);
            pointer-events: none;
        }}

        .metric-label {{
            position: relative;
            z-index: 1;
            color: var(--muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
            margin-bottom: 8px;
        }}

        .metric-value {{
            position: relative;
            z-index: 1;
            color: var(--text);
            font-size: 1.34rem;
            line-height: 1.12;
            font-weight: 850;
            letter-spacing: -0.025em;
            overflow-wrap: anywhere;
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
        .metric-orange .metric-value {{ color: var(--orange); }}

        .definition-card {{
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(11, 31, 74, 0.52);
            border-radius: 18px;
            padding: 15px 17px;
            height: 100%;
        }}

        .definition-card b {{ color: var(--text); }}
        .definition-card span {{ color: var(--muted); font-size: 0.90rem; }}

        div[data-testid="stAlert"] {{
            border-radius: 16px;
            border: 1px solid rgba(139, 61, 255, 0.25);
            background: rgba(11, 31, 74, 0.85);
        }}

        .stButton > button, .stFormSubmitButton > button {{
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(183, 140, 255, 0.65);
            background: linear-gradient(90deg, var(--primary), #5F7CFF);
            color: white;
            font-weight: 850;
            box-shadow: 0 12px 28px rgba(139, 61, 255, 0.28);
        }}

        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            border-color: var(--cyan);
            filter: brightness(1.08);
        }}

        div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(139, 61, 255, 0.18);
        }}

        div[data-baseweb="tab-list"] {{ gap: 8px; }}

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

        hr {{ border-color: rgba(139, 61, 255, 0.20) !important; }}
        .small-muted {{ color: var(--muted); font-size: 0.88rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def panel_start(title: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="input-panel">
            <div class="section-title">{title}</div>
            <div class="section-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


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


def survival_tone(pct: float) -> str:
    if pct >= 90:
        return "good"
    if pct >= 70:
        return "warn"
    return "bad"


# ============================================================
# Plot helpers
# ============================================================

def apply_plot_theme(fig: go.Figure, *, y_currency: bool = True, x_currency: bool = False) -> go.Figure:
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
        margin={"l": 72, "r": 36, "t": 68, "b": 54},
        separators=",.",
    )
    fig.update_xaxes(gridcolor="rgba(184, 196, 216, 0.12)", zerolinecolor="rgba(184, 196, 216, 0.18)")
    fig.update_yaxes(gridcolor="rgba(184, 196, 216, 0.12)", zerolinecolor="rgba(184, 196, 216, 0.18)")
    if y_currency:
        fig.update_yaxes(tickprefix="$", separatethousands=True, tickformat=",.0f")
    else:
        fig.update_yaxes(tickprefix="", separatethousands=True, tickformat=",.0f")
    if x_currency:
        fig.update_xaxes(tickprefix="$", separatethousands=True, tickformat=",.0f")
    return fig


def add_value_annotation(fig: go.Figure, x: float, y_clp: float, text: str, color: str = COLOR_CYAN, yshift: int = 12) -> None:
    fig.add_annotation(
        x=x,
        y=y_clp,
        text=text,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=color,
        font={"color": color, "size": 12},
        bgcolor="rgba(6, 24, 68, 0.86)",
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        yshift=yshift,
    )


def plot_percentile_fan(tabla: pd.DataFrame, edad_inicio_retiro: int, target_clp: Optional[float]) -> go.Figure:
    fig = go.Figure()

    for col, name in [
        ("p95_mm", "p95"),
        ("p75_mm", "p75"),
        ("p50_mediana_mm", "p50 / mediana"),
        ("p25_mm", "p25"),
        ("p5_mm", "p5"),
        ("media_mm", "media"),
    ]:
        y_clp = tabla[col] * 1_000_000
        fig.add_trace(
            go.Scatter(
                x=tabla["edad"],
                y=y_clp,
                name=name,
                mode="lines",
                line={"width": 3 if name in {"p50 / mediana", "media"} else 2, "color": PERCENTILE_COLORS[name]},
                hovertemplate="Edad %{x}<br>Patrimonio $%{y:,.0f}<extra>" + name + "</extra>",
            )
        )

    fig.add_vline(
        x=edad_inicio_retiro,
        line_dash="dash",
        line_color=COLOR_CYAN,
        annotation_text="inicio retiro",
        annotation_position="top left",
    )

    if target_clp is not None:
        fig.add_hline(
            y=target_clp,
            line_dash="dot",
            line_color=COLOR_WARN,
            annotation_text=f"meta {fmt_clp(target_clp)}",
            annotation_position="top left",
        )

    # Números clave sobre el gráfico
    row_ret = tabla.loc[tabla["edad"] == edad_inicio_retiro]
    if not row_ret.empty:
        ret_p50_clp = float(row_ret.iloc[0]["p50_mediana_mm"] * 1_000_000)
        add_value_annotation(fig, edad_inicio_retiro, ret_p50_clp, f"P50 retiro<br>{fmt_clp(ret_p50_clp)}", COLOR_CYAN)

    final_row = tabla.iloc[-1]
    final_age = int(final_row["edad"])
    final_p50_clp = float(final_row["p50_mediana_mm"] * 1_000_000)
    final_p5_clp = float(final_row["p5_mm"] * 1_000_000)
    final_p95_clp = float(final_row["p95_mm"] * 1_000_000)
    add_value_annotation(fig, final_age, final_p50_clp, f"P50 final<br>{fmt_clp(final_p50_clp)}", COLOR_PRIMARY_2)
    add_value_annotation(fig, final_age, final_p5_clp, f"P5<br>{fmt_clp(final_p5_clp)}", COLOR_BAD, yshift=-24)
    add_value_annotation(fig, final_age, final_p95_clp, f"P95<br>{fmt_clp(final_p95_clp)}", COLOR_CYAN, yshift=24)

    fig.update_layout(
        title="Evolución del patrimonio por edad",
        xaxis_title="Edad",
        yaxis_title="Patrimonio (CLP)",
        hovermode="x unified",
        legend_title="Serie",
    )
    return apply_plot_theme(fig)


def plot_cashflow_schedule(result: dict) -> go.Figure:
    inputs = result["inputs"]
    edad_inicial = inputs["edad_inicial"]
    months = inputs["months"]
    x = edad_inicial + (np.arange(months) + 1) / 12

    ahorro_clp = result["monthly_savings_mm"].mean(axis=0) * 1_000_000
    retiro_clp = -result["withdrawal_schedule_mm"] * 1_000_000
    recurrente_clp = result["recurring_cashflows_mm"] * 1_000_000
    extra_clp = result["lump_sums_mm"] * 1_000_000
    neto_clp = ahorro_clp + retiro_clp + recurrente_clp + extra_clp

    fig = go.Figure()
    series = [
        (ahorro_clp, "Ahorro promedio", COLOR_GOOD),
        (retiro_clp, "Retiro fijo", COLOR_BAD),
        (recurrente_clp, "Ingresos/egresos recurrentes", COLOR_CYAN),
        (extra_clp, "Flujos esporádicos", COLOR_ORANGE),
        (neto_clp, "Flujo neto antes de retorno", COLOR_PRIMARY_2),
    ]
    for y, name, color in series:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line={"width": 3 if name == "Flujo neto antes de retorno" else 2, "color": color},
                hovertemplate="Edad %{x:.1f}<br>Flujo $%{y:,.0f}<extra>" + name + "</extra>",
            )
        )

    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.25)")
    fig.add_vline(x=inputs["edad_inicio_retiro"], line_dash="dash", line_color=COLOR_CYAN, annotation_text="inicio retiro")

    # Etiquetas de números al final del calendario
    for y, name, color in [
        (ahorro_clp, "ahorro", COLOR_GOOD),
        (retiro_clp, "retiro", COLOR_BAD),
        (recurrente_clp, "recurrente", COLOR_CYAN),
    ]:
        nonzero = np.where(np.abs(y) > 1)[0]
        if len(nonzero) > 0:
            idx = nonzero[-1]
            add_value_annotation(fig, float(x[idx]), float(y[idx]), f"{name}<br>{fmt_clp(y[idx])}", color, yshift=10)

    fig.update_layout(
        title="Calendario mensual de flujos: plata que entra y sale",
        xaxis_title="Edad",
        yaxis_title="Flujo mensual (CLP)",
        hovermode="x unified",
        legend_title="Flujo",
    )
    return apply_plot_theme(fig)


def plot_sample_paths(result: dict, n_sample: int = 300) -> go.Figure:
    paths = result["paths_mm"]
    inputs = result["inputs"]
    edad_inicial = inputs["edad_inicial"]
    edad_inicio_retiro = inputs["edad_inicio_retiro"]
    months = inputs["months"]
    target_clp = inputs["target_mm"] * 1_000_000 if inputs["target_mm"] is not None else None

    rng = np.random.default_rng(2026)
    n_sample = min(n_sample, paths.shape[0])
    idx = rng.choice(paths.shape[0], size=n_sample, replace=False)
    sample_clp = paths[idx] * 1_000_000
    x = edad_inicial + np.arange(months + 1) / 12

    fig = go.Figure()
    for row in sample_clp:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=row,
                mode="lines",
                line={"width": 0.75, "color": COLOR_PRIMARY_2},
                opacity=0.14,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    p50 = np.percentile(paths, 50, axis=0) * 1_000_000
    p5 = np.percentile(paths, 5, axis=0) * 1_000_000
    p95 = np.percentile(paths, 95, axis=0) * 1_000_000
    fig.add_trace(go.Scatter(x=x, y=p50, mode="lines", name="mediana", line={"width": 3.2, "color": COLOR_CYAN}))
    fig.add_trace(go.Scatter(x=x, y=p5, mode="lines", name="p5", line={"width": 2, "dash": "dot", "color": COLOR_BAD}))
    fig.add_trace(go.Scatter(x=x, y=p95, mode="lines", name="p95", line={"width": 2, "dash": "dot", "color": COLOR_PRIMARY_2}))

    fig.add_vline(x=edad_inicio_retiro, line_dash="dash", line_color=COLOR_CYAN, annotation_text="inicio retiro")
    if target_clp is not None:
        fig.add_hline(y=target_clp, line_dash="dot", line_color=COLOR_WARN, annotation_text=f"meta {fmt_clp(target_clp)}")

    ret_idx = int(round((edad_inicio_retiro - edad_inicial) * 12))
    ret_idx = min(max(ret_idx, 0), len(x) - 1)
    add_value_annotation(fig, edad_inicio_retiro, float(p50[ret_idx]), f"P50 retiro<br>{fmt_clp(p50[ret_idx])}", COLOR_CYAN)
    add_value_annotation(fig, int(inputs["edad_final"]), float(p50[-1]), f"P50 final<br>{fmt_clp(p50[-1])}", COLOR_PRIMARY_2)

    fig.update_layout(
        title=f"Paths Monte Carlo simulados ({fmt_int_dot(n_sample)} paths mostrados)",
        xaxis_title="Edad",
        yaxis_title="Patrimonio (CLP)",
        hovermode="x unified",
    )
    return apply_plot_theme(fig)


def plot_final_distribution(result: dict) -> go.Figure:
    final_wealth_clp = result["final_wealth_mm"] * 1_000_000
    edad_final = result["inputs"]["edad_final"]
    p5, p50, p95 = np.percentile(final_wealth_clp, [5, 50, 95])

    fig = px.histogram(
        x=final_wealth_clp,
        nbins=80,
        labels={"x": "Patrimonio final (CLP)", "y": "Frecuencia"},
        title=f"Distribución del patrimonio final a los {edad_final} años",
        color_discrete_sequence=[COLOR_PRIMARY],
    )
    fig.update_traces(marker_line_width=0.3, marker_line_color="rgba(255,255,255,0.20)")
    fig.update_layout(showlegend=False)
    for value, label, color in [(p5, "P5", COLOR_BAD), (p50, "P50", COLOR_CYAN), (p95, "P95", COLOR_PRIMARY_2)]:
        fig.add_vline(x=value, line_dash="dash", line_color=color, annotation_text=f"{label}: {fmt_clp(value)}")
    return apply_plot_theme(fig, y_currency=False, x_currency=True)


def plot_ruin_distribution(result: dict) -> go.Figure:
    ruin_age = result["ruin_age"]
    data = ruin_age[~np.isnan(ruin_age)]
    if len(data) == 0:
        fig = go.Figure()
        fig.update_layout(title="No hubo agotamiento de patrimonio en las simulaciones")
        return apply_plot_theme(fig, y_currency=False, x_currency=False)

    median_age = float(np.median(data))
    fig = px.histogram(
        x=data,
        nbins=40,
        labels={"x": "Edad de agotamiento", "y": "Frecuencia"},
        title="Distribución de edad de agotamiento del patrimonio",
        color_discrete_sequence=[COLOR_BAD],
    )
    fig.update_traces(marker_line_width=0.3, marker_line_color="rgba(255,255,255,0.20)")
    fig.update_layout(showlegend=False)
    fig.add_vline(x=median_age, line_dash="dash", line_color=COLOR_CYAN, annotation_text=f"mediana: {median_age:,.1f} años")
    return apply_plot_theme(fig, y_currency=False, x_currency=False)


def make_display_table(tabla: pd.DataFrame) -> pd.DataFrame:
    display = tabla.copy()
    rename_map = {
        "media_mm": "media_clp",
        "p5_mm": "p5_clp",
        "p25_mm": "p25_clp",
        "p50_mediana_mm": "p50_mediana_clp",
        "p75_mm": "p75_clp",
        "p95_mm": "p95_clp",
        "ahorro_prom_mensual_mm": "ahorro_prom_mensual_clp",
        "retiro_prom_mensual_mm": "retiro_prom_mensual_clp",
        "ingreso_recurrente_prom_mensual_mm": "ingreso_recurrente_prom_mensual_clp",
        "egreso_recurrente_prom_mensual_mm": "egreso_recurrente_prom_mensual_clp",
        "flujo_recurrente_neto_mensual_mm": "flujo_recurrente_neto_mensual_clp",
        "aporte_extra_anual_mm": "aporte_extra_anual_clp",
    }
    for old, new in rename_map.items():
        display[new] = display[old].apply(lambda x: fmt_clp_from_mm(float(x)))
    display["prob_sobre_target"] = display["prob_sobre_target"].apply(lambda x: fmt_pct(float(x), 2))
    display["prob_sobre_cero"] = display["prob_sobre_cero"].apply(lambda x: fmt_pct(float(x), 2))
    cols = [
        "edad",
        "media_clp",
        "p5_clp",
        "p50_mediana_clp",
        "p95_clp",
        "prob_sobre_target",
        "prob_sobre_cero",
        "ahorro_prom_mensual_clp",
        "retiro_prom_mensual_clp",
        "ingreso_recurrente_prom_mensual_clp",
        "flujo_recurrente_neto_mensual_clp",
        "aporte_extra_anual_clp",
    ]
    return display[cols]


def make_numeric_csv_table(tabla: pd.DataFrame) -> pd.DataFrame:
    out = tabla.copy()
    for col in list(out.columns):
        if col.endswith("_mm"):
            out[col.replace("_mm", "_clp")] = (out[col] * 1_000_000).round(0)
    return out


# ============================================================
# App Streamlit
# ============================================================

st.set_page_config(
    page_title="Monte Carlo Retiro Fijo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

st.markdown(
    f"""
    <div class="quant-hero">
        <div class="quant-title">Monte Carlo patrimonial: acumulación + retiro fijo</div>
        <div class="quant-subtitle">
            Simulador en CLP para probar desde qué edad dejas de ahorrar, cuánto retiras fijo al mes,
            qué ingresos recurrentes entran después y si el patrimonio llega vivo a los {EDAD_FINAL_FIJA} años.
        </div>
        <div class="quant-pill-row">
            <div class="quant-pill">Montos con todos los ceros</div>
            <div class="quant-pill">Edad final fija: {EDAD_FINAL_FIJA}</div>
            <div class="quant-pill">Jubilación y arriendos recurrentes</div>
            <div class="quant-pill">Flujos esporádicos</div>
            <div class="quant-pill">Percentiles Monte Carlo</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Inputs en pantalla principal
# ============================================================

with st.form("formulario_simulacion"):
    panel_start("1. Supuestos base", "Edad final fija en 90 años. Todos los montos se ingresan en CLP, con todos los ceros.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        edad_inicial = st.number_input("Edad inicial", min_value=18, max_value=89, value=28, step=1)
    with c2:
        edad_final = EDAD_FINAL_FIJA
        st.number_input("Edad final", min_value=EDAD_FINAL_FIJA, max_value=EDAD_FINAL_FIJA, value=EDAD_FINAL_FIJA, step=1, disabled=True)
    with c3:
        edad_inicio_retiro = st.number_input(
            "Edad inicio retiro",
            min_value=int(edad_inicial),
            max_value=EDAD_FINAL_FIJA,
            value=min(40, EDAD_FINAL_FIJA),
            step=1,
            help="Desde esta edad el ahorro mensual se vuelve cero y comienza el retiro fijo mensual.",
        )
    with c4:
        target_clp = st.number_input(
            "Meta patrimonial CLP",
            min_value=0,
            value=1_000_000_000,
            step=50_000_000,
            format="%d",
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        initial_capital_clp = st.number_input("Capital inicial CLP", min_value=0, value=50_000_000, step=1_000_000, format="%d")
    with c6:
        withdrawal_monthly_clp = st.number_input("Retiro mensual fijo CLP", min_value=0, value=3_000_000, step=100_000, format="%d")
    with c7:
        withdrawal_indexed_to_inflation = st.checkbox("Indexar retiro por inflación", value=False)
    with c8:
        inflation_annual_pct = st.number_input("Inflación anual indexación (%)", min_value=0.0, value=3.0, step=0.25)
    panel_end()

    panel_start("2. Ahorro mensual antes del retiro", "Se modela como distribución triangular: mínimo, más probable y máximo. Desde la edad de retiro el ahorro queda en cero.")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        monthly_saving_min_clp = st.number_input("Ahorro mínimo CLP", min_value=0, value=2_500_000, step=100_000, format="%d")
    with a2:
        monthly_saving_mode_clp = st.number_input("Ahorro más probable CLP", min_value=0, value=3_000_000, step=100_000, format="%d")
    with a3:
        monthly_saving_max_clp = st.number_input("Ahorro máximo CLP", min_value=0, value=3_500_000, step=100_000, format="%d")
    with a4:
        contribution_timing_es = st.selectbox("Timing ahorro", ["Fin de mes", "Inicio de mes"], index=0)
    panel_end()

    panel_start("3. Ingresos o egresos mensuales recurrentes", "Ejemplos: jubilación, arriendo de propiedades, dividendo, gastos familiares. Se aplican todos los meses desde la edad indicada.")
    default_recurring_df = pd.DataFrame(
        {
            "descripcion": ["Jubilación", "Arriendo propiedades"],
            "tipo": ["Ingreso", "Ingreso"],
            "edad_inicio": [65, 40],
            "edad_fin": [90, 90],
            "monto_mensual_clp": [0, 0],
        }
    )
    recurring_df = st.data_editor(
        default_recurring_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "descripcion": st.column_config.TextColumn("Descripción"),
            "tipo": st.column_config.SelectboxColumn("Tipo", options=["Ingreso", "Egreso"], required=True),
            "edad_inicio": st.column_config.NumberColumn("Edad inicio", min_value=int(edad_inicial), max_value=EDAD_FINAL_FIJA, step=1),
            "edad_fin": st.column_config.NumberColumn("Edad fin", min_value=int(edad_inicial), max_value=EDAD_FINAL_FIJA, step=1),
            "monto_mensual_clp": st.column_config.NumberColumn("Monto mensual CLP", min_value=0, step=100_000, format="%d"),
        },
        key="recurring_editor",
    )
    panel_end()

    panel_start("4. Flujos esporádicos", "Plata que entra o sale una sola vez: bono, venta de activo, pie de propiedad, gasto grande, herencia, prepago, etc.")
    default_lump_df = pd.DataFrame(
        {
            "descripcion": ["Ejemplo bono / venta activo"],
            "tipo": ["Ingreso"],
            "edad_evento": [40],
            "monto_clp": [0],
        }
    )
    lump_df = st.data_editor(
        default_lump_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "descripcion": st.column_config.TextColumn("Descripción"),
            "tipo": st.column_config.SelectboxColumn("Tipo", options=["Ingreso", "Egreso"], required=True),
            "edad_evento": st.column_config.NumberColumn("Edad evento", min_value=int(edad_inicial), max_value=EDAD_FINAL_FIJA, step=1),
            "monto_clp": st.column_config.NumberColumn("Monto CLP", min_value=0, step=1_000_000, format="%d"),
        },
        key="lump_editor",
    )
    panel_end()

    panel_start("5. Retorno, riesgo y simulación", "El modo mensual IID captura mejor el riesgo de secuencia durante el retiro; el anual suavizado replica mejor el código original.")
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        return_model_es = st.selectbox("Modelo retorno", ["Mensual IID más realista para retiro", "Anual suavizado como código original"], index=0)
    with r2:
        annual_return_mean_pct = st.number_input("Retorno anual esperado (%)", value=10.0, step=0.5)
    with r3:
        annual_return_std_pct = st.number_input("Volatilidad anual (%)", min_value=0.1, value=5.0, step=0.5)
    with r4:
        annual_return_low_pct = st.number_input("Mínimo truncado (%)", value=-55.0, step=1.0)
    with r5:
        annual_return_high_pct = st.number_input("Máximo truncado (%)", value=25.0, step=1.0)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        n_paths = st.number_input("Simulaciones", min_value=1_000, max_value=100_000, value=30_000, step=5_000, format="%d")
    with s2:
        seed = st.number_input("Seed", min_value=0, max_value=999_999, value=123, step=1)
    with s3:
        floor_zero = st.checkbox("Patrimonio no negativo", value=True)
    with s4:
        mean_is_effective = st.checkbox("Calibrar media truncada", value=True)

    submitted = st.form_submit_button("Simular escenario", type="primary")
    panel_end()


# ============================================================
# Parseo de inputs
# ============================================================

def parse_lump_events(df: pd.DataFrame, edad_inicial_: int, edad_final_: int) -> tuple[tuple[int, float], ...]:
    events: list[tuple[int, float]] = []
    if df is None or df.empty:
        return tuple(events)
    for _, row in df.dropna(subset=["edad_evento", "monto_clp"]).iterrows():
        amount_clp = float(row.get("monto_clp", 0) or 0)
        if amount_clp == 0:
            continue
        age = float(row["edad_evento"])
        if age < edad_inicial_ or age > edad_final_:
            continue
        sign = 1.0 if str(row.get("tipo", "Ingreso")) == "Ingreso" else -1.0
        month_idx = int(round((age - edad_inicial_) * 12)) + 1
        month_idx = min(max(month_idx, 1), (edad_final_ - edad_inicial_) * 12)
        events.append((month_idx, sign * clp_to_mm(amount_clp)))
    return tuple(events)


def parse_recurring_events(df: pd.DataFrame, edad_inicial_: int, edad_final_: int) -> tuple[tuple[float, Optional[float], float, str], ...]:
    events: list[tuple[float, Optional[float], float, str]] = []
    if df is None or df.empty:
        return tuple(events)
    for _, row in df.dropna(subset=["edad_inicio", "monto_mensual_clp"]).iterrows():
        amount_clp = float(row.get("monto_mensual_clp", 0) or 0)
        if amount_clp == 0:
            continue
        start_age = float(row["edad_inicio"])
        end_age_raw = row.get("edad_fin", edad_final_)
        end_age = float(end_age_raw) if pd.notna(end_age_raw) else float(edad_final_)
        start_age = min(max(start_age, edad_inicial_), edad_final_)
        end_age = min(max(end_age, edad_inicial_), edad_final_)
        if end_age <= start_age:
            continue
        sign = 1.0 if str(row.get("tipo", "Ingreso")) == "Ingreso" else -1.0
        description = str(row.get("descripcion", "Flujo recurrente"))
        events.append((start_age, end_age, sign * clp_to_mm(amount_clp), description))
    return tuple(events)


if submitted:
    errors = []
    if not (monthly_saving_min_clp <= monthly_saving_mode_clp <= monthly_saving_max_clp):
        errors.append("Debe cumplirse: ahorro mínimo <= ahorro más probable <= ahorro máximo.")
    if not (annual_return_low_pct < annual_return_mean_pct < annual_return_high_pct):
        errors.append("El retorno esperado debe estar entre el mínimo y el máximo truncado.")
    if edad_inicio_retiro > EDAD_FINAL_FIJA:
        errors.append("La edad de inicio de retiro no puede ser mayor que 90.")

    if errors:
        for msg in errors:
            st.error(msg)
        st.stop()

    lump_events = parse_lump_events(lump_df, int(edad_inicial), EDAD_FINAL_FIJA)
    recurring_events = parse_recurring_events(recurring_df, int(edad_inicial), EDAD_FINAL_FIJA)
    contribution_timing = "end" if contribution_timing_es == "Fin de mes" else "begin"
    withdrawal_timing = "end"
    return_model = "monthly_iid" if return_model_es.startswith("Mensual") else "annual_smooth"

    try:
        with st.spinner("Simulando escenarios..."):
            result = monte_carlo_accumulation_withdrawal_mm(
                edad_inicial=int(edad_inicial),
                edad_final=EDAD_FINAL_FIJA,
                edad_inicio_retiro=int(edad_inicio_retiro),
                n_paths=int(n_paths),
                initial_capital_mm=clp_to_mm(initial_capital_clp),
                annual_return_mean=float(annual_return_mean_pct) / 100,
                annual_return_std=float(annual_return_std_pct) / 100,
                annual_return_low=float(annual_return_low_pct) / 100,
                annual_return_high=float(annual_return_high_pct) / 100,
                monthly_saving_min_mm=clp_to_mm(monthly_saving_min_clp),
                monthly_saving_mode_mm=clp_to_mm(monthly_saving_mode_clp),
                monthly_saving_max_mm=clp_to_mm(monthly_saving_max_clp),
                withdrawal_monthly_mm=clp_to_mm(withdrawal_monthly_clp),
                contribution_timing=contribution_timing,
                withdrawal_timing=withdrawal_timing,
                target_mm=clp_to_mm(target_clp),
                seed=int(seed),
                mean_is_effective=bool(mean_is_effective),
                lump_sum_events=lump_events,
                recurring_monthly_events=recurring_events,
                floor_zero=bool(floor_zero),
                return_model=return_model,
                withdrawal_indexed_to_inflation=bool(withdrawal_indexed_to_inflation),
                inflation_annual=float(inflation_annual_pct) / 100,
            )
            tabla = tabla_monte_carlo_por_edad(result)
            st.session_state["mc_result"] = result
            st.session_state["mc_tabla"] = tabla
            st.session_state["mc_return_model_es"] = return_model_es
            st.session_state["mc_target_clp"] = target_clp
            st.session_state["mc_recurring_df"] = recurring_df
            st.session_state["mc_lump_df"] = lump_df
    except Exception as exc:
        st.error(f"Error en la simulación: {exc}")
        st.stop()

if "mc_result" not in st.session_state:
    st.info("Ajusta los supuestos y presiona **Simular escenario**.")
    st.stop()

result = st.session_state["mc_result"]
tabla = st.session_state["mc_tabla"]
return_model_es = st.session_state["mc_return_model_es"]
target_clp = st.session_state["mc_target_clp"]

# ============================================================
# KPIs
# ============================================================

summary = result["summary"].set_index("metric")
final_p50_mm = float(summary.loc["p50", "final_wealth_mm"])
ret_p50_mm = float(summary.loc["p50", "wealth_at_retirement_mm"])
prob_no_ruin = float(result["prob_no_ruin"] * 100)
prob_target_ret = float(result["prob_reach_target_at_retirement"] * 100)
prob_target_final = float(result["prob_reach_target_final"] * 100)
prob_grow = float(result["prob_final_above_retirement_wealth"] * 100)
median_ruin_age = result["median_ruin_age"]
prob_ruin = 100 - prob_no_ruin

st.markdown("### Resultado del escenario")

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card(
        "Patrimonio mediano al iniciar retiro",
        fmt_clp_from_mm(ret_p50_mm),
        f"Edad {result['inputs']['edad_inicio_retiro']}. Es el P50 justo cuando dejas de ahorrar.",
        "primary",
    )
with k2:
    metric_card(
        "Patrimonio mediano a los 90",
        fmt_clp_from_mm(final_p50_mm),
        "P50 final después de retiros, jubilación, arriendos y flujos esporádicos.",
        "cyan",
    )
with k3:
    metric_card(
        "Probabilidad de no agotar patrimonio",
        fmt_pct(prob_no_ruin),
        "Porcentaje de simulaciones que nunca llegan a cero antes de los 90.",
        survival_tone(prob_no_ruin),
    )
with k4:
    metric_card(
        "Probabilidad de quiebre",
        fmt_pct(prob_ruin),
        "Complemento de la probabilidad de no agotarse.",
        "bad" if prob_ruin > 20 else "warn" if prob_ruin > 5 else "good",
    )

st.write("")
k5, k6, k7, k8 = st.columns(4)
with k5:
    metric_card(
        "Llegar a la meta al iniciar retiro",
        fmt_pct(prob_target_ret),
        f"Meta: {fmt_clp(target_clp)} justo al cortar el ahorro.",
        "primary",
    )
with k6:
    metric_card(
        "Terminar sobre la meta a los 90",
        fmt_pct(prob_target_final),
        f"Meta: {fmt_clp(target_clp)} después de toda la etapa de retiro.",
        "cyan",
    )
with k7:
    metric_card(
        "Gasto bruto de retiro hasta los 90",
        fmt_clp_from_mm(result["total_withdrawal_requested_mm"]),
        "Suma de todos los retiros mensuales programados, sin descontar jubilación ni arriendos.",
        "orange",
    )
with k8:
    metric_card(
        "Si falla, edad mediana de agotamiento",
        "No se agota" if np.isnan(median_ruin_age) else f"{median_ruin_age:,.1f} años".replace(".", ","),
        "Solo mira las simulaciones que sí llegan a cero; no es la edad esperada de todos los escenarios.",
        "good" if np.isnan(median_ruin_age) else "bad",
    )

st.write("")
k9, k10, k11, k12 = st.columns(4)
with k9:
    metric_card(
        "Ingresos recurrentes externos hasta los 90",
        fmt_clp_from_mm(result["total_recurring_inflows_mm"]),
        "Jubilación, arriendos u otros ingresos mensuales cargados en la tabla.",
        "good",
    )
with k10:
    metric_card(
        "Egresos recurrentes externos hasta los 90",
        fmt_clp_from_mm(result["total_recurring_outflows_mm"]),
        "Costos mensuales extra si agregaste egresos recurrentes.",
        "bad" if result["total_recurring_outflows_mm"] > 0 else "primary",
    )
with k11:
    metric_card(
        "Necesidad neta financiada con patrimonio",
        fmt_clp_from_mm(result["total_net_cash_need_mm"]),
        "Retiro bruto - ingresos recurrentes + egresos recurrentes.",
        "warn",
    )
with k12:
    metric_card(
        "Probabilidad de crecer pese al retiro",
        fmt_pct(prob_grow),
        "% donde el patrimonio final supera al patrimonio del inicio del retiro.",
        survival_tone(prob_grow) if not np.isnan(prob_grow) else "primary",
    )

st.caption(
    f"Media efectiva anual simulada: {result['inputs']['effective_truncated_mean_annualized'] * 100:,.2f}%. "
    f"Modelo de retorno: {return_model_es}. Todos los montos se muestran en CLP."
)

with st.expander("Qué significa cada métrica", expanded=False):
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(
            """
            <div class="definition-card"><b>Meta al iniciar retiro</b><br>
            <span>Porcentaje de simulaciones donde el patrimonio es mayor o igual a la meta justo cuando dejas de ahorrar.</span></div>
            """,
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            """
            <div class="definition-card"><b>Meta a los 90</b><br>
            <span>Porcentaje de simulaciones donde terminas sobre la meta después de retirar mensualmente y recibir ingresos externos.</span></div>
            """,
            unsafe_allow_html=True,
        )
    with d3:
        st.markdown(
            """
            <div class="definition-card"><b>Edad mediana de agotamiento</b><br>
            <span>Se calcula solo con las simulaciones que quebraron. Si pocas quiebran, esta edad no representa el escenario central.</span></div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Percentiles", "Flujos", "Paths", "Distribución final", "Agotamiento", "Tablas"]
)

with tab1:
    st.plotly_chart(plot_percentile_fan(tabla, int(result["inputs"]["edad_inicio_retiro"]), float(target_clp)), use_container_width=True)

with tab2:
    st.plotly_chart(plot_cashflow_schedule(result), use_container_width=True)
    if result.get("recurring_event_rows"):
        st.write("Flujos recurrentes aplicados")
        rec_events = pd.DataFrame(result["recurring_event_rows"])
        rec_events["monto_mensual_clp"] = rec_events["monto_mensual_mm"].apply(fmt_clp_from_mm)
        st.dataframe(rec_events[["descripcion", "edad_inicio", "edad_fin", "monto_mensual_clp", "mes_inicio", "mes_fin"]], use_container_width=True)

with tab3:
    n_sample = st.slider("Paths a mostrar", min_value=50, max_value=1_000, value=300, step=50)
    st.plotly_chart(plot_sample_paths(result, n_sample=n_sample), use_container_width=True)

with tab4:
    st.plotly_chart(plot_final_distribution(result), use_container_width=True)

with tab5:
    st.plotly_chart(plot_ruin_distribution(result), use_container_width=True)
    if result["prob_no_ruin"] == 1:
        st.success("En este escenario, ningún path agotó patrimonio dentro del horizonte simulado hasta los 90 años.")
    else:
        st.warning(
            f"En este escenario, {fmt_pct((1 - result['prob_no_ruin']) * 100)} de los paths agotó patrimonio "
            "dentro del horizonte simulado."
        )

with tab6:
    st.write("Tabla por edad con montos en CLP")
    display_table = make_display_table(tabla)
    st.dataframe(display_table, use_container_width=True)

    st.write("Resumen final con montos en CLP")
    summary_display = result["summary"].copy()
    for col in ["final_wealth_mm", "wealth_at_retirement_mm", "total_savings_mm"]:
        summary_display[col.replace("_mm", "_clp")] = summary_display[col].apply(fmt_clp_from_mm)
    st.dataframe(summary_display[["metric", "final_wealth_clp", "wealth_at_retirement_clp", "total_savings_clp"]], use_container_width=True)

    csv_tabla = make_numeric_csv_table(tabla).to_csv(index=False).encode("utf-8")
    csv_summary = summary_display.to_csv(index=False).encode("utf-8")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Descargar tabla por edad CSV",
            data=csv_tabla,
            file_name="tabla_montecarlo_por_edad_clp.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Descargar resumen CSV",
            data=csv_summary,
            file_name="resumen_montecarlo_clp.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Nota: esto es una herramienta de simulación, no una recomendación financiera. "
    "El motor mantiene cálculos internos en MM CLP para estabilidad numérica, pero la interfaz muestra los montos en CLP con todos los ceros."
)
