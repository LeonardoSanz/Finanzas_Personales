from __future__ import annotations

from typing import Optional
import os
import io
import json
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from montecarlo_engine import (
    monte_carlo_accumulation_withdrawal_mm,
    tabla_monte_carlo_por_edad,
    required_capital_matrix_mm,
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


def fmt_mm_from_clp(value_clp: float | int, decimals: int = 1) -> str:
    """Muestra la equivalencia en millones con separadores locales."""
    if value_clp is None or (isinstance(value_clp, float) and np.isnan(value_clp)):
        return "N/A"
    value_mm = float(value_clp) / 1_000_000
    if abs(value_mm - round(value_mm)) < 1e-9:
        return f"{fmt_int_dot(value_mm)} MM"
    text = f"{value_mm:,.{decimals}f}"
    text = text.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{text} MM"


def parse_clp_value(value, default: int = 0) -> int:
    """
    Convierte entradas tipo '$1.000.000', '1000000', '3 MM' o '3,5 MM' a CLP enteros.
    Mantiene la UI con separadores de miles sin romper los cálculos internos.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return int(default)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(round(float(value)))

    raw = str(value).strip().lower()
    if raw == "":
        return int(default)

    is_mm = ("mm" in raw) or ("mill" in raw)
    keep = "".join(ch for ch in raw if ch.isdigit() or ch in ",.-")
    if keep in {"", "-", ".", ","}:
        return int(default)

    try:
        if is_mm:
            # Para montos escritos como 3,5 MM o 3.5 MM.
            normalized = keep.replace(".", "").replace(",", ".") if "," in keep else keep
            amount = float(normalized) * 1_000_000
        else:
            # CLP sin decimales: puntos/comas se asumen como separadores de miles.
            normalized = keep.replace(".", "").replace(",", "")
            amount = float(normalized)
        return int(round(amount))
    except ValueError:
        return int(default)


def money_text_input(label: str, value: int, *, key: str, help: str | None = None, show_caption: bool = False) -> int:
    """Input de dinero con separadores visibles. Streamlit number_input no agrupa miles de forma consistente."""
    text = st.text_input(
        label,
        value=fmt_int_dot(value),
        key=key,
        help=help,
        placeholder="ej: 1.000.000.000",
    )
    parsed = max(parse_clp_value(text, default=value), 0)
    if show_caption:
        st.caption(f"= {fmt_clp(parsed)} · {fmt_mm_from_clp(parsed)}")
    return parsed


def fmt_pct(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:,.{decimals}f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def mm_series_to_clp(series: pd.Series) -> pd.Series:
    return (series.astype(float) * 1_000_000).round(0).astype("Int64")


def future_value_monthly_real_clp(
    initial_balance_clp: float,
    monthly_contribution_clp: float,
    annual_real_return: float,
    n_months: int,
    contribution_timing: str = "end",
) -> float:
    """
    Valor futuro en pesos de hoy usando retorno real.
    Sirve para estimar una pensión AFP en UF/pesos reales y luego indexarla por inflación.
    """
    initial_balance_clp = max(float(initial_balance_clp), 0.0)
    monthly_contribution_clp = max(float(monthly_contribution_clp), 0.0)
    n_months = max(int(n_months), 0)
    rm = (1 + annual_real_return) ** (1 / 12) - 1

    if n_months == 0:
        return initial_balance_clp
    if abs(rm) < 1e-14:
        return initial_balance_clp + monthly_contribution_clp * n_months

    fv_capital = initial_balance_clp * (1 + rm) ** n_months
    fv_contrib = monthly_contribution_clp * (((1 + rm) ** n_months - 1) / rm)
    if contribution_timing == "begin":
        fv_contrib *= (1 + rm)
    return fv_capital + fv_contrib


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

        .workflow-note {{
            border: 1px solid rgba(0, 209, 255, 0.22);
            background: linear-gradient(90deg, rgba(0, 209, 255, 0.10), rgba(139, 61, 255, 0.08));
            color: var(--muted);
            border-radius: 18px;
            padding: 12px 16px;
            margin: 2px 0 16px 0;
            font-size: 0.92rem;
        }}

        .workflow-note b {{ color: var(--text); }}

        .mini-card {{
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(11, 31, 74, 0.62);
            border-radius: 16px;
            padding: 12px 14px;
            min-height: 72px;
        }}

        .mini-card b {{
            color: var(--text);
            display: block;
            font-size: 0.86rem;
            margin-bottom: 5px;
        }}

        .mini-card span {{
            color: var(--muted);
            font-size: 0.90rem;
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

        .login-shell {{
            min-height: 70vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .login-card {{
            width: min(620px, 100%);
            border: 1px solid rgba(183, 140, 255, 0.42);
            background: linear-gradient(145deg, rgba(11, 31, 74, 0.96), rgba(6, 24, 68, 0.86));
            border-radius: 30px;
            padding: 30px 32px;
            box-shadow: 0 30px 80px rgba(0, 0, 0, 0.42);
        }}

        .login-eyebrow {{
            color: var(--cyan);
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 900;
            margin-bottom: 10px;
        }}

        .login-title {{
            color: var(--text);
            font-size: 2.0rem;
            font-weight: 900;
            letter-spacing: -0.04em;
            margin-bottom: 8px;
        }}

        .login-subtitle {{
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.45;
            margin-bottom: 18px;
        }}

        .mode-strip {{
            border: 1px solid rgba(0, 209, 255, 0.18);
            background: linear-gradient(90deg, rgba(0, 209, 255, 0.09), rgba(139, 61, 255, 0.10));
            border-radius: 22px;
            padding: 15px 18px;
            margin: 2px 0 18px 0;
            display: flex;
            gap: 14px;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
        }}

        .mode-strip-title {{
            color: var(--text);
            font-weight: 900;
            font-size: 1.0rem;
        }}

        .mode-strip-caption {{
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 2px;
        }}

        .step-grid {{
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 18px 0;
        }}

        .step-card {{
            border: 1px solid rgba(139, 61, 255, 0.22);
            background: linear-gradient(145deg, rgba(11, 31, 74, 0.72), rgba(6, 24, 68, 0.54));
            border-radius: 18px;
            padding: 13px 13px 12px 13px;
            min-height: 92px;
        }}

        .step-card strong {{
            display: block;
            color: var(--text);
            font-size: 0.86rem;
            margin-bottom: 4px;
        }}

        .step-card span {{
            color: var(--muted);
            font-size: 0.80rem;
            line-height: 1.25;
        }}

        .advanced-frame {{
            border: 1px solid rgba(0, 209, 255, 0.16);
            background: rgba(2, 8, 31, 0.18);
            border-radius: 28px;
            padding: 18px 18px 8px 18px;
            box-shadow: inset 0 0 0 1px rgba(139, 61, 255, 0.08);
            margin-bottom: 18px;
        }}

        .download-card {{
            border: 1px solid rgba(48, 209, 88, 0.22);
            background: linear-gradient(145deg, rgba(48, 209, 88, 0.08), rgba(0, 209, 255, 0.06));
            border-radius: 20px;
            padding: 15px 16px;
            margin: 8px 0 16px 0;
        }}

        .download-card b {{ color: var(--text); }}
        .download-card span {{ color: var(--muted); font-size: 0.90rem; }}

        @media (max-width: 1100px) {{
            .step-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
        @media (max-width: 650px) {{
            .step-grid {{ grid-template-columns: 1fr; }}
        }}
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
# Seguridad simple por clave
# ============================================================

def get_app_password() -> str:
    """Lee la clave desde Streamlit Secrets o variable de entorno.

    En Streamlit Cloud conviene crear APP_PASSWORD en Secrets.
    Si no existe, queda una clave default editable en este archivo.
    """
    try:
        secret_value = st.secrets.get("APP_PASSWORD", None)
    except Exception:
        secret_value = None
    env_value = os.environ.get("APP_PASSWORD")
    return str(secret_value or env_value or "quant2026")


def password_gate() -> None:
    """Bloquea la app hasta ingresar clave."""
    if st.session_state.get("mc_authenticated", False):
        return

    st.markdown(
        """
        <div class="login-shell">
            <div class="login-card">
                <div class="login-eyebrow">Acceso privado</div>
                <div class="login-title">Simulador patrimonial</div>
                <div class="login-subtitle">
                    Ingresa la clave para abrir el panel Monte Carlo.
                    La app queda protegida antes de mostrar supuestos, resultados o descargas.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=False):
        password = st.text_input("Clave", type="password", placeholder="Ingresa la clave")
        submitted_login = st.form_submit_button("Entrar", type="primary")

    if submitted_login:
        if password == get_app_password():
            st.session_state["mc_authenticated"] = True
            st.rerun()
        else:
            st.error("Clave incorrecta.")
    st.stop()


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
    """
    Gráfico de flujos más legible:
    - Barras positivas: plata que entra cada mes.
    - Barras negativas: plata que sale cada mes.
    - Línea morada: flujo neto mensual recurrente antes del retorno.
    - Diamantes naranjos: flujos esporádicos acumulados durante esa edad/año.

    Se agrupa por edad/año para no mostrar cientos de puntos mensuales que se pisan.
    """
    inputs = result["inputs"]
    edad_inicial = int(inputs["edad_inicial"])
    months = int(inputs["months"])
    ages_monthly = edad_inicial + np.arange(months) / 12
    age_bucket = np.floor(ages_monthly).astype(int)

    savings_mean = result.get("monthly_savings_mean_mm")
    if savings_mean is None:
        old_savings = result.get("monthly_savings_mm")
        savings_mean = np.zeros(months) if old_savings is None else old_savings.mean(axis=0)

    savings_clp = np.asarray(savings_mean, dtype=float) * 1_000_000
    withdrawal_clp = -np.asarray(result["withdrawal_schedule_mm"], dtype=float) * 1_000_000
    recurring = np.asarray(result["recurring_cashflows_mm"], dtype=float) * 1_000_000
    recurring_in_clp = np.maximum(recurring, 0)
    recurring_out_clp = np.minimum(recurring, 0)
    lump_clp = np.asarray(result["lump_sums_mm"], dtype=float) * 1_000_000

    df = pd.DataFrame(
        {
            "edad": age_bucket,
            "ahorro_mensual": savings_clp,
            "retiro_mensual": withdrawal_clp,
            "ingreso_recurrente": recurring_in_clp,
            "egreso_recurrente": recurring_out_clp,
            "extra_anual": lump_clp,
        }
    )
    grouped = df.groupby("edad", as_index=False).agg(
        ahorro_mensual=("ahorro_mensual", "mean"),
        retiro_mensual=("retiro_mensual", "mean"),
        ingreso_recurrente=("ingreso_recurrente", "mean"),
        egreso_recurrente=("egreso_recurrente", "mean"),
        extra_anual=("extra_anual", "sum"),
    )
    grouped["neto_mensual_recurrente"] = (
        grouped["ahorro_mensual"]
        + grouped["retiro_mensual"]
        + grouped["ingreso_recurrente"]
        + grouped["egreso_recurrente"]
    )

    fig = go.Figure()
    bar_series = [
        ("ahorro_mensual", "Ahorro mensual", COLOR_GOOD),
        ("ingreso_recurrente", "Ingresos recurrentes", COLOR_CYAN),
        ("retiro_mensual", "Retiro mensual", COLOR_BAD),
        ("egreso_recurrente", "Egresos recurrentes", COLOR_ORANGE),
    ]
    for col, name, color in bar_series:
        fig.add_trace(
            go.Bar(
                x=grouped["edad"],
                y=grouped[col],
                name=name,
                marker_color=color,
                opacity=0.82,
                hovertemplate="Edad %{x}<br>" + name + " $%{y:,.0f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=grouped["edad"],
            y=grouped["neto_mensual_recurrente"],
            name="Neto mensual antes de retorno",
            mode="lines+markers",
            line={"width": 4, "color": COLOR_PRIMARY_2},
            marker={"size": 7, "color": COLOR_PRIMARY_2},
            hovertemplate="Edad %{x}<br>Neto mensual $%{y:,.0f}<extra></extra>",
        )
    )

    # Los esporádicos son totales anuales, no montos mensuales. Por eso van como diamantes.
    extras = grouped[np.abs(grouped["extra_anual"]) > 1]
    if not extras.empty:
        fig.add_trace(
            go.Scatter(
                x=extras["edad"],
                y=extras["extra_anual"],
                name="Flujos esporádicos del año",
                mode="markers+text",
                marker={"size": 12, "symbol": "diamond", "color": COLOR_ORANGE},
                text=[fmt_clp(v) for v in extras["extra_anual"]],
                textposition="top center",
                hovertemplate="Edad %{x}<br>Flujo esporádico anual $%{y:,.0f}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.28)")
    fig.add_vline(
        x=inputs["edad_inicio_retiro"],
        line_dash="dash",
        line_color=COLOR_CYAN,
        annotation_text="inicio retiro",
        annotation_position="top left",
    )

    # Etiquetas clave al final, separadas para evitar que se pisen.
    if len(grouped) > 0:
        final = grouped.iloc[-1]
        final_age = float(final["edad"])
        add_value_annotation(
            fig,
            final_age,
            float(final["retiro_mensual"]),
            f"retiro<br>{fmt_clp(final['retiro_mensual'])}",
            COLOR_BAD,
            yshift=-34,
        )
        add_value_annotation(
            fig,
            final_age,
            float(final["ingreso_recurrente"]),
            f"ingresos<br>{fmt_clp(final['ingreso_recurrente'])}",
            COLOR_CYAN,
            yshift=28,
        )
        add_value_annotation(
            fig,
            final_age,
            float(final["neto_mensual_recurrente"]),
            f"neto<br>{fmt_clp(final['neto_mensual_recurrente'])}",
            COLOR_PRIMARY_2,
            yshift=0,
        )

    fig.update_layout(
        title="Flujo mensual promedio por edad: entradas, salidas y brecha que cubre el patrimonio",
        xaxis_title="Edad",
        yaxis_title="Flujo mensual / flujo esporádico anual (CLP)",
        hovermode="x unified",
        legend_title="Concepto",
        barmode="relative",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=1.10,
        showarrow=False,
        align="left",
        text="Barras sobre cero = plata que entra. Barras bajo cero = plata que sale. Línea morada = neto mensual recurrente antes del retorno. Diamantes = flujos únicos del año.",
        font={"size": 12, "color": COLOR_MUTED},
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


def plot_distribution_at_age(result: dict, selected_age: int | float | None = None) -> go.Figure:
    """Histograma del patrimonio simulado para la edad elegida."""
    inputs = result["inputs"]
    edad_inicial = int(inputs["edad_inicial"])
    edad_final = int(inputs["edad_final"])
    months = int(inputs["months"])

    if selected_age is None:
        selected_age = edad_final
    selected_age = float(min(max(float(selected_age), float(edad_inicial)), float(edad_final)))
    month_idx = int(round((selected_age - edad_inicial) * 12))
    month_idx = min(max(month_idx, 0), months)
    selected_age_exact = edad_inicial + month_idx / 12

    wealth_clp = result["paths_mm"][:, month_idx].astype(float) * 1_000_000
    p5, p50, p95 = np.percentile(wealth_clp, [5, 50, 95])

    fig = px.histogram(
        x=wealth_clp,
        nbins=80,
        labels={"x": "Patrimonio (CLP)", "y": "Frecuencia"},
        title=f"Distribución del patrimonio a los {selected_age_exact:,.0f} años",
        color_discrete_sequence=[COLOR_PRIMARY],
    )
    fig.update_traces(marker_line_width=0.3, marker_line_color="rgba(255,255,255,0.20)")
    fig.update_layout(showlegend=False)

    # Cortes separados verticalmente para no taparse.
    cuts = [
        (p5, "P5", COLOR_BAD, 1.08),
        (p50, "P50", COLOR_CYAN, 1.00),
        (p95, "P95", COLOR_PRIMARY_2, 0.92),
    ]
    for value, label, color, ypaper in cuts:
        fig.add_vline(x=value, line_dash="dash", line_color=color, line_width=3 if label == "P50" else 2)
        fig.add_annotation(
            x=value,
            y=ypaper,
            xref="x",
            yref="paper",
            text=f"{label}<br>{fmt_clp(value)}",
            showarrow=False,
            xanchor="left",
            align="left",
            font={"color": color, "size": 12},
            bgcolor="rgba(6, 24, 68, 0.88)",
            bordercolor="rgba(255,255,255,0.18)",
            borderwidth=1,
        )

    # Indicador de edad revisada y percentiles principales, para que el gráfico sea autoexplicativo.
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=1.16,
        showarrow=False,
        align="left",
        text=(
            f"Edad revisada: <b>{selected_age_exact:,.0f}</b> · "
            f"P5: <b>{fmt_clp(p5)}</b> · "
            f"P50: <b>{fmt_clp(p50)}</b> · "
            f"P95: <b>{fmt_clp(p95)}</b>"
        ),
        font={"size": 13, "color": COLOR_MUTED},
    )
    fig.update_layout(margin={"l": 72, "r": 36, "t": 122, "b": 54})
    return apply_plot_theme(fig, y_currency=False, x_currency=True)


# Alias para compatibilidad interna si alguna parte antigua lo llama.
def plot_final_distribution(result: dict) -> go.Figure:
    return plot_distribution_at_age(result, result["inputs"]["edad_final"])


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



def parse_int_list(text: str, *, min_value: int, max_value: int) -> tuple[int, ...]:
    """Parsea una lista tipo '40, 45, 50' y la deja ordenada/sin duplicados."""
    values: list[int] = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(round(float(part.replace(",", "."))))
        except ValueError:
            continue
        if min_value <= value <= max_value:
            values.append(value)
    return tuple(sorted(set(values)))


def parse_probability_pct_list(text: str) -> tuple[float, ...]:
    """Parsea probabilidades escritas como '70,80,90,95' y devuelve 0.70..."""
    probs: list[float] = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip().replace("%", "")
        if not part:
            continue
        try:
            value = float(part.replace(",", "."))
        except ValueError:
            continue
        if value > 1:
            value = value / 100
        if 0 < value < 1:
            probs.append(value)
    return tuple(sorted(set(probs)))


def format_required_matrix_clp(matrix_clp: pd.DataFrame) -> pd.DataFrame:
    """Matriz formateada para mostrar en pantalla."""
    display = matrix_clp.copy()
    display.index = [f"{idx:,.0f}%".replace(",", ".") for idx in display.index]
    display.columns = [f"Edad {int(c)}" for c in display.columns]
    return display.map(lambda x: fmt_clp(x) if pd.notna(x) else "N/A")


def plot_required_capital_heatmap(matrix_clp: pd.DataFrame) -> go.Figure:
    """Heatmap de capital requerido por edad y probabilidad de éxito."""
    z = matrix_clp.astype(float).values
    x = [f"{int(c)}" for c in matrix_clp.columns]
    y = [f"{idx:,.0f}%".replace(",", ".") for idx in matrix_clp.index]
    text = np.vectorize(lambda v: fmt_clp(v))(z)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12, "color": COLOR_TEXT},
            colorscale=[
                [0.0, "rgba(0, 209, 255, 0.30)"],
                [0.45, "rgba(139, 61, 255, 0.62)"],
                [1.0, "rgba(255, 92, 122, 0.84)"],
            ],
            colorbar={"title": "Capital CLP", "tickprefix": "$", "tickformat": ",.0f"},
            hovertemplate="Edad retiro %{x}<br>Éxito %{y}<br>Capital requerido %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Matriz de capital requerido para jubilar hasta los 90",
        xaxis_title="Edad a la que comienzas a retirar",
        yaxis_title="Probabilidad de éxito objetivo",
        height=520,
        separators=",.",
    )
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


def make_monthly_cashflow_table(result: dict) -> pd.DataFrame:
    """Calendario mensual exportable en CLP."""
    inputs = result["inputs"]
    months = int(inputs["months"])
    edad_inicial = float(inputs["edad_inicial"])
    idx = np.arange(months)
    edad = edad_inicial + idx / 12
    out = pd.DataFrame(
        {
            "mes_simulacion": idx + 1,
            "edad": np.round(edad, 4),
            "año_edad": np.floor(edad).astype(int),
            "ahorro_min_clp": np.round(result.get("saving_min_schedule_mm", np.zeros(months)) * 1_000_000, 0),
            "ahorro_probable_clp": np.round(result.get("saving_mode_schedule_mm", np.zeros(months)) * 1_000_000, 0),
            "ahorro_max_clp": np.round(result.get("saving_max_schedule_mm", np.zeros(months)) * 1_000_000, 0),
            "ahorro_promedio_simulado_clp": np.round(result.get("monthly_savings_mean_mm", np.zeros(months)) * 1_000_000, 0),
            "retiro_clp": np.round(result.get("withdrawal_schedule_mm", np.zeros(months)) * 1_000_000, 0),
            "flujo_recurrente_neto_clp": np.round(result.get("recurring_cashflows_mm", np.zeros(months)) * 1_000_000, 0),
            "flujo_esporadico_clp": np.round(result.get("lump_sums_mm", np.zeros(months)) * 1_000_000, 0),
            "retorno_mensual_promedio_simulado": result.get("monthly_returns_mean", np.zeros(months)),
        }
    )
    out["flujo_neto_antes_retorno_clp"] = (
        out["ahorro_promedio_simulado_clp"]
        + out["flujo_recurrente_neto_clp"]
        + out["flujo_esporadico_clp"]
        - out["retiro_clp"]
    )
    return out


def make_final_distribution_table(result: dict) -> pd.DataFrame:
    n = len(result["final_wealth_mm"])
    return pd.DataFrame(
        {
            "path_id": np.arange(1, n + 1),
            "patrimonio_inicio_retiro_clp": np.round(result["wealth_at_retirement_mm"].astype(float) * 1_000_000, 0),
            "patrimonio_final_clp": np.round(result["final_wealth_mm"].astype(float) * 1_000_000, 0),
            "edad_agotamiento": result["ruin_age"],
            "agotado": ~np.isnan(result["ruin_age"]),
        }
    )


def make_inputs_table(result: dict) -> pd.DataFrame:
    rows = []
    for key, value in result.get("inputs", {}).items():
        if isinstance(value, (list, tuple, dict)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        rows.append({"parametro": key, "valor": value})
    return pd.DataFrame(rows)


def make_export_zip(
    result: dict,
    tabla: pd.DataFrame,
    saving_ranges_df: pd.DataFrame | None,
    recurring_df: pd.DataFrame | None,
    lump_df: pd.DataFrame | None,
    afp_info: dict | None,
    retirement_matrix: dict | None = None,
    *,
    include_paths: bool = False,
) -> bytes:
    """Crea paquete ZIP con todos los CSV relevantes del escenario."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        def write_csv(name: str, df: pd.DataFrame) -> None:
            zf.writestr(name, df.to_csv(index=False).encode("utf-8-sig"))

        metadata = pd.DataFrame(
            [
                {
                    "exportado_en": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "edad_inicial": result["inputs"].get("edad_inicial"),
                    "edad_inicio_retiro": result["inputs"].get("edad_inicio_retiro"),
                    "edad_final": result["inputs"].get("edad_final"),
                    "n_paths": result["inputs"].get("n_paths"),
                    "modelo_retorno": result["inputs"].get("return_model"),
                }
            ]
        )
        write_csv("00_metadata.csv", metadata)
        write_csv("01_inputs_modelo.csv", make_inputs_table(result))
        write_csv("02_resumen_percentiles.csv", result["summary"].copy())
        write_csv("03_tabla_por_edad.csv", make_numeric_csv_table(tabla))
        write_csv("04_flujos_mensuales.csv", make_monthly_cashflow_table(result))
        write_csv("05_distribucion_paths_final.csv", make_final_distribution_table(result))

        if saving_ranges_df is not None:
            write_csv("06_inputs_tramos_ahorro.csv", saving_ranges_df.copy())
        if recurring_df is not None:
            write_csv("07_inputs_flujos_recurrentes.csv", recurring_df.copy())
        if lump_df is not None:
            write_csv("08_inputs_flujos_esporadicos.csv", lump_df.copy())
        if afp_info is not None:
            write_csv("09_afp_calculada.csv", pd.DataFrame([afp_info]))
        if retirement_matrix is not None:
            write_csv("10_matriz_capital_requerido_clp.csv", retirement_matrix["matrix_clp"].reset_index())
            write_csv("11_matriz_capital_requerido_largo.csv", retirement_matrix["long"].copy())
            write_csv("12_matriz_capital_requerido_percentiles.csv", retirement_matrix["distribution_by_age"].copy())

        if include_paths:
            paths_clp = np.round(result["paths_mm"].astype(float) * 1_000_000, 0)
            edad_inicial = int(result["inputs"].get("edad_inicial", 0))
            cols = [f"edad_{edad_inicial + i / 12:.2f}" for i in range(paths_clp.shape[1])]
            paths_df = pd.DataFrame(paths_clp, columns=cols)
            paths_df.insert(0, "path_id", np.arange(1, paths_clp.shape[0] + 1))
            write_csv("20_paths_completos_clp.csv", paths_df)

    return buffer.getvalue()


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
password_gate()

st.markdown(
    f"""
    <div class="quant-hero">
        <div class="quant-title">Monte Carlo patrimonial: acumulación + retiro fijo</div>
        <div class="quant-subtitle">
            Simulador en CLP para probar desde qué edad dejas de ahorrar, cuánto retiras fijo al mes,
            qué ingresos recurrentes entran después y si el patrimonio llega vivo a los {EDAD_FINAL_FIJA} años.
        </div>
        <div class="quant-pill-row">
            <div class="quant-pill">Escenario base editable</div>
            <div class="quant-pill">Edad final fija: {EDAD_FINAL_FIJA}</div>
            <div class="quant-pill">Ahorro por edad</div>
            <div class="quant-pill">AFP + arriendos indexados</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Inputs en pantalla principal
# ============================================================

with st.form("formulario_simulacion"):
    st.markdown(
        """
        <div class="mode-strip">
            <div>
                <div class="mode-strip-title">Modo avanzado premium</div>
                <div class="mode-strip-caption">Configuración por módulos. La idea es partir con tus defaults y tocar solo el bloque que quieras sensibilizar.</div>
            </div>
            <div class="quant-pill">Listo para guardar CSV</div>
        </div>
        <div class="step-grid">
            <div class="step-card"><strong>1. Base</strong><span>Edad, capital, meta y retiro mensual.</span></div>
            <div class="step-card"><strong>2. Ahorro</strong><span>Tramos por edad con distribución triangular.</span></div>
            <div class="step-card"><strong>3. AFP</strong><span>Pensión real estimada e indexada.</span></div>
            <div class="step-card"><strong>4. Flujos</strong><span>Arriendos, gastos e ingresos mensuales.</span></div>
            <div class="step-card"><strong>5. Eventos</strong><span>Entradas o salidas de una sola vez.</span></div>
            <div class="step-card"><strong>6. Mercado</strong><span>Retorno, volatilidad y simulaciones.</span></div>
        </div>
        <div class="advanced-frame">
        """,
        unsafe_allow_html=True,
    )

    input_tabs = st.tabs([
        "1. Base",
        "2. Ahorro por edad",
        "3. AFP",
        "4. Ingresos / gastos",
        "5. Eventos únicos",
        "6. Mercado",
    ])

    with input_tabs[0]:
        panel_start(
            "Supuestos base",
            "Aquí queda lo mínimo para definir el ciclo de vida: edad inicial, edad de retiro, patrimonio inicial, meta y retiro mensual. La edad final queda fija en 90 años.",
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            edad_inicial = st.number_input("Edad inicial", min_value=18, max_value=89, value=27, step=1)
        with c2:
            edad_final = EDAD_FINAL_FIJA
            st.number_input("Edad final", min_value=EDAD_FINAL_FIJA, max_value=EDAD_FINAL_FIJA, value=EDAD_FINAL_FIJA, step=1, disabled=True)
        with c3:
            default_edad_inicio_retiro = min(max(42, int(edad_inicial)), EDAD_FINAL_FIJA)
            edad_inicio_retiro = st.number_input(
                "Edad inicio retiro",
                min_value=int(edad_inicial),
                max_value=EDAD_FINAL_FIJA,
                value=default_edad_inicio_retiro,
                step=1,
                help="Desde esta edad el ahorro mensual se vuelve cero y comienza el retiro fijo mensual.",
            )
        with c4:
            target_clp = money_text_input(
                "Meta patrimonial CLP",
                1_000_000_000,
                key="target_clp_text",
                help="Puedes escribir 1.000.000.000 o 1.000 MM.",
            )

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            initial_capital_clp = money_text_input("Capital inicial CLP", 35_000_000, key="initial_capital_clp_text")
        with c6:
            withdrawal_monthly_clp = money_text_input("Retiro mensual fijo CLP", 5_000_000, key="withdrawal_monthly_clp_text")
        with c7:
            withdrawal_indexed_to_inflation = st.checkbox("Indexar retiro por inflación", value=True)
        with c8:
            inflation_annual_pct = st.number_input("Inflación anual indexación (%)", min_value=0.0, value=3.0, step=0.25)
        panel_end()

    with input_tabs[1]:
        panel_start(
            "Rangos de ahorro mensual por edad",
            "Cada tramo usa distribución triangular: mínimo, más probable y máximo. Desde la edad de retiro el ahorro patrimonial queda automáticamente en cero.",
        )
        t1, t2 = st.columns([3, 1])
        with t1:
            st.caption("Default: etapa actual fuerte, menor ahorro entre 30 y 40 por hijos/familia, y recuperación hasta la edad de retiro.")
        with t2:
            contribution_timing_es = st.selectbox("Timing ahorro", ["Fin de mes", "Inicio de mes"], index=0)

        default_saving_rows = []
        if int(edad_inicio_retiro) > int(edad_inicial):
            if int(edad_inicial) < 30:
                default_saving_rows.append(
                    {
                        "descripcion": "Etapa actual",
                        "edad_inicio": int(edad_inicial),
                        "edad_fin": min(30, int(edad_inicio_retiro)),
                        "ahorro_min_clp": "2.500.000",
                        "ahorro_probable_clp": "3.000.000",
                        "ahorro_max_clp": "3.500.000",
                    }
                )
            if int(edad_inicio_retiro) > max(30, int(edad_inicial)):
                default_saving_rows.append(
                    {
                        "descripcion": "Hijos / menor ahorro",
                        "edad_inicio": max(30, int(edad_inicial)),
                        "edad_fin": min(40, int(edad_inicio_retiro)),
                        "ahorro_min_clp": "1.000.000",
                        "ahorro_probable_clp": "1.500.000",
                        "ahorro_max_clp": "2.000.000",
                    }
                )
            if int(edad_inicio_retiro) > max(40, int(edad_inicial)):
                default_saving_rows.append(
                    {
                        "descripcion": "Recuperación ahorro",
                        "edad_inicio": max(40, int(edad_inicial)),
                        "edad_fin": int(edad_inicio_retiro),
                        "ahorro_min_clp": "2.000.000",
                        "ahorro_probable_clp": "2.500.000",
                        "ahorro_max_clp": "3.000.000",
                    }
                )
        if not default_saving_rows:
            default_saving_rows.append(
                {
                    "descripcion": "Sin ahorro previo al retiro",
                    "edad_inicio": int(edad_inicial),
                    "edad_fin": min(int(edad_inicial) + 1, EDAD_FINAL_FIJA),
                    "ahorro_min_clp": "0",
                    "ahorro_probable_clp": "0",
                    "ahorro_max_clp": "0",
                }
            )
        default_saving_df = pd.DataFrame(default_saving_rows)
        saving_ranges_df = st.data_editor(
            default_saving_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "descripcion": st.column_config.TextColumn("Descripción"),
                "edad_inicio": st.column_config.NumberColumn("Edad inicio", min_value=int(edad_inicial), max_value=EDAD_FINAL_FIJA, step=1),
                "edad_fin": st.column_config.NumberColumn("Edad fin", min_value=int(edad_inicial), max_value=EDAD_FINAL_FIJA, step=1),
                "ahorro_min_clp": st.column_config.TextColumn("Ahorro mínimo CLP", help="Ej: 1.000.000 o 1 MM"),
                "ahorro_probable_clp": st.column_config.TextColumn("Ahorro más probable CLP", help="Ej: 1.500.000 o 1,5 MM"),
                "ahorro_max_clp": st.column_config.TextColumn("Ahorro máximo CLP", help="Ej: 2.000.000 o 2 MM"),
            },
            key="saving_ranges_editor",
        )
        panel_end()

    with input_tabs[2]:
        panel_start(
            "Jubilación AFP estimada",
            "Calcula una pensión real aproximada: saldo AFP + ahorro mensual AFP a retorno real, y luego retiro anual como % del saldo al jubilar. La pensión se indexa por inflación.",
        )
        afp_enable = st.checkbox("Agregar jubilación AFP calculada como ingreso recurrente", value=True)
        afp1, afp2, afp3, afp4, afp5 = st.columns(5)
        with afp1:
            afp_balance_clp = money_text_input("Saldo AFP actual CLP", 40_000_000, key="afp_balance_clp_text")
        with afp2:
            afp_monthly_contribution_clp = money_text_input("Ahorro mensual AFP CLP", 600_000, key="afp_monthly_contribution_clp_text")
        with afp3:
            afp_retirement_age = st.number_input("Edad jubilación AFP", min_value=int(edad_inicial), max_value=EDAD_FINAL_FIJA, value=min(max(60, int(edad_inicial)), EDAD_FINAL_FIJA), step=1)
        with afp4:
            afp_real_return_pct = st.number_input("Retorno real AFP anual (%)", min_value=-5.0, max_value=15.0, value=5.0, step=0.25)
        with afp5:
            afp_withdrawal_rate_pct = st.number_input("Tasa retiro AFP anual (%)", min_value=0.0, max_value=10.0, value=3.2, step=0.1)

        months_to_afp = max(int(round((float(afp_retirement_age) - float(edad_inicial)) * 12)), 0)
        afp_fv_real_clp = future_value_monthly_real_clp(
            initial_balance_clp=afp_balance_clp,
            monthly_contribution_clp=afp_monthly_contribution_clp,
            annual_real_return=float(afp_real_return_pct) / 100,
            n_months=months_to_afp,
            contribution_timing="end",
        )
        afp_monthly_pension_real_clp = afp_fv_real_clp * (float(afp_withdrawal_rate_pct) / 100) / 12
        afp_monthly_inflation = (1 + float(inflation_annual_pct) / 100) ** (1 / 12) - 1 if float(inflation_annual_pct) > -100 else 0.0
        afp_monthly_pension_nominal_start_clp = afp_monthly_pension_real_clp * (1 + afp_monthly_inflation) ** months_to_afp

        afp_summary = st.columns(3)
        with afp_summary[0]:
            st.markdown(f"""<div class="mini-card"><b>Saldo AFP al jubilar</b><span>{fmt_clp(afp_fv_real_clp)} de hoy</span></div>""", unsafe_allow_html=True)
        with afp_summary[1]:
            st.markdown(f"""<div class="mini-card"><b>Pensión real mensual</b><span>{fmt_clp(afp_monthly_pension_real_clp)} de hoy</span></div>""", unsafe_allow_html=True)
        with afp_summary[2]:
            st.markdown(f"""<div class="mini-card"><b>Pensión nominal inicial</b><span>{fmt_clp(afp_monthly_pension_nominal_start_clp)} a los {int(afp_retirement_age)}</span></div>""", unsafe_allow_html=True)
        st.caption("Si el ahorro AFP ya está dentro de tus tramos de ahorro patrimonial, evita duplicarlo.")
        panel_end()

    with input_tabs[3]:
        panel_start(
            "Otros ingresos o egresos mensuales recurrentes",
            "Arriendos, dividendos, gastos familiares u otros flujos mensuales. Si marcas indexación, el monto escrito en pesos de hoy crece con inflación.",
        )
        default_recurring_df = pd.DataFrame(
            {
                "descripcion": ["Arriendo propiedades", "Otro ingreso", "Gasto familiar"],
                "tipo": ["Ingreso", "Ingreso", "Egreso"],
                "edad_inicio": [max(52, int(edad_inicial)), max(65, int(edad_inicial)), max(30, int(edad_inicial))],
                "edad_fin": [90, 90, 90],
                "monto_mensual_clp": ["1.200.000", "0", "0"],
                "indexar_inflacion": [True, False, True],
            }
        )
        recurring_df = st.data_editor(
            default_recurring_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "descripcion": st.column_config.TextColumn("Descripción"),
                "tipo": st.column_config.SelectboxColumn("Tipo", options=["Ingreso", "Egreso"], required=True),
                "edad_inicio": st.column_config.NumberColumn("Edad inicio", min_value=int(edad_inicial), max_value=EDAD_FINAL_FIJA, step=1),
                "edad_fin": st.column_config.NumberColumn("Edad fin", min_value=int(edad_inicial), max_value=EDAD_FINAL_FIJA, step=1),
                "monto_mensual_clp": st.column_config.TextColumn("Monto mensual CLP", help="Ej: 1.200.000 o 1,2 MM. Si marcas indexación, se interpreta como pesos de hoy."),
                "indexar_inflacion": st.column_config.CheckboxColumn("Indexar inflación", help="Si está marcado, el monto crece con inflación desde hoy."),
            },
            key="recurring_editor",
        )
        panel_end()

    with input_tabs[4]:
        panel_start("Flujos esporádicos", "Plata que entra o sale una sola vez: bono, venta de activo, pie de propiedad, gasto grande, herencia, prepago, etc.")
        default_lump_df = pd.DataFrame(
            {
                "descripcion": ["Ejemplo bono / venta activo"],
                "tipo": ["Ingreso"],
                "edad_evento": [31],
                "monto_clp": ["40.000.000"],
            }
        )
        lump_df = st.data_editor(
            default_lump_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "descripcion": st.column_config.TextColumn("Descripción"),
                "tipo": st.column_config.SelectboxColumn("Tipo", options=["Ingreso", "Egreso"], required=True),
                "edad_evento": st.column_config.NumberColumn("Edad evento", min_value=int(edad_inicial), max_value=EDAD_FINAL_FIJA, step=1),
                "monto_clp": st.column_config.TextColumn("Monto CLP", help="Ej: 25.000.000 o 25 MM."),
            },
            key="lump_editor",
        )
        panel_end()

    with input_tabs[5]:
        panel_start("Retorno, riesgo y simulación", "El modo mensual IID captura mejor el riesgo de secuencia durante el retiro; el anual suavizado replica mejor el código original.")
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
        panel_end()

    st.markdown("</div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Simular escenario", type="primary")


# ============================================================
# Parseo de inputs
# ============================================================

def parse_lump_events(df: pd.DataFrame, edad_inicial_: int, edad_final_: int) -> tuple[tuple[int, float], ...]:
    events: list[tuple[int, float]] = []
    if df is None or df.empty:
        return tuple(events)
    for _, row in df.dropna(subset=["edad_evento", "monto_clp"]).iterrows():
        amount_clp = parse_clp_value(row.get("monto_clp", 0))
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



def parse_lump_age_events(df: pd.DataFrame, edad_inicial_: int, edad_final_: int) -> tuple[tuple[float, float], ...]:
    """Convierte eventos únicos a edades absolutas para la matriz de retiro."""
    events: list[tuple[float, float]] = []
    if df is None or df.empty:
        return tuple(events)
    for _, row in df.dropna(subset=["edad_evento", "monto_clp"]).iterrows():
        amount_clp = parse_clp_value(row.get("monto_clp", 0))
        if amount_clp == 0:
            continue
        age = float(row["edad_evento"])
        if age < edad_inicial_ or age > edad_final_:
            continue
        sign = 1.0 if str(row.get("tipo", "Ingreso")) == "Ingreso" else -1.0
        events.append((age, sign * clp_to_mm(amount_clp)))
    return tuple(events)


def parse_recurring_events(df: pd.DataFrame, edad_inicial_: int, edad_final_: int) -> tuple[tuple, ...]:
    events: list[tuple] = []
    if df is None or df.empty:
        return tuple(events)
    for _, row in df.dropna(subset=["edad_inicio", "monto_mensual_clp"]).iterrows():
        amount_clp = parse_clp_value(row.get("monto_mensual_clp", 0))
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
        indexed_raw = row.get("indexar_inflacion", False)
        indexed = bool(indexed_raw) if pd.notna(indexed_raw) else False
        # edad_base_indexacion = edad inicial: el monto se interpreta como pesos de hoy.
        events.append((start_age, end_age, sign * clp_to_mm(amount_clp), description, indexed, float(edad_inicial_)))
    return tuple(events)


def parse_saving_ranges(df: pd.DataFrame, edad_inicial_: int, edad_retiro_: int) -> tuple[tuple[float, Optional[float], float, float, float, str], ...]:
    """Convierte la tabla de tramos de ahorro a MM CLP para el motor."""
    ranges: list[tuple[float, Optional[float], float, float, float, str]] = []
    if df is None or df.empty:
        return tuple(ranges)

    for _, row in df.dropna(subset=["edad_inicio", "edad_fin"]).iterrows():
        min_clp = parse_clp_value(row.get("ahorro_min_clp", 0))
        mode_clp = parse_clp_value(row.get("ahorro_probable_clp", 0))
        max_clp = parse_clp_value(row.get("ahorro_max_clp", 0))
        if min_clp == 0 and mode_clp == 0 and max_clp == 0:
            continue

        start_age = float(row["edad_inicio"])
        end_age = float(row["edad_fin"])
        start_age = min(max(start_age, edad_inicial_), EDAD_FINAL_FIJA)
        end_age = min(max(end_age, edad_inicial_), EDAD_FINAL_FIJA)
        if end_age <= start_age:
            continue

        description = str(row.get("descripcion", "Tramo ahorro"))
        ranges.append(
            (
                start_age,
                end_age,
                clp_to_mm(min_clp),
                clp_to_mm(mode_clp),
                clp_to_mm(max_clp),
                description,
            )
        )
    return tuple(ranges)


if submitted:
    errors = []
    saving_ranges = parse_saving_ranges(saving_ranges_df, int(edad_inicial), int(edad_inicio_retiro))
    for start_age, end_age, min_mm, mode_mm, max_mm, description in saving_ranges:
        if not (min_mm <= mode_mm <= max_mm):
            errors.append(f"En el tramo '{description}' debe cumplirse: ahorro mínimo <= ahorro más probable <= ahorro máximo.")
        if end_age <= start_age:
            errors.append(f"En el tramo '{description}' la edad fin debe ser mayor que la edad inicio.")
    if not (annual_return_low_pct < annual_return_mean_pct < annual_return_high_pct):
        errors.append("El retorno esperado debe estar entre el mínimo y el máximo truncado.")
    if edad_inicio_retiro > EDAD_FINAL_FIJA:
        errors.append("La edad de inicio de retiro no puede ser mayor que 90.")

    if errors:
        for msg in errors:
            st.error(msg)
        st.stop()

    lump_events = parse_lump_events(lump_df, int(edad_inicial), EDAD_FINAL_FIJA)
    lump_age_events = parse_lump_age_events(lump_df, int(edad_inicial), EDAD_FINAL_FIJA)
    recurring_events_list = list(parse_recurring_events(recurring_df, int(edad_inicial), EDAD_FINAL_FIJA))
    afp_info = {
        "enabled": bool(afp_enable),
        "saldo_actual_clp": int(afp_balance_clp),
        "ahorro_mensual_clp": int(afp_monthly_contribution_clp),
        "edad_jubilacion": int(afp_retirement_age),
        "retorno_real_anual": float(afp_real_return_pct) / 100,
        "tasa_retiro_anual": float(afp_withdrawal_rate_pct) / 100,
        "saldo_estimado_real_clp": float(afp_fv_real_clp),
        "pension_mensual_real_clp": float(afp_monthly_pension_real_clp),
        "pension_mensual_nominal_inicio_clp": float(afp_monthly_pension_nominal_start_clp),
    }
    if bool(afp_enable) and afp_monthly_pension_real_clp > 0 and int(afp_retirement_age) < EDAD_FINAL_FIJA:
        recurring_events_list.append(
            (
                float(afp_retirement_age),
                float(EDAD_FINAL_FIJA),
                clp_to_mm(afp_monthly_pension_real_clp),
                "Jubilación estimada AFP",
                True,
                float(edad_inicial),
            )
        )
    recurring_events = tuple(recurring_events_list)
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
                monthly_saving_min_mm=0.0,
                monthly_saving_mode_mm=0.0,
                monthly_saving_max_mm=0.0,
                withdrawal_monthly_mm=clp_to_mm(withdrawal_monthly_clp),
                contribution_timing=contribution_timing,
                withdrawal_timing=withdrawal_timing,
                target_mm=clp_to_mm(target_clp),
                seed=int(seed),
                mean_is_effective=bool(mean_is_effective),
                lump_sum_events=lump_events,
                recurring_monthly_events=recurring_events,
                saving_ranges=saving_ranges,
                floor_zero=bool(floor_zero),
                return_model=return_model,
                withdrawal_indexed_to_inflation=bool(withdrawal_indexed_to_inflation),
                inflation_annual=float(inflation_annual_pct) / 100,
            )
            result["afp_info"] = afp_info
            tabla = tabla_monte_carlo_por_edad(result)
            st.session_state["mc_result"] = result
            st.session_state["mc_tabla"] = tabla
            st.session_state["mc_return_model_es"] = return_model_es
            st.session_state["mc_target_clp"] = target_clp
            st.session_state["mc_recurring_df"] = recurring_df
            st.session_state["mc_lump_df"] = lump_df
            st.session_state["mc_lump_age_events"] = lump_age_events
            st.session_state["mc_recurring_events"] = recurring_events
            st.session_state["mc_saving_ranges_df"] = saving_ranges_df
            st.session_state["mc_afp_info"] = afp_info
            st.session_state["mc_export_ready_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.pop("mc_retirement_matrix", None)
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

# Alertas interpretativas para evitar leer mal los resultados.
if result["inputs"].get("return_model") == "monthly_iid":
    st.info(
        "Modo mensual IID: cada mes tiene un shock independiente. Es normal que el patrimonio se vea más castigado "
        "que en el modo anual suavizado, porque aparece riesgo de secuencia: malos meses justo al empezar a retirar "
        "pueden dañar mucho más el capital."
    )

withdrawal_schedule = result["withdrawal_schedule_mm"]
nonzero_withdrawals = withdrawal_schedule[withdrawal_schedule > 0]
if result["inputs"].get("withdrawal_indexed_to_inflation") and len(nonzero_withdrawals) > 0:
    first_w = float(nonzero_withdrawals[0])
    last_w = float(nonzero_withdrawals[-1])
    last_rec = float(result["recurring_cashflows_mm"][-1]) if len(result["recurring_cashflows_mm"]) else 0.0
    gap_last = last_rec - last_w
    st.warning(
        f"Ojo con el retiro indexado: el primer retiro es {fmt_clp_from_mm(first_w)} mensual, "
        f"pero a los {result['inputs']['edad_final']} años llega a {fmt_clp_from_mm(last_w)} mensual. "
        f"Con ingresos/egresos recurrentes netos indexados de {fmt_clp_from_mm(last_rec)}, "
        f"la brecha mensual antes de retorno queda en {fmt_clp_from_mm(gap_last)}."
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Percentiles", "Flujos", "Paths", "Distribución final", "Agotamiento", "Matriz retiro", "Tablas"]
)

with tab1:
    st.plotly_chart(plot_percentile_fan(tabla, int(result["inputs"]["edad_inicio_retiro"]), float(target_clp)), width="stretch")

with tab2:
    st.info(
        "Lectura: los flujos esporádicos positivos aumentan el patrimonio una sola vez; los negativos lo reducen una sola vez. "
        "Después de ese mes, el patrimonio resultante sigue rentando con el modelo de retorno. Los flujos recurrentes indexados, como arriendos o AFP, sí crecen mes a mes con inflación."
    )
    st.plotly_chart(plot_cashflow_schedule(result), width="stretch")
    if result.get("afp_info", {}).get("enabled"):
        afp = result["afp_info"]
        st.write("Jubilación AFP calculada")
        afp_cols = st.columns(4)
        with afp_cols[0]:
            st.markdown(f"""<div class="definition-card"><b>Saldo AFP real al jubilar</b><br><span>{fmt_clp(afp['saldo_estimado_real_clp'])}</span></div>""", unsafe_allow_html=True)
        with afp_cols[1]:
            st.markdown(f"""<div class="definition-card"><b>Pensión real mensual</b><br><span>{fmt_clp(afp['pension_mensual_real_clp'])}</span></div>""", unsafe_allow_html=True)
        with afp_cols[2]:
            st.markdown(f"""<div class="definition-card"><b>Pensión nominal inicio</b><br><span>{fmt_clp(afp['pension_mensual_nominal_inicio_clp'])}</span></div>""", unsafe_allow_html=True)
        with afp_cols[3]:
            st.markdown(f"""<div class="definition-card"><b>Supuestos</b><br><span>retorno real {fmt_pct(afp['retorno_real_anual']*100)} · retiro {fmt_pct(afp['tasa_retiro_anual']*100)}</span></div>""", unsafe_allow_html=True)
    # Se eliminaron las tablas/vistas auxiliares con montos formateados bajo el gráfico de flujos.
    # El detalle numérico queda disponible en el tab "Tablas" y en los CSV descargables.

with tab3:
    n_sample = st.slider("Paths a mostrar", min_value=50, max_value=1_000, value=300, step=50)
    st.plotly_chart(plot_sample_paths(result, n_sample=n_sample), width="stretch")

with tab4:
    dist_c1, dist_c2 = st.columns([1, 3])
    with dist_c1:
        edad_distribucion = st.slider(
            "Edad a revisar",
            min_value=int(result["inputs"]["edad_inicial"]),
            max_value=int(result["inputs"]["edad_final"]),
            value=int(result["inputs"]["edad_final"]),
            step=1,
            help="Permite ver la distribución del patrimonio en cualquier edad del horizonte, no solo a los 90.",
        )
        month_idx_dist = int(round((edad_distribucion - int(result["inputs"]["edad_inicial"])) * 12))
        month_idx_dist = min(max(month_idx_dist, 0), result["paths_mm"].shape[1] - 1)
        wealth_dist = result["paths_mm"][:, month_idx_dist].astype(float) * 1_000_000
        quick_text = (
            f'<div class="definition-card"><b>Lectura rápida</b><br>'
            f'<span>P5: {fmt_clp(np.percentile(wealth_dist, 5))}<br>'
            f'P50: {fmt_clp(np.percentile(wealth_dist, 50))}<br>'
            f'P95: {fmt_clp(np.percentile(wealth_dist, 95))}</span></div>'
        )
        st.markdown(quick_text, unsafe_allow_html=True)
    with dist_c2:
        st.plotly_chart(plot_distribution_at_age(result, edad_distribucion), width="stretch")

with tab5:
    st.plotly_chart(plot_ruin_distribution(result), width="stretch")
    if result["prob_no_ruin"] == 1:
        st.success("En este escenario, ningún path agotó patrimonio dentro del horizonte simulado hasta los 90 años.")
    else:
        st.warning(
            f"En este escenario, {fmt_pct((1 - result['prob_no_ruin']) * 100)} de los paths agotó patrimonio "
            "dentro del horizonte simulado."
        )

with tab6:
    st.markdown(
        """
        <div class="download-card">
            <b>Matriz de capital requerido</b><br>
            <span>La celda responde: si comienzo a retirar a esta edad, ¿cuánto patrimonio necesito tener justo en esa fecha para llegar a los 90 sin agotar capital con X% de éxito?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "La matriz usa los mismos supuestos de retorno, retiro mensual, inflación, AFP, arriendos y eventos únicos del escenario actual. "
        "No incluye ahorro antes de esa edad: precisamente calcula el capital que ya deberías tener acumulado en ese momento."
    )

    mx1, mx2, mx3 = st.columns([2, 2, 1])
    with mx1:
        default_matrix_ages = [35, 37, 40, 43, 45, 48, 50, 55, 60, 65]
        default_ages_text = ", ".join(
            str(x)
            for x in default_matrix_ages
            if int(result["inputs"]["edad_inicial"]) <= x < EDAD_FINAL_FIJA
        )
        if not default_ages_text:
            default_ages_text = str(min(max(int(result["inputs"]["edad_inicio_retiro"]), int(result["inputs"]["edad_inicial"])), EDAD_FINAL_FIJA - 1))
        matrix_ages_text = st.text_input(
            "Edades a evaluar",
            value=default_ages_text,
            help="Ejemplo: 35, 37, 40, 43, 45, 48, 50, 55, 60, 65",
        )
    with mx2:
        matrix_probs_text = st.text_input(
            "Probabilidades de éxito objetivo (%)",
            value="70, 80, 90, 95",
            help="Ejemplo: 80, 90, 95",
        )
    with mx3:
        matrix_n_paths = st.number_input(
            "Simulaciones matriz",
            min_value=2_000,
            max_value=50_000,
            value=10_000,
            step=2_000,
            format="%d",
            help="La matriz corre una simulación adicional. 10.000 suele ser suficiente para una primera lectura.",
        )

    calc_matrix = st.button("Calcular matriz de capital requerido", type="primary")
    if calc_matrix:
        matrix_ages = parse_int_list(
            matrix_ages_text,
            min_value=int(result["inputs"]["edad_inicial"]),
            max_value=EDAD_FINAL_FIJA - 1,
        )
        matrix_probs = parse_probability_pct_list(matrix_probs_text)
        if not matrix_ages:
            st.error("Debes ingresar al menos una edad válida menor que 90.")
        elif not matrix_probs:
            st.error("Debes ingresar al menos una probabilidad válida, por ejemplo 80, 90 o 95.")
        else:
            with st.spinner("Calculando matriz de capital requerido..."):
                retirement_matrix = required_capital_matrix_mm(
                    edad_final=EDAD_FINAL_FIJA,
                    retirement_ages=matrix_ages,
                    success_probabilities=matrix_probs,
                    n_paths=int(matrix_n_paths),
                    annual_return_mean=float(result["inputs"]["annual_return_mean_requested"]),
                    annual_return_std=float(result["inputs"]["annual_return_std"]),
                    annual_return_low=float(result["inputs"]["annual_return_low"]),
                    annual_return_high=float(result["inputs"]["annual_return_high"]),
                    withdrawal_monthly_mm=float(result["inputs"]["withdrawal_monthly_mm"]),
                    withdrawal_timing=str(result["inputs"].get("withdrawal_timing", "end")),
                    seed=int(result["inputs"].get("seed", 123)) + 2026,
                    mean_is_effective=bool(result["inputs"].get("mean_is_effective", True)),
                    lump_sum_age_events=st.session_state.get("mc_lump_age_events", tuple()),
                    recurring_monthly_events=st.session_state.get("mc_recurring_events", tuple()),
                    return_model=str(result["inputs"].get("return_model", "monthly_iid")),
                    withdrawal_indexed_to_inflation=bool(result["inputs"].get("withdrawal_indexed_to_inflation", False)),
                    inflation_annual=float(result["inputs"].get("inflation_annual", 0.0)),
                )
                st.session_state["mc_retirement_matrix"] = retirement_matrix
                st.success("Matriz calculada.")

    retirement_matrix = st.session_state.get("mc_retirement_matrix")
    if retirement_matrix is None:
        st.warning("Presiona **Calcular matriz de capital requerido** para generar la tabla.")
    else:
        matrix_clp = retirement_matrix["matrix_clp"]
        st.plotly_chart(plot_required_capital_heatmap(matrix_clp), width="stretch")
        st.write("Matriz en CLP")
        st.dataframe(format_required_matrix_clp(matrix_clp), width="stretch")

        long_csv = retirement_matrix["long"].to_csv(index=False).encode("utf-8-sig")
        matrix_csv = matrix_clp.reset_index().to_csv(index=False).encode("utf-8-sig")
        cmat1, cmat2 = st.columns(2)
        with cmat1:
            st.download_button(
                "Descargar matriz CSV",
                data=matrix_csv,
                file_name="matriz_capital_requerido_clp.csv",
                mime="text/csv",
            )
        with cmat2:
            st.download_button(
                "Descargar formato largo CSV",
                data=long_csv,
                file_name="matriz_capital_requerido_largo.csv",
                mime="text/csv",
            )

        with st.expander("Cómo leer esta matriz", expanded=False):
            st.markdown(
                """
                - **Columnas:** edad a la que empiezas el retiro.
                - **Filas:** probabilidad de éxito objetivo.
                - **Celda:** capital requerido en esa edad para llegar a los 90 sin agotar patrimonio.
                - Si la celda de **Edad 42 / 90%** dice `$X`, significa que, bajo estos supuestos, necesitarías tener aproximadamente `$X` a los 42 para que 90% de los paths sobreviva hasta los 90.
                """
            )

with tab7:
    st.write("Tabla por edad con montos en CLP")
    display_table = make_display_table(tabla)
    st.dataframe(display_table, width="stretch")

    st.write("Resumen final con montos en CLP")
    summary_display = result["summary"].copy()
    for col in ["final_wealth_mm", "wealth_at_retirement_mm", "total_savings_mm"]:
        summary_display[col.replace("_mm", "_clp")] = summary_display[col].apply(fmt_clp_from_mm)
    st.dataframe(summary_display[["metric", "final_wealth_clp", "wealth_at_retirement_clp", "total_savings_clp"]], width="stretch")

    st.markdown(
        """
        <div class="download-card">
            <b>Exportar escenario</b><br>
            <span>Descarga un paquete ZIP con CSVs: inputs, resumen, tabla por edad, flujos mensuales, distribución final y tablas editadas. Puedes incluir paths completos si quieres auditar todo, aunque el archivo será más pesado.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    csv_tabla = make_numeric_csv_table(tabla).to_csv(index=False).encode("utf-8-sig")
    csv_summary = summary_display.to_csv(index=False).encode("utf-8-sig")
    csv_flujos = make_monthly_cashflow_table(result).to_csv(index=False).encode("utf-8-sig")
    csv_dist = make_final_distribution_table(result).to_csv(index=False).encode("utf-8-sig")

    include_paths_export = st.checkbox(
        "Incluir paths completos en el ZIP",
        value=False,
        help="Puede quedar pesado si usas muchas simulaciones. Déjalo apagado para un paquete liviano.",
    )
    export_zip = make_export_zip(
        result,
        tabla,
        st.session_state.get("mc_saving_ranges_df"),
        st.session_state.get("mc_recurring_df"),
        st.session_state.get("mc_lump_df"),
        st.session_state.get("mc_afp_info"),
        st.session_state.get("mc_retirement_matrix"),
        include_paths=bool(include_paths_export),
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Descargar ZIP completo",
            data=export_zip,
            file_name="escenario_montecarlo_completo.zip",
            mime="application/zip",
        )
    with c2:
        st.download_button(
            "Tabla por edad CSV",
            data=csv_tabla,
            file_name="tabla_montecarlo_por_edad_clp.csv",
            mime="text/csv",
        )
    with c3:
        st.download_button(
            "Flujos mensuales CSV",
            data=csv_flujos,
            file_name="flujos_mensuales_clp.csv",
            mime="text/csv",
        )
    with c4:
        st.download_button(
            "Distribución final CSV",
            data=csv_dist,
            file_name="distribucion_final_paths_clp.csv",
            mime="text/csv",
        )

    with st.expander("Descargas individuales adicionales", expanded=False):
        c5, c6 = st.columns(2)
        with c5:
            st.download_button(
                "Resumen CSV",
                data=csv_summary,
                file_name="resumen_montecarlo_clp.csv",
                mime="text/csv",
            )
        with c6:
            st.download_button(
                "Inputs del modelo CSV",
                data=make_inputs_table(result).to_csv(index=False).encode("utf-8-sig"),
                file_name="inputs_modelo.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "Nota: esto es una herramienta de simulación, no una recomendación financiera. "
    "El motor mantiene cálculos internos en MM CLP para estabilidad numérica, pero la interfaz muestra los montos en CLP con todos los ceros. Los flujos marcados como indexados se interpretan como pesos de hoy y crecen con inflación."
)
