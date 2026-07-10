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
import streamlit.components.v1 as components
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
FIRE_ANALYSIS_VERSION = "sustainable_withdrawal_by_age_v3"
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

# Supuestos de retorno real anualizado entregados por el usuario
# Fuente declarada en la app: Superintendencia de Pensiones.
AFP_RETURN_ASSUMPTIONS = {
    "Fondo A": {"mean": 0.0449, "std": 0.1099},
    "Fondo B": {"mean": 0.0402, "std": 0.0853},
    "Fondo C": {"mean": 0.0338, "std": 0.0619},
    "Fondo D": {"mean": 0.0281, "std": 0.0452},
    "Fondo E": {"mean": 0.0217, "std": 0.0412},
    "Renta Vitalicia": {"mean": 0.0311, "std": 0.0065},
}

AFP_PERCENTILE_OPTIONS = {
    "Conservador P25": 25,
    "Mediano P50": 50,
    "Optimista P75": 75,
    "Muy conservador P5": 5,
}


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


def fmt_clp_compact(value_clp: float | int, decimals: int = 0) -> str:
    """Formato compacto para etiquetas de gráficos: $2.350 MM o $1,2 B."""
    if value_clp is None or (isinstance(value_clp, float) and np.isnan(value_clp)):
        return "N/A"
    value = float(value_clp)
    sign = "-" if value < 0 else ""
    value_abs = abs(value)
    if value_abs >= 1_000_000_000_000:
        scaled = value_abs / 1_000_000_000_000
        suffix = "B"
        dec = 1
    else:
        scaled = value_abs / 1_000_000
        suffix = "MM"
        dec = decimals
    text = f"{scaled:,.{dec}f}"
    text = text.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{sign}${text} {suffix}"


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




def simulate_afp_future_balance_distribution_real_clp(
    *,
    initial_balance_clp: float,
    monthly_contribution_clp: float,
    annual_real_return_mean: float,
    annual_real_return_std: float,
    n_months: int,
    n_paths: int,
    seed: int | None = 2026,
    contribution_timing: str = "end",
) -> dict:
    """
    Simula saldo AFP real en pesos de hoy.

    La SP entrega promedio y desviación estándar del retorno real anualizado.
    En esta versión cada simulación/path recibe retornos reales aleatorios mensuales
    coherentes con el promedio y la volatilidad anual del fondo seleccionado.
    """
    initial_balance_clp = max(float(initial_balance_clp), 0.0)
    monthly_contribution_clp = max(float(monthly_contribution_clp), 0.0)
    n_months = max(int(n_months), 0)
    n_paths = max(int(n_paths), 1)

    rng = np.random.default_rng(seed)
    balances = np.full(n_paths, initial_balance_clp, dtype=np.float64)

    if n_months == 0:
        annualized_returns = np.full(n_paths, annual_real_return_mean, dtype=np.float64)
    else:
        # Aproximación mensual IID: media real mensual equivalente y volatilidad anual / sqrt(12).
        # Se trunca para evitar retornos mensuales imposibles o excesivamente extremos.
        monthly_mean = (1 + float(annual_real_return_mean)) ** (1 / 12) - 1
        monthly_std = max(float(annual_real_return_std), 1e-12) / np.sqrt(12)
        monthly_return_history = np.empty((n_paths, n_months), dtype=np.float32)

        for t in range(n_months):
            r_t = rng.normal(monthly_mean, monthly_std, size=n_paths)
            r_t = np.clip(r_t, -0.50, 0.50)
            monthly_return_history[:, t] = r_t.astype(np.float32)
            if contribution_timing == "begin":
                balances = (balances + monthly_contribution_clp) * (1 + r_t)
            else:
                balances = balances * (1 + r_t) + monthly_contribution_clp

        compounded = np.prod(1 + monthly_return_history.astype(np.float64), axis=1)
        annualized_returns = np.power(np.maximum(compounded, 1e-12), 12 / n_months) - 1

    percentiles = {
        "p5": float(np.percentile(balances, 5)),
        "p25": float(np.percentile(balances, 25)),
        "p50": float(np.percentile(balances, 50)),
        "p75": float(np.percentile(balances, 75)),
        "p95": float(np.percentile(balances, 95)),
    }
    return {
        "balances_real_clp": balances,
        "annualized_returns_real": annualized_returns,
        "mean_balance_real_clp": float(np.mean(balances)),
        "std_balance_real_clp": float(np.std(balances, ddof=1)) if len(balances) > 1 else 0.0,
        "percentiles_balance_real_clp": percentiles,
    }


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

        .excel-export-hero {{
            border: 1px solid rgba(48, 209, 88, 0.42);
            background: linear-gradient(135deg, rgba(48, 209, 88, 0.18), rgba(0, 209, 255, 0.10), rgba(139, 61, 255, 0.16));
            border-radius: 26px;
            padding: 20px 22px;
            margin: 18px 0 10px 0;
            box-shadow: 0 18px 46px rgba(0, 0, 0, 0.28), inset 0 0 0 1px rgba(255, 255, 255, 0.05);
        }}

        .excel-export-title {{
            font-size: 1.22rem;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 4px;
        }}

        .excel-export-subtitle {{
            font-size: 0.94rem;
            color: var(--muted);
            line-height: 1.35;
        }}

        div[data-testid="stDownloadButton"] > button[kind="secondary"] {{
            border: 1px solid rgba(48, 209, 88, 0.78) !important;
            background: linear-gradient(90deg, rgba(48, 209, 88, 0.98), rgba(0, 209, 255, 0.94), rgba(139, 61, 255, 0.90)) !important;
            color: #031135 !important;
            font-weight: 950 !important;
            border-radius: 22px !important;
            padding: 1.35rem 1.4rem !important;
            min-height: 74px !important;
            font-size: 1.12rem !important;
            letter-spacing: 0.02em !important;
            box-shadow: 0 18px 44px rgba(0, 209, 255, 0.22), 0 12px 28px rgba(48, 209, 88, 0.16) !important;
        }}

        div[data-testid="stDownloadButton"] > button[kind="secondary"]:hover {{
            transform: translateY(-1px);
            filter: brightness(1.06);
        }}

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


def install_clp_input_formatter() -> None:
    """Formatea montos CLP en tiempo real, incluidos los editores de tabla.

    Los inputs monetarios principales y las celdas de ``st.data_editor`` pueden
    recibir valores sin puntuación (por ejemplo, 3000000). El script agrega los
    puntos de miles mientras se escribe. En tablas solo interviene cuando el
    contenido es numérico; por lo tanto, no altera descripciones u otros textos.
    """
    components.html(
        r"""
        <script>
        (function () {
            const doc = window.parent.document;
            const TEXT_EDITORS = 'input[type="text"], textarea, [contenteditable="true"]';
            let lastGridInteractionAt = 0;

            function getEditorValue(editor) {
                if (editor.isContentEditable) return editor.textContent || "";
                return editor.value || "";
            }

            function formatThousandsCLP(raw) {
                if (raw === null || raw === undefined) return "";
                const value = String(raw);

                // Se mantiene la posibilidad de pegar abreviaciones como "3 MM".
                if (/[a-zA-ZáéíóúÁÉÍÓÚñÑ]/.test(value)) return value;

                const trimmed = value.trim();
                if (trimmed === "") return "";
                const isNegative = trimmed.startsWith("-");
                const digits = value.replace(/\D/g, "");
                if (!digits) return "";
                const formatted = digits.replace(/\B(?=(\d{3})+(?!\d))/g, ".");
                return (isNegative ? "-" : "") + formatted;
            }

            function isNumericLike(value) {
                const text = String(value || "").trim();
                return text === "" || /^-?[0-9.$,\s]+$/.test(text);
            }

            function setNativeValue(editor, value) {
                if (editor.isContentEditable) {
                    editor.textContent = value;
                    return;
                }
                const proto = editor.tagName === "TEXTAREA"
                    ? window.HTMLTextAreaElement.prototype
                    : window.HTMLInputElement.prototype;
                const descriptor = Object.getOwnPropertyDescriptor(proto, "value");
                if (descriptor && descriptor.set) descriptor.set.call(editor, value);
                else editor.value = value;
            }

            function moneyHint(editor) {
                const placeholder = editor.getAttribute("placeholder") || "";
                const aria = editor.getAttribute("aria-label") || "";
                const title = editor.getAttribute("title") || "";
                const labelText = editor.closest("label") ? editor.closest("label").innerText : "";
                const txt = `${placeholder} ${aria} ${title} ${labelText}`.toLowerCase();
                return (
                    placeholder.includes("1.000.000.000") ||
                    txt.includes("clp") ||
                    txt.includes("monto mensual") ||
                    txt.includes("monto evento") ||
                    txt.includes("ahorro objetivo") ||
                    txt.includes("ahorro máximo") ||
                    txt.includes("capital inicial") ||
                    txt.includes("retiro mensual") ||
                    txt.includes("saldo afp") ||
                    txt.includes("precisión aproximada") ||
                    txt.includes("precision aproximada")
                );
            }

            function hasGridAncestry(editor) {
                const classes = String(editor.className || "").toLowerCase();
                const gridAncestor = editor.closest(
                    '[data-testid*="DataFrame"], [data-testid*="dataframe"], ' +
                    '[data-testid*="DataEditor"], [data-testid*="data_editor"], ' +
                    '[role="grid"], [class*="gdg"], [class*="glide"]'
                );
                return Boolean(
                    gridAncestor ||
                    classes.includes("gdg") ||
                    classes.includes("glide")
                );
            }

            function isInsideGridEditor(editor) {
                return Boolean(
                    editor.dataset.clpGridEditor === "1" ||
                    hasGridAncestry(editor) ||
                    (Date.now() - lastGridInteractionAt) < 4000
                );
            }

            function shouldFormatEditor(editor) {
                if (!editor || !editor.matches(TEXT_EDITORS)) return false;
                if (moneyHint(editor)) return true;
                return isInsideGridEditor(editor) && isNumericLike(getEditorValue(editor));
            }

            function applyFormat(editor) {
                if (!shouldFormatEditor(editor)) return;
                if (editor.dataset.clpFormatting === "1") return;

                const current = getEditorValue(editor);
                if (!isNumericLike(current)) return;
                const formatted = formatThousandsCLP(current);
                if (formatted === current) return;

                editor.dataset.clpFormatting = "1";
                setNativeValue(editor, formatted);
                editor.dispatchEvent(new Event("input", { bubbles: true }));
                editor.dispatchEvent(new Event("change", { bubbles: true }));
                editor.dataset.clpFormatting = "0";

                try {
                    if (!editor.isContentEditable) {
                        const end = getEditorValue(editor).length;
                        editor.setSelectionRange(end, end);
                    }
                } catch (e) {}
            }

            function attachFormatter(editor) {
                if (!editor || editor.dataset.clpFormatterAttached === "1") return;
                editor.dataset.clpFormatterAttached = "1";
                if (hasGridAncestry(editor) || (Date.now() - lastGridInteractionAt) < 4000) {
                    editor.dataset.clpGridEditor = "1";
                }
                editor.setAttribute("autocomplete", "off");

                editor.addEventListener("input", function () {
                    window.requestAnimationFrame(() => applyFormat(editor));
                });
                editor.addEventListener("blur", function () { applyFormat(editor); });
                applyFormat(editor);
            }

            function scanEditors(root) {
                const base = root && root.querySelectorAll ? root : doc;
                base.querySelectorAll(TEXT_EDITORS).forEach(attachFormatter);
                if (root && root.matches && root.matches(TEXT_EDITORS)) attachFormatter(root);
            }

            // Glide Data Grid abre el editor de celda en un portal separado del canvas.
            // Recordar la interacción con la tabla permite reconocer ese editor aunque
            // no herede directamente el aria-label de la columna.
            doc.addEventListener("pointerdown", function (event) {
                const target = event.target;
                if (!target || !target.closest) return;
                const grid = target.closest(
                    '[data-testid*="DataFrame"], [data-testid*="dataframe"], ' +
                    '[data-testid*="DataEditor"], [role="grid"], canvas'
                );
                if (grid) lastGridInteractionAt = Date.now();
            }, true);

            doc.addEventListener("focusin", function (event) {
                const editor = event.target;
                if (!editor || !editor.matches || !editor.matches(TEXT_EDITORS)) return;
                if (hasGridAncestry(editor) || (Date.now() - lastGridInteractionAt) < 4000) {
                    editor.dataset.clpGridEditor = "1";
                }
                attachFormatter(editor);
            }, true);

            // Respaldo global: captura el primer dígito incluso cuando el editor de la
            // tabla fue creado después del último escaneo del MutationObserver.
            doc.addEventListener("input", function (event) {
                const editor = event.target;
                if (!editor || !editor.matches || !editor.matches(TEXT_EDITORS)) return;
                if (!shouldFormatEditor(editor)) return;
                window.requestAnimationFrame(() => applyFormat(editor));
            }, true);

            scanEditors(doc);
            const observer = new MutationObserver(function (mutations) {
                mutations.forEach(function (mutation) {
                    mutation.addedNodes.forEach(function (node) {
                        if (node.nodeType === 1) scanEditors(node);
                    });
                });
            });
            observer.observe(doc.body, { childList: true, subtree: true });
        })();
        </script>
        """,
        height=0,
        width=0,
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
    # Si el neto mensual es negativo, esa diferencia debe salir del patrimonio invertido.
    # Se grafica bajo cero para que se lea como uso/retiro adicional del fondo.
    grouped["monto_que_saca_del_fondo"] = np.minimum(grouped["neto_mensual_recurrente"], 0.0)
    grouped["monto_que_saca_del_fondo_abs"] = -grouped["monto_que_saca_del_fondo"]

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
        add_value_annotation(
            fig,
            final_age,
            float(final["monto_que_saca_del_fondo"]),
            f"sale del fondo<br>{fmt_clp(final['monto_que_saca_del_fondo_abs'])}",
            COLOR_ORANGE,
            yshift=-62,
        )

    fig.update_layout(
        title="Flujo mensual promedio por edad: entradas y salidas",
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
        text="Barras sobre cero = plata que entra. Barras bajo cero = plata que sale. Diamantes = flujos únicos del año.",
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
            colorbar={"title": "Capital nominal CLP", "tickprefix": "$", "tickformat": ",.0f"},
            hovertemplate="Edad retiro %{x}<br>Éxito %{y}<br>Capital requerido %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Matriz nominal de capital requerido para jubilar hasta los 90",
        xaxis_title="Edad a la que comienzas a retirar",
        yaxis_title="Probabilidad de éxito objetivo",
        height=520,
        separators=",.",
    )
    return apply_plot_theme(fig, y_currency=False, x_currency=False)


def cap_saving_ranges_to_age(
    saving_ranges: tuple | None,
    stop_age: float,
) -> tuple:
    """Recorta los tramos de ahorro para modelar Coast FIRE.

    Hasta stop_age se ahorra según los tramos definidos. Después de stop_age,
    ahorro = 0, aunque el retiro empiece más adelante.
    """
    if not saving_ranges:
        return tuple()
    capped = []
    for item in saving_ranges:
        if len(item) < 5:
            continue
        start_age, end_age, min_mm, mode_mm, max_mm = item[:5]
        description = item[5] if len(item) >= 6 else "Tramo ahorro"
        if start_age is None:
            continue
        start = float(start_age)
        end = float(stop_age) if end_age is None or (isinstance(end_age, float) and np.isnan(end_age)) else float(end_age)
        end = min(end, float(stop_age))
        if end <= start:
            continue
        capped.append((start, end, float(min_mm), float(mode_mm), float(max_mm), str(description)))
    return tuple(capped)


def first_withdrawal_nominal_clp_from_result(sim_result: dict) -> float:
    """Primer retiro mensual nominal de una simulación."""
    schedule = sim_result.get("withdrawal_schedule_mm")
    if schedule is None or len(schedule) == 0:
        return np.nan
    nonzero = np.asarray(schedule, dtype=float)[np.asarray(schedule, dtype=float) > 0]
    if len(nonzero) == 0:
        return np.nan
    return float(nonzero[0]) * 1_000_000


def run_fire_scan(
    *,
    base_result: dict,
    retirement_ages: tuple[int, ...],
    n_paths: int,
    saving_ranges: tuple | None,
    recurring_events: tuple,
    lump_events_monthly: tuple,
    seed_offset: int = 9000,
) -> pd.DataFrame:
    """Evalúa FIRE realista: acumula hasta cada edad y retira desde esa edad.

    A diferencia de la matriz teórica, aquí sí se considera si con los ahorros,
    eventos y retornos simulados realmente llegas con patrimonio suficiente a esa edad.
    """
    inputs = base_result["inputs"]
    rows = []
    for age in retirement_ages:
        age = int(age)
        if age < int(inputs["edad_inicial"]) or age >= int(inputs["edad_final"]):
            continue
        sim = monte_carlo_accumulation_withdrawal_mm(
            edad_inicial=int(inputs["edad_inicial"]),
            edad_final=int(inputs["edad_final"]),
            edad_inicio_retiro=age,
            n_paths=int(n_paths),
            initial_capital_mm=float(inputs["initial_capital_mm"]),
            annual_return_mean=float(inputs["annual_return_mean_requested"]),
            annual_return_std=float(inputs["annual_return_std"]),
            annual_return_low=float(inputs["annual_return_low"]),
            annual_return_high=float(inputs["annual_return_high"]),
            monthly_saving_min_mm=0.0,
            monthly_saving_mode_mm=0.0,
            monthly_saving_max_mm=0.0,
            withdrawal_monthly_mm=float(inputs["withdrawal_monthly_mm"]),
            contribution_timing=str(inputs.get("contribution_timing", "end")),
            withdrawal_timing=str(inputs.get("withdrawal_timing", "end")),
            target_mm=float(inputs.get("target_mm")) if inputs.get("target_mm") is not None else None,
            seed=int(inputs.get("seed", 123)) + int(seed_offset) + age * 17,
            mean_is_effective=bool(inputs.get("mean_is_effective", True)),
            lump_sum_events=lump_events_monthly,
            recurring_monthly_events=recurring_events,
            saving_ranges=saving_ranges,
            floor_zero=bool(inputs.get("floor_zero", True)),
            return_model=str(inputs.get("return_model", "monthly_iid")),
            withdrawal_indexed_to_inflation=bool(inputs.get("withdrawal_indexed_to_inflation", False)),
            inflation_annual=float(inputs.get("inflation_annual", 0.0)),
            withdrawal_index_base_age=float(inputs.get("withdrawal_index_base_age", inputs.get("edad_inicial"))),
            savings_indexed_to_inflation=bool(inputs.get("savings_indexed_to_inflation", False)),
        )
        w_ret = np.asarray(sim["wealth_at_retirement_mm"], dtype=float)
        rows.append(
            {
                "edad_retiro": age,
                "prob_exito_pct": float(sim["prob_no_ruin"] * 100),
                "capital_p5_retiro_clp": float(np.percentile(w_ret, 5) * 1_000_000),
                "capital_p50_retiro_clp": float(np.percentile(w_ret, 50) * 1_000_000),
                "capital_p95_retiro_clp": float(np.percentile(w_ret, 95) * 1_000_000),
                "patrimonio_p50_90_clp": float(np.percentile(sim["final_wealth_mm"], 50) * 1_000_000),
                "primer_retiro_nominal_clp": first_withdrawal_nominal_clp_from_result(sim),
                "edad_mediana_agotamiento": sim.get("median_ruin_age", np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values("edad_retiro") if rows else pd.DataFrame()


def build_realistic_matrix(
    required_matrix_clp: pd.DataFrame,
    fire_scan_df: pd.DataFrame,
) -> pd.DataFrame:
    """Matriz nominal requerida, pero solo muestra celdas alcanzables con el plan simulado."""
    if required_matrix_clp is None or required_matrix_clp.empty or fire_scan_df is None or fire_scan_df.empty:
        return pd.DataFrame()
    success_by_age = dict(zip(fire_scan_df["edad_retiro"].astype(int), fire_scan_df["prob_exito_pct"].astype(float)))
    out = required_matrix_clp.copy().astype(float)
    for prob_pct in out.index:
        for age in out.columns:
            achieved = success_by_age.get(int(age), np.nan)
            if np.isnan(achieved) or achieved + 1e-12 < float(prob_pct):
                out.loc[prob_pct, age] = np.nan
    return out


def format_realistic_matrix_clp(realistic_matrix_clp: pd.DataFrame, required_matrix_clp: pd.DataFrame | None = None, fire_scan_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Matriz display: muestra capital si es alcanzable; si no, muestra No alcanza."""
    if realistic_matrix_clp is None or realistic_matrix_clp.empty:
        return pd.DataFrame()
    display = realistic_matrix_clp.copy()
    display.index = [f"{idx:,.0f}%".replace(",", ".") for idx in display.index]
    display.columns = [f"Edad {int(c)}" for c in display.columns]
    formatted = display.copy().astype(object)
    for r in formatted.index:
        for c in formatted.columns:
            val = formatted.loc[r, c]
            formatted.loc[r, c] = fmt_clp(val) if pd.notna(val) else "No alcanza"
    return formatted


def plot_realistic_required_capital_heatmap(realistic_matrix_clp: pd.DataFrame) -> go.Figure:
    """Heatmap legible de matriz realista.

    Las celdas viables se colorean por magnitud y muestran montos compactos.
    Las celdas no alcanzables se muestran como "—" para no saturar el gráfico.
    """
    if realistic_matrix_clp is None or realistic_matrix_clp.empty:
        fig = go.Figure()
        fig.update_layout(title="Matriz realista no disponible")
        return apply_plot_theme(fig, y_currency=False, x_currency=False)

    raw = realistic_matrix_clp.astype(float).values
    valid = np.isfinite(raw)
    x = [f"{int(c)}" for c in realistic_matrix_clp.columns]
    y = [f"{idx:,.0f}%".replace(",", ".") for idx in realistic_matrix_clp.index]

    z = np.zeros_like(raw, dtype=float)
    if valid.any():
        vmin = float(np.nanmin(raw))
        vmax = float(np.nanmax(raw))
        if abs(vmax - vmin) < 1e-9:
            z[valid] = 0.70
        else:
            z[valid] = 0.25 + 0.75 * (raw[valid] - vmin) / (vmax - vmin)

    text = np.empty(raw.shape, dtype=object)
    hover = np.empty(raw.shape, dtype=object)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            if valid[i, j]:
                text[i, j] = fmt_clp_compact(raw[i, j], decimals=0)
                hover[i, j] = fmt_clp(raw[i, j])
            else:
                text[i, j] = "—"
                hover[i, j] = "No alcanza con el plan actual"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            text=text,
            customdata=hover,
            texttemplate="%{text}",
            textfont={"size": 11, "color": COLOR_TEXT},
            xgap=3,
            ygap=3,
            colorscale=[
                [0.0, "rgba(20, 31, 66, 0.55)"],
                [0.249, "rgba(20, 31, 66, 0.55)"],
                [0.25, "rgba(0, 209, 255, 0.38)"],
                [0.55, "rgba(139, 61, 255, 0.70)"],
                [1.0, "rgba(255, 92, 122, 0.90)"],
            ],
            showscale=False,
            hovertemplate="Edad retiro %{x}<br>Éxito objetivo %{y}<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Matriz FIRE realista — capital nominal requerido",
        xaxis_title="Edad a la que empiezas a retirar",
        yaxis_title="Probabilidad de éxito objetivo",
        height=520,
        separators=",.",
        margin=dict(l=70, r=30, t=75, b=75),
        annotations=[
            dict(
                x=0,
                y=1.08,
                xref="paper",
                yref="paper",
                text="— = tu plan simulado no alcanza esa combinación de edad y éxito",
                showarrow=False,
                align="left",
                font=dict(size=12, color=COLOR_MUTED),
            )
        ],
    )
    fig.update_xaxes(type="category", tickangle=0)
    fig.update_yaxes(type="category")
    return apply_plot_theme(fig, y_currency=False, x_currency=False)


def find_fire_row(fire_scan_df: pd.DataFrame, target_success_pct: float) -> pd.Series | None:
    if fire_scan_df is None or fire_scan_df.empty:
        return None
    ok = fire_scan_df[fire_scan_df["prob_exito_pct"] >= float(target_success_pct)].sort_values("edad_retiro")
    if ok.empty:
        return None
    return ok.iloc[0]


def run_coast_scan(
    *,
    base_result: dict,
    fire_age: int,
    target_success_pct: float,
    n_paths: int,
    saving_ranges: tuple | None,
    recurring_events: tuple,
    lump_events_monthly: tuple,
) -> pd.DataFrame:
    """Busca la edad mínima en que puedes dejar de ahorrar, manteniendo el retiro en fire_age."""
    inputs = base_result["inputs"]
    edad_inicial = int(inputs["edad_inicial"])
    if fire_age < edad_inicial:
        return pd.DataFrame()
    rows = []
    for coast_age in range(edad_inicial, int(fire_age) + 1):
        capped_savings = cap_saving_ranges_to_age(saving_ranges, coast_age)
        sim = monte_carlo_accumulation_withdrawal_mm(
            edad_inicial=edad_inicial,
            edad_final=int(inputs["edad_final"]),
            edad_inicio_retiro=int(fire_age),
            n_paths=int(n_paths),
            initial_capital_mm=float(inputs["initial_capital_mm"]),
            annual_return_mean=float(inputs["annual_return_mean_requested"]),
            annual_return_std=float(inputs["annual_return_std"]),
            annual_return_low=float(inputs["annual_return_low"]),
            annual_return_high=float(inputs["annual_return_high"]),
            monthly_saving_min_mm=0.0,
            monthly_saving_mode_mm=0.0,
            monthly_saving_max_mm=0.0,
            withdrawal_monthly_mm=float(inputs["withdrawal_monthly_mm"]),
            contribution_timing=str(inputs.get("contribution_timing", "end")),
            withdrawal_timing=str(inputs.get("withdrawal_timing", "end")),
            target_mm=float(inputs.get("target_mm")) if inputs.get("target_mm") is not None else None,
            seed=int(inputs.get("seed", 123)) + 12000 + coast_age * 19 + int(fire_age) * 23,
            mean_is_effective=bool(inputs.get("mean_is_effective", True)),
            lump_sum_events=lump_events_monthly,
            recurring_monthly_events=recurring_events,
            saving_ranges=capped_savings,
            floor_zero=bool(inputs.get("floor_zero", True)),
            return_model=str(inputs.get("return_model", "monthly_iid")),
            withdrawal_indexed_to_inflation=bool(inputs.get("withdrawal_indexed_to_inflation", False)),
            inflation_annual=float(inputs.get("inflation_annual", 0.0)),
            withdrawal_index_base_age=float(inputs.get("withdrawal_index_base_age", inputs.get("edad_inicial"))),
            savings_indexed_to_inflation=bool(inputs.get("savings_indexed_to_inflation", False)),
        )
        coast_month = int(round((coast_age - edad_inicial) * 12))
        coast_month = min(max(coast_month, 0), sim["paths_mm"].shape[1] - 1)
        coast_wealth = sim["paths_mm"][:, coast_month].astype(float)
        rows.append(
            {
                "edad_dejar_ahorrar": coast_age,
                "edad_fire": int(fire_age),
                "prob_exito_pct": float(sim["prob_no_ruin"] * 100),
                "capital_p50_al_dejar_ahorrar_clp": float(np.percentile(coast_wealth, 50) * 1_000_000),
                "capital_p50_al_fire_clp": float(np.percentile(sim["wealth_at_retirement_mm"], 50) * 1_000_000),
                "cumple_objetivo": bool(sim["prob_no_ruin"] * 100 >= float(target_success_pct)),
            }
        )
    return pd.DataFrame(rows)


def run_minimum_saving_calculator(
    *,
    base_result: dict,
    monthly_saving_low_clp: float,
    monthly_saving_high_clp: float,
    target_success_pct: float,
    n_paths: int,
    recurring_events: tuple,
    lump_events_monthly: tuple,
    max_iter: int = 14,
) -> dict:
    """Busca el ahorro mensual central mínimo para cumplir meta y no quebrar.

    El ahorro probado se interpreta como pesos de hoy si el modelo tiene activada
    la indexación del ahorro. El motor interpreta el monto como ahorro objetivo:
    80% de meses normales entre objetivo-$500.000 y objetivo, 15% de meses malos
    bajo ese rango y 5% de meses muy buenos hasta objetivo+$500.000.
    
    Criterio de éxito conjunto:
      1) patrimonio al inicio del retiro >= meta, y
      2) el patrimonio nunca se agota antes de los 90.
    """
    inputs = base_result["inputs"]
    edad_inicial = int(inputs["edad_inicial"])
    edad_retiro = int(inputs["edad_inicio_retiro"])
    edad_final = int(inputs["edad_final"])
    target_mm = float(inputs.get("target_mm")) if inputs.get("target_mm") is not None else None
    band_clp = 500_000.0

    def simulate(center_clp: float, seed_extra: int = 0) -> dict:
        center_clp = max(float(center_clp), 0.0)
        min_clp = max(center_clp - band_clp, 0.0)
        max_clp = center_clp + band_clp if center_clp > 0 else 0.0
        saving_ranges = ((
            float(edad_inicial),
            float(edad_retiro),
            clp_to_mm(min_clp),
            clp_to_mm(center_clp),
            clp_to_mm(max_clp),
            "Ahorro calculado mínimo",
        ),)
        sim = monte_carlo_accumulation_withdrawal_mm(
            edad_inicial=edad_inicial,
            edad_final=edad_final,
            edad_inicio_retiro=edad_retiro,
            n_paths=int(n_paths),
            initial_capital_mm=float(inputs["initial_capital_mm"]),
            annual_return_mean=float(inputs["annual_return_mean_requested"]),
            annual_return_std=float(inputs["annual_return_std"]),
            annual_return_low=float(inputs["annual_return_low"]),
            annual_return_high=float(inputs["annual_return_high"]),
            monthly_saving_min_mm=0.0,
            monthly_saving_mode_mm=0.0,
            monthly_saving_max_mm=0.0,
            withdrawal_monthly_mm=float(inputs["withdrawal_monthly_mm"]),
            contribution_timing=str(inputs.get("contribution_timing", "end")),
            withdrawal_timing=str(inputs.get("withdrawal_timing", "end")),
            target_mm=target_mm,
            seed=int(inputs.get("seed", 123)) + 31000 + int(seed_extra),
            mean_is_effective=bool(inputs.get("mean_is_effective", True)),
            lump_sum_events=lump_events_monthly,
            recurring_monthly_events=recurring_events,
            saving_ranges=saving_ranges,
            floor_zero=bool(inputs.get("floor_zero", True)),
            return_model=str(inputs.get("return_model", "monthly_iid")),
            withdrawal_indexed_to_inflation=bool(inputs.get("withdrawal_indexed_to_inflation", False)),
            inflation_annual=float(inputs.get("inflation_annual", 0.0)),
            withdrawal_index_base_age=float(inputs.get("withdrawal_index_base_age", inputs.get("edad_inicial"))),
            savings_indexed_to_inflation=bool(inputs.get("savings_indexed_to_inflation", False)),
        )
        no_ruin_mask = np.isnan(np.asarray(sim["ruin_month"], dtype=float))
        if target_mm is None:
            target_mask = np.ones_like(no_ruin_mask, dtype=bool)
        else:
            target_mask = np.asarray(sim["wealth_at_retirement_mm"], dtype=float) >= target_mm
        joint_success = float(np.mean(no_ruin_mask & target_mask))
        return {
            "saving_center_clp": center_clp,
            "saving_min_clp": min_clp,
            "saving_max_clp": max_clp,
            "joint_success_pct": joint_success * 100,
            "prob_no_ruin_pct": float(sim["prob_no_ruin"] * 100),
            "prob_target_retirement_pct": float(sim.get("prob_reach_target_at_retirement", np.nan) * 100),
            "capital_p50_retiro_clp": float(np.percentile(sim["wealth_at_retirement_mm"], 50) * 1_000_000),
            "capital_p5_retiro_clp": float(np.percentile(sim["wealth_at_retirement_mm"], 5) * 1_000_000),
            "patrimonio_p50_90_clp": float(np.percentile(sim["final_wealth_mm"], 50) * 1_000_000),
            "median_ruin_age": sim.get("median_ruin_age", np.nan),
        }

    target_success_pct = float(target_success_pct)
    low = max(float(monthly_saving_low_clp), 0.0)
    high = max(float(monthly_saving_high_clp), low)

    low_result = simulate(low, seed_extra=0)
    high_result = simulate(high, seed_extra=1)

    if high_result["joint_success_pct"] < target_success_pct:
        return {
            "status": "no_alcanza",
            "target_success_pct": target_success_pct,
            "low_result": low_result,
            "high_result": high_result,
            "best_result": high_result,
            "iterations": [],
        }

    iterations = []
    best = high_result
    for i in range(int(max_iter)):
        mid = (low + high) / 2
        mid_result = simulate(mid, seed_extra=100 + i)
        iterations.append(mid_result)
        if mid_result["joint_success_pct"] >= target_success_pct:
            best = mid_result
            high = mid
        else:
            low = mid

    return {
        "status": "ok",
        "target_success_pct": target_success_pct,
        "low_result": low_result,
        "high_result": high_result,
        "best_result": best,
        "iterations": iterations,
    }




def nominal_to_today_clp(value_clp: float | int, age: float | int, inputs: dict) -> float:
    """Deflacta un monto nominal de cierta edad a pesos de hoy."""
    if value_clp is None or (isinstance(value_clp, float) and np.isnan(value_clp)):
        return np.nan
    inflation = float(inputs.get("inflation_annual", 0.0))
    base_age = float(inputs.get("edad_inicial", age))
    years = float(age) - base_age
    if inflation <= -1:
        return float(value_clp)
    return float(value_clp) / ((1 + inflation) ** years)


def today_to_nominal_clp(value_clp: float | int, age: float | int, inputs: dict) -> float:
    """Lleva un monto en pesos de hoy a pesos nominales de cierta edad."""
    if value_clp is None or (isinstance(value_clp, float) and np.isnan(value_clp)):
        return np.nan
    inflation = float(inputs.get("inflation_annual", 0.0))
    base_age = float(inputs.get("edad_inicial", age))
    years = float(age) - base_age
    if inflation <= -1:
        return float(value_clp)
    return float(value_clp) * ((1 + inflation) ** years)


def _event_priority(event_types: list[str]) -> str:
    priority = ["agotamiento", "retiro", "afp", "arriendo", "esporadico", "hijos", "ahorro", "flujo", "ninguno"]
    for p in priority:
        if p in event_types:
            return p
    return "ninguno"


def build_event_timeline_table(
    result: dict,
    tabla: pd.DataFrame,
    saving_ranges_df: pd.DataFrame | None,
    recurring_df: pd.DataFrame | None,
    lump_df: pd.DataFrame | None,
    afp_info: dict | None,
) -> pd.DataFrame:
    """Construye un calendario anual coloreable con hitos del plan."""
    inputs = result["inputs"]
    edad_inicial = int(inputs.get("edad_inicial", 0))
    edad_retiro = int(inputs.get("edad_inicio_retiro", edad_inicial))
    current_year = datetime.now().year
    out_rows = []

    saving_events: dict[int, list[str]] = {}
    if saving_ranges_df is not None and not saving_ranges_df.empty:
        for _, row in saving_ranges_df.iterrows():
            if pd.isna(row.get("edad_inicio")):
                continue
            age = int(round(float(row.get("edad_inicio"))))
            desc = str(row.get("descripcion", "Tramo de ahorro"))
            amount = parse_clp_value(row.get("ahorro_esperado_clp", 0)) if "ahorro_esperado_clp" in saving_ranges_df.columns else 0
            label = f"Cambio ahorro: {desc}"
            if amount:
                label += f" ({fmt_clp(amount)}/mes objetivo)"
            saving_events.setdefault(age, []).append(label)

    lump_events: dict[int, list[str]] = {}
    if lump_df is not None and not lump_df.empty:
        for _, row in lump_df.iterrows():
            if pd.isna(row.get("edad_evento")):
                continue
            amount = parse_clp_value(row.get("monto_clp", 0))
            if amount == 0:
                continue
            age = int(round(float(row.get("edad_evento"))))
            tipo = str(row.get("tipo", "Ingreso"))
            sign_txt = "+" if tipo == "Ingreso" else "-"
            desc = str(row.get("descripcion", "Evento único"))
            lump_events.setdefault(age, []).append(f"{tipo} único: {desc} ({sign_txt}{fmt_clp(amount)})")

    recurring_events: dict[int, list[tuple[str, str]]] = {}
    if recurring_df is not None and not recurring_df.empty:
        for _, row in recurring_df.iterrows():
            if pd.isna(row.get("edad_inicio")):
                continue
            amount = parse_clp_value(row.get("monto_mensual_clp", 0))
            if amount == 0:
                continue
            age = int(round(float(row.get("edad_inicio"))))
            tipo = str(row.get("tipo", "Ingreso"))
            desc = str(row.get("descripcion", "Flujo recurrente"))
            indexed = " indexado" if bool(row.get("indexar_inflacion", False)) else ""
            label = f"Comienza {tipo.lower()} recurrente: {desc} ({fmt_clp(amount)}/mes{indexed})"
            event_kind = "arriendo" if "arriendo" in desc.lower() or tipo == "Ingreso" else "flujo"
            recurring_events.setdefault(age, []).append((label, event_kind))

    afp_age = None
    if afp_info and afp_info.get("enabled"):
        try:
            afp_age = int(round(float(afp_info.get("edad_jubilacion"))))
        except Exception:
            afp_age = None

    median_ruin_age = result.get("median_ruin_age", np.nan)
    ruin_age_marker = None if pd.isna(median_ruin_age) else int(round(float(median_ruin_age)))

    tabla_sorted = tabla.sort_values("edad").reset_index(drop=True)
    for i, row in tabla_sorted.iterrows():
        age = int(row["edad"])
        year = current_year + (age - edad_inicial)
        p50_start = float(row.get("p50_mediana_mm", 0.0)) * 1_000_000
        if i + 1 < len(tabla_sorted):
            p50_end = float(tabla_sorted.iloc[i + 1].get("p50_mediana_mm", 0.0)) * 1_000_000
        else:
            p50_end = p50_start
        retiro_nominal = float(row.get("retiro_prom_mensual_mm", 0.0)) * 1_000_000
        retiro_hoy = nominal_to_today_clp(retiro_nominal, age, inputs) if retiro_nominal else 0.0
        ingresos_rec = float(row.get("ingreso_recurrente_prom_mensual_mm", 0.0)) * 1_000_000
        egresos_rec = float(row.get("egreso_recurrente_prom_mensual_mm", 0.0)) * 1_000_000
        extra_anual = float(row.get("aporte_extra_anual_mm", 0.0)) * 1_000_000

        event_texts: list[str] = []
        event_types: list[str] = []
        if age in saving_events:
            event_texts.extend(saving_events[age])
            # Si el usuario etiqueta hijos en la descripción, se pinta como hito familiar.
            if any("hij" in x.lower() for x in saving_events[age]):
                event_types.append("hijos")
            else:
                event_types.append("ahorro")
        if age in lump_events:
            event_texts.extend(lump_events[age])
            event_types.append("esporadico")
        if age == edad_retiro:
            event_texts.append("Inicio del retiro elegido")
            event_types.append("retiro")
        if age in recurring_events:
            for label, kind in recurring_events[age]:
                event_texts.append(label)
                event_types.append(kind)
        if afp_age is not None and age == afp_age:
            pension = afp_info.get("pension_mensual_real_clp", np.nan)
            event_texts.append(f"Inicio pensión AFP estimada ({fmt_clp(pension)} de hoy)")
            event_types.append("afp")
        if ruin_age_marker is not None and age == ruin_age_marker:
            event_texts.append("Edad mediana de agotamiento si el plan falla")
            event_types.append("agotamiento")

        event_type = _event_priority(event_types)
        out_rows.append(
            {
                "Año": year,
                "Edad": age,
                "Patrimonio P50 inicio": p50_start,
                "Patrimonio P50 fin": p50_end,
                "Retiro mensual nominal": retiro_nominal,
                "Retiro mensual en pesos de hoy": retiro_hoy,
                "Ingresos recurrentes mensuales": ingresos_rec,
                "Egresos recurrentes mensuales": egresos_rec,
                "Flujo único anual": extra_anual,
                "Evento": " | ".join(event_texts) if event_texts else "",
                "Tipo evento": event_type,
            }
        )
    return pd.DataFrame(out_rows)


def format_event_timeline_table(timeline: pd.DataFrame) -> pd.DataFrame:
    if timeline is None or timeline.empty:
        return pd.DataFrame()
    display = timeline.copy()
    money_cols = [
        "Patrimonio P50 inicio",
        "Patrimonio P50 fin",
        "Retiro mensual nominal",
        "Retiro mensual en pesos de hoy",
        "Ingresos recurrentes mensuales",
        "Egresos recurrentes mensuales",
        "Flujo único anual",
    ]
    for col in money_cols:
        if col in display.columns:
            display[col] = display[col].apply(fmt_clp)
    if "Tipo evento" in display.columns:
        display = display.drop(columns=["Tipo evento"])
    return display


def style_event_timeline_table(timeline: pd.DataFrame):
    display = format_event_timeline_table(timeline)
    if timeline is None or timeline.empty:
        return display
    color_map = {
        "hijos": "background-color: rgba(255, 209, 102, 0.28); color: #FFFFFF;",
        "ahorro": "background-color: rgba(255, 209, 102, 0.16); color: #FFFFFF;",
        "esporadico": "background-color: rgba(255, 184, 107, 0.26); color: #FFFFFF;",
        "retiro": "background-color: rgba(0, 209, 255, 0.24); color: #FFFFFF;",
        "arriendo": "background-color: rgba(48, 209, 88, 0.22); color: #FFFFFF;",
        "afp": "background-color: rgba(183, 140, 255, 0.28); color: #FFFFFF;",
        "flujo": "background-color: rgba(139, 61, 255, 0.16); color: #FFFFFF;",
        "agotamiento": "background-color: rgba(255, 92, 122, 0.32); color: #FFFFFF;",
        "ninguno": "",
    }
    types = list(timeline.get("Tipo evento", pd.Series(["ninguno"] * len(display))))

    def row_style(row):
        kind = types[row.name] if row.name < len(types) else "ninguno"
        style = color_map.get(kind, "")
        return [style for _ in row]

    return display.style.apply(row_style, axis=1)


def adapt_saving_ranges_for_retirement_age(saving_ranges: tuple | None, planned_retirement_age: float, candidate_retirement_age: float) -> tuple:
    """Hace que los tramos que terminaban en la edad de retiro elegida terminen en la edad candidata."""
    if not saving_ranges:
        return tuple()
    out = []
    for item in saving_ranges:
        if len(item) < 5:
            continue
        start_age, end_age, min_mm, mode_mm, max_mm = item[:5]
        description = item[5] if len(item) >= 6 else "Tramo ahorro"
        if start_age is None:
            continue
        start = float(start_age)
        if start >= float(candidate_retirement_age):
            continue
        if end_age is None or (isinstance(end_age, float) and np.isnan(end_age)):
            end = float(candidate_retirement_age)
        else:
            end = float(end_age)
            if abs(end - float(planned_retirement_age)) < 1e-8:
                end = float(candidate_retirement_age)
            else:
                end = min(end, float(candidate_retirement_age))
        if end <= start:
            continue
        out.append((start, end, float(min_mm), float(mode_mm), float(max_mm), str(description)))
    return tuple(out)


def lump_age_events_to_monthly_from_start(lump_age_events: tuple | None, start_age: int, edad_final: int) -> tuple[tuple[int, float], ...]:
    events = []
    if not lump_age_events:
        return tuple(events)
    for age, amount_mm in lump_age_events:
        try:
            age_f = float(age)
            amount_f = float(amount_mm)
        except Exception:
            continue
        if amount_f == 0 or age_f < float(start_age) or age_f > float(edad_final):
            continue
        month_idx = int(round((age_f - float(start_age)) * 12)) + 1
        max_month = int((int(edad_final) - int(start_age)) * 12)
        if 1 <= month_idx <= max_month:
            events.append((month_idx, amount_f))
    return tuple(events)



def run_sustainable_withdrawal_by_retirement_age(
    *,
    base_result: dict,
    saving_ranges: tuple | None,
    recurring_events: tuple,
    lump_events_monthly: tuple,
    lump_age_events: tuple,
    n_paths: int,
    withdrawal_rate_annual: float = 0.04,
) -> pd.DataFrame:
    """Calcula retiro máximo sostenible por edad de jubilación.

    Para cada edad se busca por simulación el retiro real máximo viable.
    Para cada edad posible de jubilación, busca por bisección el mayor retiro mensual
    expresado en pesos de hoy que permite llegar a los 90 años con al menos el umbral
    objetivo de éxito. El benchmark 4% se mantiene solo como referencia secundaria.
    """
    inputs = base_result["inputs"]
    edad_inicial = int(inputs.get("edad_inicial"))
    edad_final = int(inputs.get("edad_final"))
    planned_age = int(inputs.get("edad_inicio_retiro"))
    target_success_pct = float(st.session_state.get("mc_fire_analysis_target_success_pct", 90.0))
    target_success = target_success_pct / 100.0
    desired_monthly_today_clp = float(inputs.get("withdrawal_monthly_mm", 0.0)) * 1_000_000
    rows = []

    # La tabla evalúa todas las edades, por lo que se acota el número de simulaciones
    # para mantener la app usable en Streamlit Cloud. La matriz FIRE principal puede
    # usar más paths en edades específicas.
    n_paths = int(max(700, min(int(n_paths), 2_500)))
    seed_base = int(inputs.get("seed", 123))

    def simulate_candidate(age: int, withdrawal_today_clp: float, seed_offset: int) -> dict:
        """Simula el escenario completo retirándose a una edad candidata."""
        candidate_savings = adapt_saving_ranges_for_retirement_age(saving_ranges, planned_age, age)
        return monte_carlo_accumulation_withdrawal_mm(
            edad_inicial=edad_inicial,
            edad_final=edad_final,
            edad_inicio_retiro=age,
            n_paths=n_paths,
            initial_capital_mm=float(inputs["initial_capital_mm"]),
            annual_return_mean=float(inputs["annual_return_mean_requested"]),
            annual_return_std=float(inputs["annual_return_std"]),
            annual_return_low=float(inputs["annual_return_low"]),
            annual_return_high=float(inputs["annual_return_high"]),
            monthly_saving_min_mm=0.0,
            monthly_saving_mode_mm=0.0,
            monthly_saving_max_mm=0.0,
            withdrawal_monthly_mm=clp_to_mm(withdrawal_today_clp),
            contribution_timing=str(inputs.get("contribution_timing", "end")),
            withdrawal_timing=str(inputs.get("withdrawal_timing", "end")),
            target_mm=None,
            seed=seed_base + 73000 + age * 101 + int(seed_offset),
            mean_is_effective=bool(inputs.get("mean_is_effective", True)),
            lump_sum_events=lump_events_monthly,
            recurring_monthly_events=recurring_events,
            saving_ranges=candidate_savings,
            floor_zero=bool(inputs.get("floor_zero", True)),
            return_model=str(inputs.get("return_model", "monthly_iid")),
            withdrawal_indexed_to_inflation=bool(inputs.get("withdrawal_indexed_to_inflation", True)),
            inflation_annual=float(inputs.get("inflation_annual", 0.0)),
            # El retiro probado está en pesos de hoy, por lo que se indexa desde la edad actual
            # hasta la edad evaluada y luego continúa indexándose hasta los 90.
            withdrawal_index_base_age=float(inputs.get("edad_inicial", edad_inicial)),
            savings_indexed_to_inflation=bool(inputs.get("savings_indexed_to_inflation", False)),
        )

    for age in range(edad_inicial, edad_final):
        # 1) Simulación sin retiro para estimar patrimonio disponible al jubilar.
        zero = simulate_candidate(age, 0.0, seed_offset=0)
        wealth_ret = np.asarray(zero.get("wealth_at_retirement_mm", []), dtype=float)
        capital_p50_mm = float(np.percentile(wealth_ret, 50)) if wealth_ret.size else 0.0
        capital_p50_clp = capital_p50_mm * 1_000_000
        capital_p50_today_clp = nominal_to_today_clp(capital_p50_clp, age, inputs)

        retiro_4pct_anual_nominal_clp = capital_p50_clp * float(withdrawal_rate_annual)
        retiro_4pct_mensual_nominal_clp = retiro_4pct_anual_nominal_clp / 12
        retiro_4pct_mensual_hoy_clp = nominal_to_today_clp(retiro_4pct_mensual_nominal_clp, age, inputs)

        # 2) Búsqueda del retiro mensual máximo en pesos de hoy.
        #    El límite superior crece si el caso todavía sobrevive, para no quedar corto.
        high = max(float(desired_monthly_today_clp) * 1.8, float(retiro_4pct_mensual_hoy_clp) * 2.0, 1_000_000.0)
        low = 0.0
        high_result = simulate_candidate(age, high, seed_offset=77)
        high_success = float(high_result.get("prob_no_ruin", np.nan))
        expand_count = 0
        while np.isfinite(high_success) and high_success >= target_success and expand_count < 5:
            high *= 1.8
            high_result = simulate_candidate(age, high, seed_offset=77)
            high_success = float(high_result.get("prob_no_ruin", np.nan))
            expand_count += 1

        best_withdrawal_today = 0.0
        best_result = simulate_candidate(age, 0.0, seed_offset=77)
        # Si aun con retiro cero el plan falla, queda en cero y se marca como no viable.
        zero_success = float(best_result.get("prob_no_ruin", np.nan))
        if np.isfinite(zero_success) and zero_success >= target_success:
            for it in range(8):
                mid = 0.5 * (low + high)
                sim_mid = simulate_candidate(age, mid, seed_offset=77)
                success_mid = float(sim_mid.get("prob_no_ruin", np.nan))
                if np.isfinite(success_mid) and success_mid >= target_success:
                    best_withdrawal_today = mid
                    best_result = sim_mid
                    low = mid
                else:
                    high = mid

        sustainable_success_pct = float(best_result.get("prob_no_ruin", np.nan) * 100)
        first_nominal_clp = today_to_nominal_clp(best_withdrawal_today, age, inputs)
        desired_gap_clp = best_withdrawal_today - desired_monthly_today_clp
        if best_withdrawal_today >= desired_monthly_today_clp:
            estado = "Alcanza"
        elif best_withdrawal_today >= 0.85 * desired_monthly_today_clp:
            estado = "Cerca"
        else:
            estado = "No alcanza"

        rows.append(
            {
                "Edad jubilación": int(age),
                "Año calendario": datetime.now().year + (int(age) - edad_inicial),
                "Patrimonio P50 nominal": capital_p50_clp,
                "Patrimonio P50 pesos de hoy": capital_p50_today_clp,
                "Retiro sostenible mensual pesos de hoy": best_withdrawal_today,
                "Primer retiro sostenible nominal": first_nominal_clp,
                "Retiro deseado mensual pesos de hoy": desired_monthly_today_clp,
                "Brecha vs retiro deseado": desired_gap_clp,
                "Retiro mensual 4% pesos de hoy": retiro_4pct_mensual_hoy_clp,
                "Retiro mensual 4% nominal": retiro_4pct_mensual_nominal_clp,
                "Prob. no agotar hasta 90": sustainable_success_pct,
                "Edad mediana agotamiento si falla": best_result.get("median_ruin_age", np.nan),
                "Estado": estado,
            }
        )
    return pd.DataFrame(rows)


def format_sustainable_withdrawal_by_retirement_age(df: pd.DataFrame) -> pd.DataFrame:
    """Formatea la tabla de retiro sostenible por edad para pantalla."""
    if df is None or df.empty:
        return pd.DataFrame()
    display = df.copy()
    preferred_order = [
        "Edad jubilación",
        "Año calendario",
        "Estado",
        "Patrimonio P50 pesos de hoy",
        "Patrimonio P50 nominal",
        "Retiro sostenible mensual pesos de hoy",
        "Primer retiro sostenible nominal",
        "Retiro deseado mensual pesos de hoy",
        "Brecha vs retiro deseado",
        "Prob. no agotar hasta 90",
        "Edad mediana agotamiento si falla",
    ]
    display = display[[c for c in preferred_order if c in display.columns]]
    money_cols = [
        "Patrimonio P50 nominal",
        "Patrimonio P50 pesos de hoy",
        "Retiro sostenible mensual pesos de hoy",
        "Primer retiro sostenible nominal",
        "Retiro deseado mensual pesos de hoy",
        "Brecha vs retiro deseado",
    ]
    for col in money_cols:
        if col in display.columns:
            display[col] = display[col].apply(fmt_clp)
    if "Prob. no agotar hasta 90" in display.columns:
        display["Prob. no agotar hasta 90"] = display["Prob. no agotar hasta 90"].apply(lambda x: fmt_pct(float(x), 1))
    if "Edad mediana agotamiento si falla" in display.columns:
        display["Edad mediana agotamiento si falla"] = display["Edad mediana agotamiento si falla"].apply(lambda x: "No se agota" if pd.isna(x) else f"{float(x):,.1f} años".replace(".", ","))
    return display


def style_sustainable_withdrawal_by_retirement_age(df: pd.DataFrame):
    display = format_sustainable_withdrawal_by_retirement_age(df)
    if df is None or df.empty:
        return display
    estados = list(df.get("Estado", pd.Series([""] * len(df))))
    colors = {
        "Alcanza": "background-color: rgba(48, 209, 88, 0.22); color: #FFFFFF;",
        "Cerca": "background-color: rgba(255, 209, 102, 0.24); color: #FFFFFF;",
        "No alcanza": "background-color: rgba(255, 92, 122, 0.26); color: #FFFFFF;",
        "Aguanta": "background-color: rgba(48, 209, 88, 0.22); color: #FFFFFF;",
        "Frágil": "background-color: rgba(255, 209, 102, 0.24); color: #FFFFFF;",
        "No aguanta": "background-color: rgba(255, 92, 122, 0.26); color: #FFFFFF;",
    }
    def row_style(row):
        state = estados[row.name] if row.name < len(estados) else ""
        style = colors.get(state, "")
        return [style for _ in row]
    return display.style.apply(row_style, axis=1)

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
    out["monto_que_debe_salir_del_fondo_clp"] = np.maximum(-out["flujo_neto_antes_retorno_clp"], 0)
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
    fire_analysis: dict | None = None,
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
            write_csv("10_matriz_capital_requerido_largo.csv", retirement_matrix["long"].copy())
            zf.writestr("11_matriz_capital_requerido_clp.csv", retirement_matrix["matrix_clp"].reset_index().to_csv(index=False).encode("utf-8-sig"))
            write_csv("12_matriz_capital_requerido_distribucion.csv", retirement_matrix["distribution_by_age"].copy())
        if fire_analysis is not None:
            if isinstance(fire_analysis.get("fire_scan"), pd.DataFrame):
                write_csv("13_fire_realista_por_edad.csv", fire_analysis["fire_scan"].copy())
            if isinstance(fire_analysis.get("coast_scan"), pd.DataFrame):
                write_csv("14_coast_fire.csv", fire_analysis["coast_scan"].copy())
            if isinstance(fire_analysis.get("realistic_matrix_clp"), pd.DataFrame):
                zf.writestr("15_matriz_fire_realista_clp.csv", fire_analysis["realistic_matrix_clp"].reset_index().to_csv(index=False).encode("utf-8-sig"))

        if include_paths:
            paths_clp = np.round(result["paths_mm"].astype(float) * 1_000_000, 0)
            edad_inicial = int(result["inputs"].get("edad_inicial", 0))
            cols = [f"edad_{edad_inicial + i / 12:.2f}" for i in range(paths_clp.shape[1])]
            paths_df = pd.DataFrame(paths_clp, columns=cols)
            paths_df.insert(0, "path_id", np.arange(1, paths_clp.shape[0] + 1))
            write_csv("13_paths_completos_clp.csv", paths_df)

    return buffer.getvalue()


def make_executive_excel_report(
    result: dict,
    tabla: pd.DataFrame,
    saving_ranges_df: pd.DataFrame | None,
    recurring_df: pd.DataFrame | None,
    lump_df: pd.DataFrame | None,
    afp_info: dict | None,
    fire_analysis: dict | None = None,
) -> bytes:
    """Reporte ejecutivo en Excel, pensado para una persona no financiera.

    Orden de lectura:
    1) Inputs del escenario.
    2) Flujos proyectados.
    3) FIRE / Coast FIRE.
    4) Matriz FIRE realista.
    5) Tabla de percentiles por edad.
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter", engine_kwargs={"options": {"nan_inf_to_errors": True}}) as writer:
        wb = writer.book

        # Paleta compatible con la app.
        fmt_title = wb.add_format({"bold": True, "font_size": 20, "font_color": "#FFFFFF", "bg_color": "#041F5F"})
        fmt_subtitle = wb.add_format({"font_size": 11, "font_color": "#B8C4D8", "bg_color": "#041F5F", "text_wrap": True})
        fmt_section = wb.add_format({"bold": True, "font_size": 13, "font_color": "#FFFFFF", "bg_color": "#8B3DFF", "border": 0})
        fmt_header = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#102B66", "border": 1, "border_color": "#2E4A84"})
        fmt_text = wb.add_format({"font_color": "#0B1F4A", "border": 1, "border_color": "#D9E2F3"})
        fmt_note = wb.add_format({"font_color": "#102B66", "bg_color": "#EAF2FF", "text_wrap": True, "border": 1, "border_color": "#B8C4D8"})
        fmt_money = wb.add_format({"num_format": '[$$-es-CL] #,##0', "font_color": "#0B1F4A", "border": 1, "border_color": "#D9E2F3"})
        fmt_pct_cell = wb.add_format({"num_format": "0.0%", "font_color": "#0B1F4A", "border": 1, "border_color": "#D9E2F3"})
        fmt_num = wb.add_format({"num_format": "#,##0.0", "font_color": "#0B1F4A", "border": 1, "border_color": "#D9E2F3"})
        fmt_int = wb.add_format({"num_format": "#,##0", "font_color": "#0B1F4A", "border": 1, "border_color": "#D9E2F3"})
        fmt_good = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#30D158", "border": 1, "border_color": "#30D158"})
        fmt_warn = wb.add_format({"bold": True, "font_color": "#061844", "bg_color": "#FFD166", "border": 1, "border_color": "#FFD166"})
        fmt_bad = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#FF5C7A", "border": 1, "border_color": "#FF5C7A"})
        fmt_card_label = wb.add_format({"bold": True, "font_color": "#B8C4D8", "bg_color": "#0B1F4A", "border": 1, "border_color": "#8B3DFF"})
        fmt_card_value = wb.add_format({"bold": True, "font_size": 16, "font_color": "#FFFFFF", "bg_color": "#0B1F4A", "border": 1, "border_color": "#8B3DFF", "num_format": '[$$-es-CL] #,##0'})
        fmt_card_pct = wb.add_format({"bold": True, "font_size": 16, "font_color": "#FFFFFF", "bg_color": "#0B1F4A", "border": 1, "border_color": "#8B3DFF", "num_format": "0.0%"})
        fmt_blank_dark = wb.add_format({"bg_color": "#041F5F"})

        def add_sheet(name: str):
            ws = wb.add_worksheet(name)
            writer.sheets[name] = ws
            ws.hide_gridlines(2)
            ws.set_tab_color("#8B3DFF")
            return ws

        def write_title(ws, title: str, subtitle: str):
            ws.set_row(0, 30)
            ws.merge_range(0, 0, 0, 8, title, fmt_title)
            ws.merge_range(1, 0, 2, 8, subtitle, fmt_subtitle)
            ws.set_row(1, 24)
            ws.set_row(2, 24)

        def safe_df(df: pd.DataFrame | None) -> pd.DataFrame:
            if df is None:
                return pd.DataFrame()
            out = df.copy()
            out = out.replace([np.inf, -np.inf], np.nan)
            return out

        def write_table(ws, df: pd.DataFrame, start_row: int, start_col: int = 0, title: str | None = None, money_cols: set[str] | None = None, pct_cols: set[str] | None = None, int_cols: set[str] | None = None) -> int:
            df = safe_df(df)
            money_cols = money_cols or set()
            pct_cols = pct_cols or set()
            int_cols = int_cols or set()
            row = start_row
            if title:
                ws.merge_range(row, start_col, row, max(start_col + len(df.columns) - 1, start_col + 3), title, fmt_section)
                row += 2
            if df.empty:
                ws.write(row, start_col, "Sin datos para esta sección", fmt_note)
                return row + 3
            for j, col in enumerate(df.columns):
                ws.write(row, start_col + j, str(col), fmt_header)
            for i, (_, data_row) in enumerate(df.iterrows(), start=1):
                for j, col in enumerate(df.columns):
                    val = data_row[col]
                    fmt = fmt_text
                    if col in money_cols:
                        fmt = fmt_money
                    elif col in pct_cols:
                        fmt = fmt_pct_cell
                    elif col in int_cols:
                        fmt = fmt_int
                    elif isinstance(val, (int, float, np.integer, np.floating)) and pd.notna(val):
                        fmt = fmt_num
                    if pd.isna(val):
                        ws.write_blank(row + i, start_col + j, None, fmt)
                    elif isinstance(val, (int, float, np.integer, np.floating)):
                        ws.write_number(row + i, start_col + j, float(val), fmt)
                    else:
                        ws.write(row + i, start_col + j, str(val), fmt)
            ws.autofilter(row, start_col, row + len(df), start_col + len(df.columns) - 1)
            return row + len(df) + 3

        # Valores clave.
        inputs = result.get("inputs", {})
        summary = result["summary"].set_index("metric")
        ret_p50_clp = float(summary.loc["p50", "wealth_at_retirement_mm"] * 1_000_000)
        final_p50_clp = float(summary.loc["p50", "final_wealth_mm"] * 1_000_000)
        prob_no_ruin_excel = float(result.get("prob_no_ruin", np.nan))
        median_ruin_age = result.get("median_ruin_age", np.nan)
        retirement_age = int(inputs.get("edad_inicio_retiro", 0))
        retirement_month = int(inputs.get("retirement_start_month", 0))
        withdrawal_schedule = result.get("withdrawal_schedule_mm", np.array([]))
        first_withdrawal_nominal_clp = np.nan
        if len(withdrawal_schedule) and 0 <= retirement_month < len(withdrawal_schedule):
            first_withdrawal_nominal_clp = float(withdrawal_schedule[retirement_month] * 1_000_000)

        # FIRE analysis derivado.
        fire_scan = safe_df(fire_analysis.get("fire_scan") if fire_analysis else None)
        coast_scan = safe_df(fire_analysis.get("coast_scan") if fire_analysis else None)
        realistic_matrix = safe_df(fire_analysis.get("realistic_matrix_clp") if fire_analysis else None)
        retirement_sustainable = safe_df(fire_analysis.get("sustainable_withdrawal_by_age") if fire_analysis else None)
        target_success_pct = float(fire_analysis.get("target_success_pct", 90.0)) if fire_analysis else 90.0
        fire_row = find_fire_row(fire_scan, target_success_pct) if not fire_scan.empty else None
        coast_row = None
        if not coast_scan.empty:
            coast_ok = coast_scan[coast_scan["prob_exito_pct"] >= target_success_pct].sort_values("edad_dejar_ahorrar")
            coast_row = coast_ok.iloc[0] if not coast_ok.empty else None

        # ----------------------------------------------------
        # 01 Inputs
        # ----------------------------------------------------
        ws = add_sheet("01 Inputs")
        write_title(
            ws,
            "Reporte de simulación FIRE",
            "Lectura simple: primero revisa los supuestos cargados, luego los flujos, después la edad FIRE / Coast FIRE y finalmente la matriz. Todos los montos están en CLP nominales salvo cuando se indique 'pesos de hoy'.",
        )
        ws.set_column("A:A", 34)
        ws.set_column("B:B", 24)
        ws.set_column("C:C", 72)
        inputs_df = pd.DataFrame(
            [
                ["Edad inicial", inputs.get("edad_inicial"), "Edad desde la cual empieza la simulación."],
                ["Edad retiro elegida", inputs.get("edad_inicio_retiro"), "Escenario base cargado por ti."],
                ["Edad final", inputs.get("edad_final"), "Horizonte de supervivencia del patrimonio."],
                ["Capital inicial", fmt_clp(float(inputs.get("initial_capital_mm", 0)) * 1_000_000), "Patrimonio invertible inicial."],
                ["Retiro mensual real deseado", fmt_clp(float(inputs.get("withdrawal_monthly_mm", 0)) * 1_000_000), "Monto en pesos de hoy que quieres recibir cada mes."],
                ["Primer retiro nominal", fmt_clp(first_withdrawal_nominal_clp), "Monto mensual al inicio del retiro elegido, ya indexado si corresponde."],
                ["Inflación anual", fmt_pct(float(inputs.get("inflation_annual", 0)) * 100), "Usada para indexar retiro, ahorro e ingresos marcados como indexados."],
                ["Retorno anual esperado", fmt_pct(float(inputs.get("annual_return_mean_requested", 0)) * 100), "Media anual de retorno del portafolio."],
                ["Volatilidad anual", fmt_pct(float(inputs.get("annual_return_std", 0)) * 100), "Riesgo anual simulado."],
                ["Modelo de retorno", inputs.get("return_model", ""), "Regímenes realistas mezcla años normales, malos y crisis; mensual IID usa shocks independientes."],
                ["Simulaciones", inputs.get("n_paths"), "Cantidad de caminos Monte Carlo."],
            ],
            columns=["Input", "Valor", "Explicación"],
        )
        r = write_table(ws, inputs_df, 4, title="1. Inputs principales")
        ws.write(r, 0, "Nota", fmt_header)
        ws.merge_range(r, 1, r, 8, "El retiro que ingresas se interpreta como poder de compra de hoy. Si está indexado, la app lo lleva a pesos nominales de la edad de retiro y lo sigue indexando hasta los 90.", fmt_note)
        r += 3
        sr_df = safe_df(saving_ranges_df)
        if not sr_df.empty:
            r = write_table(ws, sr_df, r, title="2. Ahorro por edad ingresado", money_cols={"ahorro_esperado_clp"}, int_cols={"edad_inicio", "edad_fin"})
        if afp_info:
            afp_df = pd.DataFrame([afp_info])
            r = write_table(ws, afp_df, r, title="3. AFP calculada", money_cols={c for c in afp_df.columns if "clp" in c.lower()})
        rec_df = safe_df(recurring_df)
        if not rec_df.empty:
            r = write_table(ws, rec_df, r, title="4. Ingresos / gastos recurrentes", money_cols={"monto_mensual_clp"}, int_cols={"edad_inicio", "edad_fin"})
        lump_in = safe_df(lump_df)
        if not lump_in.empty:
            r = write_table(ws, lump_in, r, title="5. Eventos únicos", money_cols={"monto_clp"}, int_cols={"edad_evento"})

        # ----------------------------------------------------
        # 02 Flujos
        # ----------------------------------------------------
        ws = add_sheet("02 Flujos")
        write_title(ws, "Tabla de flujos", "Resume cuánta plata entra y sale por edad. Las cifras se muestran en CLP nominales de cada año/edad.")
        ws.set_column("A:A", 14)
        ws.set_column("B:I", 22)
        flows = make_monthly_cashflow_table(result)
        flows_by_age = flows.groupby("año_edad", as_index=False).agg(
            ahorro_promedio_simulado_clp=("ahorro_promedio_simulado_clp", "sum"),
            retiro_clp=("retiro_clp", "sum"),
            flujo_recurrente_neto_clp=("flujo_recurrente_neto_clp", "sum"),
            flujo_esporadico_clp=("flujo_esporadico_clp", "sum"),
            flujo_neto_antes_retorno_clp=("flujo_neto_antes_retorno_clp", "sum"),
            monto_que_debe_salir_del_fondo_clp=("monto_que_debe_salir_del_fondo_clp", "sum"),
        )
        flows_by_age.rename(columns={"año_edad": "edad"}, inplace=True)
        r = write_table(
            ws,
            flows_by_age,
            4,
            title="Flujos anuales por edad",
            money_cols={"ahorro_promedio_simulado_clp", "retiro_clp", "flujo_recurrente_neto_clp", "flujo_esporadico_clp", "flujo_neto_antes_retorno_clp", "monto_que_debe_salir_del_fondo_clp"},
            int_cols={"edad"},
        )
        # Formato condicional para flujo neto.
        last_row = 6 + len(flows_by_age)
        ws.conditional_format(6, 5, last_row, 5, {"type": "cell", "criteria": ">=", "value": 0, "format": fmt_good})
        ws.conditional_format(6, 5, last_row, 5, {"type": "cell", "criteria": "<", "value": 0, "format": fmt_bad})
        ws.conditional_format(6, 6, last_row, 6, {"type": "cell", "criteria": ">", "value": 0, "format": fmt_warn})
        # Gráfico ejecutivo de flujos anuales.
        if len(flows_by_age) > 0:
            chart = wb.add_chart({"type": "line"})
            data_first = 7
            data_last = 6 + len(flows_by_age)
            chart.add_series({
                "name": "Ahorro anual",
                "categories": ["02 Flujos", data_first, 0, data_last, 0],
                "values": ["02 Flujos", data_first, 1, data_last, 1],
                "line": {"color": "#30D158", "width": 2.25},
            })
            chart.add_series({
                "name": "Retiro anual",
                "categories": ["02 Flujos", data_first, 0, data_last, 0],
                "values": ["02 Flujos", data_first, 2, data_last, 2],
                "line": {"color": "#FF5C7A", "width": 2.25},
            })
            chart.add_series({
                "name": "Flujo neto antes de retorno",
                "categories": ["02 Flujos", data_first, 0, data_last, 0],
                "values": ["02 Flujos", data_first, 5, data_last, 5],
                "line": {"color": "#00D1FF", "width": 2.75},
            })
            chart.add_series({
                "name": "Monto que debe salir del fondo",
                "categories": ["02 Flujos", data_first, 0, data_last, 0],
                "values": ["02 Flujos", data_first, 6, data_last, 6],
                "line": {"color": "#FFC857", "width": 2.75, "dash_type": "dash"},
            })
            chart.set_title({"name": "Flujos anuales"})
            chart.set_x_axis({"name": "Edad"})
            chart.set_y_axis({"name": "CLP nominal"})
            chart.set_legend({"position": "bottom"})
            chart.set_size({"width": 720, "height": 360})
            ws.insert_chart("J5", chart)

        ws.write(r, 0, "Cómo leer", fmt_header)
        ws.merge_range(r, 1, r + 1, 8, "Si el flujo neto anual es negativo, el patrimonio invertido debe financiar esa diferencia además del efecto de mercado. Si es positivo, entra más plata de la que sale antes de retorno.", fmt_note)

        # ----------------------------------------------------
        # 03 FIRE / Coast
        # ----------------------------------------------------
        ws = add_sheet("03 FIRE Coast")
        write_title(ws, "FIRE y Coast FIRE por simulación", "No usa regla 4%. Usa tu retiro mensual deseado, ahorro, AFP, arriendos, eventos, inflación y retornos simulados.")
        ws.set_column("A:A", 30)
        ws.set_column("B:B", 24)
        ws.set_column("C:C", 80)
        fire_age = "No alcanza" if fire_row is None else int(fire_row["edad_retiro"])
        fire_capital = np.nan if fire_row is None else float(fire_row["capital_p50_retiro_clp"])
        fire_success = np.nan if fire_row is None else float(fire_row["prob_exito_pct"]) / 100
        fire_first_withdrawal = np.nan if fire_row is None else float(fire_row["primer_retiro_nominal_clp"])
        coast_age = "No alcanza" if coast_row is None else int(coast_row["edad_dejar_ahorrar"])
        coast_capital = np.nan if coast_row is None else float(coast_row["capital_p50_al_dejar_ahorrar_clp"])
        coast_success = np.nan if coast_row is None else float(coast_row["prob_exito_pct"]) / 100
        kpi = pd.DataFrame(
            [
                ["Edad FIRE anticipada", fire_age, "Primera edad evaluada donde puedes empezar a retirar y llegar a los 90 con el éxito objetivo."],
                ["Patrimonio para FIRE", fmt_clp(fire_capital), "Capital mediano esperado justo al comenzar ese FIRE."],
                ["Éxito FIRE", "N/A" if pd.isna(fire_success) else fmt_pct(fire_success * 100), "Probabilidad simulada de no agotar patrimonio."],
                ["Primer retiro nominal FIRE", fmt_clp(fire_first_withdrawal), "Retiro mensual nominal al iniciar el FIRE."],
                ["Edad Coast FIRE", coast_age, "Primera edad donde puedes dejar de ahorrar y aún retirarte a la edad elegida."],
                ["Patrimonio al Coast FIRE", fmt_clp(coast_capital), "Capital mediano esperado al momento de dejar de ahorrar."],
                ["Éxito Coast FIRE", "N/A" if pd.isna(coast_success) else fmt_pct(coast_success * 100), "Probabilidad simulada de no agotar patrimonio al retirarte en la edad elegida."],
            ],
            columns=["Métrica", "Valor", "Explicación simple"],
        )
        r = write_table(ws, kpi, 4, title="Resumen ejecutivo FIRE")
        # Resaltar visualmente los valores clave.
        ws.conditional_format(6, 1, 12, 1, {"type": "no_blanks", "format": fmt_card_label})
        ws.write(r, 0, "Interpretación", fmt_header)
        ws.merge_range(r, 1, r + 2, 8, "FIRE anticipado responde si puedes jubilar antes de la edad elegida. Coast FIRE responde desde qué edad puedes dejar de ahorrar, dejar invertido el patrimonio, y aun así jubilarte en la edad que elegiste.", fmt_note)
        r += 4
        if not fire_scan.empty:
            fire_display = fire_scan.copy()
            fire_display["prob_exito"] = fire_display["prob_exito_pct"] / 100
            fire_table_start = r
            fire_table_df = fire_display.drop(columns=[c for c in ["prob_exito_pct"] if c in fire_display.columns])
            r = write_table(
                ws,
                fire_table_df,
                r,
                title="Detalle FIRE por edad evaluada",
                money_cols={"capital_p5_retiro_clp", "capital_p50_retiro_clp", "capital_p95_retiro_clp", "patrimonio_p50_90_clp", "primer_retiro_nominal_clp"},
                pct_cols={"prob_exito"},
                int_cols={"edad_retiro"},
            )
            # Gráfico: probabilidad de éxito por edad evaluada.
            if len(fire_table_df) > 0 and "edad_retiro" in fire_table_df.columns and "prob_exito" in fire_table_df.columns:
                header_row = fire_table_start + 2
                first_row = header_row + 1
                last_row = header_row + len(fire_table_df)
                edad_col = list(fire_table_df.columns).index("edad_retiro")
                prob_col = list(fire_table_df.columns).index("prob_exito")
                chart = wb.add_chart({"type": "column"})
                chart.add_series({
                    "name": "Probabilidad de éxito",
                    "categories": ["03 FIRE Coast", first_row, edad_col, last_row, edad_col],
                    "values": ["03 FIRE Coast", first_row, prob_col, last_row, prob_col],
                    "fill": {"color": "#8B3DFF"},
                    "border": {"color": "#8B3DFF"},
                })
                chart.set_title({"name": "Éxito simulado por edad FIRE"})
                chart.set_x_axis({"name": "Edad de retiro"})
                chart.set_y_axis({"name": "Probabilidad", "num_format": "0%", "min": 0, "max": 1})
                chart.set_legend({"none": True})
                chart.set_size({"width": 720, "height": 330})
                ws.insert_chart("J5", chart)
        if not coast_scan.empty:
            coast_display = coast_scan.copy()
            coast_display["prob_exito"] = coast_display["prob_exito_pct"] / 100
            r = write_table(
                ws,
                coast_display.drop(columns=[c for c in ["prob_exito_pct"] if c in coast_display.columns]),
                r,
                title="Detalle Coast FIRE",
                money_cols={"capital_p50_al_dejar_ahorrar_clp", "capital_p50_al_fire_clp"},
                pct_cols={"prob_exito"},
                int_cols={"edad_dejar_ahorrar", "edad_fire"},
            )

        # ----------------------------------------------------
        # 04 Matriz FIRE
        # ----------------------------------------------------
        ws = add_sheet("04 Matriz FIRE")
        write_title(ws, "Matriz FIRE realista", "Cada celda muestra el capital nominal requerido a esa edad solo si tu plan simulado realmente alcanza esa combinación de edad y éxito. Si no alcanza, queda marcado como 'No alcanza'.")
        ws.set_column("A:A", 18)
        ws.set_column("B:Z", 18)
        if not realistic_matrix.empty:
            matrix = realistic_matrix.copy()
            matrix.index = [f"{idx:.0f}%" for idx in matrix.index]
            matrix.insert(0, "Éxito objetivo", matrix.index)
            r = 4
            ws.merge_range(r, 0, r, matrix.shape[1] - 1, "Capital nominal requerido CLP", fmt_section)
            r += 2
            # Manual para mostrar No alcanza sin romper formatos.
            for j, col in enumerate(matrix.columns):
                ws.write(r, j, str(col), fmt_header)
            for i in range(len(matrix)):
                for j, col in enumerate(matrix.columns):
                    val = matrix.iloc[i, j]
                    if j == 0:
                        ws.write(r + 1 + i, j, val, fmt_header)
                    elif pd.notna(val):
                        ws.write_number(r + 1 + i, j, float(val), fmt_money)
                    else:
                        ws.write(r + 1 + i, j, "No alcanza", fmt_bad)
            if matrix.shape[1] > 1 and len(matrix) > 0:
                ws.conditional_format(r + 1, 1, r + len(matrix), matrix.shape[1] - 1, {"type": "3_color_scale", "min_color": "#00D1FF", "mid_color": "#8B3DFF", "max_color": "#FF5C7A"})
            ws.write(r + len(matrix) + 3, 0, "Nota", fmt_header)
            ws.merge_range(r + len(matrix) + 3, 1, r + len(matrix) + 5, 8, "La matriz está en capital nominal de la edad evaluada. Puede crecer con la edad por inflación acumulada. Para comparar poder adquisitivo entre edades, hay que deflactar a pesos de hoy.", fmt_note)
        else:
            ws.write(4, 0, "La matriz FIRE no fue calculada todavía en la app.", fmt_note)

        # ----------------------------------------------------
        # 05 Calendario
        # ----------------------------------------------------
        ws = add_sheet("05 Calendario")
        write_title(ws, "Calendario anual del plan", "Línea de tiempo con hitos del plan: cambios de ahorro, flujos únicos, retiro, arriendos, AFP y edad de agotamiento si falla.")
        timeline = build_event_timeline_table(result, tabla, saving_ranges_df, recurring_df, lump_df, afp_info)
        timeline_excel = timeline.drop(columns=["Tipo evento"], errors="ignore")
        r = write_table(
            ws,
            timeline_excel,
            4,
            title="Tabla cronológica por edad",
            money_cols={
                "Patrimonio P50 inicio", "Patrimonio P50 fin", "Retiro mensual nominal",
                "Retiro mensual en pesos de hoy", "Ingresos recurrentes mensuales",
                "Egresos recurrentes mensuales", "Flujo único anual"
            },
            int_cols={"Año", "Edad"},
        )
        ws.write(r, 0, "Nota", fmt_header)
        ws.merge_range(r, 1, r + 1, 8, "Los montos nominales corresponden a pesos de cada año/edad. La columna en pesos de hoy deflacta por la inflación anual del escenario para facilitar la comparación temporal.", fmt_note)

        # ----------------------------------------------------
        # 06 Retiro sostenible
        # ----------------------------------------------------
        ws = add_sheet("06 Retiro sostenible")
        write_title(ws, "Retiro sostenible por edad de jubilación", "Para cada edad se busca por simulación el retiro mensual máximo en pesos de hoy que permite llegar a los 90 con la probabilidad objetivo. La regla 4% queda solo como referencia secundaria en el Excel.")
        if not retirement_sustainable.empty:
            ret4 = retirement_sustainable.copy()
            if "Prob. no agotar hasta 90" in ret4.columns:
                ret4["Probabilidad no agotar"] = ret4["Prob. no agotar hasta 90"] / 100
                ret4 = ret4.drop(columns=["Prob. no agotar hasta 90"])
            r = write_table(
                ws,
                ret4,
                4,
                title="Todas las edades evaluadas",
                money_cols={
                    "Patrimonio P50 nominal", "Patrimonio P50 pesos de hoy",
                    "Retiro sostenible mensual pesos de hoy", "Primer retiro sostenible nominal",
                    "Retiro deseado mensual pesos de hoy", "Brecha vs retiro deseado",
                    "Retiro mensual 4% pesos de hoy", "Retiro mensual 4% nominal"
                },
                pct_cols={"Probabilidad no agotar"},
                int_cols={"Edad jubilación", "Año calendario"},
            )
            ws.write(r, 0, "Interpretación", fmt_header)
            ws.merge_range(r, 1, r + 2, 8, "La columna principal es el retiro sostenible mensual en pesos de hoy: el mayor monto real que, según la simulación, permite llegar a los 90 con la probabilidad objetivo. El 4% se mantiene solo como referencia intuitiva y no controla la decisión.", fmt_note)
        else:
            ws.write(4, 0, "Calcula FIRE / Coast / Matriz en la app para generar la tabla de retiro sostenible.", fmt_note)

        # ----------------------------------------------------
        # 07 Percentiles
        # ----------------------------------------------------
        ws = add_sheet("07 Percentiles")
        write_title(ws, "Percentiles de patrimonio por edad", "Evolución anual del patrimonio simulado. P50 es la mediana; P5/P95 muestran escenarios pesimista/optimista.")
        percentiles = make_numeric_csv_table(tabla)
        r = write_table(
            ws,
            percentiles,
            4,
            title="Tabla anual por edad",
            money_cols={c for c in percentiles.columns if c.endswith("_clp")},
            int_cols={"edad", "año_simulación"},
        )
        # Gráfico de percentiles patrimoniales.
        needed_cols = ["edad", "p5_clp", "p50_mediana_clp", "p95_clp"]
        if len(percentiles) > 0 and all(c in percentiles.columns for c in needed_cols):
            header_row = 6
            first_row = 7
            last_row = 6 + len(percentiles)
            edad_col = list(percentiles.columns).index("edad")
            p5_col = list(percentiles.columns).index("p5_clp")
            p50_col = list(percentiles.columns).index("p50_mediana_clp")
            p95_col = list(percentiles.columns).index("p95_clp")
            chart = wb.add_chart({"type": "line"})
            chart.add_series({
                "name": "P5",
                "categories": ["07 Percentiles", first_row, edad_col, last_row, edad_col],
                "values": ["07 Percentiles", first_row, p5_col, last_row, p5_col],
                "line": {"color": "#FF5C7A", "width": 2.0},
            })
            chart.add_series({
                "name": "P50 mediana",
                "categories": ["07 Percentiles", first_row, edad_col, last_row, edad_col],
                "values": ["07 Percentiles", first_row, p50_col, last_row, p50_col],
                "line": {"color": "#00D1FF", "width": 2.75},
            })
            chart.add_series({
                "name": "P95",
                "categories": ["07 Percentiles", first_row, edad_col, last_row, edad_col],
                "values": ["07 Percentiles", first_row, p95_col, last_row, p95_col],
                "line": {"color": "#30D158", "width": 2.0},
            })
            chart.set_title({"name": "Evolución del patrimonio por percentiles"})
            chart.set_x_axis({"name": "Edad"})
            chart.set_y_axis({"name": "CLP nominal"})
            chart.set_legend({"position": "bottom"})
            chart.set_size({"width": 780, "height": 380})
            ws.insert_chart("J5", chart)

        # Anchos razonables globales.
        for sheet_name, ws in writer.sheets.items():
            ws.freeze_panes(4, 0)
            ws.set_zoom(90)

    output.seek(0)
    return output.getvalue()


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
install_clp_input_formatter()

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
            <div class="step-card"><strong>2. Ahorro</strong><span>Tramos por edad, en pesos de hoy.</span></div>
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

        f1, f2 = st.columns([1, 2])
        with f1:
            st.markdown(
                f"""<div class="mini-card"><b>Retiro mensual real deseado</b><span>{fmt_clp(withdrawal_monthly_clp)} en pesos de hoy</span></div>""",
                unsafe_allow_html=True,
            )
        with f2:
            st.caption(
                "El retiro que ingresas se interpreta como poder de compra de hoy. "
                "Si está indexado, la app lo lleva a monto nominal desde hoy hasta la edad de retiro y luego lo sigue indexando hasta los 90. "
                "FIRE y Coast FIRE se evalúan por simulación, no con una tasa fija tipo 4%."
            )
        panel_end()

    with input_tabs[1]:
        panel_start(
            "Ahorro mensual por edad",
            "Define tramos de ahorro antes del retiro. En cada tramo ingresas el ahorro objetivo mensual en pesos de hoy; la mayoría de los meses queda entre ese monto y $500.000 menos, algunos meses son peores y pocos meses superan el objetivo hasta $500.000.",
        )
        t1, t2 = st.columns([1, 1])
        with t1:
            contribution_timing_es = st.selectbox("Timing ahorro", ["Fin de mes", "Inicio de mes"], index=0)
        with t2:
            savings_indexed_to_inflation = st.checkbox(
                "Indexar ahorro por inflación",
                value=True,
                help="Si está activo, cada monto de ahorro se interpreta como pesos de hoy y sube con inflación hasta el retiro.",
            )

        SAVING_BAND_CLP = 500_000
        first_cut_age = min(35, int(edad_inicio_retiro))
        default_saving_rows = []
        if int(edad_inicial) < first_cut_age:
            default_saving_rows.append(
                {
                    "descripcion": "Hasta 35",
                    "edad_inicio": int(edad_inicial),
                    "edad_fin": int(first_cut_age),
                    "ahorro_esperado_clp": "3.000.000",
                }
            )
        if int(edad_inicio_retiro) > first_cut_age:
            default_saving_rows.append(
                {
                    "descripcion": "35 a retiro",
                    "edad_inicio": int(first_cut_age),
                    "edad_fin": int(edad_inicio_retiro),
                    "ahorro_esperado_clp": "2.000.000",
                }
            )
        if not default_saving_rows:
            default_saving_rows.append(
                {
                    "descripcion": "Ahorro hasta retiro",
                    "edad_inicio": int(edad_inicial),
                    "edad_fin": int(edad_inicio_retiro),
                    "ahorro_esperado_clp": "3.000.000",
                }
            )

        saving_ranges_input_df = pd.DataFrame(default_saving_rows)
        saving_ranges_df = st.data_editor(
            saving_ranges_input_df,
            width="stretch",
            num_rows="dynamic",
            hide_index=True,
            key="saving_ranges_editor",
            column_config={
                "descripcion": st.column_config.TextColumn("Descripción"),
                "edad_inicio": st.column_config.NumberColumn(
                    "Edad inicio",
                    min_value=int(edad_inicial),
                    max_value=EDAD_FINAL_FIJA,
                    step=1,
                ),
                "edad_fin": st.column_config.NumberColumn(
                    "Edad fin",
                    min_value=int(edad_inicial),
                    max_value=EDAD_FINAL_FIJA,
                    step=1,
                    help="El motor corta automáticamente el ahorro en la edad de retiro aunque pongas una edad mayor.",
                ),
                "ahorro_esperado_clp": st.column_config.TextColumn(
                    "Ahorro objetivo CLP",
                    help="Ej: 3.000.000. Mes normal típico: entre 2.500.000 y 3.000.000; pocos meses buenos pueden llegar a 3.500.000.",
                ),
            },
        )

        panel_end()

    with input_tabs[2]:
        panel_start(
            "Jubilación AFP estimada",
            "Usa los retornos reales anualizados informados por la Superintendencia de Pensiones: promedio y desviación estándar por fondo. La pensión se calcula por simulación y luego se indexa por inflación.",
        )
        afp_enable = st.checkbox("Agregar jubilación AFP calculada como ingreso recurrente", value=True)

        sp_returns_df = pd.DataFrame(
            [
                {
                    "Fondo": fondo,
                    "Retorno real anual promedio": f"{v['mean']*100:.2f}%".replace(".", ","),
                    "Desv. est. anual": f"{v['std']*100:.2f}%".replace(".", ","),
                }
                for fondo, v in AFP_RETURN_ASSUMPTIONS.items()
            ]
        )
        with st.expander("Ver supuestos SP usados para AFP", expanded=False):
            st.dataframe(sp_returns_df, width="stretch", hide_index=True)
            st.caption(
                "La tabla de la SP entrega retornos reales anualizados como promedio y desviación estándar. "
                "La app simula retornos aleatorios por path para proyectar el saldo AFP al jubilar. "
                "El paso de saldo AFP a pensión depende de factores actuariales/CNU/beneficiarios; por eso se usa un factor de conversión anual configurable."
            )

        afp1, afp2, afp3 = st.columns(3)
        with afp1:
            afp_balance_clp = money_text_input("Saldo AFP actual CLP", 40_000_000, key="afp_balance_clp_text")
        with afp2:
            afp_monthly_contribution_clp = money_text_input("Ahorro mensual AFP CLP", 600_000, key="afp_monthly_contribution_clp_text")
        with afp3:
            afp_retirement_age = st.number_input(
                "Edad jubilación AFP",
                min_value=int(edad_inicial),
                max_value=EDAD_FINAL_FIJA,
                value=min(max(65, int(edad_inicial)), EDAD_FINAL_FIJA),
                step=1,
            )

        afp4, afp5 = st.columns([2.4, 1.0])
        with afp4:
            afp_fund = st.radio(
                "Fondo AFP / supuesto SP",
                list(AFP_RETURN_ASSUMPTIONS.keys()),
                index=2,
                horizontal=True,
                help="Cada simulación toma retornos reales aleatorios usando el promedio y la desviación estándar del fondo elegido.",
            )
        with afp5:
            afp_conversion_factor_pct_input = st.number_input(
                "Factor conversión AFP anual (%)",
                min_value=0.0,
                max_value=12.0,
                value=5.0,
                step=0.1,
                help="Aproxima la pensión anual como porcentaje del saldo AFP al jubilar. El simulador SP suele incorporar CNU, edad, sexo y beneficiarios; por eso este factor puede ser mayor que 3,2%.",
            )

        afp_pension_percentile_label = st.radio(
            "Escenario de saldo AFP usado para pensión",
            list(AFP_PERCENTILE_OPTIONS.keys()),
            index=1,
            horizontal=True,
            help="Define qué percentil del saldo AFP simulado se usa para calcular la pensión mensual que entra al flujo patrimonial.",
        )

        with st.expander("Calibrar contra el simulador de la SP", expanded=False):
            sp_reference_pension_clp = money_text_input(
                "Pensión esperada SP mensual CLP opcional",
                0,
                key="afp_sp_reference_pension_clp_text",
                help="Si ingresas la pensión esperada que muestra el simulador de la SP, la app recalcula automáticamente el factor de conversión anual para empatar ese monto.",
            )
            st.caption(
                "Úsalo si quieres que esta app parta desde el mismo número AFP que te entrega el simulador oficial. "
                "Ejemplo: si la SP muestra $2.701.927, escríbelo acá."
            )

        afp_return_mean = float(AFP_RETURN_ASSUMPTIONS[afp_fund]["mean"])
        afp_return_std = float(AFP_RETURN_ASSUMPTIONS[afp_fund]["std"])
        afp_selected_percentile = int(AFP_PERCENTILE_OPTIONS[afp_pension_percentile_label])

        months_to_afp = max(int(round((float(afp_retirement_age) - float(edad_inicial)) * 12)), 0)
        afp_sim = simulate_afp_future_balance_distribution_real_clp(
            initial_balance_clp=afp_balance_clp,
            monthly_contribution_clp=afp_monthly_contribution_clp,
            annual_real_return_mean=afp_return_mean,
            annual_real_return_std=afp_return_std,
            n_months=months_to_afp,
            n_paths=20_000,
            seed=2026,
            contribution_timing="end",
        )
        afp_balance_percentiles = afp_sim["percentiles_balance_real_clp"]
        afp_selected_key = f"p{afp_selected_percentile}"
        afp_fv_real_clp = float(afp_balance_percentiles.get(afp_selected_key, afp_balance_percentiles["p50"]))
        afp_fv_real_mean_clp = float(afp_sim["mean_balance_real_clp"])

        if float(sp_reference_pension_clp) > 0 and afp_fv_real_clp > 0:
            afp_conversion_factor_pct = float(sp_reference_pension_clp) * 12 / afp_fv_real_clp * 100
            afp_conversion_source = "Calibrado contra simulador SP"
        else:
            afp_conversion_factor_pct = float(afp_conversion_factor_pct_input)
            afp_conversion_source = "Manual / aproximado"

        afp_monthly_pension_real_clp = afp_fv_real_clp * (float(afp_conversion_factor_pct) / 100) / 12
        afp_monthly_inflation = (1 + float(inflation_annual_pct) / 100) ** (1 / 12) - 1 if float(inflation_annual_pct) > -100 else 0.0
        afp_monthly_pension_nominal_start_clp = afp_monthly_pension_real_clp * (1 + afp_monthly_inflation) ** months_to_afp

        afp_summary = st.columns(4)
        with afp_summary[0]:
            st.markdown(f"""<div class="mini-card"><b>Fondo usado</b><span>{afp_fund}<br>μ {fmt_pct(afp_return_mean*100, 2)} · σ {fmt_pct(afp_return_std*100, 2)}</span></div>""", unsafe_allow_html=True)
        with afp_summary[1]:
            st.markdown(f"""<div class="mini-card"><b>Saldo AFP al jubilar</b><span>{fmt_clp(afp_fv_real_clp)} de hoy<br>{afp_pension_percentile_label}</span></div>""", unsafe_allow_html=True)
        with afp_summary[2]:
            st.markdown(f"""<div class="mini-card"><b>Pensión real mensual</b><span>{fmt_clp(afp_monthly_pension_real_clp)} de hoy<br>Factor {fmt_pct(afp_conversion_factor_pct, 2)}</span></div>""", unsafe_allow_html=True)
        with afp_summary[3]:
            st.markdown(f"""<div class="mini-card"><b>Pensión nominal inicial</b><span>{fmt_clp(afp_monthly_pension_nominal_start_clp)} a los {int(afp_retirement_age)}<br>{afp_conversion_source}</span></div>""", unsafe_allow_html=True)

        afp_dist_df = pd.DataFrame(
            {
                "Percentil": ["P5", "P25", "P50", "P75", "P95", "Promedio"],
                "Saldo AFP real al jubilar": [
                    fmt_clp(afp_balance_percentiles["p5"]),
                    fmt_clp(afp_balance_percentiles["p25"]),
                    fmt_clp(afp_balance_percentiles["p50"]),
                    fmt_clp(afp_balance_percentiles["p75"]),
                    fmt_clp(afp_balance_percentiles["p95"]),
                    fmt_clp(afp_fv_real_mean_clp),
                ],
                "Pensión mensual real aprox.": [
                    fmt_clp(afp_balance_percentiles["p5"] * (float(afp_conversion_factor_pct) / 100) / 12),
                    fmt_clp(afp_balance_percentiles["p25"] * (float(afp_conversion_factor_pct) / 100) / 12),
                    fmt_clp(afp_balance_percentiles["p50"] * (float(afp_conversion_factor_pct) / 100) / 12),
                    fmt_clp(afp_balance_percentiles["p75"] * (float(afp_conversion_factor_pct) / 100) / 12),
                    fmt_clp(afp_balance_percentiles["p95"] * (float(afp_conversion_factor_pct) / 100) / 12),
                    fmt_clp(afp_fv_real_mean_clp * (float(afp_conversion_factor_pct) / 100) / 12),
                ],
            }
        )
        with st.expander("Ver distribución AFP simulada", expanded=False):
            st.dataframe(afp_dist_df, width="stretch", hide_index=True)
        st.caption("Si el ahorro AFP ya está dentro de tus tramos de ahorro patrimonial, evita duplicarlo. La pensión AFP que entra al flujo patrimonial es el escenario seleccionado arriba y se indexa por inflación.")
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
        panel_start("Retorno, riesgo y simulación", "El modelo de regímenes usa retorno nominal esperado, volatilidad anual y años normales/malos/crisis. Dentro de cada año los meses también varían, con colas pesadas.")
        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            return_model_es = st.selectbox(
                "Modelo retorno",
                [
                    "Regímenes realistas: 75% normal / 20% malo / 5% crisis",
                    "Mensual IID más realista para retiro",
                    "Anual suavizado como código original",
                ],
                index=0,
            )
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
    """Convierte la tabla de tramos de ahorro a MM CLP para el motor.

    Formato actual de UI:
      - ahorro_esperado_clp: ahorro objetivo por tramo.
      - el motor usa una mezcla asimétrica: 80% normal entre objetivo-$500.000 y objetivo; 15% malo; 5% muy bueno.

    También mantiene compatibilidad con versiones antiguas que tenían
    ahorro_min_clp / ahorro_probable_clp / ahorro_max_clp.
    """
    ranges: list[tuple[float, Optional[float], float, float, float, str]] = []
    if df is None or df.empty:
        return tuple(ranges)

    saving_band_clp = 500_000

    for _, row in df.dropna(subset=["edad_inicio", "edad_fin"]).iterrows():
        if "ahorro_esperado_clp" in df.columns:
            mode_clp = parse_clp_value(row.get("ahorro_esperado_clp", 0))
            min_clp = max(mode_clp - saving_band_clp, 0)
            max_clp = mode_clp + saving_band_clp if mode_clp > 0 else 0
        else:
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
        "fondo": str(afp_fund),
        "fuente_retornos": "Superintendencia de Pensiones",
        "retorno_real_anual": float(afp_return_mean),
        "desv_est_real_anual": float(afp_return_std),
        "escenario_pension": str(afp_pension_percentile_label),
        "percentil_pension": int(afp_selected_percentile),
        "factor_conversion_anual": float(afp_conversion_factor_pct) / 100,
        "factor_conversion_fuente": str(afp_conversion_source),
        "pension_sp_referencia_clp": float(sp_reference_pension_clp),
        "tasa_retiro_anual": float(afp_conversion_factor_pct) / 100,  # alias compatible
        "saldo_estimado_real_clp": float(afp_fv_real_clp),
        "saldo_promedio_real_clp": float(afp_fv_real_mean_clp),
        "saldo_p5_real_clp": float(afp_balance_percentiles["p5"]),
        "saldo_p25_real_clp": float(afp_balance_percentiles["p25"]),
        "saldo_p50_real_clp": float(afp_balance_percentiles["p50"]),
        "saldo_p75_real_clp": float(afp_balance_percentiles["p75"]),
        "saldo_p95_real_clp": float(afp_balance_percentiles["p95"]),
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
    if return_model_es.startswith("Regímenes"):
        return_model = "regime_realistic"
    elif return_model_es.startswith("Mensual"):
        return_model = "monthly_iid"
    else:
        return_model = "annual_smooth"

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
                withdrawal_index_base_age=float(edad_inicial),
                savings_indexed_to_inflation=bool(savings_indexed_to_inflation),
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
            st.session_state["mc_fire_info"] = {
                "retiro_mensual_real_clp": int(withdrawal_monthly_clp),
                "retiro_anual_real_clp": int(withdrawal_monthly_clp) * 12,
                "retiro_indexado": bool(withdrawal_indexed_to_inflation),
                "inflacion_anual": float(inflation_annual_pct) / 100,
            }
            st.session_state["mc_export_ready_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.pop("mc_retirement_matrix", None)
            st.session_state.pop("mc_fire_analysis", None)
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

# Vista ejecutiva: solo las 4 métricas principales solicitadas.
kpi_cols = st.columns(4)
with kpi_cols[0]:
    metric_card(
        "Patrimonio mediano al iniciar retiro",
        fmt_clp_from_mm(ret_p50_mm),
        "P50 del patrimonio justo cuando se corta el ahorro e inicia el retiro.",
        "primary",
    )
with kpi_cols[1]:
    metric_card(
        "Patrimonio mediano a los 90",
        fmt_clp_from_mm(final_p50_mm),
        "P50 del patrimonio al final del horizonte de simulación.",
        "cyan",
    )
with kpi_cols[2]:
    metric_card(
        "Probabilidad de no agotar patrimonio",
        fmt_pct(prob_no_ruin),
        "Porcentaje de simulaciones que llegan a los 90 sin caer a cero.",
        survival_tone(prob_no_ruin),
    )
with kpi_cols[3]:
    metric_card(
        "Si falla, edad mediana de agotamiento",
        "No se agota" if np.isnan(median_ruin_age) else f"{median_ruin_age:,.1f} años".replace(".", ","),
        "Solo mira las simulaciones que sí llegan a cero; no es la edad esperada de todos los escenarios.",
        "good" if np.isnan(median_ruin_age) else "bad",
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
elif result["inputs"].get("return_model") == "regime_realistic":
    st.info(
        "Modo regímenes realistas: cada año se clasifica como normal, malo o crisis con probabilidades 75% / 20% / 5%. "
        "Dentro de cada año los meses también varían y usan colas pesadas, por lo que pueden aparecer meses malos incluso en años normales."
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
    ["Percentiles", "Flujos", "Paths", "Distribución final", "Agotamiento", "FIRE / Coast / Matriz", "Calculadora ahorro"]
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
            st.markdown(f"""<div class="definition-card"><b>Supuestos</b><br><span>{afp.get('fondo', 'AFP')} · μ {fmt_pct(afp['retorno_real_anual']*100, 2)} · σ {fmt_pct(afp.get('desv_est_real_anual', 0)*100, 2)} · retiro {fmt_pct(afp['tasa_retiro_anual']*100)}</span></div>""", unsafe_allow_html=True)
    # Se eliminaron las tablas/vistas auxiliares con montos formateados bajo el gráfico de flujos.
    # El detalle numérico queda disponible dentro del reporte ejecutivo Excel.

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
            <b>FIRE anticipado / Coast FIRE / Matriz realista</b><br>
            <span>Busca por simulación la primera edad en que puedes retirar el monto mensual deseado —expresado en plata de hoy e indexado si corresponde— sin agotar patrimonio hasta los 90.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Esta sección no usa tasa FIRE. Usa el retiro mensual que ingresaste, tus tramos de ahorro, AFP, arriendos, inflación, eventos únicos y retornos. "
        "La matriz muestra capital nominal requerido solo en las edades donde el plan simulado realmente alcanza el éxito objetivo; si no, marca 'No alcanza'."
    )

    current_retirement_age = int(result["inputs"]["edad_inicio_retiro"])
    current_retirement_month = int(result["inputs"].get("retirement_start_month", 0))
    current_withdrawal_schedule = result.get("withdrawal_schedule_mm")
    current_first_withdrawal_nominal = np.nan
    if current_withdrawal_schedule is not None and 0 <= current_retirement_month < len(current_withdrawal_schedule):
        current_first_withdrawal_nominal = float(current_withdrawal_schedule[current_retirement_month]) * 1_000_000
    current_capital_at_retirement = float(np.percentile(result["wealth_at_retirement_mm"], 50)) * 1_000_000

    chosen_cols = st.columns(4)
    with chosen_cols[0]:
        metric_card(
            "Retiro elegido en inputs",
            f"{current_retirement_age} años",
            "Este es el escenario principal cargado arriba; abajo buscamos si puede ser antes.",
            "primary",
        )
    with chosen_cols[1]:
        metric_card(
            "Éxito del retiro elegido",
            fmt_pct(result.get("prob_no_ruin", np.nan) * 100),
            "Probabilidad de llegar a los 90 sin agotar patrimonio con la edad elegida.",
            survival_tone(result.get("prob_no_ruin", np.nan) * 100),
        )
    with chosen_cols[2]:
        metric_card(
            "Capital mediano al retiro elegido",
            fmt_clp(current_capital_at_retirement),
            "P50 del patrimonio justo al comenzar el retiro elegido.",
            "cyan",
        )
    with chosen_cols[3]:
        metric_card(
            "Primer retiro nominal",
            fmt_clp(current_first_withdrawal_nominal),
            "Tu retiro deseado en plata de hoy, llevado a pesos nominales de esa edad.",
            "orange",
        )

    mx1, mx2, mx3, mx4 = st.columns([2.2, 1.4, 1.4, 1.2])
    with mx1:
        default_matrix_ages = [35, 37, 40, 43, 45, 48, 50, 55, 60, 65]
        default_ages_text = ", ".join(
            str(x)
            for x in default_matrix_ages
            if int(result["inputs"]["edad_inicial"]) <= x < EDAD_FINAL_FIJA
        )
        if not default_ages_text:
            default_ages_text = str(min(max(current_retirement_age, int(result["inputs"]["edad_inicial"])), EDAD_FINAL_FIJA - 1))
        matrix_ages_text = st.text_input(
            "Edades a evaluar",
            value=default_ages_text,
            help="Ejemplo: 35, 37, 40, 43, 45, 48, 50, 55, 60, 65",
        )
    with mx2:
        matrix_probs_text = st.text_input(
            "Éxitos objetivo (%)",
            value="70, 80, 90, 95",
            help="Filas de la matriz y umbrales de FIRE.",
        )
    with mx3:
        fire_target_success_pct = st.selectbox(
            "Éxito para FIRE/Coast",
            options=[70, 80, 90, 95],
            index=2,
            help="Umbral usado para encontrar la edad FIRE mínima y la edad Coast FIRE.",
        )
    with mx4:
        matrix_n_paths = st.number_input(
            "Simulaciones análisis",
            min_value=2_000,
            max_value=50_000,
            value=10_000,
            step=2_000,
            format="%d",
            help="Corre simulaciones adicionales para cada edad. Si queda lento, baja este número.",
        )

    calc_matrix = st.button("Calcular FIRE / Coast / matriz realista", type="primary")
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
            with st.spinner("Calculando FIRE realista, Coast FIRE y matriz..."):
                saving_ranges_for_scan = result["inputs"].get("saving_ranges") or tuple()
                recurring_events_for_scan = st.session_state.get("mc_recurring_events", tuple())
                lump_events_monthly_for_scan = result["inputs"].get("lump_sum_events") or tuple()
                lump_events_age_for_matrix = st.session_state.get("mc_lump_age_events", tuple())

                fire_scan_df = run_fire_scan(
                    base_result=result,
                    retirement_ages=matrix_ages,
                    n_paths=int(matrix_n_paths),
                    saving_ranges=saving_ranges_for_scan,
                    recurring_events=recurring_events_for_scan,
                    lump_events_monthly=lump_events_monthly_for_scan,
                )

                required_matrix = required_capital_matrix_mm(
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
                    lump_sum_age_events=lump_events_age_for_matrix,
                    recurring_monthly_events=recurring_events_for_scan,
                    return_model=str(result["inputs"].get("return_model", "monthly_iid")),
                    withdrawal_indexed_to_inflation=bool(result["inputs"].get("withdrawal_indexed_to_inflation", False)),
                    inflation_annual=float(result["inputs"].get("inflation_annual", 0.0)),
                    withdrawal_index_base_age=float(result["inputs"].get("withdrawal_index_base_age", result["inputs"].get("edad_inicial"))),
                )
                realistic_matrix = build_realistic_matrix(required_matrix["matrix_clp"], fire_scan_df)

                # FIRE anticipado: busca la primera edad evaluada donde podrías retirarte
                # con el retiro mensual deseado y el éxito objetivo.
                target_row = find_fire_row(fire_scan_df, float(fire_target_success_pct))

                # Coast FIRE / Cost FIRE: responde otra pregunta distinta.
                # Mantiene la edad de retiro que el usuario eligió en el escenario base,
                # y busca desde qué edad podría dejar de ahorrar y aun así llegar a esa
                # edad de retiro con el éxito objetivo.
                planned_fire_age = int(result["inputs"].get("edad_inicio_retiro"))
                coast_df = run_coast_scan(
                    base_result=result,
                    fire_age=planned_fire_age,
                    target_success_pct=float(fire_target_success_pct),
                    n_paths=int(matrix_n_paths),
                    saving_ranges=saving_ranges_for_scan,
                    recurring_events=recurring_events_for_scan,
                    lump_events_monthly=lump_events_monthly_for_scan,
                )

                # Umbral usado por la tabla de retiro sostenible por edad.
                st.session_state["mc_fire_analysis_target_success_pct"] = float(fire_target_success_pct)
                sustainable_withdrawal_df = run_sustainable_withdrawal_by_retirement_age(
                    base_result=result,
                    saving_ranges=saving_ranges_for_scan,
                    recurring_events=recurring_events_for_scan,
                    lump_events_monthly=lump_events_monthly_for_scan,
                    lump_age_events=lump_events_age_for_matrix,
                    n_paths=min(int(matrix_n_paths), 3_000),
                    withdrawal_rate_annual=0.04,
                )

                analysis = {
                    "required_matrix": required_matrix,
                    "realistic_matrix_clp": realistic_matrix,
                    "fire_scan": fire_scan_df,
                    "coast_scan": coast_df,
                    "sustainable_withdrawal_by_age": sustainable_withdrawal_df,
                    "target_success_pct": float(fire_target_success_pct),
                    "analysis_version": FIRE_ANALYSIS_VERSION,
                    "planned_fire_age": planned_fire_age,
                    "planned_fire_success_pct": float(result["prob_no_ruin"] * 100),
                    "ages": matrix_ages,
                    "probabilities": matrix_probs,
                }
                st.session_state["mc_retirement_matrix"] = required_matrix
                st.session_state["mc_fire_analysis"] = analysis
                st.success("Análisis FIRE / Coast / matriz calculado.")

    analysis = st.session_state.get("mc_fire_analysis")
    if analysis is not None and analysis.get("analysis_version") != FIRE_ANALYSIS_VERSION:
        st.session_state.pop("mc_fire_analysis", None)
        analysis = None
    if analysis is None:
        st.warning("Presiona **Calcular FIRE / Coast / matriz realista** para generar el análisis actualizado.")
    else:
        fire_scan_df = analysis.get("fire_scan", pd.DataFrame())
        realistic_matrix_clp = analysis.get("realistic_matrix_clp", pd.DataFrame())
        required_matrix = analysis.get("required_matrix")
        target_success_pct = float(analysis.get("target_success_pct", 90))
        target_row = find_fire_row(fire_scan_df, target_success_pct)
        coast_scan_df = analysis.get("coast_scan", pd.DataFrame())
        coast_ok = coast_scan_df[coast_scan_df["cumple_objetivo"]].sort_values("edad_dejar_ahorrar") if coast_scan_df is not None and not coast_scan_df.empty else pd.DataFrame()
        coast_row = coast_ok.iloc[0] if not coast_ok.empty else None

        planned_fire_age = int(analysis.get("planned_fire_age", result["inputs"].get("edad_inicio_retiro")))
        planned_fire_success_pct = float(analysis.get("planned_fire_success_pct", result["prob_no_ruin"] * 100))

        st.markdown("#### Resultado FIRE realista")
        st.caption(
            f"FIRE anticipado busca la primera edad evaluada donde podrías retirarte con {fmt_pct(target_success_pct)} de éxito. "
            f"Coast FIRE usa tu edad de retiro elegida ({planned_fire_age} años) y busca desde qué edad puedes dejar de ahorrar."
        )
        out_cols = st.columns(4)
        if target_row is None:
            with out_cols[0]:
                metric_card("Edad FIRE mínima", "No alcanza", f"Ninguna edad evaluada llega a {fmt_pct(target_success_pct)} de éxito.", "bad")
            with out_cols[1]:
                metric_card("Capital mediano al FIRE", "N/A", "No hay edad viable dentro de las evaluadas.", "bad")
            with out_cols[2]:
                metric_card("Primer retiro nominal", "N/A", "No hay edad viable dentro de las evaluadas.", "bad")
            with out_cols[3]:
                if coast_row is None:
                    metric_card(
                        "Coast FIRE / Cost FIRE",
                        "No alcanza",
                        f"Retiro objetivo a los {planned_fire_age} años: {fmt_pct(planned_fire_success_pct, 2)} de éxito.",
                        "bad",
                    )
                else:
                    metric_card(
                        "Coast FIRE / Cost FIRE",
                        f"{int(coast_row['edad_dejar_ahorrar'])} años",
                        f"Para retirarte a los {planned_fire_age} años con {fmt_pct(target_success_pct)}+ de éxito.",
                        "primary",
                    )
        else:
            with out_cols[0]:
                metric_card(
                    "Edad FIRE mínima",
                    f"{int(target_row['edad_retiro'])} años",
                    f"Primera edad evaluada con al menos {fmt_pct(target_success_pct)} de éxito.",
                    "good",
                )
            with out_cols[1]:
                metric_card(
                    "Capital mediano al FIRE",
                    fmt_clp(target_row["capital_p50_retiro_clp"]),
                    "P50 del patrimonio acumulado justo al empezar a retirar.",
                    "cyan",
                )
            with out_cols[2]:
                metric_card(
                    "Primer retiro nominal",
                    fmt_clp(target_row["primer_retiro_nominal_clp"]),
                    "Monto mensual de inicio en pesos de esa edad.",
                    "orange",
                )
            with out_cols[3]:
                if coast_row is None:
                    metric_card(
                        "Coast FIRE / Cost FIRE",
                        "No alcanza",
                        f"Aun ahorrando hasta los {planned_fire_age} años, el escenario llega a {fmt_pct(planned_fire_success_pct, 2)} de éxito.",
                        "bad",
                    )
                else:
                    metric_card(
                        "Coast FIRE / Cost FIRE",
                        f"{int(coast_row['edad_dejar_ahorrar'])} años",
                        f"Puedes dejar de ahorrar ahí y retirarte a los {planned_fire_age} años. Capital P50: {fmt_clp(coast_row['capital_p50_al_dejar_ahorrar_clp'])}.",
                        "primary",
                    )

        st.markdown("#### Matriz realista de capital nominal requerido")
        st.caption(
            "Cada celda muestra el capital nominal requerido en esa edad solo si el escenario completo logra el éxito de esa fila. "
            "Si dice 'No alcanza', con tus supuestos de ahorro/capital/ingresos no llegarías a esa condición en esa edad."
        )
        st.plotly_chart(plot_realistic_required_capital_heatmap(realistic_matrix_clp), width="stretch")
        st.dataframe(format_realistic_matrix_clp(realistic_matrix_clp), width="stretch")

        st.markdown("#### NUEVO · Retiro máximo sostenible por edad de jubilación")
        st.caption(
            "Esta ya no es una tabla 4%. Para cada edad posible de jubilación se corre una búsqueda por simulación: encuentra el retiro mensual máximo en pesos de hoy que permite llegar a los 90 con la probabilidad objetivo. Luego lo compara contra tu retiro mensual deseado y muestra si Alcanza, Cerca o No alcanza."
        )
        sustainable_withdrawal_df = analysis.get("sustainable_withdrawal_by_age", pd.DataFrame())
        if sustainable_withdrawal_df is None or sustainable_withdrawal_df.empty:
            st.info("La tabla de retiro sostenible no está disponible todavía. Vuelve a calcular FIRE / Coast / matriz.")
        else:
            st.dataframe(style_sustainable_withdrawal_by_retirement_age(sustainable_withdrawal_df), width="stretch", hide_index=True, height=760)

        excel_fire_report = make_executive_excel_report(
            result,
            tabla,
            st.session_state.get("mc_saving_ranges_df"),
            st.session_state.get("mc_recurring_df"),
            st.session_state.get("mc_lump_df"),
            st.session_state.get("mc_afp_info"),
            analysis,
        )
        st.markdown(
            """
            <div class="excel-export-hero">
                <div>
                    <div class="excel-export-title">Descarga el reporte ejecutivo completo</div>
                    <div class="excel-export-subtitle">Un solo Excel con inputs, flujos, FIRE, Coast FIRE, matriz realista, percentiles y gráficos. Ideal para entregar o revisar el escenario con alguien que no mira el código.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇️ DESCARGAR EXCEL EJECUTIVO COMPLETO",
            data=excel_fire_report,
            file_name="reporte_fire_cliente.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Archivo único con colores, tablas explicativas, matriz FIRE y gráficos.",
            key="download_excel_fire_unico",
            width="stretch",
        )
        st.divider()

        st.markdown("#### Auditoría por edad")
        audit = fire_scan_df.copy()
        if not audit.empty:
            audit_display = audit.copy()
            for col in ["capital_p5_retiro_clp", "capital_p50_retiro_clp", "capital_p95_retiro_clp", "patrimonio_p50_90_clp", "primer_retiro_nominal_clp"]:
                audit_display[col] = audit_display[col].apply(fmt_clp)
            audit_display["prob_exito_pct"] = audit_display["prob_exito_pct"].apply(lambda x: fmt_pct(x, 2))
            st.dataframe(audit_display, width="stretch", hide_index=True)

        if coast_scan_df is not None and not coast_scan_df.empty:
            with st.expander("Detalle Coast FIRE / Cost FIRE", expanded=False):
                coast_display = coast_scan_df.copy()
                for col in ["capital_p50_al_dejar_ahorrar_clp", "capital_p50_al_fire_clp"]:
                    coast_display[col] = coast_display[col].apply(fmt_clp)
                coast_display["prob_exito_pct"] = coast_display["prob_exito_pct"].apply(lambda x: fmt_pct(x, 2))
                st.dataframe(coast_display, width="stretch", hide_index=True)

        with st.expander("Cómo leer esta sección", expanded=False):
            st.markdown(
                """
                - **Edad FIRE mínima:** primera edad evaluada donde podrías empezar a retirar el monto mensual deseado y llegar a los 90 con el éxito objetivo. Puede ser antes que tu edad de retiro elegida.
                - **Coast FIRE / Cost FIRE:** primera edad donde podrías dejar de ahorrar, mantener el capital invertido, y aun así retirarte a la edad que elegiste en el escenario base.
                - **Matriz realista:** muestra capital nominal requerido solo en las edades donde tu plan simulado realmente logra el éxito de esa fila. Si dice **No alcanza**, no basta con mirar el capital requerido teórico: con tus supuestos no llegarías a esa condición.
                - El retiro mensual ingresado se interpreta como plata de hoy. Si está indexado, el primer retiro nominal aumenta según la inflación acumulada hasta la edad de retiro.
                """
            )


with tab7:
    st.markdown(
        """
        <div class="download-card">
            <b>Calculadora de ahorro mínimo mensual</b><br>
            <span>Busca cuánto deberías ahorrar cada mes, en pesos de hoy, para llegar a tu meta de patrimonio al iniciar retiro y además no agotar el patrimonio hasta los 90 con el nivel de confianza elegido.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Esta calculadora usa los mismos supuestos del escenario principal: capital inicial, edad de retiro, retiro mensual deseado, inflación, AFP, arriendos, eventos únicos y retornos. "
        "Solo reemplaza los tramos de ahorro por un ahorro objetivo mensual único; el motor aplica la mezcla realista asimétrica."
    )

    calc_cols = st.columns([1.1, 1.1, 1.1, 1.1])
    with calc_cols[0]:
        calc_target_success_pct = st.selectbox(
            "Confianza objetivo",
            options=[80, 85, 90, 95],
            index=2,
            help="Se exige cumplir simultáneamente la meta de patrimonio al retiro y no agotar el patrimonio hasta los 90.",
        )
    with calc_cols[1]:
        calc_max_saving_clp = money_text_input(
            "Ahorro máximo a probar",
            8_000_000,
            key="calc_max_saving_clp",
            help="Si ni este ahorro alcanza, la calculadora marcará que no alcanza con los supuestos actuales.",
        )
    with calc_cols[2]:
        calc_n_paths = st.number_input(
            "Simulaciones calculadora",
            min_value=2_000,
            max_value=50_000,
            value=10_000,
            step=2_000,
            format="%d",
        )
    with calc_cols[3]:
        calc_precision_clp = money_text_input(
            "Precisión aproximada",
            50_000,
            key="calc_precision_clp",
            help="La búsqueda binaria se detiene cerca de este orden de magnitud. Montos más finos tardan más.",
        )

    st.markdown(
        f"""
        <div class="workflow-note">
            <b>Lectura:</b> si el resultado dice {fmt_clp(3_200_000)}, significa que el modelo estima que deberías ahorrar cerca de ese monto mensual, en pesos de hoy, hasta los {int(result['inputs']['edad_inicio_retiro'])} años. Si tienes activada la indexación del ahorro, ese monto sube con inflación en la simulación.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Calcular ahorro mínimo", type="primary"):
        if calc_max_saving_clp <= 0:
            st.error("El ahorro máximo a probar debe ser mayor que cero.")
        else:
            with st.spinner("Buscando ahorro mensual mínimo con simulaciones..."):
                # Iteraciones suficientes para llegar a una precisión práctica sin sobrecargar Streamlit Cloud.
                span = max(float(calc_max_saving_clp), 1.0)
                max_iter_calc = int(np.ceil(np.log2(span / max(float(calc_precision_clp), 1.0)))) + 2
                max_iter_calc = min(max(max_iter_calc, 8), 16)
                calc_result = run_minimum_saving_calculator(
                    base_result=result,
                    monthly_saving_low_clp=0.0,
                    monthly_saving_high_clp=float(calc_max_saving_clp),
                    target_success_pct=float(calc_target_success_pct),
                    n_paths=int(calc_n_paths),
                    recurring_events=st.session_state.get("mc_recurring_events", tuple()),
                    lump_events_monthly=tuple(st.session_state.get("mc_lump_age_events", tuple())),
                    max_iter=max_iter_calc,
                )
                st.session_state["mc_min_saving_calc"] = calc_result

    calc_result = st.session_state.get("mc_min_saving_calc")
    if calc_result:
        best = calc_result.get("best_result", {})
        if calc_result.get("status") == "ok":
            st.success("Se encontró un ahorro mensual que cumple la condición de confianza.")
            rcols = st.columns(4)
            with rcols[0]:
                metric_card(
                    "Ahorro mensual mínimo",
                    fmt_clp(best.get("saving_center_clp", np.nan)),
                    "Ahorro objetivo en pesos de hoy. La mayoría de los meses queda entre este monto y $500.000 menos; pocos meses superan el objetivo.",
                    "good",
                )
            with rcols[1]:
                metric_card(
                    "Éxito conjunto",
                    fmt_pct(best.get("joint_success_pct", np.nan), 2),
                    "Meta al retiro + no agotar patrimonio hasta los 90.",
                    survival_tone(best.get("joint_success_pct", np.nan)),
                )
            with rcols[2]:
                metric_card(
                    "Capital P50 al retiro",
                    fmt_clp(best.get("capital_p50_retiro_clp", np.nan)),
                    "Patrimonio mediano estimado al iniciar el retiro.",
                    "cyan",
                )
            with rcols[3]:
                metric_card(
                    "Patrimonio P50 a los 90",
                    fmt_clp(best.get("patrimonio_p50_90_clp", np.nan)),
                    "Patrimonio mediano final después de retiros e ingresos.",
                    "primary",
                )

            band_cols = st.columns(3)
            with band_cols[0]:
                st.markdown(f"""<div class="definition-card"><b>Banda usada por el motor</b><br><span>Mínimo: {fmt_clp(best.get('saving_min_clp', np.nan))}<br>Esperado: {fmt_clp(best.get('saving_center_clp', np.nan))}<br>Máximo: {fmt_clp(best.get('saving_max_clp', np.nan))}</span></div>""", unsafe_allow_html=True)
            with band_cols[1]:
                st.markdown(f"""<div class="definition-card"><b>Validación separada</b><br><span>Meta al retiro: {fmt_pct(best.get('prob_target_retirement_pct', np.nan), 2)}<br>No agotar: {fmt_pct(best.get('prob_no_ruin_pct', np.nan), 2)}</span></div>""", unsafe_allow_html=True)
            with band_cols[2]:
                st.markdown(f"""<div class="definition-card"><b>Meta del escenario</b><br><span>Capital objetivo: {fmt_clp_from_mm(float(result['inputs'].get('target_mm', np.nan)))}<br>Retiro desde: {int(result['inputs']['edad_inicio_retiro'])} años<br>Confianza exigida: {fmt_pct(calc_result.get('target_success_pct', np.nan), 2)}</span></div>""", unsafe_allow_html=True)

            history = pd.DataFrame(calc_result.get("iterations", []))
            if not history.empty:
                history_display = history[["saving_center_clp", "joint_success_pct", "prob_no_ruin_pct", "prob_target_retirement_pct", "capital_p50_retiro_clp"]].copy()
                history_display["ahorro_mensual"] = history_display["saving_center_clp"].apply(fmt_clp)
                history_display["exito_conjunto"] = history_display["joint_success_pct"].apply(lambda x: fmt_pct(x, 2))
                history_display["no_agotar"] = history_display["prob_no_ruin_pct"].apply(lambda x: fmt_pct(x, 2))
                history_display["meta_retiro"] = history_display["prob_target_retirement_pct"].apply(lambda x: fmt_pct(x, 2))
                history_display["capital_p50_retiro"] = history_display["capital_p50_retiro_clp"].apply(fmt_clp)
                st.dataframe(
                    history_display[["ahorro_mensual", "exito_conjunto", "no_agotar", "meta_retiro", "capital_p50_retiro"]],
                    width="stretch",
                    hide_index=True,
                )
        else:
            st.error("No alcanza con el ahorro máximo probado.")
            high = calc_result.get("high_result", {})
            hcols = st.columns(3)
            with hcols[0]:
                metric_card("Ahorro máximo probado", fmt_clp(high.get("saving_center_clp", np.nan)), "Sube el máximo o revisa retiro/meta/edad.", "bad")
            with hcols[1]:
                metric_card("Éxito conjunto máximo", fmt_pct(high.get("joint_success_pct", np.nan), 2), "Meta al retiro + no agotar hasta 90.", "bad")
            with hcols[2]:
                metric_card("Capital P50 al retiro", fmt_clp(high.get("capital_p50_retiro_clp", np.nan)), "Resultado con el ahorro máximo probado.", "orange")

    else:
        st.markdown(
            """
            <div class="definition-card"><b>Qué calcula</b><br>
            <span>Busca el menor ahorro mensual que cumple simultáneamente: llegar a la meta de patrimonio al comenzar el retiro y no quedarse sin patrimonio antes de los 90 con la confianza objetivo.</span></div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "Nota: esto es una herramienta de simulación, no una recomendación financiera. "
    "El motor mantiene cálculos internos en MM CLP para estabilidad numérica, pero la interfaz muestra los montos en CLP con todos los ceros. Los flujos marcados como indexados se interpretan como pesos de hoy y crecen con inflación."
)
