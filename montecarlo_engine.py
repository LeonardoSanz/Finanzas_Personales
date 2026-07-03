import math
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy.stats import truncnorm


# ============================================================
# Utilidades de retorno truncado
# ============================================================

def _truncnorm_mean(loc: float, scale: float, low: float, high: float) -> float:
    """
    Media efectiva de una normal truncada entre [low, high].
    Se usa para calibrar el 'loc' si queremos que la media observada
    luego del truncamiento sea igual a la media objetivo.
    """
    if scale <= 0:
        raise ValueError("scale debe ser > 0")
    if low >= high:
        raise ValueError("low debe ser menor que high")

    a = (low - loc) / scale
    b = (high - loc) / scale

    phi_a = np.exp(-0.5 * a**2) / np.sqrt(2 * np.pi)
    phi_b = np.exp(-0.5 * b**2) / np.sqrt(2 * np.pi)

    cdf_a = 0.5 * (1 + math.erf(a / np.sqrt(2)))
    cdf_b = 0.5 * (1 + math.erf(b / np.sqrt(2)))

    z = cdf_b - cdf_a
    if z <= 0:
        raise ValueError("La masa de probabilidad truncada es demasiado pequeña")

    return loc + scale * (phi_a - phi_b) / z


def _find_loc_for_effective_truncated_mean(
    target_mean: float,
    scale: float,
    low: float,
    high: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """
    Encuentra el loc de la normal original para que la media efectiva
    de la distribución truncada sea exactamente target_mean.
    """
    if not (low < target_mean < high):
        raise ValueError("La media objetivo debe estar entre low y high")

    left = low - 10 * scale
    right = high + 10 * scale

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        m = _truncnorm_mean(mid, scale, low, high)

        if abs(m - target_mean) < tol:
            return mid
        if m < target_mean:
            left = mid
        else:
            right = mid

    return 0.5 * (left + right)


def _sample_truncated_normal(
    rng: np.random.Generator,
    target_mean: float,
    std: float,
    low: float,
    high: float,
    size: tuple[int, int],
    mean_is_effective: bool = True,
) -> tuple[np.ndarray, float, float]:
    """
    Muestra una normal truncada y retorna:
    - muestras
    - loc usado antes del truncamiento
    - media efectiva luego del truncamiento
    """
    if std <= 0:
        raise ValueError("La volatilidad debe ser > 0")
    if low <= -1:
        raise ValueError("El retorno mínimo no puede ser <= -100%")
    if not low < target_mean < high:
        raise ValueError("La media debe estar entre el mínimo y el máximo")

    loc = (
        _find_loc_for_effective_truncated_mean(target_mean, std, low, high)
        if mean_is_effective
        else target_mean
    )

    a = (low - loc) / std
    b = (high - loc) / std

    draws = truncnorm.rvs(
        a=a,
        b=b,
        loc=loc,
        scale=std,
        size=size,
        random_state=rng,
    )

    effective_mean = _truncnorm_mean(loc, std, low, high)
    return draws, loc, effective_mean


# ============================================================
# Motor Monte Carlo: acumulación + retiro fijo
# ============================================================

def monte_carlo_accumulation_withdrawal_mm(
    *,
    edad_inicial: int,
    edad_final: int,
    edad_inicio_retiro: int,
    n_paths: int,
    initial_capital_mm: float,
    annual_return_mean: float,
    annual_return_std: float,
    annual_return_low: float,
    annual_return_high: float,
    monthly_saving_min_mm: float,
    monthly_saving_mode_mm: float,
    monthly_saving_max_mm: float,
    withdrawal_monthly_mm: float,
    contribution_timing: Literal["begin", "end"] = "end",
    withdrawal_timing: Literal["begin", "end"] = "end",
    target_mm: Optional[float] = 1_000.0,
    seed: Optional[int] = 123,
    mean_is_effective: bool = True,
    lump_sum_events: Optional[tuple[tuple[int, float], ...]] = None,
    floor_zero: bool = True,
    return_model: Literal["annual_smooth", "monthly_iid"] = "annual_smooth",
    withdrawal_indexed_to_inflation: bool = False,
    inflation_annual: float = 0.0,
) -> dict:
    """
    Simula patrimonio en MM CLP con dos fases:

    1) Acumulación: ahorro mensual positivo hasta edad_inicio_retiro.
    2) Retiro: ahorro mensual = 0 y retiro fijo mensual desde edad_inicio_retiro.

    La simulación trabaja en meses, pero los snapshots se muestran por edad/año.
    """
    if edad_final <= edad_inicial:
        raise ValueError("edad_final debe ser mayor que edad_inicial")
    if edad_inicio_retiro < edad_inicial:
        raise ValueError("edad_inicio_retiro no puede ser menor que edad_inicial")
    if n_paths <= 0:
        raise ValueError("n_paths debe ser > 0")
    if initial_capital_mm < 0:
        raise ValueError("initial_capital_mm no puede ser negativo")
    if withdrawal_monthly_mm < 0:
        raise ValueError("withdrawal_monthly_mm no puede ser negativo")
    if not (monthly_saving_min_mm <= monthly_saving_mode_mm <= monthly_saving_max_mm):
        raise ValueError("Debe cumplirse ahorro mínimo <= moda <= máximo")
    if contribution_timing not in {"begin", "end"}:
        raise ValueError("contribution_timing debe ser 'begin' o 'end'")
    if withdrawal_timing not in {"begin", "end"}:
        raise ValueError("withdrawal_timing debe ser 'begin' o 'end'")

    rng = np.random.default_rng(seed)

    years = edad_final - edad_inicial
    months = years * 12
    retirement_start_month = int(round((edad_inicio_retiro - edad_inicial) * 12))
    retirement_start_month = min(max(retirement_start_month, 0), months)

    # -----------------------------
    # Retornos
    # -----------------------------
    if return_model == "annual_smooth":
        annual_returns, loc_used, effective_mean = _sample_truncated_normal(
            rng=rng,
            target_mean=annual_return_mean,
            std=annual_return_std,
            low=annual_return_low,
            high=annual_return_high,
            size=(n_paths, years),
            mean_is_effective=mean_is_effective,
        )
        monthly_returns_by_year = (1 + annual_returns) ** (1 / 12) - 1
        monthly_returns = np.repeat(monthly_returns_by_year, repeats=12, axis=1)
    elif return_model == "monthly_iid":
        # Aproximación: transforma parámetros anuales a mensuales.
        # Esto captura mejor el sequence risk que suavizar el retorno anual.
        monthly_mean_target = (1 + annual_return_mean) ** (1 / 12) - 1
        monthly_std = annual_return_std / np.sqrt(12)
        monthly_low = (1 + annual_return_low) ** (1 / 12) - 1
        monthly_high = (1 + annual_return_high) ** (1 / 12) - 1

        monthly_returns, loc_used, effective_monthly_mean = _sample_truncated_normal(
            rng=rng,
            target_mean=monthly_mean_target,
            std=monthly_std,
            low=monthly_low,
            high=monthly_high,
            size=(n_paths, months),
            mean_is_effective=mean_is_effective,
        )
        effective_mean = (1 + effective_monthly_mean) ** 12 - 1
    else:
        raise ValueError("return_model debe ser 'annual_smooth' o 'monthly_iid'")

    # -----------------------------
    # Ahorros mensuales
    # -----------------------------
    monthly_savings = rng.triangular(
        left=monthly_saving_min_mm,
        mode=monthly_saving_mode_mm,
        right=monthly_saving_max_mm,
        size=(n_paths, months),
    )

    # Desde el inicio del retiro, ahorro = 0.
    if retirement_start_month < months:
        monthly_savings[:, retirement_start_month:] = 0.0

    # -----------------------------
    # Aportes extraordinarios
    # lump_sum_events viene como tupla de (mes_simulacion_1_based, monto_mm)
    # -----------------------------
    lump_sums = np.zeros(months, dtype=np.float64)
    if lump_sum_events is not None:
        for month_idx, amount_mm in lump_sum_events:
            if amount_mm == 0:
                continue
            if 1 <= month_idx <= months:
                lump_sums[month_idx - 1] += amount_mm
            else:
                raise ValueError(f"El mes {month_idx} está fuera del horizonte de simulación")

    # -----------------------------
    # Retiros mensuales fijos
    # -----------------------------
    withdrawal_schedule = np.zeros(months, dtype=np.float64)
    if retirement_start_month < months and withdrawal_monthly_mm > 0:
        if withdrawal_indexed_to_inflation:
            monthly_inflation = (1 + inflation_annual) ** (1 / 12) - 1
            for t in range(retirement_start_month, months):
                k = t - retirement_start_month
                withdrawal_schedule[t] = withdrawal_monthly_mm * (1 + monthly_inflation) ** k
        else:
            withdrawal_schedule[retirement_start_month:] = withdrawal_monthly_mm

    # -----------------------------
    # Paths
    # -----------------------------
    paths = np.empty((n_paths, months + 1), dtype=np.float64)
    paths[:, 0] = initial_capital_mm

    ruin_month = np.full(n_paths, np.nan, dtype=np.float64)

    for t in range(months):
        r_t = monthly_returns[:, t]
        savings_t = monthly_savings[:, t]
        lump_t = lump_sums[t]
        withdrawal_t = withdrawal_schedule[t]

        wealth = paths[:, t].copy()

        if contribution_timing == "begin":
            wealth = wealth + savings_t + lump_t

        if withdrawal_timing == "begin":
            wealth = wealth - withdrawal_t

        wealth = wealth * (1 + r_t)

        if contribution_timing == "end":
            wealth = wealth + savings_t + lump_t

        if withdrawal_timing == "end":
            wealth = wealth - withdrawal_t

        newly_ruined = (
            np.isnan(ruin_month)
            & (t >= retirement_start_month)
            & (withdrawal_t > 0)
            & (wealth <= 0)
        )
        ruin_month[newly_ruined] = t + 1

        if floor_zero:
            wealth = np.maximum(wealth, 0.0)

        paths[:, t + 1] = wealth

    final_wealth = paths[:, -1]
    wealth_at_retirement = paths[:, retirement_start_month]
    total_savings = monthly_savings.sum(axis=1)
    total_lump_sums = lump_sums.sum()
    total_withdrawal_requested = withdrawal_schedule.sum()

    def pct(x: np.ndarray, q: float) -> float:
        return float(np.percentile(x, q))

    final_summary = pd.DataFrame(
        {
            "metric": ["mean", "std", "p5", "p25", "p50", "p75", "p95"],
            "final_wealth_mm": [
                float(np.mean(final_wealth)),
                float(np.std(final_wealth, ddof=1)),
                pct(final_wealth, 5),
                pct(final_wealth, 25),
                pct(final_wealth, 50),
                pct(final_wealth, 75),
                pct(final_wealth, 95),
            ],
            "wealth_at_retirement_mm": [
                float(np.mean(wealth_at_retirement)),
                float(np.std(wealth_at_retirement, ddof=1)),
                pct(wealth_at_retirement, 5),
                pct(wealth_at_retirement, 25),
                pct(wealth_at_retirement, 50),
                pct(wealth_at_retirement, 75),
                pct(wealth_at_retirement, 95),
            ],
            "total_savings_mm": [
                float(np.mean(total_savings)),
                float(np.std(total_savings, ddof=1)),
                pct(total_savings, 5),
                pct(total_savings, 25),
                pct(total_savings, 50),
                pct(total_savings, 75),
                pct(total_savings, 95),
            ],
        }
    )

    prob_reach_target_final = np.nan
    prob_reach_target_at_retirement = np.nan
    if target_mm is not None:
        prob_reach_target_final = float(np.mean(final_wealth >= target_mm))
        prob_reach_target_at_retirement = float(np.mean(wealth_at_retirement >= target_mm))

    prob_no_ruin = float(np.mean(np.isnan(ruin_month)))
    ruin_age = edad_inicial + ruin_month / 12
    ruin_age_clean = ruin_age[~np.isnan(ruin_age)]
    median_ruin_age = float(np.median(ruin_age_clean)) if len(ruin_age_clean) > 0 else np.nan

    if retirement_start_month < months:
        prob_final_above_retirement_wealth = float(np.mean(final_wealth > wealth_at_retirement))
    else:
        prob_final_above_retirement_wealth = np.nan

    return {
        "inputs": {
            "edad_inicial": edad_inicial,
            "edad_final": edad_final,
            "edad_inicio_retiro": edad_inicio_retiro,
            "years": years,
            "months": months,
            "retirement_start_month": retirement_start_month,
            "n_paths": n_paths,
            "initial_capital_mm": initial_capital_mm,
            "annual_return_mean_requested": annual_return_mean,
            "annual_return_std": annual_return_std,
            "annual_return_low": annual_return_low,
            "annual_return_high": annual_return_high,
            "monthly_saving_min_mm": monthly_saving_min_mm,
            "monthly_saving_mode_mm": monthly_saving_mode_mm,
            "monthly_saving_max_mm": monthly_saving_max_mm,
            "withdrawal_monthly_mm": withdrawal_monthly_mm,
            "contribution_timing": contribution_timing,
            "withdrawal_timing": withdrawal_timing,
            "target_mm": target_mm,
            "seed": seed,
            "mean_is_effective": mean_is_effective,
            "loc_used_pre_truncation": loc_used,
            "effective_truncated_mean_annualized": effective_mean,
            "lump_sum_events": lump_sum_events,
            "floor_zero": floor_zero,
            "return_model": return_model,
            "withdrawal_indexed_to_inflation": withdrawal_indexed_to_inflation,
            "inflation_annual": inflation_annual,
        },
        "summary": final_summary,
        "paths_mm": paths,
        "final_wealth_mm": final_wealth,
        "wealth_at_retirement_mm": wealth_at_retirement,
        "total_savings_mm": total_savings,
        "monthly_savings_mm": monthly_savings,
        "monthly_returns": monthly_returns,
        "lump_sums_mm": lump_sums,
        "withdrawal_schedule_mm": withdrawal_schedule,
        "ruin_month": ruin_month,
        "ruin_age": ruin_age,
        "prob_reach_target_final": prob_reach_target_final,
        "prob_reach_target_at_retirement": prob_reach_target_at_retirement,
        "prob_no_ruin": prob_no_ruin,
        "median_ruin_age": median_ruin_age,
        "prob_final_above_retirement_wealth": prob_final_above_retirement_wealth,
        "total_lump_sums_mm": total_lump_sums,
        "total_withdrawal_requested_mm": total_withdrawal_requested,
    }


def tabla_monte_carlo_por_edad(result: dict) -> pd.DataFrame:
    """
    Tabla anual con percentiles, probabilidad sobre target y flujo anual promedio.
    """
    paths = result["paths_mm"]
    inputs = result["inputs"]
    edad_inicial = inputs["edad_inicial"]
    years = inputs["years"]
    target_mm = inputs["target_mm"]
    monthly_savings = result["monthly_savings_mm"]
    withdrawal_schedule = result["withdrawal_schedule_mm"]
    lump_sums = result["lump_sums_mm"]

    rows = []
    for year in range(0, years + 1):
        month = year * 12
        data = paths[:, month]

        if year < years:
            m0 = year * 12
            m1 = (year + 1) * 12
            ahorro_prom_mensual = float(np.mean(monthly_savings[:, m0:m1]))
            retiro_prom_mensual = float(np.mean(withdrawal_schedule[m0:m1]))
            aporte_extra_anual = float(np.sum(lump_sums[m0:m1]))
        else:
            ahorro_prom_mensual = 0.0
            retiro_prom_mensual = 0.0
            aporte_extra_anual = 0.0

        prob_sobre_target = np.nan if target_mm is None else float(np.mean(data >= target_mm))
        prob_sobre_cero = float(np.mean(data > 0))

        rows.append(
            {
                "edad": edad_inicial + year,
                "año_simulación": year,
                "media_mm": np.mean(data),
                "p5_mm": np.percentile(data, 5),
                "p25_mm": np.percentile(data, 25),
                "p50_mediana_mm": np.percentile(data, 50),
                "p75_mm": np.percentile(data, 75),
                "p95_mm": np.percentile(data, 95),
                "prob_sobre_target": prob_sobre_target,
                "prob_sobre_cero": prob_sobre_cero,
                "ahorro_prom_mensual_mm": ahorro_prom_mensual,
                "retiro_prom_mensual_mm": retiro_prom_mensual,
                "aporte_extra_anual_mm": aporte_extra_anual,
            }
        )

    tabla = pd.DataFrame(rows)
    monto_cols = [
        "media_mm",
        "p5_mm",
        "p25_mm",
        "p50_mediana_mm",
        "p75_mm",
        "p95_mm",
        "ahorro_prom_mensual_mm",
        "retiro_prom_mensual_mm",
        "aporte_extra_anual_mm",
    ]
    tabla[monto_cols] = tabla[monto_cols].round(2)
    tabla["prob_sobre_target"] = (tabla["prob_sobre_target"] * 100).round(2)
    tabla["prob_sobre_cero"] = (tabla["prob_sobre_cero"] * 100).round(2)
    return tabla


