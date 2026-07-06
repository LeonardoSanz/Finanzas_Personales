import math
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy.stats import truncnorm


# ============================================================
# Utilidades de retorno truncado
# ============================================================

def _truncnorm_mean(loc: float, scale: float, low: float, high: float) -> float:
    """Media efectiva de una normal truncada entre [low, high]."""
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
    """Encuentra el loc pre-truncamiento para que la media truncada sea target_mean."""
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


def _calibrate_truncated_normal(
    target_mean: float,
    std: float,
    low: float,
    high: float,
    mean_is_effective: bool = True,
) -> tuple[float, float, float, float]:
    """
    Retorna loc, a, b, effective_mean para una normal truncada.
    Se separa de la generación para poder simular mes a mes sin crear matrices gigantes.
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
    effective_mean = _truncnorm_mean(loc, std, low, high)
    return loc, a, b, effective_mean


def _sample_truncated_normal(
    rng: np.random.Generator,
    target_mean: float,
    std: float,
    low: float,
    high: float,
    size: tuple[int, int],
    mean_is_effective: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Muestra una normal truncada y retorna muestras, loc usado y media efectiva."""
    loc, a, b, effective_mean = _calibrate_truncated_normal(
        target_mean=target_mean,
        std=std,
        low=low,
        high=high,
        mean_is_effective=mean_is_effective,
    )
    draws = truncnorm.rvs(a=a, b=b, loc=loc, scale=std, size=size, random_state=rng)
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
    recurring_monthly_events: Optional[tuple[tuple, ...]] = None,
    saving_ranges: Optional[tuple[tuple[float, Optional[float], float, float, float, str], ...]] = None,
    floor_zero: bool = True,
    return_model: Literal["annual_smooth", "monthly_iid"] = "annual_smooth",
    withdrawal_indexed_to_inflation: bool = False,
    inflation_annual: float = 0.0,
    withdrawal_index_base_age: Optional[float] = None,
) -> dict:
    """
    Simula patrimonio en MM CLP con dos fases:

    1) Acumulación: ahorro mensual positivo hasta edad_inicio_retiro.
       El ahorro puede ser un único triangular global o tramos por edad.
    2) Retiro: ahorro mensual = 0 y retiro fijo mensual desde edad_inicio_retiro.

    Esta versión evita guardar matrices gigantes de retornos/ahorros. Solo guarda paths,
    flujos medios y vectores necesarios para reportes. Eso mejora la estabilidad en
    Streamlit Cloud, especialmente en modo mensual IID y edad final 90.
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

    years = int(edad_final - edad_inicial)
    months = int(years * 12)
    retirement_start_month = int(round((edad_inicio_retiro - edad_inicial) * 12))
    retirement_start_month = min(max(retirement_start_month, 0), months)

    # -----------------------------
    # Retornos: se calibran una vez, se generan mes a mes.
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
        monthly_loc_used = np.nan
        monthly_effective_mean = np.nan
        monthly_a = monthly_b = monthly_std = np.nan
    elif return_model == "monthly_iid":
        monthly_mean_target = (1 + annual_return_mean) ** (1 / 12) - 1
        monthly_std = annual_return_std / np.sqrt(12)
        monthly_low = (1 + annual_return_low) ** (1 / 12) - 1
        monthly_high = (1 + annual_return_high) ** (1 / 12) - 1
        monthly_loc_used, monthly_a, monthly_b, monthly_effective_mean = _calibrate_truncated_normal(
            target_mean=monthly_mean_target,
            std=monthly_std,
            low=monthly_low,
            high=monthly_high,
            mean_is_effective=mean_is_effective,
        )
        loc_used = monthly_loc_used
        effective_mean = (1 + monthly_effective_mean) ** 12 - 1
        annual_returns = None
        monthly_returns_by_year = None
    else:
        raise ValueError("return_model debe ser 'annual_smooth' o 'monthly_iid'")

    # -----------------------------
    # Aportes extraordinarios: monto positivo = entra plata; negativo = sale plata.
    # -----------------------------
    lump_sums = np.zeros(months, dtype=np.float64)
    if lump_sum_events is not None:
        for month_idx, amount_mm in lump_sum_events:
            if amount_mm == 0:
                continue
            if 1 <= month_idx <= months:
                lump_sums[month_idx - 1] += float(amount_mm)
            else:
                raise ValueError(f"El mes {month_idx} está fuera del horizonte de simulación")

    # -----------------------------
    # Ingresos / egresos mensuales recurrentes externos
    # Formato compatible:
    #   Antiguo: (edad_inicio, edad_fin_opcional, monto_mensual_mm, descripción)
    #   Nuevo:   (edad_inicio, edad_fin_opcional, monto_mensual_mm, descripción, indexado_inflacion, edad_base_indexacion)
    #
    # Si indexado_inflacion=True, el monto se interpreta en pesos de hoy / MM de hoy
    # y se transforma a monto nominal de cada mes usando inflación anual.
    # Esto permite que jubilación y arriendos crezcan igual que el retiro indexado.
    # -----------------------------
    recurring_cashflows = np.zeros(months, dtype=np.float64)
    recurring_event_rows = []
    monthly_inflation_for_cashflows = (1 + inflation_annual) ** (1 / 12) - 1 if inflation_annual > -1 else 0.0

    if recurring_monthly_events is not None:
        for item in recurring_monthly_events:
            if len(item) == 4:
                start_age, end_age, monthly_amount_mm, description = item
                indexed_to_inflation = False
                index_base_age = start_age
            elif len(item) == 6:
                start_age, end_age, monthly_amount_mm, description, indexed_to_inflation, index_base_age = item
            else:
                raise ValueError("Cada flujo recurrente debe tener 4 o 6 campos")

            if monthly_amount_mm == 0:
                continue
            if start_age is None:
                continue

            start_month = int(round((float(start_age) - edad_inicial) * 12))
            start_month = min(max(start_month, 0), months)

            if end_age is None or (isinstance(end_age, float) and np.isnan(end_age)):
                end_month = months
                end_age_clean = edad_final
            else:
                end_month = int(round((float(end_age) - edad_inicial) * 12))
                end_month = min(max(end_month, 0), months)
                end_age_clean = float(end_age)

            if end_month <= start_month:
                continue

            if index_base_age is None or (isinstance(index_base_age, float) and np.isnan(index_base_age)):
                index_base_age = edad_inicial
            base_month = int(round((float(index_base_age) - edad_inicial) * 12))

            if indexed_to_inflation:
                month_index = np.arange(start_month, end_month, dtype=np.float64)
                k = month_index - float(base_month)
                cashflow_vector = float(monthly_amount_mm) * (1 + monthly_inflation_for_cashflows) ** k
            else:
                cashflow_vector = np.full(end_month - start_month, float(monthly_amount_mm), dtype=np.float64)

            recurring_cashflows[start_month:end_month] += cashflow_vector
            recurring_event_rows.append(
                {
                    "descripcion": description,
                    "edad_inicio": float(start_age),
                    "edad_fin": end_age_clean,
                    "monto_mensual_mm": float(monthly_amount_mm),
                    "monto_inicio_nominal_mm": float(cashflow_vector[0]) if len(cashflow_vector) else 0.0,
                    "monto_fin_nominal_mm": float(cashflow_vector[-1]) if len(cashflow_vector) else 0.0,
                    "indexado_inflacion": bool(indexed_to_inflation),
                    "edad_base_indexacion": float(index_base_age),
                    "mes_inicio": int(start_month + 1),
                    "mes_fin": int(end_month),
                }
            )

    # -----------------------------
    # Rangos de ahorro por edad
    # Formato: (edad_inicio, edad_fin_opcional, ahorro_min_mm, ahorro_mode_mm, ahorro_max_mm, descripción)
    # Si no se informa, se usa el triangular global. Desde edad_inicio_retiro el ahorro siempre queda en cero.
    # -----------------------------
    saving_min_schedule = np.zeros(months, dtype=np.float64)
    saving_mode_schedule = np.zeros(months, dtype=np.float64)
    saving_max_schedule = np.zeros(months, dtype=np.float64)
    saving_range_rows = []

    if saving_ranges is None or len(saving_ranges) == 0:
        saving_min_schedule[:retirement_start_month] = float(monthly_saving_min_mm)
        saving_mode_schedule[:retirement_start_month] = float(monthly_saving_mode_mm)
        saving_max_schedule[:retirement_start_month] = float(monthly_saving_max_mm)
        if retirement_start_month > 0:
            saving_range_rows.append(
                {
                    "descripcion": "Ahorro global",
                    "edad_inicio": float(edad_inicial),
                    "edad_fin": float(edad_inicio_retiro),
                    "ahorro_min_mm": float(monthly_saving_min_mm),
                    "ahorro_mode_mm": float(monthly_saving_mode_mm),
                    "ahorro_max_mm": float(monthly_saving_max_mm),
                    "mes_inicio": 1,
                    "mes_fin": int(retirement_start_month),
                }
            )
    else:
        for item in saving_ranges:
            if len(item) == 6:
                start_age, end_age, saving_min_mm, saving_mode_mm, saving_max_mm, description = item
            elif len(item) == 5:
                start_age, end_age, saving_min_mm, saving_mode_mm, saving_max_mm = item
                description = "Tramo ahorro"
            else:
                raise ValueError("Cada rango de ahorro debe tener 5 o 6 campos")
            if start_age is None:
                continue
            if not (saving_min_mm <= saving_mode_mm <= saving_max_mm):
                raise ValueError(
                    f"Rango de ahorro inválido en '{description}': debe cumplirse mínimo <= más probable <= máximo"
                )
            start_month = int(round((float(start_age) - edad_inicial) * 12))
            start_month = min(max(start_month, 0), months)

            if end_age is None or (isinstance(end_age, float) and np.isnan(end_age)):
                end_month = retirement_start_month
                end_age_clean = float(edad_inicio_retiro)
            else:
                end_month = int(round((float(end_age) - edad_inicial) * 12))
                end_month = min(max(end_month, 0), months)
                end_age_clean = float(end_age)

            # El ahorro nunca cruza la edad de retiro: desde ahí queda en cero.
            end_month = min(end_month, retirement_start_month)
            if end_month <= start_month:
                continue

            saving_min_schedule[start_month:end_month] = float(saving_min_mm)
            saving_mode_schedule[start_month:end_month] = float(saving_mode_mm)
            saving_max_schedule[start_month:end_month] = float(saving_max_mm)
            saving_range_rows.append(
                {
                    "descripcion": description,
                    "edad_inicio": float(start_age),
                    "edad_fin": min(end_age_clean, float(edad_inicio_retiro)),
                    "ahorro_min_mm": float(saving_min_mm),
                    "ahorro_mode_mm": float(saving_mode_mm),
                    "ahorro_max_mm": float(saving_max_mm),
                    "mes_inicio": int(start_month + 1),
                    "mes_fin": int(end_month),
                }
            )

    # -----------------------------
    # Retiros mensuales fijos
    # -----------------------------
    withdrawal_schedule = np.zeros(months, dtype=np.float64)
    if retirement_start_month < months and withdrawal_monthly_mm > 0:
        if withdrawal_indexed_to_inflation:
            # El monto ingresado se interpreta como pesos de hoy / poder de compra actual.
            # Por eso el primer retiro a edad_inicio_retiro ya viene inflado desde edad_inicial
            # hasta esa edad, y luego sigue indexándose hasta los 90.
            monthly_inflation = (1 + inflation_annual) ** (1 / 12) - 1
            base_age = float(edad_inicial if withdrawal_index_base_age is None else withdrawal_index_base_age)
            base_month = int(round((base_age - float(edad_inicial)) * 12))
            k = np.arange(retirement_start_month, months, dtype=np.float64) - float(base_month)
            withdrawal_schedule[retirement_start_month:] = withdrawal_monthly_mm * (1 + monthly_inflation) ** k
        else:
            withdrawal_schedule[retirement_start_month:] = withdrawal_monthly_mm

    # -----------------------------
    # Simulación path by path en matriz de patrimonio, pero flujos/retornos se generan por mes.
    # paths en float32 reduce memoria y es suficiente para MM CLP.
    # -----------------------------
    paths = np.empty((n_paths, months + 1), dtype=np.float32)
    paths[:, 0] = np.float32(initial_capital_mm)
    ruin_month = np.full(n_paths, np.nan, dtype=np.float32)
    total_savings = np.zeros(n_paths, dtype=np.float32)
    monthly_savings_mean = np.zeros(months, dtype=np.float64)
    monthly_return_mean = np.zeros(months, dtype=np.float64)

    for t in range(months):
        if return_model == "annual_smooth":
            year_idx = min(t // 12, years - 1)
            r_t = monthly_returns_by_year[:, year_idx].astype(np.float32, copy=False)
        else:
            r_t = truncnorm.rvs(
                a=monthly_a,
                b=monthly_b,
                loc=monthly_loc_used,
                scale=monthly_std,
                size=n_paths,
                random_state=rng,
            ).astype(np.float32, copy=False)
        monthly_return_mean[t] = float(np.mean(r_t))

        if t < retirement_start_month and saving_max_schedule[t] > 0:
            savings_t = rng.triangular(
                left=saving_min_schedule[t],
                mode=saving_mode_schedule[t],
                right=saving_max_schedule[t],
                size=n_paths,
            ).astype(np.float32, copy=False)
        else:
            savings_t = np.zeros(n_paths, dtype=np.float32)

        monthly_savings_mean[t] = float(np.mean(savings_t))
        total_savings += savings_t

        lump_t = np.float32(lump_sums[t])
        recurring_t = np.float32(recurring_cashflows[t])
        withdrawal_t = np.float32(withdrawal_schedule[t])

        wealth = paths[:, t].astype(np.float32, copy=True)

        if contribution_timing == "begin":
            wealth = wealth + savings_t + lump_t
        if withdrawal_timing == "begin":
            wealth = wealth - withdrawal_t

        wealth = wealth * (1 + r_t)

        if contribution_timing == "end":
            wealth = wealth + savings_t + lump_t
        if withdrawal_timing == "end":
            wealth = wealth - withdrawal_t

        # Ingresos/egresos recurrentes externos se aplican al final del mes.
        wealth = wealth + recurring_t

        newly_ruined = (
            np.isnan(ruin_month)
            & (t >= retirement_start_month)
            & (withdrawal_t > 0)
            & (wealth <= 0)
        )
        ruin_month[newly_ruined] = np.float32(t + 1)

        if floor_zero:
            wealth = np.maximum(wealth, 0.0)

        paths[:, t + 1] = wealth.astype(np.float32, copy=False)

    final_wealth = paths[:, -1].astype(np.float64)
    wealth_at_retirement = paths[:, retirement_start_month].astype(np.float64)
    total_savings = total_savings.astype(np.float64)

    total_lump_sums = float(lump_sums.sum())
    total_withdrawal_requested = float(withdrawal_schedule.sum())
    total_recurring_inflows = float(recurring_cashflows[recurring_cashflows > 0].sum())
    total_recurring_outflows = float(-recurring_cashflows[recurring_cashflows < 0].sum())
    total_recurring_net = float(recurring_cashflows.sum())

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
    ruin_age = edad_inicial + ruin_month.astype(np.float64) / 12
    ruin_age_clean = ruin_age[~np.isnan(ruin_age)]
    median_ruin_age = float(np.median(ruin_age_clean)) if len(ruin_age_clean) > 0 else np.nan

    if retirement_start_month < months:
        prob_final_above_retirement_wealth = float(np.mean(final_wealth > wealth_at_retirement))
    else:
        prob_final_above_retirement_wealth = np.nan

    # Solo informativo: cantidad aproximada de memoria del path principal.
    path_memory_mb = float(paths.nbytes / (1024 ** 2))

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
            "recurring_monthly_events": recurring_monthly_events,
            "saving_ranges": saving_ranges,
            "floor_zero": floor_zero,
            "return_model": return_model,
            "withdrawal_indexed_to_inflation": withdrawal_indexed_to_inflation,
            "inflation_annual": inflation_annual,
            "withdrawal_index_base_age": float(edad_inicial if withdrawal_index_base_age is None else withdrawal_index_base_age),
            "path_memory_mb": path_memory_mb,
        },
        "summary": final_summary,
        "paths_mm": paths,
        "final_wealth_mm": final_wealth,
        "wealth_at_retirement_mm": wealth_at_retirement,
        "total_savings_mm": total_savings,
        # Nueva forma liviana: vector promedio mensual, no matriz n_paths x months.
        "monthly_savings_mean_mm": monthly_savings_mean,
        "monthly_returns_mean": monthly_return_mean,
        # Compatibilidad: ya no guardamos la matriz gigante.
        "monthly_savings_mm": None,
        "monthly_returns": None,
        "lump_sums_mm": lump_sums,
        "recurring_cashflows_mm": recurring_cashflows,
        "recurring_event_rows": recurring_event_rows,
        "saving_min_schedule_mm": saving_min_schedule,
        "saving_mode_schedule_mm": saving_mode_schedule,
        "saving_max_schedule_mm": saving_max_schedule,
        "saving_range_rows": saving_range_rows,
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
        "total_recurring_inflows_mm": total_recurring_inflows,
        "total_recurring_outflows_mm": total_recurring_outflows,
        "total_recurring_net_mm": total_recurring_net,
        "total_net_cash_need_mm": total_withdrawal_requested - total_recurring_inflows + total_recurring_outflows,
    }


def tabla_monte_carlo_por_edad(result: dict) -> pd.DataFrame:
    """Tabla anual con percentiles, probabilidad sobre target y flujo anual promedio."""
    paths = result["paths_mm"]
    inputs = result["inputs"]
    edad_inicial = inputs["edad_inicial"]
    years = inputs["years"]
    target_mm = inputs["target_mm"]
    monthly_savings_mean = result.get("monthly_savings_mean_mm")
    withdrawal_schedule = result["withdrawal_schedule_mm"]
    lump_sums = result["lump_sums_mm"]
    recurring_cashflows = result.get("recurring_cashflows_mm", np.zeros_like(lump_sums))

    if monthly_savings_mean is None:
        old_monthly_savings = result.get("monthly_savings_mm")
        if old_monthly_savings is None:
            monthly_savings_mean = np.zeros_like(lump_sums)
        else:
            monthly_savings_mean = old_monthly_savings.mean(axis=0)

    rows = []
    for year in range(0, years + 1):
        month = year * 12
        data = paths[:, month].astype(np.float64)

        if year < years:
            m0 = year * 12
            m1 = (year + 1) * 12
            ahorro_prom_mensual = float(np.mean(monthly_savings_mean[m0:m1]))
            retiro_prom_mensual = float(np.mean(withdrawal_schedule[m0:m1]))
            aporte_extra_anual = float(np.sum(lump_sums[m0:m1]))
            flujo_recurrente_prom_mensual = float(np.mean(recurring_cashflows[m0:m1]))
            ingreso_recurrente_prom_mensual = float(np.mean(np.maximum(recurring_cashflows[m0:m1], 0)))
            egreso_recurrente_prom_mensual = float(np.mean(np.maximum(-recurring_cashflows[m0:m1], 0)))
        else:
            ahorro_prom_mensual = 0.0
            retiro_prom_mensual = 0.0
            aporte_extra_anual = 0.0
            flujo_recurrente_prom_mensual = 0.0
            ingreso_recurrente_prom_mensual = 0.0
            egreso_recurrente_prom_mensual = 0.0

        prob_sobre_target = np.nan if target_mm is None else float(np.mean(data >= target_mm))
        prob_sobre_cero = float(np.mean(data > 0))

        rows.append(
            {
                "edad": edad_inicial + year,
                "año_simulación": year,
                "media_mm": float(np.mean(data)),
                "p5_mm": float(np.percentile(data, 5)),
                "p25_mm": float(np.percentile(data, 25)),
                "p50_mediana_mm": float(np.percentile(data, 50)),
                "p75_mm": float(np.percentile(data, 75)),
                "p95_mm": float(np.percentile(data, 95)),
                "prob_sobre_target": prob_sobre_target,
                "prob_sobre_cero": prob_sobre_cero,
                "ahorro_prom_mensual_mm": ahorro_prom_mensual,
                "retiro_prom_mensual_mm": retiro_prom_mensual,
                "ingreso_recurrente_prom_mensual_mm": ingreso_recurrente_prom_mensual,
                "egreso_recurrente_prom_mensual_mm": egreso_recurrente_prom_mensual,
                "flujo_recurrente_neto_mensual_mm": flujo_recurrente_prom_mensual,
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
        "ingreso_recurrente_prom_mensual_mm",
        "egreso_recurrente_prom_mensual_mm",
        "flujo_recurrente_neto_mensual_mm",
        "aporte_extra_anual_mm",
    ]
    tabla[monto_cols] = tabla[monto_cols].round(2)
    tabla["prob_sobre_target"] = (tabla["prob_sobre_target"] * 100).round(2)
    tabla["prob_sobre_cero"] = (tabla["prob_sobre_cero"] * 100).round(2)
    return tabla


# Matriz de capital requerido por edad y probabilidad de éxito
# ============================================================

def _build_retirement_cashflow_schedule_mm(
    *,
    edad_inicio: int,
    edad_final: int,
    withdrawal_monthly_mm: float,
    withdrawal_timing: Literal["begin", "end"] = "end",
    withdrawal_indexed_to_inflation: bool = False,
    inflation_annual: float = 0.0,
    withdrawal_index_base_age: Optional[float] = None,
    recurring_monthly_events: Optional[tuple[tuple, ...]] = None,
    lump_sum_age_events: Optional[tuple[tuple[float, float], ...]] = None,
) -> dict:
    """Construye flujos futuros desde una edad específica.

    Los eventos recurrentes usan edades absolutas. Los eventos únicos también.
    Monto positivo = entrada de caja; monto negativo = salida de caja.
    El retiro se trata como salida de caja.
    """
    if edad_final <= edad_inicio:
        return {
            "months": 0,
            "withdrawal_schedule_mm": np.array([], dtype=np.float64),
            "recurring_cashflows_mm": np.array([], dtype=np.float64),
            "lump_sums_mm": np.array([], dtype=np.float64),
            "net_end_cashflows_mm": np.array([], dtype=np.float64),
        }

    months = int((edad_final - edad_inicio) * 12)
    monthly_inflation = (1 + inflation_annual) ** (1 / 12) - 1 if inflation_annual > -1 else 0.0

    withdrawal_schedule = np.zeros(months, dtype=np.float64)
    if withdrawal_monthly_mm > 0:
        if withdrawal_indexed_to_inflation:
            # Monto deseado expresado en pesos de hoy. Para una edad de retiro X,
            # el primer retiro nominal debe llevar inflación desde la edad base hasta X.
            base_age = float(edad_inicio if withdrawal_index_base_age is None else withdrawal_index_base_age)
            k = np.arange(months, dtype=np.float64) + (float(edad_inicio) - base_age) * 12
            withdrawal_schedule[:] = float(withdrawal_monthly_mm) * (1 + monthly_inflation) ** k
        else:
            withdrawal_schedule[:] = float(withdrawal_monthly_mm)

    recurring_cashflows = np.zeros(months, dtype=np.float64)
    if recurring_monthly_events is not None:
        for item in recurring_monthly_events:
            if len(item) == 4:
                start_age, end_age, monthly_amount_mm, _description = item
                indexed_to_inflation = False
                index_base_age = start_age
            elif len(item) == 6:
                start_age, end_age, monthly_amount_mm, _description, indexed_to_inflation, index_base_age = item
            else:
                raise ValueError("Cada flujo recurrente debe tener 4 o 6 campos")

            if monthly_amount_mm == 0 or start_age is None:
                continue

            start_age_f = float(start_age)
            end_age_f = float(edad_final) if end_age is None or (isinstance(end_age, float) and np.isnan(end_age)) else float(end_age)
            start_age_f = max(start_age_f, float(edad_inicio))
            end_age_f = min(end_age_f, float(edad_final))
            if end_age_f <= start_age_f:
                continue

            start_month = int(round((start_age_f - edad_inicio) * 12))
            end_month = int(round((end_age_f - edad_inicio) * 12))
            start_month = min(max(start_month, 0), months)
            end_month = min(max(end_month, 0), months)
            if end_month <= start_month:
                continue

            if index_base_age is None or (isinstance(index_base_age, float) and np.isnan(index_base_age)):
                index_base_age = edad_inicio

            if indexed_to_inflation:
                month_index = np.arange(start_month, end_month, dtype=np.float64)
                k = month_index + (float(edad_inicio) - float(index_base_age)) * 12
                flow = float(monthly_amount_mm) * (1 + monthly_inflation) ** k
            else:
                flow = np.full(end_month - start_month, float(monthly_amount_mm), dtype=np.float64)
            recurring_cashflows[start_month:end_month] += flow

    lump_sums = np.zeros(months, dtype=np.float64)
    if lump_sum_age_events is not None:
        for event_age, amount_mm in lump_sum_age_events:
            if amount_mm == 0:
                continue
            event_age_f = float(event_age)
            if event_age_f < edad_inicio or event_age_f > edad_final:
                continue
            month_idx = int(round((event_age_f - edad_inicio) * 12))
            month_idx = min(max(month_idx, 0), months - 1)
            lump_sums[month_idx] += float(amount_mm)

    # Flujo neto aplicado al final del mes para el caso estándar de la app.
    # Positivo ayuda al patrimonio; negativo lo consume.
    net_end_cashflows = recurring_cashflows + lump_sums - withdrawal_schedule

    return {
        "months": months,
        "withdrawal_schedule_mm": withdrawal_schedule,
        "recurring_cashflows_mm": recurring_cashflows,
        "lump_sums_mm": lump_sums,
        "net_end_cashflows_mm": net_end_cashflows,
    }


def _simulate_required_capital_by_path_mm(
    *,
    rng: np.random.Generator,
    n_paths: int,
    months: int,
    years: int,
    annual_return_mean: float,
    annual_return_std: float,
    annual_return_low: float,
    annual_return_high: float,
    mean_is_effective: bool,
    return_model: Literal["annual_smooth", "monthly_iid"],
    net_end_cashflows_mm: np.ndarray,
    withdrawal_schedule_mm: np.ndarray,
    recurring_plus_lump_mm: np.ndarray,
    withdrawal_timing: Literal["begin", "end"] = "end",
) -> np.ndarray:
    """Calcula capital mínimo por path mediante recursión hacia atrás.

    Para cada secuencia de retornos, devuelve el capital inicial requerido para que
    el patrimonio nunca sea negativo hasta el final del horizonte.
    """
    if months <= 0:
        return np.zeros(n_paths, dtype=np.float64)

    if return_model == "annual_smooth":
        annual_returns, _, _ = _sample_truncated_normal(
            rng=rng,
            target_mean=annual_return_mean,
            std=annual_return_std,
            low=annual_return_low,
            high=annual_return_high,
            size=(n_paths, years),
            mean_is_effective=mean_is_effective,
        )
        monthly_returns_by_year = ((1 + annual_returns) ** (1 / 12) - 1).astype(np.float32, copy=False)

        required = np.zeros(n_paths, dtype=np.float64)
        for t in range(months - 1, -1, -1):
            year_idx = min(t // 12, years - 1)
            r_t = monthly_returns_by_year[:, year_idx].astype(np.float64, copy=False)
            growth = 1.0 + r_t
            if withdrawal_timing == "begin":
                # W_{t+1} = (W_t - retiro_t) * (1+r_t) + recurrente/lump_t
                required = withdrawal_schedule_mm[t] + np.maximum(
                    0.0,
                    (required - recurring_plus_lump_mm[t]) / growth,
                )
            else:
                # W_{t+1} = W_t * (1+r_t) + recurrente/lump_t - retiro_t
                required = np.maximum(0.0, (required - net_end_cashflows_mm[t]) / growth)
        return required

    if return_model == "monthly_iid":
        monthly_mean_target = (1 + annual_return_mean) ** (1 / 12) - 1
        monthly_std = annual_return_std / np.sqrt(12)
        monthly_low = (1 + annual_return_low) ** (1 / 12) - 1
        monthly_high = (1 + annual_return_high) ** (1 / 12) - 1
        monthly_loc, monthly_a, monthly_b, _ = _calibrate_truncated_normal(
            target_mean=monthly_mean_target,
            std=monthly_std,
            low=monthly_low,
            high=monthly_high,
            mean_is_effective=mean_is_effective,
        )
        # Se guarda la matriz completa para poder recorrer hacia atrás.
        monthly_returns = truncnorm.rvs(
            a=monthly_a,
            b=monthly_b,
            loc=monthly_loc,
            scale=monthly_std,
            size=(n_paths, months),
            random_state=rng,
        ).astype(np.float32, copy=False)

        required = np.zeros(n_paths, dtype=np.float64)
        for t in range(months - 1, -1, -1):
            growth = 1.0 + monthly_returns[:, t].astype(np.float64, copy=False)
            if withdrawal_timing == "begin":
                required = withdrawal_schedule_mm[t] + np.maximum(
                    0.0,
                    (required - recurring_plus_lump_mm[t]) / growth,
                )
            else:
                required = np.maximum(0.0, (required - net_end_cashflows_mm[t]) / growth)
        return required

    raise ValueError("return_model debe ser 'annual_smooth' o 'monthly_iid'")


def required_capital_matrix_mm(
    *,
    edad_final: int,
    retirement_ages: tuple[int, ...],
    success_probabilities: tuple[float, ...],
    n_paths: int,
    annual_return_mean: float,
    annual_return_std: float,
    annual_return_low: float,
    annual_return_high: float,
    withdrawal_monthly_mm: float,
    withdrawal_timing: Literal["begin", "end"] = "end",
    seed: Optional[int] = 456,
    mean_is_effective: bool = True,
    lump_sum_age_events: Optional[tuple[tuple[float, float], ...]] = None,
    recurring_monthly_events: Optional[tuple[tuple, ...]] = None,
    return_model: Literal["annual_smooth", "monthly_iid"] = "monthly_iid",
    withdrawal_indexed_to_inflation: bool = False,
    inflation_annual: float = 0.0,
    withdrawal_index_base_age: Optional[float] = None,
) -> dict:
    """Calcula matriz de capital requerido para jubilar a distintas edades.

    La celda (edad, probabilidad) es el capital inicial requerido en esa edad
    para que el patrimonio no se agote antes de edad_final con esa probabilidad.

    Matemáticamente, para cada path se calcula el capital mínimo que permite
    sobrevivir hasta el final; luego la probabilidad deseada corresponde al
    percentil de esa distribución de capital requerido.
    """
    if n_paths <= 0:
        raise ValueError("n_paths debe ser > 0")
    if not retirement_ages:
        raise ValueError("Debes entregar al menos una edad")
    if not success_probabilities:
        raise ValueError("Debes entregar al menos una probabilidad de éxito")

    ages = tuple(sorted({int(a) for a in retirement_ages if int(a) < int(edad_final)}))
    probs = tuple(sorted({float(p) for p in success_probabilities}))
    if any(p <= 0 or p >= 1 for p in probs):
        raise ValueError("Las probabilidades de éxito deben estar entre 0 y 1")
    if not ages:
        raise ValueError("Todas las edades deben ser menores que edad_final")

    rng_master = np.random.default_rng(seed)
    long_rows = []
    distribution_rows = []

    for age in ages:
        years = int(edad_final - age)
        schedules = _build_retirement_cashflow_schedule_mm(
            edad_inicio=age,
            edad_final=edad_final,
            withdrawal_monthly_mm=withdrawal_monthly_mm,
            withdrawal_timing=withdrawal_timing,
            withdrawal_indexed_to_inflation=withdrawal_indexed_to_inflation,
            inflation_annual=inflation_annual,
            withdrawal_index_base_age=withdrawal_index_base_age,
            recurring_monthly_events=recurring_monthly_events,
            lump_sum_age_events=lump_sum_age_events,
        )
        months = int(schedules["months"])
        seed_age = int(rng_master.integers(0, 2**32 - 1))
        rng = np.random.default_rng(seed_age)
        recurring_plus_lump = schedules["recurring_cashflows_mm"] + schedules["lump_sums_mm"]
        required_by_path = _simulate_required_capital_by_path_mm(
            rng=rng,
            n_paths=n_paths,
            months=months,
            years=years,
            annual_return_mean=annual_return_mean,
            annual_return_std=annual_return_std,
            annual_return_low=annual_return_low,
            annual_return_high=annual_return_high,
            mean_is_effective=mean_is_effective,
            return_model=return_model,
            net_end_cashflows_mm=schedules["net_end_cashflows_mm"],
            withdrawal_schedule_mm=schedules["withdrawal_schedule_mm"],
            recurring_plus_lump_mm=recurring_plus_lump,
            withdrawal_timing=withdrawal_timing,
        )

        for p in probs:
            required_mm = float(np.percentile(required_by_path, p * 100))
            long_rows.append(
                {
                    "edad_jubilacion": age,
                    "prob_exito": p,
                    "prob_exito_pct": p * 100,
                    "capital_requerido_mm": required_mm,
                    "capital_requerido_clp": round(required_mm * 1_000_000, 0),
                    "n_paths": int(n_paths),
                    "modelo_retorno": return_model,
                }
            )

        # Percentiles auxiliares para auditar la distribución por edad.
        pcts = np.percentile(required_by_path, [50, 70, 80, 90, 95])
        distribution_rows.append(
            {
                "edad_jubilacion": age,
                "p50_mm": float(pcts[0]),
                "p70_mm": float(pcts[1]),
                "p80_mm": float(pcts[2]),
                "p90_mm": float(pcts[3]),
                "p95_mm": float(pcts[4]),
                "promedio_mm": float(np.mean(required_by_path)),
                "max_mm": float(np.max(required_by_path)),
            }
        )

    long_df = pd.DataFrame(long_rows)
    matrix_mm = long_df.pivot(index="prob_exito_pct", columns="edad_jubilacion", values="capital_requerido_mm").sort_index()
    matrix_clp = long_df.pivot(index="prob_exito_pct", columns="edad_jubilacion", values="capital_requerido_clp").sort_index()
    matrix_mm.columns = [int(c) for c in matrix_mm.columns]
    matrix_clp.columns = [int(c) for c in matrix_clp.columns]
    matrix_mm.index.name = "prob_exito_pct"
    matrix_clp.index.name = "prob_exito_pct"

    return {
        "matrix_mm": matrix_mm,
        "matrix_clp": matrix_clp,
        "long": long_df,
        "distribution_by_age": pd.DataFrame(distribution_rows),
        "inputs": {
            "edad_final": edad_final,
            "retirement_ages": ages,
            "success_probabilities": probs,
            "n_paths": n_paths,
            "return_model": return_model,
            "withdrawal_monthly_mm": withdrawal_monthly_mm,
            "withdrawal_indexed_to_inflation": withdrawal_indexed_to_inflation,
            "inflation_annual": inflation_annual,
            "withdrawal_index_base_age": withdrawal_index_base_age,
            "seed": seed,
        },
    }
