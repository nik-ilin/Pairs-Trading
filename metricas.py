"""
metricas.py — Métricas de riesgo y rendimiento para evaluación del modelo.

Modelos implementados:
  - Sharpe Ratio        : rentabilidad ajustada por volatilidad total
  - Sortino Ratio       : penaliza solo volatilidad negativa (downside)
  - Calmar Ratio        : CAGR / Máximo Drawdown
  - Omega Ratio         : probabilidad de ganancia ponderada vs pérdida
  - VaR (histórico)     : pérdida máxima al nivel de confianza dado
  - CVaR / ES           : pérdida esperada en el peor % de casos (cola)
  - Máximo Drawdown     : mayor caída pico-a-valle de la curva de capital
  - CAGR                : tasa de crecimiento anual compuesta
  - Win Rate            : porcentaje de trades ganadores
  - Profit Factor       : ratio ganancias brutas / pérdidas brutas
"""

import numpy as np
import pandas as pd

from config import DIAS_ANIO


# ── Rendimiento ───────────────────────────────────────────────────────────────

def cagr(retornos: pd.Series) -> float:
    """Tasa de crecimiento anual compuesta a partir de retornos diarios."""
    n_anos = len(retornos) / DIAS_ANIO
    if n_anos <= 0:
        return 0.0
    valor_final = (1 + retornos).prod()
    return float(valor_final ** (1 / n_anos) - 1)


def sharpe_ratio(retornos: pd.Series, tasa_libre: float = 0.0) -> float:
    """
    Sharpe Ratio anualizado.
    Mide rentabilidad por unidad de riesgo total (volatilidad).
    Objetivo: > 1.0
    """
    exceso = retornos - tasa_libre / DIAS_ANIO
    if exceso.std() == 0:
        return 0.0
    return float((exceso.mean() / exceso.std()) * np.sqrt(DIAS_ANIO))


def sortino_ratio(retornos: pd.Series, tasa_libre: float = 0.0) -> float:
    """
    Sortino Ratio anualizado.
    Como el Sharpe pero solo penaliza la volatilidad negativa (downside).
    Más justo para estrategias asimétricas.
    """
    exceso = retornos - tasa_libre / DIAS_ANIO
    retornos_negativos = exceso[exceso < 0]
    if len(retornos_negativos) == 0:
        return np.inf
    downside_std = np.sqrt((retornos_negativos ** 2).mean()) * np.sqrt(DIAS_ANIO)
    if downside_std == 0:
        return 0.0
    return float(exceso.mean() * DIAS_ANIO / downside_std)


def calmar_ratio(retornos: pd.Series) -> float:
    """
    Calmar Ratio = CAGR / |Máximo Drawdown|.
    Mide cuánta rentabilidad se obtiene por unidad de riesgo de caída extrema.
    """
    mdd = max_drawdown(retornos)
    if mdd == 0:
        return np.inf
    return float(cagr(retornos) / abs(mdd))


def omega_ratio(retornos: pd.Series, umbral: float = 0.0) -> float:
    """
    Omega Ratio: suma de ganancias sobre suma de pérdidas por encima/debajo del umbral.
    Omega > 1 indica que las ganancias superan las pérdidas en términos ponderados.
    """
    ganancias = retornos[retornos > umbral] - umbral
    perdidas  = umbral - retornos[retornos <= umbral]
    if perdidas.sum() == 0:
        return np.inf
    return float(ganancias.sum() / perdidas.sum())


# ── Riesgo ────────────────────────────────────────────────────────────────────

def max_drawdown(retornos: pd.Series) -> float:
    """
    Máximo Drawdown (MDD): mayor caída porcentual desde un pico hasta un valle.
    Objetivo: < 15%  (expresado como valor negativo, ej. -0.12 = -12%)
    """
    curva = (1 + retornos).cumprod()
    pico  = curva.cummax()
    dd    = (curva - pico) / pico
    return float(dd.min())


def curva_drawdown(retornos: pd.Series) -> pd.Series:
    """Serie temporal del drawdown (útil para graficar)."""
    curva = (1 + retornos).cumprod()
    pico  = curva.cummax()
    return (curva - pico) / pico


def var_historico(retornos: pd.Series, nivel: float = 0.05) -> float:
    """
    Value at Risk histórico al nivel de confianza (1 - nivel).
    VaR 5% = pérdida que no se supera el 95% de los días.
    Valor negativo, ej. -0.02 = -2% en un día.
    """
    return float(np.percentile(retornos, nivel * 100))


def cvar(retornos: pd.Series, nivel: float = 0.05) -> float:
    """
    CVaR (Conditional VaR) / Expected Shortfall al nivel dado.
    Media de pérdidas en el peor % de casos. Mejor que VaR para capturar colas.
    """
    umbral = var_historico(retornos, nivel)
    cola   = retornos[retornos <= umbral]
    if len(cola) == 0:
        return umbral
    return float(cola.mean())


# ── Estadísticas de trades ────────────────────────────────────────────────────

def win_rate(pnl_trades: pd.Series) -> float:
    """Porcentaje de trades con resultado positivo."""
    if len(pnl_trades) == 0:
        return 0.0
    return float((pnl_trades > 0).mean())


def profit_factor(pnl_trades: pd.Series) -> float:
    """
    Profit Factor = ganancias brutas / pérdidas brutas.
    > 1.5 se considera bueno; > 2.0 excelente.
    """
    ganancias = pnl_trades[pnl_trades > 0].sum()
    perdidas  = abs(pnl_trades[pnl_trades < 0].sum())
    if perdidas == 0:
        return np.inf
    return float(ganancias / perdidas)


def duracion_media_trade(fechas_entrada: pd.Series, fechas_salida: pd.Series) -> float:
    """Duración media de un trade en días."""
    if len(fechas_entrada) == 0:
        return 0.0
    duraciones = (fechas_salida - fechas_entrada).dt.days
    return float(duraciones.mean())


# ── Reporte completo ──────────────────────────────────────────────────────────

def reporte_completo(
    retornos: pd.Series,
    pnl_trades: pd.Series | None = None,
    tasa_libre: float = 0.0,
    nombre: str = "Estrategia",
) -> pd.DataFrame:
    """
    Genera un DataFrame con todas las métricas clave del modelo.
    Incluye semáforos de cumplimiento de los objetivos SMART.
    """
    metricas = {
        "CAGR (%)":             round(cagr(retornos) * 100, 2),
        "Sharpe Ratio":         round(sharpe_ratio(retornos, tasa_libre), 3),
        "Sortino Ratio":        round(sortino_ratio(retornos, tasa_libre), 3),
        "Calmar Ratio":         round(calmar_ratio(retornos), 3),
        "Omega Ratio":          round(omega_ratio(retornos), 3),
        "Máx. Drawdown (%)":    round(max_drawdown(retornos) * 100, 2),
        "VaR 95% (%)":          round(var_historico(retornos, 0.05) * 100, 2),
        "CVaR 95% (%)":         round(cvar(retornos, 0.05) * 100, 2),
        "Volatilidad Anual (%)": round(retornos.std() * np.sqrt(DIAS_ANIO) * 100, 2),
        "Retorno Total (%)":    round(((1 + retornos).prod() - 1) * 100, 2),
        "N° Días":              len(retornos),
    }

    if pnl_trades is not None and len(pnl_trades) > 0:
        metricas.update({
            "N° Trades":    len(pnl_trades),
            "Win Rate (%)": round(win_rate(pnl_trades) * 100, 2),
            "Profit Factor": round(profit_factor(pnl_trades), 3),
        })

    df = pd.DataFrame.from_dict(metricas, orient="index", columns=[nombre])

    # Semáforos SMART: Sharpe > 1, MDD < 15%
    sharpe_ok = metricas["Sharpe Ratio"] > 1.0
    mdd_ok    = metricas["Máx. Drawdown (%)"] > -15.0

    print(f"\n{'='*50}")
    print(f"  REPORTE: {nombre}")
    print(f"{'='*50}")
    print(df.to_string())
    print(f"\n  OBJETIVO SMART — Sharpe > 1:   {'✓ CUMPLE' if sharpe_ok else '✗ NO CUMPLE'}")
    print(f"  OBJETIVO SMART — MDD < 15%:    {'✓ CUMPLE' if mdd_ok    else '✗ NO CUMPLE'}")
    print(f"{'='*50}\n")

    return df
