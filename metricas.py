"""Performance and risk metrics for daily return series."""

import numpy as np
import pandas as pd

from config import DIAS_ANIO


def cagr(retornos: pd.Series) -> float:
    n_anos = len(retornos) / DIAS_ANIO
    if n_anos <= 0:
        return 0.0
    valor_final = (1 + retornos).prod()
    return float(valor_final ** (1 / n_anos) - 1)


def sharpe_ratio(retornos: pd.Series, tasa_libre: float = 0.0) -> float:
    exceso = retornos - tasa_libre / DIAS_ANIO
    if exceso.std() == 0:
        return 0.0
    return float((exceso.mean() / exceso.std()) * np.sqrt(DIAS_ANIO))


def sortino_ratio(retornos: pd.Series, tasa_libre: float = 0.0) -> float:
    exceso = retornos - tasa_libre / DIAS_ANIO
    retornos_negativos = exceso[exceso < 0]
    if len(retornos_negativos) == 0:
        return np.inf
    downside_std = np.sqrt((retornos_negativos**2).mean()) * np.sqrt(DIAS_ANIO)
    if downside_std == 0:
        return 0.0
    return float(exceso.mean() * DIAS_ANIO / downside_std)


def calmar_ratio(retornos: pd.Series) -> float:
    mdd = max_drawdown(retornos)
    if mdd == 0:
        return np.inf
    return float(cagr(retornos) / abs(mdd))


def omega_ratio(retornos: pd.Series, umbral: float = 0.0) -> float:
    ganancias = retornos[retornos > umbral] - umbral
    perdidas = umbral - retornos[retornos <= umbral]
    if perdidas.sum() == 0:
        return np.inf
    return float(ganancias.sum() / perdidas.sum())


def max_drawdown(retornos: pd.Series) -> float:
    curva = (1 + retornos).cumprod()
    pico = curva.cummax()
    dd = (curva - pico) / pico
    return float(dd.min())


def curva_drawdown(retornos: pd.Series) -> pd.Series:
    curva = (1 + retornos).cumprod()
    pico = curva.cummax()
    return (curva - pico) / pico


def var_historico(retornos: pd.Series, nivel: float = 0.05) -> float:
    return float(np.percentile(retornos, nivel * 100))


def cvar(retornos: pd.Series, nivel: float = 0.05) -> float:
    umbral = var_historico(retornos, nivel)
    cola = retornos[retornos <= umbral]
    if len(cola) == 0:
        return umbral
    return float(cola.mean())


def win_rate(pnl_trades: pd.Series) -> float:
    if len(pnl_trades) == 0:
        return 0.0
    return float((pnl_trades > 0).mean())


def profit_factor(pnl_trades: pd.Series) -> float:
    ganancias = pnl_trades[pnl_trades > 0].sum()
    perdidas = abs(pnl_trades[pnl_trades < 0].sum())
    if perdidas == 0:
        return np.inf
    return float(ganancias / perdidas)


def duracion_media_trade(fechas_entrada: pd.Series, fechas_salida: pd.Series) -> float:
    if len(fechas_entrada) == 0:
        return 0.0
    duraciones = (fechas_salida - fechas_entrada).dt.days
    return float(duraciones.mean())


def reporte_completo(
    retornos: pd.Series,
    pnl_trades: pd.Series | None = None,
    tasa_libre: float = 0.0,
    nombre: str = "Estrategia",
) -> pd.DataFrame:
    metricas = {
        "CAGR (%)": round(cagr(retornos) * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio(retornos, tasa_libre), 3),
        "Sortino Ratio": round(sortino_ratio(retornos, tasa_libre), 3),
        "Calmar Ratio": round(calmar_ratio(retornos), 3),
        "Omega Ratio": round(omega_ratio(retornos), 3),
        "Máx. Drawdown (%)": round(max_drawdown(retornos) * 100, 2),
        "VaR 95% (%)": round(var_historico(retornos, 0.05) * 100, 2),
        "CVaR 95% (%)": round(cvar(retornos, 0.05) * 100, 2),
        "Volatilidad Anual (%)": round(retornos.std() * np.sqrt(DIAS_ANIO) * 100, 2),
        "Retorno Total (%)": round(((1 + retornos).prod() - 1) * 100, 2),
        "N° Días": len(retornos),
    }

    if pnl_trades is not None and len(pnl_trades) > 0:
        metricas.update(
            {
                "N° Trades": len(pnl_trades),
                "Win Rate (%)": round(win_rate(pnl_trades) * 100, 2),
                "Profit Factor": round(profit_factor(pnl_trades), 3),
            }
        )

    df = pd.DataFrame.from_dict(metricas, orient="index", columns=[nombre])

    sharpe_ok = metricas["Sharpe Ratio"] > 1.0
    mdd_ok = metricas["Máx. Drawdown (%)"] > -15.0

    print(f"\n{'=' * 50}")
    print(f"  REPORTE: {nombre}")
    print(f"{'=' * 50}")
    print(df.to_string())
    print(f"\n  OBJETIVO SMART - Sharpe > 1:   {'OK CUMPLE' if sharpe_ok else 'FAIL NO CUMPLE'}")
    print(f"  OBJETIVO SMART - MDD < 15%:    {'OK CUMPLE' if mdd_ok else 'FAIL NO CUMPLE'}")
    print(f"{'=' * 50}\n")

    return df
