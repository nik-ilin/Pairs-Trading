"""Pair discovery and rolling cointegration diagnostics."""

import itertools
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from config import (
    HORAS_DIA,
    MIN_OBS_HISTORICO,
    MIN_OBS_ROLLING,
    MIN_SCORE_JOHANSEN,
    UMBRAL_EG,
    VENTANA_ROLLING,
)


def _bpd(index: pd.DatetimeIndex) -> float:
    if len(index) < 10:
        return 1.0
    n_dias = pd.Series(index).dt.date.nunique()
    return HORAS_DIA if (len(index) / max(n_dias, 1)) > 2 else 1.0


warnings.filterwarnings("ignore")

RESULTADOS_PATH = os.path.join(os.path.dirname(__file__), "pares_cointegrados.csv")


def test_engle_granger(s1: pd.Series, s2: pd.Series, umbral_pvalue: float = UMBRAL_EG) -> dict:
    log_s1 = np.log(s1)
    log_s2 = np.log(s2)
    try:
        stat, pvalue, _ = coint(log_s1, log_s2)
        return {"pasa": pvalue < umbral_pvalue, "p_value_eg": pvalue, "stat_eg": stat}
    except Exception:
        return {"pasa": False, "p_value_eg": 1.0, "stat_eg": 0.0}


def test_johansen(s1: pd.Series, s2: pd.Series, min_observaciones: int = MIN_OBS_ROLLING) -> dict:
    log_par = np.log(pd.concat([s1, s2], axis=1)).dropna()
    if len(log_par) < min_observaciones:
        return {"cointegrado": False, "traza": 0.0, "critico": 0.0, "score": 0.0}
    try:
        res = coint_johansen(log_par, det_order=0, k_ar_diff=1)
        traza = float(res.lr1[0])
        critico = float(res.cvt[0, 1])
        score = traza / critico if critico > 0 else 0.0
        return {
            "cointegrado": traza > critico,
            "traza": traza,
            "critico": critico,
            "score": score,
        }
    except Exception:
        return {"cointegrado": False, "traza": 0.0, "critico": 0.0, "score": 0.0}


def escanear_todos_los_pares(
    df_precios: pd.DataFrame,
    umbral_eg: float = UMBRAL_EG,
    min_score_johansen: float = MIN_SCORE_JOHANSEN,
    max_pares: int | None = None,
    verbose: bool = True,
    min_observaciones: int = MIN_OBS_HISTORICO,
) -> pd.DataFrame:
    tickers = df_precios.columns.tolist()
    todos_pares = list(itertools.combinations(tickers, 2))
    if max_pares:
        todos_pares = todos_pares[:max_pares]
    min_obs_req = min_observaciones

    if verbose:
        print(f"\n{'=' * 60}")
        print("  ESCÁNER DE PARES COINTEGRADOS")
        print(f"{'=' * 60}")
        print(f"  Universo:        {len(tickers):>6} tickers")
        print(f"  Pares a evaluar: {len(todos_pares):>6,}")
        print("  EG + Johansen en curso...")

    resultados = []
    candidatos = []

    for t1, t2 in todos_pares:
        s1 = df_precios[t1].dropna()
        s2 = df_precios[t2].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < min_obs_req:
            continue
        s1, s2 = s1.loc[idx], s2.loc[idx]

        eg = test_engle_granger(s1, s2, umbral_eg)
        candidatos.append((t1, t2, s1, s2, eg))

    pvalues = np.array([c[4]["p_value_eg"] for c in candidatos])
    orden = np.argsort(pvalues)
    aceptados = np.zeros(len(pvalues), dtype=bool)
    limites = umbral_eg * np.arange(1, len(pvalues) + 1) / max(len(pvalues), 1)
    validos = np.where(pvalues[orden] <= limites)[0]
    if len(validos):
        aceptados[orden[: validos[-1] + 1]] = True
    n_eg_pasan = int(aceptados.sum())

    for aceptado, (t1, t2, s1, s2, eg) in zip(aceptados, candidatos):
        if not aceptado:
            continue
        joh = test_johansen(s1, s2, min_observaciones=min_obs_req)
        if not joh["cointegrado"] or joh["score"] < min_score_johansen:
            continue

        resultados.append(
            {
                "ticker1": t1,
                "ticker2": t2,
                "score": round(joh["score"], 4),
                "traza": round(joh["traza"], 4),
                "critico": round(joh["critico"], 4),
                "p_value_eg": round(eg["p_value_eg"], 4),
                "fdr_bh": True,
                "n_obs": len(idx),
            }
        )

    if verbose:
        print("\n[RESUMEN]")
        print(f"  Pares evaluados:           {len(todos_pares):>6,}")
        print(f"  Pares que pasan EG:        {n_eg_pasan:>6,}")
        print(f"  Pares confirmados (final): {len(resultados):>6,}")
        print(f"{'=' * 60}\n")

    if not resultados:
        return pd.DataFrame()

    return pd.DataFrame(resultados).sort_values("score", ascending=False).reset_index(drop=True)


def diagnostico_madurez_cointegracion(rolling_df: pd.DataFrame) -> dict:
    if rolling_df.empty or len(rolling_df) < 10:
        return {
            "estado": "DESCONOCIDO",
            "descripcion": "Datos de estabilidad rolling insuficientes.",
            "score_reciente": 0.0,
            "fraccion_activa": 0.0,
            "tendencia": "—",
        }

    n = len(rolling_df)
    n_q = max(5, n // 4)
    scores = rolling_df["score"]

    score_inicial = float(scores.iloc[:n_q].mean())
    score_rec_medio = float(scores.iloc[-n_q:].mean())
    score_hist_medio = float(scores.iloc[:-n_q].mean()) if n > n_q else score_inicial
    fraccion_activa = float((scores > 1.0).mean())
    fraccion_hist = float((scores.iloc[:-n_q] > 1.0).mean()) if n > n_q else 0.0

    diff = score_rec_medio - score_inicial
    if diff > 0.10:
        tendencia = "↑ subiendo"
    elif diff < -0.10:
        tendencia = "↓ bajando"
    else:
        tendencia = "→ estable"

    if score_rec_medio < 0.85 and fraccion_hist >= 0.30:
        estado = "AGOTADA"
        descripcion = (
            "Cointegración debilitada — la relación estadística que existía se ha deteriorado "
            "significativamente. No iniciar nuevas posiciones; cerrar las abiertas si procede."
        )
    elif fraccion_hist < 0.40 and score_rec_medio >= 1.10 and diff > 0:
        estado = "RECIENTE"
        descripcion = (
            "Cointegración emergente — la relación estadística es nueva y está ganando fuerza. "
            "Ventana óptima para iniciar el seguimiento con tamaño de posición reducido."
        )
    elif fraccion_activa >= 0.60 and score_rec_medio >= 1.05 and score_rec_medio >= score_hist_medio * 0.88:
        estado = "CONSOLIDADA"
        descripcion = (
            "Cointegración consolidada — relación robusta y estable a lo largo del histórico. "
            "Alta confianza estadística. Condiciones óptimas para operar."
        )
    elif fraccion_activa >= 0.45 and score_rec_medio < score_hist_medio * 0.85:
        estado = "MADURA"
        descripcion = (
            "Cointegración madura — larga historia pero el score está en declive. "
            "Probabilidad de ruptura moderada. Reducir tamaño de posición y vigilar de cerca."
        )
    else:
        estado = "INESTABLE"
        descripcion = (
            "Cointegración intermitente — la relación estadística es débil o poco estable en el tiempo. "
            "Operar con precaución y reducir el tamaño de posición."
        )

    return {
        "estado": estado,
        "descripcion": descripcion,
        "score_reciente": round(score_rec_medio, 4),
        "fraccion_activa": round(fraccion_activa, 3),
        "tendencia": tendencia,
    }


def diagnostico_madurez_simple(s1: pd.Series, s2: pd.Series) -> dict:
    idx = s1.dropna().index.intersection(s2.dropna().index)
    n = len(idx)

    if n < 90:
        return {
            "estado": "DESCONOCIDO",
            "descripcion": "Datos insuficientes para diagnóstico de madurez.",
            "fraccion_activa": 0.0,
            "tendencia": "—",
        }

    tercio = n // 3
    p_temprano = test_engle_granger(s1.loc[idx[:tercio]], s2.loc[idx[:tercio]])["p_value_eg"]
    p_intermedio = test_engle_granger(s1.loc[idx[tercio : 2 * tercio]], s2.loc[idx[tercio : 2 * tercio]])["p_value_eg"]
    p_reciente = test_engle_granger(s1.loc[idx[2 * tercio :]], s2.loc[idx[2 * tercio :]])["p_value_eg"]

    n_coints = sum(p < UMBRAL_EG for p in [p_temprano, p_intermedio, p_reciente])
    fraccion = n_coints / 3.0

    if p_reciente < p_temprano * 0.6:
        tendencia = "↑ subiendo"
    elif p_reciente > p_temprano * 1.6:
        tendencia = "↓ bajando"
    else:
        tendencia = "→ estable"

    if p_temprano < UMBRAL_EG and p_reciente > 0.10:
        estado = "AGOTADA"
        descripcion = (
            "Cointegración debilitada — la relación estadística que existía se ha deteriorado. "
            "No iniciar nuevas posiciones; cerrar las abiertas si procede."
        )
    elif p_temprano > 0.10 and p_reciente < UMBRAL_EG:
        estado = "RECIENTE"
        descripcion = (
            "Cointegración emergente — la relación estadística es nueva. "
            "Ventana óptima para iniciar el seguimiento con tamaño de posición reducido."
        )
    elif n_coints == 3 and p_reciente < 0.03:
        estado = "CONSOLIDADA"
        descripcion = (
            "Cointegración consolidada — relación robusta y estable a lo largo del histórico. "
            "Alta confianza estadística. Condiciones óptimas para operar."
        )
    elif p_temprano < UMBRAL_EG and p_intermedio < UMBRAL_EG and p_reciente >= 0.04:
        estado = "MADURA"
        descripcion = (
            "Cointegración madura — larga historia pero señales de debilitamiento reciente. "
            "Reducir tamaño de posición y vigilar de cerca."
        )
    else:
        estado = "INESTABLE"
        descripcion = (
            "Cointegración intermitente — la relación estadística es débil o poco estable. Operar con precaución."
        )

    return {
        "estado": estado,
        "descripcion": descripcion,
        "fraccion_activa": round(fraccion, 3),
        "tendencia": tendencia,
    }


def estabilidad_rolling(
    precios: pd.DataFrame,
    t1: str,
    t2: str,
    ventana: int | None = None,
) -> pd.DataFrame:
    if t1 not in precios.columns or t2 not in precios.columns:
        return pd.DataFrame()

    if ventana is None:
        ventana = max(30, int(VENTANA_ROLLING / HORAS_DIA * _bpd(precios.index)))

    log_par = np.log(precios[[t1, t2]]).dropna()
    if len(log_par) < ventana + 10:
        return pd.DataFrame()

    filas = []
    for i in range(ventana, len(log_par)):
        subset = log_par.iloc[i - ventana : i]
        try:
            res = coint_johansen(subset, det_order=0, k_ar_diff=1)
            traza = float(res.lr1[0])
            critico = float(res.cvt[0, 1])
            filas.append(
                {
                    "fecha": log_par.index[i],
                    "traza": traza,
                    "critico": critico,
                    "score": traza / critico if critico > 0 else 0.0,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(filas).set_index("fecha") if filas else pd.DataFrame()


def guardar_pares(df: pd.DataFrame, path: str = RESULTADOS_PATH) -> None:
    df.to_csv(path, index=False)
    print(f"[OK] {len(df)} pares guardados en {path}")


def cargar_pares(path: str = RESULTADOS_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta primero el escáner.")
    return pd.read_csv(path)


def top_pares(n: int = 20, path: str = RESULTADOS_PATH) -> pd.DataFrame:
    return cargar_pares(path).head(n)
