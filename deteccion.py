"""
deteccion.py — Detección de pares cointegrados con datos horarios (12 meses).

Pipeline de 2 etapas, de menor a mayor coste computacional:
  1. Test Engle-Granger: cribado rápido de cointegración actual (O(n) por par)
  2. Test Johansen     : validación robusta con puntuación

Usa datos horarios de Alpaca (últimos 12 meses ≈ 1638 barras).
El objetivo es detectar pares cointegrados AHORA, no históricamente.
"""

import itertools
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from config import (
    MIN_OBS_HORARIO, UMBRAL_EG, MIN_SCORE_JOHANSEN,
    VENTANA_ROLLING, HORAS_DIA,
)


def _bpd(index: pd.DatetimeIndex) -> float:
    """Barras por día de negociación: HORAS_DIA para horario, 1.0 para diario."""
    if len(index) < 10:
        return 1.0
    n_dias = pd.Series(index).dt.date.nunique()
    return HORAS_DIA if (len(index) / max(n_dias, 1)) > 2 else 1.0


def _min_obs(index: pd.DatetimeIndex) -> int:
    """Mínimo de observaciones escalado según la frecuencia de los datos (margen 20%)."""
    return max(30, int(MIN_OBS_HORARIO / HORAS_DIA * _bpd(index) * 0.8))

warnings.filterwarnings("ignore")

RESULTADOS_PATH = os.path.join(os.path.dirname(__file__), "pares_cointegrados.csv")


# ── Etapa 1: Test Engle-Granger ───────────────────────────────────────────────

def test_engle_granger(s1: pd.Series, s2: pd.Series, umbral_pvalue: float = UMBRAL_EG) -> dict:
    """
    Test de cointegración de Engle-Granger en dos pasos. Cribado rápido O(n).

    Returns:
        dict con pasa (bool), p_value_eg, stat_eg.
    """
    log_s1 = np.log(s1)
    log_s2 = np.log(s2)
    try:
        stat, pvalue, _ = coint(log_s1, log_s2)
        return {"pasa": pvalue < umbral_pvalue, "p_value_eg": pvalue, "stat_eg": stat}
    except Exception:
        return {"pasa": False, "p_value_eg": 1.0, "stat_eg": 0.0}


# ── Etapa 3: Test Johansen ────────────────────────────────────────────────────

def test_johansen(s1: pd.Series, s2: pd.Series) -> dict:
    """
    Test de cointegración de Johansen. Estadístico de traza al 95% de confianza.
    Score = traza / valor_crítico — cuanto mayor, más fuerte la cointegración.

    Returns:
        dict con cointegrado (bool), traza, critico, score.
    """
    log_par = np.log(pd.concat([s1, s2], axis=1)).dropna()
    if len(log_par) < _min_obs(s1.index):
        return {"cointegrado": False, "traza": 0.0, "critico": 0.0, "score": 0.0}
    try:
        res     = coint_johansen(log_par, det_order=0, k_ar_diff=1)
        traza   = float(res.lr1[0])
        critico = float(res.cvt[0, 1])
        score   = traza / critico if critico > 0 else 0.0
        return {
            "cointegrado": traza > critico,
            "traza":        traza,
            "critico":      critico,
            "score":        score,
        }
    except Exception:
        return {"cointegrado": False, "traza": 0.0, "critico": 0.0, "score": 0.0}


# ── Escáner principal ─────────────────────────────────────────────────────────

def escanear_todos_los_pares(
    df_precios: pd.DataFrame,
    umbral_eg: float = UMBRAL_EG,
    min_score_johansen: float = MIN_SCORE_JOHANSEN,
    max_pares: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Escanea todos los pares posibles en 2 etapas:

      1. Test Engle-Granger: cribado rápido (O(n) por par)
      2. Test Johansen:      validación robusta

    Args:
        df_precios:         DataFrame de precios horarios (filas=hora, columnas=tickers).
        umbral_eg:          P-value máximo para el test EG.
        min_score_johansen: Score mínimo Johansen (traza / valor_crítico).
        max_pares:          Límite de pares a evaluar (útil para pruebas).
        verbose:            Imprime resumen de cada etapa.

    Returns:
        DataFrame ordenado por score descendente con columnas:
        ticker1, ticker2, score, traza, critico, p_value_eg, n_obs.
    """
    tickers     = df_precios.columns.tolist()
    todos_pares = list(itertools.combinations(tickers, 2))
    if max_pares:
        todos_pares = todos_pares[:max_pares]
    min_obs_req = _min_obs(df_precios.index)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ESCÁNER DE PARES COINTEGRADOS")
        print(f"{'='*60}")
        print(f"  Universo:        {len(tickers):>6} tickers")
        print(f"  Pares a evaluar: {len(todos_pares):>6,}")
        print(f"  EG + Johansen en curso...")

    resultados = []
    n_eg_pasan = 0

    for t1, t2 in todos_pares:
        s1 = df_precios[t1].dropna()
        s2 = df_precios[t2].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < min_obs_req:
            continue
        s1, s2 = s1.loc[idx], s2.loc[idx]

        # Etapa 1: Engle-Granger
        eg = test_engle_granger(s1, s2, umbral_eg)
        if not eg["pasa"]:
            continue
        n_eg_pasan += 1

        # Etapa 2: Johansen
        joh = test_johansen(s1, s2)
        if not joh["cointegrado"] or joh["score"] < min_score_johansen:
            continue

        resultados.append({
            "ticker1":    t1,
            "ticker2":    t2,
            "score":      round(joh["score"], 4),
            "traza":      round(joh["traza"], 4),
            "critico":    round(joh["critico"], 4),
            "p_value_eg": round(eg["p_value_eg"], 4),
            "n_obs":      len(idx),
        })

    if verbose:
        print(f"\n[RESUMEN]")
        print(f"  Pares evaluados:           {len(todos_pares):>6,}")
        print(f"  Pares que pasan EG:        {n_eg_pasan:>6,}")
        print(f"  Pares confirmados (final): {len(resultados):>6,}")
        print(f"{'='*60}\n")

    if not resultados:
        return pd.DataFrame()

    return (
        pd.DataFrame(resultados)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


# ── Diagnóstico de madurez de cointegración ──────────────────────────────────

def diagnostico_madurez_cointegracion(rolling_df: pd.DataFrame) -> dict:
    """
    Diagnostica la madurez de una cointegración a partir del histórico rolling
    de scores Johansen (salida de estabilidad_rolling).

    Compara el score del último cuarto del período con el inicial y calcula
    qué fracción del tiempo el par estuvo cointegrado.

    Estados:
      RECIENTE    — relación nueva y en ascenso: ventana óptima de entrada
      CONSOLIDADA — relación robusta y estable a lo largo del histórico
      MADURA      — larga historia pero score en declive: vigilar ruptura
      AGOTADA     — relación estadística deteriorada: no operar
      INESTABLE   — cointegración débil o intermitente: precaución
    """
    if rolling_df.empty or len(rolling_df) < 10:
        return {
            "estado": "DESCONOCIDO",
            "descripcion": "Datos de estabilidad rolling insuficientes.",
            "score_reciente": 0.0,
            "fraccion_activa": 0.0,
            "tendencia": "—",
        }

    n      = len(rolling_df)
    n_q    = max(5, n // 4)
    scores = rolling_df["score"]

    score_inicial    = float(scores.iloc[:n_q].mean())
    score_rec_medio  = float(scores.iloc[-n_q:].mean())
    score_hist_medio = float(scores.iloc[:-n_q].mean()) if n > n_q else score_inicial
    fraccion_activa  = float((scores > 1.0).mean())
    fraccion_hist    = float((scores.iloc[:-n_q] > 1.0).mean()) if n > n_q else 0.0

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
        "estado":          estado,
        "descripcion":     descripcion,
        "score_reciente":  round(score_rec_medio, 4),
        "fraccion_activa": round(fraccion_activa, 3),
        "tendencia":       tendencia,
    }


def diagnostico_madurez_simple(s1: pd.Series, s2: pd.Series) -> dict:
    """
    Diagnóstico rápido de madurez sin rolling Johansen completo.
    Divide el histórico en tres tercios y compara los p-values de EG entre períodos.
    Diseñado para el pipeline diario donde el rendimiento es crítico.
    """
    idx = s1.dropna().index.intersection(s2.dropna().index)
    n   = len(idx)

    if n < 90:
        return {
            "estado":          "DESCONOCIDO",
            "descripcion":     "Datos insuficientes para diagnóstico de madurez.",
            "fraccion_activa": 0.0,
            "tendencia":       "—",
        }

    tercio       = n // 3
    p_temprano   = test_engle_granger(s1.loc[idx[:tercio]],         s2.loc[idx[:tercio]])["p_value_eg"]
    p_intermedio = test_engle_granger(s1.loc[idx[tercio:2*tercio]], s2.loc[idx[tercio:2*tercio]])["p_value_eg"]
    p_reciente   = test_engle_granger(s1.loc[idx[2*tercio:]],       s2.loc[idx[2*tercio:]])["p_value_eg"]

    n_coints = sum(p < UMBRAL_EG for p in [p_temprano, p_intermedio, p_reciente])
    fraccion  = n_coints / 3.0

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
            "Cointegración intermitente — la relación estadística es débil o poco estable. "
            "Operar con precaución."
        )

    return {
        "estado":          estado,
        "descripcion":     descripcion,
        "fraccion_activa": round(fraccion, 3),
        "tendencia":       tendencia,
    }


# ── Estabilidad rolling para gráficos (evaluacion.py) ─────────────────────────

def estabilidad_rolling(
    precios: pd.DataFrame,
    t1: str,
    t2: str,
    ventana: int | None = None,
) -> pd.DataFrame:
    """
    Evalúa la cointegración en ventanas rolling para visualización.
    Usa Johansen para obtener el score continuo.

    Returns:
        DataFrame indexado por fecha con columnas: traza, critico, score.
    """
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
            res     = coint_johansen(subset, det_order=0, k_ar_diff=1)
            traza   = float(res.lr1[0])
            critico = float(res.cvt[0, 1])
            filas.append({
                "fecha":   log_par.index[i],
                "traza":   traza,
                "critico": critico,
                "score":   traza / critico if critico > 0 else 0.0,
            })
        except Exception:
            continue

    return pd.DataFrame(filas).set_index("fecha") if filas else pd.DataFrame()


# ── Guardar y cargar resultados ───────────────────────────────────────────────

def guardar_pares(df: pd.DataFrame, path: str = RESULTADOS_PATH) -> None:
    """Exporta el listado de pares cointegrados a CSV."""
    df.to_csv(path, index=False)
    print(f"[OK] {len(df)} pares guardados en {path}")


def cargar_pares(path: str = RESULTADOS_PATH) -> pd.DataFrame:
    """Carga el listado de pares cointegrados desde CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta primero el escáner.")
    return pd.read_csv(path)


def top_pares(n: int = 20, path: str = RESULTADOS_PATH) -> pd.DataFrame:
    """Devuelve los N mejores pares por score de cointegración."""
    return cargar_pares(path).head(n)
