"""
deteccion.py — Detección de pares cointegrados (in-sample 2008–2020).

Pipeline de dos etapas:
  1. Pre-filtro Engle-Granger (rápido, O(n) por par):
       Rechaza pares con p-value > umbral. Reduce el espacio de búsqueda ~10x.
  2. Validación Johansen (robusto, multivariante):
       Confirma la cointegración sin asumir dirección de causalidad.
       Devuelve el estadístico de traza y el score de calidad.

Refactoriza las funciones del archivo original cointegración.py y añade:
  - Pre-filtro EG para eficiencia computacional
  - Exportación de resultados a CSV
  - Evaluación rolling para verificar estabilidad temporal del par
"""

import itertools
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from datos import preparar_par, MIN_OBS

warnings.filterwarnings("ignore")

RESULTADOS_PATH = os.path.join(os.path.dirname(__file__), "pares_cointegrados.csv")


# ── Test de Engle-Granger (pre-filtro rápido) ─────────────────────────────────

def test_engle_granger(s1: pd.Series, s2: pd.Series, umbral_pvalue: float = 0.05) -> dict:
    """
    Test de cointegración de Engle-Granger en dos pasos.
    Más rápido que Johansen; sirve como cribado inicial.

    Devuelve dict con: pasa (bool), p_value, estadístico.
    """
    log_s1 = np.log(s1)
    log_s2 = np.log(s2)
    try:
        stat, pvalue, _ = coint(log_s1, log_s2)
        return {"pasa": pvalue < umbral_pvalue, "p_value_eg": pvalue, "stat_eg": stat}
    except Exception:
        return {"pasa": False, "p_value_eg": 1.0, "stat_eg": 0.0}


# ── Test de Johansen (validador robusto) ──────────────────────────────────────

def test_johansen(s1: pd.Series, s2: pd.Series) -> dict:
    """
    Test de cointegración de Johansen.
    Utiliza el estadístico de traza al nivel de confianza del 95%.

    Score = traza / valor_crítico_95% → cuanto mayor, más fuerte la cointegración.
    """
    log_par = np.log(pd.concat([s1, s2], axis=1)).dropna()
    if len(log_par) < MIN_OBS:
        return {"cointegrado": False, "traza": 0.0, "critico": 0.0, "score": 0.0}
    try:
        res     = coint_johansen(log_par, det_order=0, k_ar_diff=1)
        traza   = float(res.lr1[0])
        critico = float(res.cvt[0, 1])  # valor crítico al 95%
        score   = traza / critico if critico > 0 else 0.0
        return {
            "cointegrado": traza > critico,
            "traza":        traza,
            "critico":      critico,
            "score":        score,
        }
    except Exception:
        return {"cointegrado": False, "traza": 0.0, "critico": 0.0, "score": 0.0}


# ── Estabilidad temporal (ventana rolling) ────────────────────────────────────

def estabilidad_rolling(
    precios: pd.DataFrame,
    t1: str,
    t2: str,
    ventana: int = 252,
) -> pd.DataFrame:
    """
    Evalúa la cointegración en ventanas rodantes para verificar que el par
    mantiene su relación a lo largo del tiempo (no solo en un periodo).

    Devuelve DataFrame con fecha, estadístico de traza y valor crítico.
    Útil para el gráfico rolling del módulo de evaluación.
    """
    par = preparar_par(precios, t1, t2)
    if par is None:
        return pd.DataFrame()

    log_par  = np.log(par)
    filas    = []

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


# ── Escáner principal ─────────────────────────────────────────────────────────

def escanear_todos_los_pares(
    df_precios: pd.DataFrame,
    umbral_eg: float = 0.05,
    min_score_johansen: float = 1.0,
    max_pares: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Escanea todas las combinaciones de pares posibles en dos etapas:
      1. Pre-filtro Engle-Granger  (descarta rápidamente)
      2. Validación Johansen       (confirma y puntúa)

    Devuelve DataFrame ordenado por score descendente.
    """
    tickers = df_precios.columns.tolist()
    pares   = list(itertools.combinations(tickers, 2))
    if max_pares:
        pares = pares[:max_pares]

    if verbose:
        print(f"\n[INFO] Escaneando {len(pares)} pares posibles...")
        print(f"       Universo: {len(tickers)} tickers")

    resultados = []
    n_eg_pasaron = 0

    for i, (t1, t2) in enumerate(pares):
        par = preparar_par(df_precios, t1, t2)
        if par is None:
            continue

        # Etapa 1: Pre-filtro Engle-Granger
        eg = test_engle_granger(par[t1], par[t2], umbral_eg)
        if not eg["pasa"]:
            continue
        n_eg_pasaron += 1

        # Etapa 2: Validación Johansen
        joh = test_johansen(par[t1], par[t2])
        if not joh["cointegrado"] or joh["score"] < min_score_johansen:
            continue

        resultados.append({
            "ticker1":    t1,
            "ticker2":    t2,
            "score":      round(joh["score"], 4),
            "traza":      round(joh["traza"], 4),
            "critico":    round(joh["critico"], 4),
            "p_value_eg": round(eg["p_value_eg"], 4),
            "n_obs":      len(par),
        })

        if verbose and len(resultados) % 10 == 0:
            print(f"  -> {len(resultados)} pares cointegrados encontrados hasta ahora...")

    if verbose:
        print(f"\n[RESUMEN] {len(pares)} pares evaluados")
        print(f"           {n_eg_pasaron} pasaron el pre-filtro EG")
        print(f"           {len(resultados)} confirmados por Johansen")

    if not resultados:
        return pd.DataFrame()

    df = pd.DataFrame(resultados).sort_values("score", ascending=False).reset_index(drop=True)
    return df


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
    df = cargar_pares(path)
    return df.head(n)
