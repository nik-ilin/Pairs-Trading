"""
automatizacion.py — Ejecución autónoma diaria de señales (SMART Objetivo 3).

Genera señales reproducibles basadas en desviaciones estadísticas del spread
para los pares validados en el backtesting. Puede ejecutarse de forma programada
(scheduler) al cierre del mercado americano (16:05 EST).

Evaluaciones estadísticas adicionales en tiempo real:
  - Test ADF sobre el spread reciente (confirma estacionariedad en la ventana activa)
  - Z-score con banda de Bollinger para visualización
  - Régimen de volatilidad (baja / normal / alta) según percentil histórico
  - Alerta de ruptura de cointegración (Johansen sobre ventana reciente)
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from datos import descargar_precios, preparar_par
from spread import (
    calcular_spread_kalman,
    calcular_zscore,
    calcular_half_life,
    parametros_ou,
    generar_señales,
    Señal,
)
from deteccion import test_johansen, cargar_pares

warnings.filterwarnings("ignore")

SEÑALES_PATH = os.path.join(os.path.dirname(__file__), "señales_diarias.csv")
ESTADO_PATH  = os.path.join(os.path.dirname(__file__), "estado_posiciones.json")

# Días de historial necesarios para calcular el z-score con ventana OU
BUFFER_DIAS = 180


# ── Tests estadísticos en tiempo real ────────────────────────────────────────

def test_adf_spread(spread: pd.Series) -> dict:
    """
    Test de Dickey-Fuller aumentado sobre el spread reciente.
    Confirma que el spread sigue siendo estacionario (condición de trading).
    p-value < 0.05 → estacionario → spread válido para operar.
    """
    try:
        resultado = adfuller(spread.dropna(), autolag="AIC")
        return {
            "adf_stat":    round(float(resultado[0]), 4),
            "p_value_adf": round(float(resultado[1]), 4),
            "estacionario": resultado[1] < 0.05,
        }
    except Exception:
        return {"adf_stat": 0.0, "p_value_adf": 1.0, "estacionario": False}


def regimen_volatilidad(spread: pd.Series, ventana_vol: int = 20) -> str:
    """
    Clasifica el régimen de volatilidad del spread en: BAJA / NORMAL / ALTA.
    Basado en el percentil histórico de la volatilidad rolling actual.
    Útil para escalar el tamaño de posición o pausar operaciones en alta vol.
    """
    vol_rolling = spread.rolling(ventana_vol).std()
    vol_actual  = vol_rolling.iloc[-1]
    p25, p75    = vol_rolling.quantile(0.25), vol_rolling.quantile(0.75)

    if vol_actual <= p25:
        return "BAJA"
    elif vol_actual >= p75:
        return "ALTA"
    else:
        return "NORMAL"


def bollinger_spread(spread: pd.Series, window: int = 20, n_std: float = 2.0) -> dict:
    """
    Calcula las bandas de Bollinger sobre el spread para visualización.
    El spread operando fuera de las bandas indica oportunidad de reversión.
    """
    media  = spread.rolling(window).mean()
    std    = spread.rolling(window).std()
    return {
        "media":  media,
        "banda_sup": media + n_std * std,
        "banda_inf": media - n_std * std,
    }


def verificar_cointegración_activa(
    precios: pd.DataFrame,
    t1: str,
    t2: str,
    ventana: int = 126,  # ~6 meses
) -> dict:
    """
    Verifica que el par sigue cointegrado en la ventana más reciente.
    Una ruptura de cointegración invalida la señal (falsa reversión).
    """
    par = preparar_par(precios, t1, t2)
    if par is None or len(par) < ventana:
        return {"cointegrado": False, "alerta": "Datos insuficientes"}

    subset = par.tail(ventana)
    joh    = test_johansen(subset[t1], subset[t2])
    return {
        "cointegrado":   joh["cointegrado"],
        "score_johansen": joh["score"],
        "alerta":        "" if joh["cointegrado"] else "RUPTURA DE COINTEGRACION — NO OPERAR",
    }


# ── Evaluación de un par activo ───────────────────────────────────────────────

def evaluar_par(
    precios: pd.DataFrame,
    t1: str,
    t2: str,
    entrada_z: float = 2.0,
    salida_z: float = 0.5,
    stop_z: float = 3.5,
) -> dict:
    """
    Evalúa el estado actual de un par cointegrado y genera la señal del día.

    Devuelve un diccionario completo con:
      señal, z-score actual, parámetros OU, tests estadísticos, régimen.
    """
    par = preparar_par(precios, t1, t2)
    if par is None:
        return {"par": f"{t1}/{t2}", "señal": "ERROR", "motivo": "Datos insuficientes"}

    # Calcular spread con Kalman sobre todo el historial disponible
    spread, beta, alpha = calcular_spread_kalman(precios, t1, t2)

    # Parámetros OU sobre la ventana reciente
    ventana_ou = min(252, len(spread))
    ou         = parametros_ou(spread.tail(ventana_ou))
    hl         = ou["half_life"]
    window     = max(5, int(np.ceil(hl)))

    zscore  = calcular_zscore(spread, window=window)
    señales = generar_señales(zscore, entrada_z, salida_z, stop_z)

    señal_hoy  = señales.iloc[-1]
    z_actual   = float(zscore.iloc[-1]) if not np.isnan(zscore.iloc[-1]) else 0.0
    beta_actual = float(beta.iloc[-1])

    # Tests estadísticos
    adf       = test_adf_spread(spread.tail(ventana_ou))
    regimen   = regimen_volatilidad(spread)
    coint_ok  = verificar_cointegración_activa(precios, t1, t2)

    # Mapear señal numérica a texto
    mapa_señal = {
        Señal.COMPRAR_SPREAD: "LONG_SPREAD",
        Señal.VENDER_SPREAD:  "SHORT_SPREAD",
        Señal.CERRAR:         "CERRAR",
        Señal.NINGUNA:        "HOLD",
    }
    señal_texto = mapa_señal.get(señal_hoy, "HOLD")

    # Invalidar señal si la cointegración está rota
    if not coint_ok["cointegrado"]:
        señal_texto = "SUSPENDIDO"

    return {
        "par":              f"{t1}/{t2}",
        "ticker1":          t1,
        "ticker2":          t2,
        "fecha":            str(precios.index[-1].date()),
        "señal":            señal_texto,
        "z_score":          round(z_actual, 4),
        "beta_kalman":      round(beta_actual, 4),
        "half_life_dias":   round(hl, 1),
        "window_zscore":    window,
        "regimen_vol":      regimen,
        "adf_p_value":      adf["p_value_adf"],
        "spread_estac":     adf["estacionario"],
        "coint_activa":     coint_ok["cointegrado"],
        "score_johansen":   coint_ok["score_johansen"],
        "alerta":           coint_ok["alerta"],
        "precio_t1":        round(float(precios[t1].iloc[-1]), 2),
        "precio_t2":        round(float(precios[t2].iloc[-1]), 2),
        "spread_actual":    round(float(spread.iloc[-1]), 6),
    }


# ── Pipeline diario ───────────────────────────────────────────────────────────

def ejecutar_pipeline_diario(
    pares_path: str | None = None,
    top_n: int = 20,
    días_historial: int = BUFFER_DIAS,
    guardar: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pipeline completo de señales diarias:
      1. Carga los pares validados (CSV)
      2. Descarga precios recientes (historial + buffer)
      3. Evalúa cada par y genera señales
      4. Exporta CSV y JSON de señales

    Diseñado para ejecutarse a las 16:05 EST (cierre del mercado US).
    """
    # 1. Cargar pares validados
    pares_df = cargar_pares(pares_path) if pares_path else cargar_pares()
    pares_df = pares_df.head(top_n)

    tickers  = list(set(pares_df["ticker1"].tolist() + pares_df["ticker2"].tolist()))

    # 2. Descargar precios recientes
    desde = pd.Timestamp.today() - pd.Timedelta(days=días_historial + 60)
    if verbose:
        print(f"\n[INFO] Descargando datos para {len(tickers)} tickers ({desde.date()} → hoy)...")

    precios = descargar_precios(tickers, inicio=str(desde.date()), fin=str(pd.Timestamp.today().date()))

    # 3. Evaluar cada par
    señales = []
    for _, fila in pares_df.iterrows():
        t1, t2 = fila["ticker1"], fila["ticker2"]
        try:
            resultado = evaluar_par(precios, t1, t2)
            señales.append(resultado)
            if verbose:
                icono = {"LONG_SPREAD": "▲", "SHORT_SPREAD": "▼", "CERRAR": "✕",
                          "HOLD": "—", "SUSPENDIDO": "⚠"}.get(resultado["señal"], "?")
                print(f"  {icono} {t1}/{t2:8} | Z={resultado['z_score']:+.2f} | "
                      f"{resultado['señal']:12} | Vol={resultado['regimen_vol']} | "
                      f"{'✓ cointegrado' if resultado['coint_activa'] else '✗ RUPTURA'}")
        except Exception as e:
            if verbose:
                print(f"  [ERR] {t1}/{t2}: {e}")

    df_señales = pd.DataFrame(señales)

    # 4. Guardar resultados
    if guardar and not df_señales.empty:
        df_señales.to_csv(SEÑALES_PATH, index=False)
        if verbose:
            print(f"\n[OK] Señales guardadas en {SEÑALES_PATH}")

    return df_señales


# ── Gestión de estado de posiciones abiertas ──────────────────────────────────

def cargar_estado() -> dict:
    """Carga el estado de posiciones abiertas desde JSON."""
    if os.path.exists(ESTADO_PATH):
        with open(ESTADO_PATH) as f:
            return json.load(f)
    return {}


def guardar_estado(estado: dict) -> None:
    """Persiste el estado de posiciones abiertas."""
    with open(ESTADO_PATH, "w") as f:
        json.dump(estado, f, indent=2, default=str)


def actualizar_estado(df_señales: pd.DataFrame) -> dict:
    """
    Actualiza el estado de posiciones abiertas basándose en las señales del día.
    Registra fecha de apertura, dirección y z-score de entrada.
    """
    estado = cargar_estado()
    hoy    = str(datetime.today().date())

    for _, fila in df_señales.iterrows():
        par    = fila["par"]
        señal  = fila["señal"]

        if señal in ("LONG_SPREAD", "SHORT_SPREAD"):
            estado[par] = {
                "direccion":     señal,
                "fecha_entrada": hoy,
                "z_entrada":     fila["z_score"],
                "beta_entrada":  fila["beta_kalman"],
            }
        elif señal in ("CERRAR", "SUSPENDIDO") and par in estado:
            del estado[par]

    guardar_estado(estado)
    return estado


# ── Resumen para revisión humana ──────────────────────────────────────────────

def imprimir_resumen_diario(df_señales: pd.DataFrame) -> None:
    """Imprime un resumen estructurado para revisión humana."""
    if df_señales.empty:
        print("[INFO] Sin señales activas hoy.")
        return

    activas    = df_señales[df_señales["señal"].isin(["LONG_SPREAD", "SHORT_SPREAD"])]
    cierres    = df_señales[df_señales["señal"] == "CERRAR"]
    suspendidas = df_señales[df_señales["señal"] == "SUSPENDIDO"]

    print(f"\n{'='*60}")
    print(f"  REPORTE DIARIO — {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    print(f"  Nuevas entradas  : {len(activas)}")
    print(f"  Cierres          : {len(cierres)}")
    print(f"  Suspendidas      : {len(suspendidas)}")
    print(f"  Total pares eval : {len(df_señales)}")

    if not activas.empty:
        print(f"\n  ENTRADAS NUEVAS:")
        for _, r in activas.iterrows():
            print(f"    {r['señal']:12} | {r['par']:12} | Z={r['z_score']:+.2f} | "
                  f"HL={r['half_life_dias']:.0f}d | Vol={r['regimen_vol']}")

    if not suspendidas.empty:
        print(f"\n  ALERTAS:")
        for _, r in suspendidas.iterrows():
            print(f"    ⚠ {r['par']} — {r['alerta']}")
    print(f"{'='*60}\n")
