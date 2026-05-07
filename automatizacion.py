"""
automatizacion.py — Pipelines de ejecución automática.

Dos pipelines con frecuencias distintas:

  Pipeline diario (cada día, apertura de mercado):
    1. Verifica que los pares activos siguen cointegrados (EG sobre datos horarios)
    2. Genera señales del día para los pares que siguen válidos

  Pipeline semanal (cada 7 días, fin de semana):
    1. Descarga datos horarios de los últimos 12 meses para todo el S&P 500
    2. Ejecuta el scan completo EG + Johansen
    3. Actualiza pares_cointegrados.csv
"""

import os
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from datos import descargar_ohlcv_horario, obtener_sp500
from spread import (
    calcular_spread_kalman,
    calcular_zscore,
    calcular_half_life,
    parametros_ou,
    generar_señales,
    Señal,
)
from deteccion import (
    escanear_todos_los_pares, guardar_pares, cargar_pares, test_johansen,
    diagnostico_madurez_simple,
)
from config import (
    MIN_OBS_HORARIO, UMBRAL_EG, VENTANA_COINT_ACTIVA, HORAS_DIA,
)


def _bpd(index: pd.DatetimeIndex) -> float:
    """Barras por día de negociación: HORAS_DIA para horario, 1.0 para diario."""
    if len(index) < 10:
        return 1.0
    n_dias = pd.Series(index).dt.date.nunique()
    return HORAS_DIA if (len(index) / max(n_dias, 1)) > 2 else 1.0

warnings.filterwarnings("ignore")

SEÑALES_PATH = os.path.join(os.path.dirname(__file__), "señales_diarias.csv")
ESTADO_PATH  = os.path.join(os.path.dirname(__file__), "estado_posiciones.json")
PARES_PATH   = os.path.join(os.path.dirname(__file__), "pares_cointegrados.csv")

DIAS_ENTRE_SCANS = 7   # días entre scans completos de nuevos pares


# ── Scheduling: ¿cuándo ejecutar cada pipeline? ───────────────────────────────

def scan_necesario(path: str = PARES_PATH, dias: int = DIAS_ENTRE_SCANS) -> bool:
    """
    Devuelve True si el CSV de pares no existe o tiene más de `dias` días de antigüedad.
    Determina si hay que ejecutar el scan semanal de nuevos pares.
    """
    if not os.path.exists(path):
        return True
    edad = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return edad > timedelta(days=dias)


def dias_desde_ultimo_scan(path: str = PARES_PATH) -> float | None:
    """Devuelve los días desde el último scan, o None si no existe el CSV."""
    if not os.path.exists(path):
        return None
    return (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days


# ── Pipeline semanal — scan de nuevos pares ───────────────────────────────────

def ejecutar_scan_semanal(
    tickers: list[str] | None = None,
    forzar: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan completo para encontrar pares cointegrados. Se ejecuta semanalmente.

    Descarga datos horarios de los últimos 12 meses y aplica EG + Johansen
    sobre todos los pares posibles. Guarda el resultado en pares_cointegrados.csv.

    Args:
        tickers: Lista de tickers. Si None, usa el S&P 500 completo.
        forzar:  Si True, ejecuta aunque el CSV sea reciente.
        verbose: Imprime progreso.
    """
    if not forzar and not scan_necesario():
        dias = dias_desde_ultimo_scan()
        if verbose:
            print(f"[INFO] Scan no necesario. Último scan hace {dias} días "
                  f"(próximo en {DIAS_ENTRE_SCANS - dias} días).")
        return cargar_pares()

    if verbose:
        print(f"\n[SCAN SEMANAL] {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Descargando datos horarios (últimos 12 meses)...")

    tickers = tickers or obtener_sp500()

    # Datos horarios: 365 días hacia atrás cubre los 12 meses (≈1638 barras)
    ohlcv = descargar_ohlcv_horario(tickers, dias_atras=365)
    df_close = ohlcv.get("close", pd.DataFrame())

    if df_close.empty:
        print("[ERROR] Sin datos horarios. Verifica las credenciales de Alpaca.")
        return pd.DataFrame()

    # La caché puede contener más tickers que los solicitados — filtrar al universo pedido
    tickers_validos = [t for t in tickers if t in df_close.columns]
    if tickers_validos:
        df_close = df_close[tickers_validos]

    pares = escanear_todos_los_pares(df_close, verbose=verbose)

    if not pares.empty:
        guardar_pares(pares)

    return pares


# ── Tests estadísticos en tiempo real ────────────────────────────────────────

def test_adf_spread(spread: pd.Series) -> dict:
    """
    Test ADF sobre el spread reciente.
    p-value < 0.05 → estacionario → spread válido para operar.
    """
    try:
        resultado = adfuller(spread.dropna(), autolag="AIC")
        return {
            "adf_stat":     round(float(resultado[0]), 4),
            "p_value_adf":  round(float(resultado[1]), 4),
            "estacionario": resultado[1] < 0.05,
        }
    except Exception:
        return {"adf_stat": 0.0, "p_value_adf": 1.0, "estacionario": False}


def regimen_volatilidad(spread: pd.Series, ventana_vol: int = 20) -> str:
    """Clasifica el régimen de volatilidad del spread: BAJA / NORMAL / ALTA."""
    vol_rolling = spread.rolling(ventana_vol).std()
    vol_actual  = vol_rolling.iloc[-1]
    p25, p75    = vol_rolling.quantile(0.25), vol_rolling.quantile(0.75)
    if vol_actual <= p25:
        return "BAJA"
    elif vol_actual >= p75:
        return "ALTA"
    return "NORMAL"


# ── Verificación diaria de cointegración activa ───────────────────────────────

def verificar_cointegración_activa(
    df_close: pd.DataFrame,
    t1: str,
    t2: str,
    ventana: int = VENTANA_COINT_ACTIVA,
) -> dict:
    """
    Verifica que el par sigue cointegrado en las últimas `ventana` barras.
    Usa EG (rápido) sobre los datos más recientes disponibles.

    Una ruptura invalida la señal: el spread ya no revierte a la media.
    """
    if t1 not in df_close.columns or t2 not in df_close.columns:
        return {"cointegrado": False, "alerta": "Tickers no disponibles en los datos"}

    s1 = df_close[t1].dropna()
    s2 = df_close[t2].dropna()
    idx = s1.index.intersection(s2.index)

    ventana_barras = int(ventana * _bpd(df_close.index))
    # Tolerancia del 10% para cubrir festivos y fuentes con menos barras que el horario de Alpaca
    min_barras = max(30, int(ventana_barras * 0.90))
    if len(idx) < min_barras:
        return {"cointegrado": False, "alerta": "Datos insuficientes para verificar"}

    ventana_efectiva = min(ventana_barras, len(idx))
    s1_rec = np.log(s1.loc[idx].iloc[-ventana_efectiva:])
    s2_rec = np.log(s2.loc[idx].iloc[-ventana_efectiva:])

    try:
        _, pvalue, _ = coint(s1_rec, s2_rec)
        cointegrado  = pvalue < UMBRAL_EG
        return {
            "cointegrado":   cointegrado,
            "p_value_eg":    round(pvalue, 4),
            "alerta":        "" if cointegrado else "RUPTURA DE COINTEGRACIÓN — NO OPERAR",
        }
    except Exception:
        return {"cointegrado": False, "alerta": "Error en el test EG"}


def verificar_pares_activos(
    df_close: pd.DataFrame,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Comprueba la cointegración de todos los pares con posición abierta.
    Se ejecuta cada día antes de generar señales.

    Lee el estado de posiciones desde estado_posiciones.json y para cada par
    activo verifica si sigue cointegrado con los datos horarios recientes.

    Returns:
        dict {par: {"cointegrado": bool, "alerta": str, ...}}
    """
    estado  = cargar_estado()
    if not estado:
        if verbose:
            print("[INFO] Sin posiciones abiertas que verificar.")
        return {}

    if verbose:
        print(f"\n[VERIFICACIÓN DIARIA] {len(estado)} pares activos...")

    resultados = {}
    for par, info in estado.items():
        t1, t2 = par.split("/")
        check  = verificar_cointegración_activa(df_close, t1, t2)
        resultados[par] = check

        if verbose:
            estado_str = "✓ cointegrado" if check["cointegrado"] else f"✗ {check['alerta']}"
            print(f"  {par:16} | {estado_str}")

    rupturas = [p for p, r in resultados.items() if not r["cointegrado"]]
    if rupturas and verbose:
        print(f"\n  [ALERTA] {len(rupturas)} par(es) con cointegración rota: {rupturas}")
        print(f"  Posiciones correspondientes deben cerrarse.")

    return resultados


# ── Evaluación de un par — señal del día ──────────────────────────────────────

def evaluar_par(
    df_close: pd.DataFrame,
    t1: str,
    t2: str,
    entrada_z: float = 2.0,
    salida_z: float = 0.5,
    stop_z: float = 3.5,
) -> dict:
    """
    Evalúa el estado actual de un par cointegrado y genera la señal del día.
    Primero verifica cointegración activa; si está rota devuelve SUSPENDIDO.
    """
    if t1 not in df_close.columns or t2 not in df_close.columns:
        return {"par": f"{t1}/{t2}", "señal": "ERROR", "motivo": "Datos insuficientes"}

    # Verificar cointegración antes de generar señal
    coint_ok = verificar_cointegración_activa(df_close, t1, t2)
    if not coint_ok["cointegrado"]:
        return {
            "par":          f"{t1}/{t2}",
            "ticker1":      t1,
            "ticker2":      t2,
            "fecha":        str(df_close.index[-1]),
            "señal":        "SUSPENDIDO",
            "coint_activa": False,
            "alerta":       coint_ok["alerta"],
        }

    spread, beta, alpha = calcular_spread_kalman(df_close, t1, t2)

    min_obs    = max(30, int(MIN_OBS_HORARIO / HORAS_DIA * _bpd(df_close.index)))
    ventana_ou = min(min_obs, len(spread))
    ou         = parametros_ou(spread.tail(ventana_ou))
    hl         = ou["half_life"]
    window     = max(5, int(np.ceil(hl)))

    zscore  = calcular_zscore(spread, window=window)
    señales = generar_señales(zscore, entrada_z, salida_z, stop_z)

    señal_hoy   = señales.iloc[-1]
    z_actual    = float(zscore.iloc[-1]) if not np.isnan(zscore.iloc[-1]) else 0.0
    beta_actual = float(beta.iloc[-1])

    adf     = test_adf_spread(spread.tail(ventana_ou))
    regimen = regimen_volatilidad(spread)
    madurez = diagnostico_madurez_simple(df_close[t1], df_close[t2])

    mapa_señal = {
        Señal.COMPRAR_SPREAD: "LONG_SPREAD",
        Señal.VENDER_SPREAD:  "SHORT_SPREAD",
        Señal.CERRAR:         "CERRAR",
        Señal.NINGUNA:        "HOLD",
    }

    return {
        "par":                f"{t1}/{t2}",
        "ticker1":            t1,
        "ticker2":            t2,
        "fecha":              str(df_close.index[-1]),
        "señal":              mapa_señal.get(señal_hoy, "HOLD"),
        "z_score":            round(z_actual, 4),
        "beta_kalman":        round(beta_actual, 4),
        "half_life_bars":     round(hl, 1),
        "window_zscore":      window,
        "regimen_vol":        regimen,
        "adf_p_value":        adf["p_value_adf"],
        "spread_estac":       adf["estacionario"],
        "coint_activa":       True,
        "p_value_eg":         coint_ok["p_value_eg"],
        "alerta":             "",
        "precio_t1":          round(float(df_close[t1].iloc[-1]), 2),
        "precio_t2":          round(float(df_close[t2].iloc[-1]), 2),
        "spread_actual":      round(float(spread.iloc[-1]), 6),
        "madurez_estado":     madurez["estado"],
        "madurez_descripcion": madurez["descripcion"],
        "madurez_tendencia":  madurez["tendencia"],
    }


# ── Pipeline diario — señales para pares activos ──────────────────────────────

def ejecutar_pipeline_diario(
    top_n: int = 20,
    pares_lista: list[dict] | None = None,
    guardar: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pipeline diario completo:
      1. Carga los pares del último scan semanal (o usa pares_lista si se pasa)
      2. Descarga datos horarios recientes (365 días)
      3. Verifica cointegración de pares con posición abierta
      4. Genera señales para todos los pares válidos
      5. Exporta señales a CSV

    Diseñado para ejecutarse en la apertura del mercado (9:30 EST).
    """
    if verbose:
        print(f"\n[PIPELINE DIARIO] {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if pares_lista is not None:
        pares_df = pd.DataFrame(pares_lista)
    else:
        # Avisar si el scan está desactualizado (pero no bloqueamos)
        dias = dias_desde_ultimo_scan()
        if dias is None:
            print("[WARN] No hay pares detectados. Ejecuta primero el scan semanal.")
            return pd.DataFrame()
        if dias > DIAS_ENTRE_SCANS and verbose:
            print(f"[WARN] El scan tiene {dias} días de antigüedad. "
                  f"Considera ejecutar ejecutar_scan_semanal().")
        pares_df = cargar_pares().head(top_n)

    tickers  = list(set(pares_df["ticker1"].tolist() + pares_df["ticker2"].tolist()))

    if verbose:
        print(f"  Descargando datos horarios para {len(tickers)} tickers...")

    ohlcv    = descargar_ohlcv_horario(tickers, dias_atras=365)
    df_close = ohlcv.get("close", pd.DataFrame())

    if df_close.empty:
        print("[ERROR] Sin datos horarios.")
        return pd.DataFrame()

    # Paso 1: verificar cointegración de posiciones abiertas
    verificar_pares_activos(df_close, verbose=verbose)

    # Paso 2: generar señales
    if verbose:
        print(f"\n  Generando señales para {len(pares_df)} pares...")

    señales = []
    for _, fila in pares_df.iterrows():
        t1, t2 = fila["ticker1"], fila["ticker2"]
        try:
            resultado = evaluar_par(df_close, t1, t2)
            señales.append(resultado)
            if verbose:
                icono = {"LONG_SPREAD": "▲", "SHORT_SPREAD": "▼", "CERRAR": "✕",
                         "HOLD": "—", "SUSPENDIDO": "⚠"}.get(resultado["señal"], "?")
                madurez_tag = f"[{resultado.get('madurez_estado', '?')} {resultado.get('madurez_tendencia', '')}]"
                print(f"  {icono} {t1}/{t2:8} | Z={resultado.get('z_score', 0):+.2f} | "
                      f"{resultado['señal']:12} | "
                      f"{'✓' if resultado.get('coint_activa') else '✗ RUPTURA'} | {madurez_tag}")
        except Exception as e:
            if verbose:
                print(f"  [ERR] {t1}/{t2}: {e}")

    df_señales = pd.DataFrame(señales)

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
        par   = fila["par"]
        señal = fila["señal"]

        if señal in ("LONG_SPREAD", "SHORT_SPREAD"):
            estado[par] = {
                "direccion":     señal,
                "fecha_entrada": hoy,
                "z_entrada":     fila.get("z_score", 0),
                "beta_entrada":  fila.get("beta_kalman", 0),
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

    activas     = df_señales[df_señales["señal"].isin(["LONG_SPREAD", "SHORT_SPREAD"])]
    cierres     = df_señales[df_señales["señal"] == "CERRAR"]
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
                  f"HL={r['half_life_bars']:.0f}b | Vol={r['regimen_vol']}")
            if "madurez_estado" in r and r["madurez_estado"] != "DESCONOCIDO":
                print(f"    {'':12}   Cointegración: {r['madurez_estado']} {r.get('madurez_tendencia', '')}")
                print(f"    {'':12}   {r['madurez_descripcion']}")

    if not suspendidas.empty:
        print(f"\n  ALERTAS:")
        for _, r in suspendidas.iterrows():
            print(f"    ⚠ {r['par']} — {r['alerta']}")
    print(f"{'='*60}\n")
