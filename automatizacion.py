"""Manual pair scanning and latest-signal snapshot workflows."""

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from config import (
    HORAS_DIA,
    MIN_OBS_SEÑAL,
    OBJETIVO_MDD,
    OBJETIVO_SHARPE,
    UMBRAL_EG,
    VENTANA_COINT_ACTIVA,
)
from datos import descargar_ohlcv_horario, descargar_precios, filtrar_datos, obtener_sp500
from deteccion import (
    cargar_pares,
    diagnostico_madurez_simple,
    escanear_todos_los_pares,
    guardar_pares,
)
from spread import (
    Señal,
    calcular_spread_kalman,
    calcular_zscore,
    generar_señales,
    parametros_ou,
)


def _bpd(index: pd.DatetimeIndex) -> float:
    if len(index) < 10:
        return 1.0
    n_dias = pd.Series(index).dt.date.nunique()
    return HORAS_DIA if (len(index) / max(n_dias, 1)) > 2 else 1.0


warnings.filterwarnings("ignore")

SEÑALES_PATH = os.path.join(os.path.dirname(__file__), "señales_diarias.csv")
PARES_PATH = os.path.join(os.path.dirname(__file__), "pares_cointegrados.csv")

DIAS_ENTRE_SCANS = 7


def scan_necesario(path: str = PARES_PATH, dias: int = DIAS_ENTRE_SCANS) -> bool:
    if not os.path.exists(path):
        return True
    edad = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return edad > timedelta(days=dias)


def dias_desde_ultimo_scan(path: str = PARES_PATH) -> float | None:
    if not os.path.exists(path):
        return None
    return (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days


def ejecutar_scan_semanal(
    tickers: list[str] | None = None,
    forzar: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    if not forzar and not scan_necesario():
        dias = dias_desde_ultimo_scan()
        if verbose:
            print(
                f"[INFO] Scan no necesario. Último scan hace {dias} días (próximo en {DIAS_ENTRE_SCANS - dias} días)."
            )
        return cargar_pares()

    if verbose:
        print(f"\n[SCAN HISTÓRICO MANUAL] {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("  Descargando datos diarios históricos de Yahoo Finance...")

    tickers = tickers or obtener_sp500()

    ohlcv = descargar_ohlcv_horario(tickers, dias_atras=365 * 7)
    df_close = ohlcv.get("close", pd.DataFrame())

    if df_close.empty:
        print("[ERROR] Yahoo Finance no devolvió datos diarios.")
        return pd.DataFrame()

    tickers_validos = [t for t in tickers if t in df_close.columns]
    if tickers_validos:
        df_close = df_close[tickers_validos]

    pares = escanear_todos_los_pares(df_close, verbose=verbose)

    if not pares.empty:
        guardar_pares(pares)

    return pares


def filtrar_por_backtest_smart(
    pares_df: pd.DataFrame,
    precios_diarios: pd.DataFrame | None = None,
    obj_sharpe: float | None = None,
    obj_mdd_pct: float | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from backtesting import MotorBacktest, ParametrosBacktest

    min_sharpe = obj_sharpe if obj_sharpe is not None else OBJETIVO_SHARPE
    max_mdd = obj_mdd_pct if obj_mdd_pct is not None else abs(OBJETIVO_MDD) * 100

    if precios_diarios is None or precios_diarios.empty:
        tickers = list(set(pares_df["ticker1"].tolist() + pares_df["ticker2"].tolist()))
        if verbose:
            print(f"  Descargando datos diarios para {len(tickers)} tickers...")
        precios_diarios = filtrar_datos(descargar_precios(tickers))

    if verbose:
        print(
            f"\n[FILTRO SMART] Backtest rápido sobre {len(pares_df)} pares "
            f"(Sharpe -- {min_sharpe:.1f}, MDD -- {max_mdd:.0f}%)..."
        )

    resultados = []
    for _, row in pares_df.iterrows():
        t1, t2 = row["ticker1"], row["ticker2"]
        nombre = f"{t1}/{t2}"

        if t1 not in precios_diarios.columns or t2 not in precios_diarios.columns:
            if verbose:
                print(f"  [SKIP] {nombre}: sin datos diarios")
            continue

        try:
            motor = MotorBacktest(precios_diarios[[t1, t2]], t1, t2, ParametrosBacktest())
            res = motor.ejecutar()
            m = res["metricas"]
            pasa = m["sharpe"] >= min_sharpe and abs(m["mdd"]) <= max_mdd

            resultados.append(
                {
                    **row.to_dict(),
                    "sharpe": m["sharpe"],
                    "sortino": m["sortino"],
                    "mdd": m["mdd"],
                    "cagr": m["cagr"],
                    "n_trades": m["n_trades"],
                    "win_rate": m["win_rate"],
                    "profit_factor": m["profit_factor"],
                    "pasa_smart": pasa,
                }
            )

            if verbose:
                marca = "OK" if pasa else "FAIL"
                tag = "PASA   " if pasa else "NO PASA"
                print(
                    f"  {marca} {tag} | {nombre:<14} | "
                    f"Sharpe={m['sharpe']:+.2f} | MDD={m['mdd']:.1f}% | "
                    f"CAGR={m['cagr']:.1f}%"
                )

        except Exception as e:
            if verbose:
                print(f"  [ERR] {nombre}: {e}")

    if not resultados:
        return pd.DataFrame(), pd.DataFrame()

    todos_df = pd.DataFrame(resultados)
    pasaron_df = todos_df[todos_df["pasa_smart"]].drop(columns=["pasa_smart"]).reset_index(drop=True)
    return pasaron_df, todos_df


def test_adf_spread(spread: pd.Series) -> dict:
    try:
        resultado = adfuller(spread.dropna(), autolag="AIC")
        return {
            "adf_stat": round(float(resultado[0]), 4),
            "p_value_adf": round(float(resultado[1]), 4),
            "estacionario": resultado[1] < 0.05,
        }
    except Exception:
        return {"adf_stat": 0.0, "p_value_adf": 1.0, "estacionario": False}


def regimen_volatilidad(spread: pd.Series, ventana_vol: int = 20) -> str:
    vol_rolling = spread.rolling(ventana_vol).std()
    vol_actual = vol_rolling.iloc[-1]
    p25, p75 = vol_rolling.quantile(0.25), vol_rolling.quantile(0.75)
    if vol_actual <= p25:
        return "BAJA"
    elif vol_actual >= p75:
        return "ALTA"
    return "NORMAL"


def verificar_cointegración_activa(
    df_close: pd.DataFrame,
    t1: str,
    t2: str,
    ventana: int = VENTANA_COINT_ACTIVA,
) -> dict:
    if t1 not in df_close.columns or t2 not in df_close.columns:
        return {"cointegrado": False, "alerta": "Tickers no disponibles en los datos"}

    s1 = df_close[t1].dropna()
    s2 = df_close[t2].dropna()
    idx = s1.index.intersection(s2.index)

    ventana_barras = int(ventana * _bpd(df_close.index))
    min_barras = max(30, int(ventana_barras * 0.90))
    if len(idx) < min_barras:
        return {"cointegrado": False, "alerta": "Datos insuficientes para verificar"}

    ventana_efectiva = min(ventana_barras, len(idx))
    s1_rec = np.log(s1.loc[idx].iloc[-ventana_efectiva:])
    s2_rec = np.log(s2.loc[idx].iloc[-ventana_efectiva:])

    try:
        _, pvalue, _ = coint(s1_rec, s2_rec)
        cointegrado = pvalue < UMBRAL_EG
        return {
            "cointegrado": cointegrado,
            "p_value_eg": round(pvalue, 4),
            "alerta": "" if cointegrado else "RUPTURA DE COINTEGRACIÓN - NO OPERAR",
        }
    except Exception:
        return {"cointegrado": False, "alerta": "Error en el test EG"}


def evaluar_par(
    df_close: pd.DataFrame,
    t1: str,
    t2: str,
    entrada_z: float = 2.0,
    salida_z: float = 0.5,
    stop_z: float = 3.5,
) -> dict:
    if t1 not in df_close.columns or t2 not in df_close.columns:
        return {"par": f"{t1}/{t2}", "señal": "ERROR", "motivo": "Datos insuficientes"}

    coint_ok = verificar_cointegración_activa(df_close, t1, t2)
    if not coint_ok["cointegrado"]:
        return {
            "par": f"{t1}/{t2}",
            "ticker1": t1,
            "ticker2": t2,
            "fecha": str(df_close.index[-1]),
            "señal": "SUSPENDIDO",
            "coint_activa": False,
            "alerta": coint_ok["alerta"],
        }

    spread, beta = calcular_spread_kalman(df_close, t1, t2)

    min_obs = max(30, int(MIN_OBS_SEÑAL * _bpd(df_close.index)))
    ventana_ou = min(min_obs, len(spread))
    ou = parametros_ou(spread.tail(ventana_ou))
    hl = ou["half_life"]
    window = max(5, int(np.ceil(hl)))

    zscore = calcular_zscore(spread, window=window)
    señales = generar_señales(zscore, entrada_z, salida_z, stop_z)

    señal_hoy = señales.iloc[-1]
    z_actual = float(zscore.iloc[-1]) if not np.isnan(zscore.iloc[-1]) else 0.0
    beta_actual = float(beta.iloc[-1])

    adf = test_adf_spread(spread.tail(ventana_ou))
    regimen = regimen_volatilidad(spread)
    madurez = diagnostico_madurez_simple(df_close[t1], df_close[t2])

    mapa_señal = {
        Señal.COMPRAR_SPREAD: "LONG_SPREAD",
        Señal.VENDER_SPREAD: "SHORT_SPREAD",
        Señal.CERRAR: "CERRAR",
        Señal.NINGUNA: "HOLD",
    }

    return {
        "par": f"{t1}/{t2}",
        "ticker1": t1,
        "ticker2": t2,
        "fecha": str(df_close.index[-1]),
        "señal": mapa_señal.get(señal_hoy, "HOLD"),
        "z_score": round(z_actual, 4),
        "beta_kalman": round(beta_actual, 4),
        "half_life_bars": round(hl, 1),
        "window_zscore": window,
        "regimen_vol": regimen,
        "adf_p_value": adf["p_value_adf"],
        "spread_estac": adf["estacionario"],
        "coint_activa": True,
        "p_value_eg": coint_ok["p_value_eg"],
        "alerta": "",
        "precio_t1": round(float(df_close[t1].iloc[-1]), 2),
        "precio_t2": round(float(df_close[t2].iloc[-1]), 2),
        "spread_actual": round(float(spread.iloc[-1]), 6),
        "madurez_estado": madurez["estado"],
        "madurez_descripcion": madurez["descripcion"],
        "madurez_tendencia": madurez["tendencia"],
        "ejecucion_mas_temprana": "siguiente barra diaria",
    }


def ejecutar_pipeline_diario(
    top_n: int = 20,
    pares_lista: list[dict] | None = None,
    guardar: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    if verbose:
        print(f"\n[PIPELINE DIARIO] {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if pares_lista is not None:
        pares_df = pd.DataFrame(pares_lista)
    else:
        dias = dias_desde_ultimo_scan()
        if dias is None:
            print("[WARN] No hay pares detectados. Ejecuta primero el scan semanal.")
            return pd.DataFrame()
        if dias > DIAS_ENTRE_SCANS and verbose:
            print(f"[WARN] El scan tiene {dias} días de antigüedad. Considera ejecutar ejecutar_scan_semanal().")
        pares_df = cargar_pares().head(top_n)

    tickers = list(set(pares_df["ticker1"].tolist() + pares_df["ticker2"].tolist()))

    if verbose:
        print(f"  Descargando datos diarios para {len(tickers)} tickers...")

    ohlcv = descargar_ohlcv_horario(tickers, dias_atras=365)
    df_close = ohlcv.get("close", pd.DataFrame())

    if df_close.empty:
        print("[ERROR] Sin datos diarios.")
        return pd.DataFrame()

    if verbose:
        print(f"\n  Generando señales para {len(pares_df)} pares...")

    señales = []
    for _, fila in pares_df.iterrows():
        t1, t2 = fila["ticker1"], fila["ticker2"]
        try:
            resultado = evaluar_par(df_close, t1, t2)
            señales.append(resultado)
            if verbose:
                icono = {
                    "LONG_SPREAD": "OK",
                    "SHORT_SPREAD": "FAIL",
                    "CERRAR": "FAIL",
                    "HOLD": "-",
                    "SUSPENDIDO": "WARNING",
                }.get(resultado["señal"], "?")
                madurez_tag = f"[{resultado.get('madurez_estado', '?')} {resultado.get('madurez_tendencia', '')}]"
                print(
                    f"  {icono} {t1}/{t2:8} | Z={resultado.get('z_score', 0):+.2f} | "
                    f"{resultado['señal']:12} | "
                    f"{'OK' if resultado.get('coint_activa') else 'FAIL RUPTURA'} | {madurez_tag}"
                )
        except Exception as e:
            if verbose:
                print(f"  [ERR] {t1}/{t2}: {e}")

    df_señales = pd.DataFrame(señales)

    if guardar and not df_señales.empty:
        df_señales.to_csv(SEÑALES_PATH, index=False)
        if verbose:
            print(f"\n[OK] Señales guardadas en {SEÑALES_PATH}")

    return df_señales


def imprimir_resumen_diario(df_señales: pd.DataFrame) -> None:
    if df_señales.empty:
        print("[INFO] Sin señales activas hoy.")
        return

    activas = df_señales[df_señales["señal"].isin(["LONG_SPREAD", "SHORT_SPREAD"])]
    cierres = df_señales[df_señales["señal"] == "CERRAR"]
    suspendidas = df_señales[df_señales["señal"] == "SUSPENDIDO"]

    print(f"\n{'=' * 60}")
    print(f"  REPORTE DIARIO - {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 60}")
    print(f"  Nuevas entradas  : {len(activas)}")
    print(f"  Cierres          : {len(cierres)}")
    print(f"  Suspendidas      : {len(suspendidas)}")
    print(f"  Total pares eval : {len(df_señales)}")

    if not activas.empty:
        print("\n  ENTRADAS NUEVAS:")
        for _, r in activas.iterrows():
            print(
                f"    {r['señal']:12} | {r['par']:12} | Z={r['z_score']:+.2f} | "
                f"HL={r['half_life_bars']:.0f}b | Vol={r['regimen_vol']}"
            )
            if "madurez_estado" in r and r["madurez_estado"] != "DESCONOCIDO":
                print(f"    {'':12}   Cointegración: {r['madurez_estado']} {r.get('madurez_tendencia', '')}")
                print(f"    {'':12}   {r['madurez_descripcion']}")

    if not suspendidas.empty:
        print("\n  ALERTAS:")
        for _, r in suspendidas.iterrows():
            print(f"    WARNING {r['par']} - {r['alerta']}")
    print(f"{'=' * 60}\n")
