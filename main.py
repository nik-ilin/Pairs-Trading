"""
main.py — Orquestador principal del sistema de pairs trading.

Modos de ejecución (--modo):
  scan     : detecta pares cointegrados con datos horarios de Alpaca (últimos 12 meses)
  diario   : verifica pares activos y genera señales del día
  backtest : backtesting out-of-sample de un par con datos Alpaca diarios
  evaluar  : genera gráficos de análisis para un par específico
  full     : scan + backtest de los mejores pares

Uso:
  python main.py --modo scan
  python main.py --modo diario
  python main.py --modo backtest --par AAPL MSFT
  python main.py --modo backtest --par AAPL MSFT --optimizar --walk-forward --graficos
  python main.py --modo evaluar --par KO PEP
  python main.py --modo full
"""

import argparse
import warnings

import pandas as pd

from datos import (
    obtener_sp500,
    filtrar_universo_interactivo,
    descargar_precios_alpaca,
    descargar_ohlcv_horario,
    filtrar_datos,
)
from deteccion import (
    escanear_todos_los_pares,
    guardar_pares,
    cargar_pares,
    top_pares,
    estabilidad_rolling,
)
from backtesting import (
    backtest_completo,
    ParametrosBacktest,
    optimizar_parametros,
    walk_forward,
)
from automatizacion import (
    ejecutar_pipeline_diario,
    ejecutar_scan_semanal,
    actualizar_estado,
    imprimir_resumen_diario,
)
from metricas import reporte_completo

warnings.filterwarnings("ignore")


# ── Helpers interactivos ──────────────────────────────────────────────────────

def _seleccionar_pares_interactivo(pares_df: pd.DataFrame) -> pd.DataFrame:
    """Muestra los pares encontrados y permite elegir cuáles guardar."""
    if pares_df.empty:
        return pares_df

    pares_df = pares_df.reset_index(drop=True)

    print(f"\n{'='*72}")
    print(f"  PARES COINTEGRADOS ENCONTRADOS: {len(pares_df)}")
    print(f"{'='*72}")
    print(f"  {'#':>4}  {'Par':<14}  {'Score':>7}  {'Traza':>7}  {'p-EG':>7}  {'Obs':>6}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}")

    for i, row in pares_df.iterrows():
        par = f"{row['ticker1']}/{row['ticker2']}"
        print(f"  {i+1:>4}  {par:<14}  {row['score']:>7.4f}  {row['traza']:>7.2f}"
              f"  {row['p_value_eg']:>7.4f}  {row['n_obs']:>6}")

    print()
    print("  Score  = fuerza de cointegración Johansen (>1 válido, mayor = mejor)")
    print("  p-EG   = p-value Engle-Granger (<0.05 = cointegrado)")
    print()
    print("  ¿Qué pares deseas guardar?")
    print("    Números por coma  →  ej: 1,3,5")
    print("    Rango con guión   →  ej: 1-10")
    print("    ENTER             →  guardar todos")
    print()

    entrada = input("  Tu selección: ").strip()
    if not entrada:
        print(f"  [OK] Guardando todos los {len(pares_df)} pares.")
        return pares_df

    indices: set[int] = set()
    try:
        for parte in entrada.split(","):
            parte = parte.strip()
            if "-" in parte:
                a, b = parte.split("-", 1)
                indices.update(range(int(a) - 1, int(b)))
            else:
                indices.add(int(parte) - 1)
        validos = [i for i in sorted(indices) if 0 <= i < len(pares_df)]
        resultado = pares_df.iloc[validos].reset_index(drop=True)
        print(f"  [OK] {len(resultado)} pares seleccionados.")
        return resultado
    except (ValueError, IndexError):
        print("  [WARN] Selección inválida. Guardando todos.")
        return pares_df


def _seleccionar_de_csv(top_n: int) -> list[dict]:
    """Carga el CSV de pares y permite elegir cuáles backtestar."""
    try:
        pares_df = top_pares(n=top_n)
    except FileNotFoundError:
        print("[!] Ejecuta primero --modo scan para generar pares_cointegrados.csv")
        return []

    pares_df = pares_df.reset_index(drop=True)

    print(f"\n{'='*72}")
    print(f"  PARES DISPONIBLES (top {top_n})")
    print(f"{'='*72}")
    print(f"  {'#':>4}  {'Par':<14}  {'Score':>7}  {'p-EG':>7}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*7}  {'─'*7}")

    for i, row in pares_df.iterrows():
        par = f"{row['ticker1']}/{row['ticker2']}"
        print(f"  {i+1:>4}  {par:<14}  {row['score']:>7.4f}  {row['p_value_eg']:>7.4f}")

    print()
    print("  ¿Qué pares deseas backtestar?")
    print("    Números por coma  →  ej: 1,3")
    print("    ENTER             →  backtestar todos")
    print()

    entrada = input("  Tu selección: ").strip()
    if not entrada:
        return pares_df.to_dict("records")

    indices: set[int] = set()
    try:
        for parte in entrada.split(","):
            parte = parte.strip()
            if "-" in parte:
                a, b = parte.split("-", 1)
                indices.update(range(int(a) - 1, int(b)))
            else:
                indices.add(int(parte) - 1)
        validos = [i for i in sorted(indices) if 0 <= i < len(pares_df)]
        return pares_df.iloc[validos].to_dict("records")
    except (ValueError, IndexError):
        print("  [WARN] Selección inválida. Usando todos.")
        return pares_df.to_dict("records")


# ── Modos de ejecución ────────────────────────────────────────────────────────

def modo_scan(args) -> None:
    """
    Scan semanal: descarga datos horarios y busca pares cointegrados ahora.
    Siempre fuerza un scan nuevo independientemente de la antigüedad del CSV.
    """
    print("\n[MODO: SCAN]")

    if getattr(args, "interactivo", True):
        tickers = filtrar_universo_interactivo()
    else:
        tickers = obtener_sp500()

    pares = ejecutar_scan_semanal(tickers=tickers, forzar=True, verbose=True)

    if pares.empty:
        print("[!] No se encontraron pares cointegrados.")
        return

    if getattr(args, "interactivo", True):
        pares = _seleccionar_pares_interactivo(pares)
        guardar_pares(pares)


def modo_diario(args) -> None:
    """
    Pipeline diario: verifica cointegración de pares activos y genera señales.
    Ejecutar cada día en la apertura del mercado.
    """
    print("\n[MODO: DIARIO]")
    df_señales = ejecutar_pipeline_diario(top_n=args.top_n, verbose=True)
    imprimir_resumen_diario(df_señales)
    if not df_señales.empty:
        estado = actualizar_estado(df_señales)
        print(f"[INFO] Posiciones abiertas: {len(estado)}")


def modo_backtest(args) -> None:
    """Backtesting out-of-sample con datos diarios de Alpaca (2020-presente)."""
    print("\n[MODO: BACKTEST]")

    # Determinar qué pares backtestar
    if args.par:
        pares_lista = [{"ticker1": args.par[0], "ticker2": args.par[1]}]
    elif getattr(args, "interactivo", True):
        pares_lista = _seleccionar_de_csv(args.top_n)
        if not pares_lista:
            return
    else:
        try:
            pares_lista = top_pares(n=args.top_n).to_dict("records")
        except FileNotFoundError:
            print("[!] Ejecuta primero --modo scan.")
            return

    # Descargar datos diarios Alpaca para los tickers necesarios
    tickers = list({t for p in pares_lista for t in (p["ticker1"], p["ticker2"])})
    print(f"\n  Descargando datos diarios para {len(tickers)} tickers...")
    precios = descargar_precios_alpaca(tickers)
    precios = filtrar_datos(precios)

    if precios.empty:
        print("[!] Sin datos. Verifica las credenciales de Alpaca.")
        return

    for fila in pares_lista:
        t1, t2 = fila["ticker1"], fila["ticker2"]
        nombre = f"{t1}/{t2}"

        if t1 not in precios.columns or t2 not in precios.columns:
            print(f"  [WARN] {nombre}: datos insuficientes. Omitiendo.")
            continue

        print(f"\n{'─'*50}")
        print(f"  {nombre}")
        print(f"{'─'*50}")

        params = ParametrosBacktest()
        if args.optimizar:
            try:
                params, _ = optimizar_parametros(precios[[t1, t2]], t1, t2)
            except Exception as e:
                print(f"  [WARN] Grid search falló: {e}. Usando parámetros por defecto.")

        resultado = backtest_completo(precios[[t1, t2]], t1, t2, params,
                                      imprimir_reporte=True)

        if args.walk_forward:
            print(f"\n  Walk-Forward: {nombre}")
            walk_forward(precios[[t1, t2]], t1, t2, params, verbose=True)

        if args.graficos:
            try:
                from evaluacion import generar_informe_completo, grafico_rolling_cointegracion
                ohlcv   = descargar_ohlcv_horario(tickers, dias_atras=365)
                rolling = estabilidad_rolling(ohlcv["close"], t1, t2)
                grafico_rolling_cointegracion(rolling, nombre_par=nombre)
                generar_informe_completo(resultado, nombre_par=nombre)
            except ImportError:
                print("  [WARN] evaluacion.py no disponible. Omitiendo gráficos.")


def modo_evaluar(args) -> None:
    """Genera gráficos de análisis completos para un par específico."""
    print("\n[MODO: EVALUAR]")

    if not args.par:
        print("[!] Indica un par con --par TICKER1 TICKER2")
        return

    t1, t2 = args.par[0], args.par[1]
    nombre = f"{t1}/{t2}"

    print(f"  Descargando datos para {nombre}...")
    precios = descargar_precios_alpaca([t1, t2])

    if t1 not in precios.columns or t2 not in precios.columns:
        print(f"[!] Sin datos para {nombre}.")
        return

    params    = ParametrosBacktest()
    resultado = backtest_completo(precios[[t1, t2]], t1, t2, params, imprimir_reporte=True)

    try:
        from evaluacion import generar_informe_completo, grafico_rolling_cointegracion
        ohlcv   = descargar_ohlcv_horario([t1, t2], dias_atras=365)
        rolling = estabilidad_rolling(ohlcv["close"], t1, t2)
        grafico_rolling_cointegracion(rolling, nombre_par=nombre)
        generar_informe_completo(resultado, nombre_par=nombre)
    except ImportError:
        print("  [WARN] evaluacion.py no disponible.")


def modo_full(args) -> None:
    """Pipeline completo no interactivo: scan → backtest top 5 pares."""
    print("\n[MODO: FULL PIPELINE]")
    args.interactivo  = False
    args.top_n        = 5
    args.optimizar    = True
    args.walk_forward = False
    args.graficos     = False
    args.par          = None
    modo_scan(args)
    modo_backtest(args)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Pairs Trading — Arbitraje Estadístico",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--modo",
        choices=["scan", "diario", "backtest", "evaluar", "full"],
        default="diario",
        help=(
            "scan     : detectar pares cointegrados (datos horarios, 12 meses)\n"
            "diario   : verificar pares activos y generar señales del día\n"
            "backtest : backtesting out-of-sample de un par\n"
            "evaluar  : gráficos de análisis para un par\n"
            "full     : scan + backtest de los mejores pares"
        ),
    )
    parser.add_argument(
        "--par", nargs=2, metavar=("TICKER1", "TICKER2"),
        help="Par específico (ej: --par AAPL MSFT)"
    )
    parser.add_argument(
        "--top-n", type=int, default=10, dest="top_n",
        help="Número de pares a evaluar (default: 10)"
    )
    parser.add_argument(
        "--optimizar", action="store_true",
        help="Optimizar parámetros con grid search (in-sample/out-of-sample)"
    )
    parser.add_argument(
        "--walk-forward", action="store_true", dest="walk_forward",
        help="Ejecutar walk-forward validation"
    )
    parser.add_argument(
        "--graficos", action="store_true",
        help="Generar gráficos de evaluación"
    )

    args = parser.parse_args()

    modos = {
        "scan":     modo_scan,
        "diario":   modo_diario,
        "backtest": modo_backtest,
        "evaluar":  modo_evaluar,
        "full":     modo_full,
    }

    modos[args.modo](args)


if __name__ == "__main__":
    main()
