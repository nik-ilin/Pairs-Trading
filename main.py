"""
main.py — Orquestador principal del sistema de arbitraje estadístico.

Modos de ejecución (--modo):
  scan      : detecta pares cointegrados en el universo (in-sample 2008-2020)
  backtest  : ejecuta backtesting completo sobre los mejores pares (2020-2026)
  evaluar   : genera todos los gráficos de presentación
  señales   : genera señales del día para los pares validados
  full      : ejecuta el pipeline completo (scan → backtest → evaluar)

Uso:
  python main.py --modo full
  python main.py --modo backtest --par AAPL MSFT
  python main.py --modo señales
"""

import argparse
import warnings

import pandas as pd

from datos import (
    obtener_sp500,
    descargar_precios,
    filtrar_datos,
    dividir_muestra,
    INICIO_DEFAULT,
    FIN_DEFAULT,
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
    actualizar_estado,
    imprimir_resumen_diario,
)
from evaluacion import (
    generar_informe_completo,
    grafico_rolling_cointegracion,
    grafico_walk_forward,
)
from metricas import reporte_completo

warnings.filterwarnings("ignore")


# ── Modos de ejecución ────────────────────────────────────────────────────────

def modo_scan(args) -> None:
    """Detecta pares cointegrados en el universo (in-sample 2008-2020)."""
    print("\n[MODO: SCAN]")
    tickers  = obtener_sp500()
    precios  = descargar_precios(tickers, INICIO_DEFAULT, FIN_DEFAULT)
    precios  = filtrar_datos(precios)
    in_sample, _ = dividir_muestra(precios, corte="2020-01-01")

    pares = escanear_todos_los_pares(
        in_sample,
        umbral_eg=0.05,
        min_score_johansen=1.0,
        verbose=True,
    )

    if pares.empty:
        print("[!] No se encontraron pares cointegrados.")
        return

    guardar_pares(pares)
    print(f"\nTop 10 pares cointegrados:")
    print(pares.head(10).to_string(index=False))


def modo_backtest(args) -> None:
    """Ejecuta backtesting completo sobre un par específico o los top-N."""
    print("\n[MODO: BACKTEST]")

    tickers = obtener_sp500()
    precios = descargar_precios(tickers, INICIO_DEFAULT, FIN_DEFAULT)
    precios = filtrar_datos(precios)
    _, out_sample = dividir_muestra(precios, corte="2020-01-01")

    # Par específico o top pares del CSV
    if args.par:
        t1, t2 = args.par[0], args.par[1]
        pares_lista = [{"ticker1": t1, "ticker2": t2}]
    else:
        try:
            pares_df = top_pares(n=args.top_n)
            pares_lista = pares_df.to_dict("records")
        except FileNotFoundError:
            print("[!] Ejecuta primero --modo scan para generar pares_cointegrados.csv")
            return

    for fila in pares_lista:
        t1, t2 = fila["ticker1"], fila["ticker2"]
        nombre = f"{t1}/{t2}"
        print(f"\n{'─'*50}")
        print(f"Backtesting: {nombre}")
        print(f"{'─'*50}")

        if t1 not in out_sample.columns or t2 not in out_sample.columns:
            print(f"  [WARN] {nombre} sin datos en out-of-sample. Omitiendo.")
            continue

        # Optimizar parámetros si se solicita
        if args.optimizar:
            params, _ = optimizar_parametros(out_sample, t1, t2)
        else:
            params = ParametrosBacktest()

        # Backtest completo con Monte Carlo y tests estadísticos
        resultado = backtest_completo(out_sample, t1, t2, params, imprimir_reporte=True)

        # Walk-forward
        if args.walk_forward:
            print(f"\n[Walk-Forward] {nombre}")
            wf = walk_forward(precios, t1, t2, params, verbose=True)
            grafico_walk_forward(wf, nombre_par=nombre)

        # Informe gráfico completo
        if args.graficos:
            rolling = estabilidad_rolling(precios, t1, t2)
            grafico_rolling_cointegracion(rolling, nombre_par=nombre)
            generar_informe_completo(resultado, nombre_par=nombre)


def modo_evaluar(args) -> None:
    """Genera todos los gráficos para el par indicado."""
    print("\n[MODO: EVALUAR]")

    if not args.par:
        print("[!] Indica un par con --par TICKER1 TICKER2")
        return

    t1, t2 = args.par[0], args.par[1]
    nombre = f"{t1}/{t2}"

    tickers = [t1, t2]
    precios = descargar_precios(tickers, INICIO_DEFAULT, FIN_DEFAULT)
    precios = filtrar_datos(precios)
    _, out_sample = dividir_muestra(precios)

    params   = ParametrosBacktest()
    resultado = backtest_completo(out_sample, t1, t2, params, imprimir_reporte=True)

    rolling = estabilidad_rolling(precios, t1, t2)
    grafico_rolling_cointegracion(rolling, nombre_par=nombre)
    generar_informe_completo(resultado, nombre_par=nombre)


def modo_señales(args) -> None:
    """Genera señales del día para los pares validados."""
    print("\n[MODO: SEÑALES DIARIAS]")
    df_señales = ejecutar_pipeline_diario(top_n=args.top_n, verbose=True)
    imprimir_resumen_diario(df_señales)
    if not df_señales.empty:
        estado = actualizar_estado(df_señales)
        print(f"[INFO] Posiciones abiertas: {len(estado)}")


def modo_full(args) -> None:
    """Pipeline completo: scan → backtest → evaluar los mejores pares."""
    print("\n[MODO: FULL PIPELINE]")
    modo_scan(args)
    args.top_n     = 5
    args.optimizar = True
    args.walk_forward = True
    args.graficos  = True
    args.par       = None
    modo_backtest(args)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Arbitraje Estadístico — Pairs Trading",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--modo",
        choices=["scan", "backtest", "evaluar", "señales", "full"],
        default="full",
        help=(
            "scan     : detectar pares cointegrados (in-sample)\n"
            "backtest : ejecutar backtesting out-of-sample\n"
            "evaluar  : generar gráficos de presentación\n"
            "señales  : señales diarias de trading\n"
            "full     : pipeline completo"
        ),
    )
    parser.add_argument(
        "--par", nargs=2, metavar=("TICKER1", "TICKER2"),
        help="Par específico a analizar (ej: --par AAPL MSFT)"
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Número de mejores pares a evaluar (default: 10)"
    )
    parser.add_argument(
        "--optimizar", action="store_true",
        help="Optimizar parámetros con grid search"
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Ejecutar walk-forward validation"
    )
    parser.add_argument(
        "--graficos", action="store_true",
        help="Generar gráficos de evaluación"
    )

    args = parser.parse_args()

    modos = {
        "scan":     modo_scan,
        "backtest": modo_backtest,
        "evaluar":  modo_evaluar,
        "señales":  modo_señales,
        "full":     modo_full,
    }

    modos[args.modo](args)


if __name__ == "__main__":
    main()
