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
    filtrar_universo_interactivo,
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


# ── Helpers interactivos ──────────────────────────────────────────────────────

def _seleccionar_pares_interactivo(pares_df: pd.DataFrame) -> pd.DataFrame:
    """
    Muestra todos los pares cointegrados encontrados con sus métricas clave
    y permite al usuario elegir cuáles conservar para el análisis posterior.
    """
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
    print("  Score  = fuerza de cointegración Johansen (>1 = válido, mayor = mejor)")
    print("  Traza  = estadístico de traza de Johansen")
    print("  p-EG   = p-value Engle-Granger (<0.05 = cointegrado)")
    print()
    print("  ¿Qué pares deseas guardar para el análisis?")
    print("    Números separados por coma  →  ej: 1,3,5")
    print("    Rango con guión             →  ej: 1-10")
    print("    ENTER sin texto             →  guardar todos")
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
        print("  [WARN] Selección inválida. Guardando todos los pares.")
        return pares_df


def _seleccionar_de_csv(top_n: int) -> list[dict]:
    """
    Carga el CSV de pares, muestra una tabla resumida y
    permite al usuario elegir cuáles backtestar interactivamente.
    """
    try:
        pares_df = top_pares(n=top_n)
    except FileNotFoundError:
        print("[!] Ejecuta primero --modo scan para generar pares_cointegrados.csv")
        return []

    pares_df = pares_df.reset_index(drop=True)

    print(f"\n{'='*72}")
    print(f"  PARES DISPONIBLES EN CSV (top {top_n})")
    print(f"{'='*72}")
    print(f"  {'#':>4}  {'Par':<14}  {'Score':>7}  {'Traza':>7}  {'p-EG':>7}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*7}  {'─'*7}  {'─'*7}")

    for i, row in pares_df.iterrows():
        par = f"{row['ticker1']}/{row['ticker2']}"
        print(f"  {i+1:>4}  {par:<14}  {row['score']:>7.4f}  {row['traza']:>7.2f}"
              f"  {row['p_value_eg']:>7.4f}")

    print()
    print("  ¿Qué pares deseas backtestar?")
    print("    Números separados por coma  →  ej: 1,3")
    print("    Rango con guión             →  ej: 1-5")
    print("    ENTER sin texto             →  backtestar todos los mostrados")
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
        print("  [WARN] Selección inválida. Usando todos los pares.")
        return pares_df.to_dict("records")


# ── Modos de ejecución ────────────────────────────────────────────────────────

def modo_scan(args) -> None:
    """Detecta pares cointegrados en el universo (in-sample 2008-2020)."""
    print("\n[MODO: SCAN]")

    # En modo full el filtro interactivo se omite para no bloquear el pipeline
    if getattr(args, "interactivo", True):
        tickers = filtrar_universo_interactivo()
    else:
        tickers = obtener_sp500()

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

    # Selección interactiva de qué pares conservar
    if getattr(args, "interactivo", True):
        pares = _seleccionar_pares_interactivo(pares)

    guardar_pares(pares)


def modo_backtest(args) -> None:
    """Ejecuta backtesting completo sobre un par específico o los top-N."""
    print("\n[MODO: BACKTEST]")

    tickers = obtener_sp500()
    precios = descargar_precios(tickers, INICIO_DEFAULT, FIN_DEFAULT)
    precios = filtrar_datos(precios)
    _, out_sample = dividir_muestra(precios, corte="2020-01-01")

    # Par específico o selección interactiva desde CSV
    if args.par:
        t1, t2 = args.par[0], args.par[1]
        pares_lista = [{"ticker1": t1, "ticker2": t2}]
    elif getattr(args, "interactivo", True):
        pares_lista = _seleccionar_de_csv(args.top_n)
        if not pares_lista:
            return
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
    """Pipeline completo no interactivo: scan → backtest top-5 → gráficos."""
    print("\n[MODO: FULL PIPELINE]")
    args.interactivo  = False   # omitir todos los diálogos interactivos
    args.top_n        = 5
    args.optimizar    = True
    args.walk_forward = True
    args.graficos     = True
    args.par          = None
    modo_scan(args)
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
