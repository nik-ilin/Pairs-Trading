"""
backtesting.py — Motor de backtesting parametrizable.

Usa datos diarios de Alpaca (2020-presente, out-of-sample).
Implementa walk-forward validation para evitar overfitting.

Modelos incluidos:
  - Walk-Forward Validation : ventanas deslizantes de detección + operación
  - Grid Search de parámetros: optimiza en in-sample, evalúa en out-of-sample
  - Monte Carlo Simulation   : distribución de posibles resultados futuros
  - Bootstrap Sharpe         : intervalo de confianza del Sharpe Ratio
  - Test de permutaciones    : valida que los retornos no son fruto del azar
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from spread import (
    calcular_spread_kalman,
    calcular_zscore,
    calcular_half_life,
    generar_señales,
    tamaño_posicion_volatilidad,
    Señal,
)
from metricas import (
    sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
    cagr, win_rate, profit_factor, var_historico, cvar, reporte_completo,
)
from config import (
    DIAS_ANIO,
    ENTRADA_Z, SALIDA_Z, STOP_Z,
    CAPITAL_INICIAL, FRACCION_RIESGO, SLIPPAGE, COMISION, USAR_LOG,
    MC_N_SIMULACIONES, MC_HORIZONTE_DIAS, MC_SEMILLA,
    BS_N_BOOTSTRAP, BS_NIVEL_CONFIANZA,
    PERM_N, PERM_SEMILLA,
    WF_ANOS_DETECCION, WF_ANOS_OPERACION,
    GRID_ENTRADA_Z, GRID_SALIDA_Z, GRID_WINDOW,
    MIN_OBS, HL_MAX_DIAS, UMBRAL_EG,
)

warnings.filterwarnings("ignore")


# ── Parámetros del backtest ───────────────────────────────────────────────────

@dataclass
class ParametrosBacktest:
    entrada_z:       float    = ENTRADA_Z
    salida_z:        float    = SALIDA_Z
    stop_z:          float    = STOP_Z
    window_zscore:   int|None = None      # None = half-life OU automático
    capital:         float    = CAPITAL_INICIAL
    fraccion_riesgo: float    = FRACCION_RIESGO
    slippage:        float    = SLIPPAGE
    comision:        float    = COMISION
    usar_log:        bool     = USAR_LOG


# ── Motor principal ───────────────────────────────────────────────────────────

class MotorBacktest:
    """
    Ejecuta el backtest completo para un par de activos.

    Uso:
        motor = MotorBacktest(precios, "AAPL", "MSFT", params)
        resultado = motor.ejecutar()
    """

    def __init__(
        self,
        precios: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        params: ParametrosBacktest | None = None,
    ):
        self.precios  = precios[[ticker1, ticker2]].dropna()
        self.t1       = ticker1
        self.t2       = ticker2
        self.params   = params or ParametrosBacktest()

    def ejecutar(self) -> dict:
        """
        Ejecuta el backtest completo y devuelve un diccionario con:
          retornos_diarios, curva_capital, trades, metricas
        """
        p = self.params

        # 1. Calcular spread con Kalman
        spread, beta = calcular_spread_kalman(
            self.precios, self.t1, self.t2, usar_log=p.usar_log
        )

        # 2. Half-life OU para la ventana del z-score
        hl = calcular_half_life(spread)
        if not np.isfinite(hl):
            hl = float(HL_MAX_DIAS)
        window = p.window_zscore or max(5, int(np.ceil(hl)))

        # 3. Z-score y señales
        zscore  = calcular_zscore(spread, window=window)
        señales = generar_señales(zscore, p.entrada_z, p.salida_z, p.stop_z)
        tamaños = tamaño_posicion_volatilidad(
            spread, p.capital, fraccion=p.fraccion_riesgo
        )

        # 4. Simulación de retornos día a día
        retornos_diarios = pd.Series(0.0, index=self.precios.index, name="retorno")
        trades           = []
        posicion_actual  = 0
        precio_entrada   = None
        fecha_entrada    = None
        tam_entrada      = 0.0
        coste_total      = (p.slippage + p.comision) * 2  # 2 patas

        precios_t1 = self.precios[self.t1]
        precios_t2 = self.precios[self.t2]

        # Precalculado fuera del bucle — evita recomputar O(n) en cada iteración
        ret_t1_serie = precios_t1.pct_change()
        ret_t2_serie = precios_t2.pct_change()

        for i in range(len(self.precios)):
            fecha  = self.precios.index[i]
            señal  = señales.iloc[i]
            tam    = tamaños.iloc[i] if not np.isnan(tamaños.iloc[i]) else tam_entrada
            z      = zscore.iloc[i]
            b      = beta.iloc[i]

            if posicion_actual != 0:
                ret_t1 = ret_t1_serie.iloc[i]
                ret_t2 = ret_t2_serie.iloc[i]

                if posicion_actual == Señal.COMPRAR_SPREAD:
                    ret_dia = ret_t1 - b * ret_t2
                else:
                    ret_dia = -ret_t1 + b * ret_t2

                retornos_diarios.iloc[i] = ret_dia * tam_entrada / p.capital

            # Apertura de posición
            if señal in (Señal.COMPRAR_SPREAD, Señal.VENDER_SPREAD) and posicion_actual == 0:
                posicion_actual = señal
                precio_entrada  = spread.iloc[i]
                fecha_entrada   = fecha
                tam_entrada     = tam
                # Descontar coste de entrada
                retornos_diarios.iloc[i] -= coste_total * tam / p.capital

            # Cierre de posición
            elif señal == Señal.CERRAR and posicion_actual != 0:
                pnl = (spread.iloc[i] - precio_entrada) * posicion_actual * tam_entrada
                trades.append({
                    "fecha_entrada": fecha_entrada,
                    "fecha_salida":  fecha,
                    "direccion":     "LONG" if posicion_actual == 1 else "SHORT",
                    "z_entrada":     float(zscore.loc[fecha_entrada]) if fecha_entrada in zscore.index else np.nan,
                    "z_salida":      float(z),
                    "pnl":           float(pnl),
                    "duracion_dias": (fecha - fecha_entrada).days,
                })
                # Descontar coste de salida
                retornos_diarios.iloc[i] -= coste_total * tam_entrada / p.capital
                posicion_actual = 0
                precio_entrada  = None
                tam_entrada     = 0.0

        # Cerrar posición abierta al final del periodo
        if posicion_actual != 0 and precio_entrada is not None:
            pnl = (spread.iloc[-1] - precio_entrada) * posicion_actual * tam_entrada
            trades.append({
                "fecha_entrada": fecha_entrada,
                "fecha_salida":  self.precios.index[-1],
                "direccion":     "LONG" if posicion_actual == 1 else "SHORT",
                "z_entrada":     float(zscore.loc[fecha_entrada]) if fecha_entrada in zscore.index else np.nan,
                "z_salida":      float(zscore.iloc[-1]),
                "pnl":           float(pnl),
                "duracion_dias": (self.precios.index[-1] - fecha_entrada).days,
            })

        curva_capital  = (1 + retornos_diarios).cumprod() * p.capital
        df_trades      = pd.DataFrame(trades) if trades else pd.DataFrame()
        pnl_trades     = df_trades["pnl"] if not df_trades.empty else pd.Series(dtype=float)

        metricas_dict = {
            "par":          f"{self.t1}/{self.t2}",
            "half_life":    round(hl, 1),
            "window_usado": window,
            "sharpe":       round(sharpe_ratio(retornos_diarios), 3),
            "sortino":      round(sortino_ratio(retornos_diarios), 3),
            "calmar":       round(calmar_ratio(retornos_diarios), 3),
            "mdd":          round(max_drawdown(retornos_diarios) * 100, 2),
            "cagr":         round(cagr(retornos_diarios) * 100, 2),
            "var_95":       round(var_historico(retornos_diarios) * 100, 2),
            "cvar_95":      round(cvar(retornos_diarios) * 100, 2),
            "n_trades":     len(trades),
            "win_rate":     round(win_rate(pnl_trades) * 100, 2) if len(pnl_trades) > 0 else 0.0,
            "profit_factor": round(profit_factor(pnl_trades), 3) if len(pnl_trades) > 0 else 0.0,
        }

        return {
            "retornos":      retornos_diarios,
            "curva_capital": curva_capital,
            "spread":        spread,
            "zscore":        zscore,
            "beta":          beta,
            "señales":       señales,
            "trades":        df_trades,
            "metricas":      metricas_dict,
        }


# ── Walk-Forward Validation ───────────────────────────────────────────────────

def walk_forward(
    precios: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    params: ParametrosBacktest | None = None,
    años_deteccion: int = WF_ANOS_DETECCION,
    años_operacion: int = WF_ANOS_OPERACION,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Validación walk-forward: ventanas deslizantes de (detección + operación).
    Más realista que un único backtest; detecta si el modelo degenera con el tiempo.

    Ventana: detectar pares en N años → operar el siguiente año → avanzar 1 año.
    """
    fechas     = precios.index
    inicio_año = fechas[0].year
    fin_año    = fechas[-1].year
    resultados = []

    año_actual = inicio_año
    while año_actual + años_deteccion + años_operacion <= fin_año + 1:
        f_inicio_det = f"{año_actual}-01-01"
        f_fin_det    = f"{año_actual + años_deteccion}-01-01"
        f_fin_op     = f"{año_actual + años_deteccion + años_operacion}-01-01"

        oos = precios[
            (precios.index >= f_fin_det) & (precios.index < f_fin_op)
        ]

        if len(oos) < 50:
            año_actual += 1
            continue

        motor = MotorBacktest(oos, ticker1, ticker2, params)
        res   = motor.ejecutar()
        m     = res["metricas"]

        resultados.append({
            "ventana":    f"{año_actual}-{año_actual + años_deteccion} → op. {año_actual + años_deteccion}",
            "sharpe":     m["sharpe"],
            "sortino":    m["sortino"],
            "mdd":        m["mdd"],
            "cagr":       m["cagr"],
            "n_trades":   m["n_trades"],
            "win_rate":   m["win_rate"],
        })

        if verbose:
            print(f"  Ventana {resultados[-1]['ventana']} | Sharpe: {m['sharpe']:.2f} | MDD: {m['mdd']:.1f}%")

        año_actual += 1

    return pd.DataFrame(resultados) if resultados else pd.DataFrame()


# ── Optimización de parámetros (Grid Search) ─────────────────────────────────

def optimizar_parametros(
    precios: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    grid: dict | None = None,
    metrica: str = "sharpe",
    fraccion_is: float = 0.7,
) -> tuple[ParametrosBacktest, pd.DataFrame]:
    """
    Grid search sobre parámetros críticos de la estrategia.

    Divide los datos en in-sample (primeros fraccion_is) para optimizar
    y out-of-sample (resto) para validar. Así los parámetros no se eligen
    viendo los datos sobre los que luego se evalúan.

    Args:
        fraccion_is: fracción del histórico usada para optimizar (default 70%).
    """
    if grid is None:
        grid = {
            "entrada_z":     GRID_ENTRADA_Z,
            "salida_z":      GRID_SALIDA_Z,
            "window_zscore": GRID_WINDOW,
        }

    n_is       = int(len(precios) * fraccion_is)
    precios_is = precios.iloc[:n_is]

    if len(precios_is) < MIN_OBS:
        raise ValueError(
            f"Datos insuficientes para grid search: {len(precios_is)} barras "
            f"(mínimo {MIN_OBS}). Reduce fraccion_is o usa más histórico."
        )

    combinaciones = [
        (e, s, w)
        for e in grid["entrada_z"]
        for s in grid["salida_z"]
        for w in grid["window_zscore"]
        if s < e
    ]

    print(f"[INFO] Grid search: {len(combinaciones)} combinaciones "
          f"(IS: {len(precios_is)} barras | OOS: {len(precios) - n_is} barras)...")

    resultados = []
    for entrada, salida, window in combinaciones:
        p = ParametrosBacktest(entrada_z=entrada, salida_z=salida, window_zscore=window)
        try:
            res = MotorBacktest(precios_is, ticker1, ticker2, p).ejecutar()
            m   = res["metricas"]
            resultados.append({
                "entrada_z":     entrada,
                "salida_z":      salida,
                "window_zscore": window,
                "sharpe":        m["sharpe"],
                "sortino":       m["sortino"],
                "calmar":        m["calmar"],
                "mdd":           m["mdd"],
                "n_trades":      m["n_trades"],
            })
        except Exception:
            continue

    if not resultados:
        raise RuntimeError("Ninguna combinación del grid produjo resultados válidos.")

    df_grid = pd.DataFrame(resultados).sort_values(metrica, ascending=False)
    mejor   = df_grid.iloc[0]

    params_optimos = ParametrosBacktest(
        entrada_z=mejor["entrada_z"],
        salida_z=mejor["salida_z"],
        window_zscore=None if pd.isna(mejor["window_zscore"]) else int(mejor["window_zscore"]),
    )

    print(f"[OK] Mejores parametros (IS): entrada={mejor['entrada_z']}, "
          f"salida={mejor['salida_z']}, window={mejor['window_zscore']} "
          f"-> {metrica.upper()} = {mejor[metrica]:.3f}")

    return params_optimos, df_grid


# ── Simulación de Monte Carlo ─────────────────────────────────────────────────

def simulacion_monte_carlo(
    retornos: pd.Series,
    n_simulaciones: int = MC_N_SIMULACIONES,
    horizonte_dias: int = MC_HORIZONTE_DIAS,
    capital_inicial: float = CAPITAL_INICIAL,
    semilla: int = MC_SEMILLA,
) -> dict:
    """
    Simula N trayectorias de capital futuras mediante bootstrapping de retornos.
    Asume que los retornos históricos son representativos del futuro.

    Devuelve distribución de: capital final, MDD, Sharpe por simulación.
    """
    rng = np.random.default_rng(semilla)
    ret_arr = retornos.dropna().values

    finales  = np.zeros(n_simulaciones)
    ddrawdwn = np.zeros(n_simulaciones)

    for i in range(n_simulaciones):
        muestra = rng.choice(ret_arr, size=horizonte_dias, replace=True)
        curva   = capital_inicial * np.cumprod(1 + muestra)
        finales[i]  = curva[-1]
        pico        = np.maximum.accumulate(curva)
        ddrawdwn[i] = np.min((curva - pico) / pico)

    percentiles = [5, 25, 50, 75, 95]
    return {
        "capital_final":     pd.Series(finales),
        "max_drawdown_sim":  pd.Series(ddrawdwn),
        "percentiles_capital": dict(zip(
            [f"p{p}" for p in percentiles],
            np.percentile(finales, percentiles)
        )),
        "prob_ganancia":     float((finales > capital_inicial).mean()),
        "capital_medio":     float(np.mean(finales)),
        "n_simulaciones":    n_simulaciones,
    }


# ── Bootstrap del Sharpe Ratio ────────────────────────────────────────────────

def bootstrap_sharpe(
    retornos: pd.Series,
    n_bootstrap: int = BS_N_BOOTSTRAP,
    nivel_confianza: float = BS_NIVEL_CONFIANZA,
    semilla: int = MC_SEMILLA,
) -> dict:
    """
    Calcula el intervalo de confianza del Sharpe Ratio mediante bootstrapping.
    Valida que el Sharpe observado no sea un artefacto del periodo elegido.
    """
    rng     = np.random.default_rng(semilla)
    ret_arr = retornos.dropna().values
    sharpes = []

    for _ in range(n_bootstrap):
        muestra = rng.choice(ret_arr, size=len(ret_arr), replace=True)
        serie   = pd.Series(muestra)
        sharpes.append(sharpe_ratio(serie))

    alpha = (1 - nivel_confianza) / 2
    return {
        "sharpe_observado": round(sharpe_ratio(retornos), 3),
        "ci_inferior":      round(float(np.percentile(sharpes, alpha * 100)), 3),
        "ci_superior":      round(float(np.percentile(sharpes, (1 - alpha) * 100)), 3),
        "sharpe_medio":     round(float(np.mean(sharpes)), 3),
    }


# ── Test de permutaciones ─────────────────────────────────────────────────────

def test_permutaciones(
    retornos: pd.Series,
    n_permutaciones: int = PERM_N,
    semilla: int = PERM_SEMILLA,
) -> dict:
    """
    Valida estadísticamente que el Sharpe de la estrategia no es fruto del azar.
    Permuta los retornos aleatoriamente y construye la distribución nula.
    p-value < 0.05 indica que el resultado es estadísticamente significativo.
    """
    rng            = np.random.default_rng(semilla)
    ret_arr        = retornos.dropna().values
    sharpe_real    = sharpe_ratio(retornos)
    sharpes_nulos  = []

    for _ in range(n_permutaciones):
        permutado = rng.permutation(ret_arr)
        sharpes_nulos.append(sharpe_ratio(pd.Series(permutado)))

    p_value = float((np.array(sharpes_nulos) >= sharpe_real).mean())
    return {
        "sharpe_estrategia": round(sharpe_real, 3),
        "sharpe_nulo_medio": round(float(np.mean(sharpes_nulos)), 3),
        "p_value":           round(p_value, 4),
        "significativo":     p_value < 0.05,
    }


# ── Paper Trading Histórico con detección dinámica de régimen ────────────────

def _imprimir_reporte_paper(
    metricas: dict,
    periodos: list[tuple],
    df_trades: pd.DataFrame,
    fecha_inicio: str,
    fecha_fin: str,
) -> None:
    """Reporte visual del paper trading histórico."""
    m     = metricas
    ancho = 62

    print(f"\n{'='*ancho}")
    print(f"  PAPER TRADING HISTÓRICO — {m['par']}")
    print(f"{'='*ancho}")
    print(f"  Período analizado    : {fecha_inicio} → {fecha_fin}")
    print(f"  Half-life estimado   : {m['half_life']:.1f} barras")
    print(f"  Ventana z-score      : {m['window_usado']} barras")
    print()

    print(f"  PERÍODOS COINTEGRADOS ({m['n_periodos']} detectados):")
    if periodos:
        for i, (ini, fin) in enumerate(periodos, 1):
            dias = (fin - ini).days
            print(f"    {i}. {str(ini)[:10]} → {str(fin)[:10]}  ({dias} días)")
    else:
        print(f"    (ninguno confirmado en el período)")
    print(f"  Tiempo operando: {m['dias_activos']} días "
          f"({m['pct_tiempo_activo']:.1f}% del histórico)")

    print(f"\n  {'─'*58}")
    print(f"  {'MÉTRICAS':30}  {'PAPER':>10}  {'NAIVE':>10}")
    print(f"  {'─'*58}")
    print(f"  {'Trades totales':30}  {m['n_trades']:>10}  {m['n_trades_naive']:>10}")
    print(f"  {'Win rate (%)':30}  {m['win_rate']:>10.1f}  {m['win_rate_naive']:>10.1f}")
    print(f"  {'Profit factor':30}  {m['profit_factor']:>10.3f}  {m['profit_factor_naive']:>10.3f}")
    print(f"  {'CAGR (%)':30}  {m['cagr']:>+10.2f}  {m['cagr_naive']:>+10.2f}")
    print(f"  {'Sharpe':30}  {m['sharpe']:>10.3f}  {m['sharpe_naive']:>10.3f}")
    print(f"  {'Sortino':30}  {m['sortino']:>10.3f}  {m['sortino_naive']:>10.3f}")
    print(f"  {'Max Drawdown (%)':30}  {m['mdd']:>10.2f}  {m['mdd_naive']:>10.2f}")
    print(f"  {'─'*58}")

    if not df_trades.empty:
        print(f"\n  TRADES ({len(df_trades)} total):")
        print(f"  {'Entrada':<12} {'Salida':<12} {'Dir':<6} {'PnL':>9} {'Días':>5}  Cierre")
        print(f"  {'─'*60}")
        for _, r in df_trades.iterrows():
            pnl_str = f"${r['pnl']:>+,.0f}"
            print(f"  {str(r['fecha_entrada'])[:10]:<12} "
                  f"{str(r['fecha_salida'])[:10]:<12} "
                  f"{r['direccion']:<6} "
                  f"{pnl_str:>9} "
                  f"{r['duracion_dias']:>5}  "
                  f"{r['cierre_por']}")

    capital_final = m['capital_final']
    ganancia      = capital_final - m['capital_inicial']
    print(f"\n  Capital inicial : ${m['capital_inicial']:>12,.0f}")
    print(f"  Capital final   : ${capital_final:>12,.0f}")
    print(f"  Ganancia neta   : ${ganancia:>+12,.0f}  ({ganancia/m['capital_inicial']*100:+.1f}%)")
    print(f"{'='*ancho}\n")


def paper_trading_historico(
    precios: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    params: ParametrosBacktest | None = None,
    ventana_coint: int = 126,
    n_confirmar: int = 4,
    n_romper: int = 2,
    freq_check: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Simulación de paper trading usando detección dinámica del régimen de cointegración.

    A diferencia del backtest estándar (que opera siempre), este modo:
      1. Solo activa el trading cuando N tests EG consecutivos confirman cointegración
         → evita operar en falsas alarmas al exigir estabilidad temporal
      2. Cierra la posición y pausa cuando M tests consecutivos detectan ruptura
      3. Reanuda automáticamente si la cointegración se reestablece

    Parámetros de régimen:
      ventana_coint : barras para cada test EG rolling (default 126 ≈ 6 meses diarios)
      n_confirmar   : checks consecutivos para ACTIVAR el trading (default 4 ≈ 1 mes)
      n_romper      : checks consecutivos para SUSPENDER el trading (default 2 ≈ 2 semanas)
      freq_check    : frecuencia del test EG en barras (default 5 = semanal)

    Returns el mismo dict que backtest_completo() más:
      "regimen"          : Series bool — True cuando el par está activo
      "periodos_activos" : lista de tuplas (fecha_inicio, fecha_fin) por período
    """
    from deteccion import test_engle_granger

    p       = params or ParametrosBacktest()
    precios = precios[[ticker1, ticker2]].dropna()
    n       = len(precios)

    if n < ventana_coint + freq_check * n_confirmar:
        raise ValueError(
            f"Datos insuficientes: {n} barras. Se necesitan al menos "
            f"{ventana_coint + freq_check * n_confirmar} para la detección de régimen."
        )

    # ── 1. Kalman continuo sobre todo el histórico ────────────────────────────
    spread, beta = calcular_spread_kalman(precios, ticker1, ticker2, usar_log=p.usar_log)

    hl = calcular_half_life(spread)
    if not np.isfinite(hl):
        hl = float(HL_MAX_DIAS)
    window  = p.window_zscore or max(5, int(np.ceil(hl)))
    zscore  = calcular_zscore(spread, window=window)
    tamaños = tamaño_posicion_volatilidad(spread, p.capital, fraccion=p.fraccion_riesgo)

    # ── 2. Detección de régimen: EG rolling cada freq_check barras ───────────
    regime         = np.zeros(n, dtype=bool)
    n_conf_consec  = 0
    n_break_consec = 0
    activo         = False

    for i in range(ventana_coint, n, freq_check):
        idx_win = precios.index[i - ventana_coint : i]
        s1_win  = np.log(precios[ticker1].loc[idx_win])
        s2_win  = np.log(precios[ticker2].loc[idx_win])
        p_val   = test_engle_granger(s1_win, s2_win)["p_value_eg"]

        if p_val < UMBRAL_EG:
            n_conf_consec  += 1
            n_break_consec  = 0
            if not activo and n_conf_consec >= n_confirmar:
                activo = True
        else:
            n_break_consec += 1
            n_conf_consec   = 0
            if activo and n_break_consec >= n_romper:
                activo = False

        fin_bloque           = min(i + freq_check, n)
        regime[i:fin_bloque] = activo

    # ── 3. Simulación de trades (solo durante régimen activo) ─────────────────
    coste_total      = (p.slippage + p.comision) * 2
    retornos_diarios = pd.Series(0.0, index=precios.index, name="retorno")
    trades           = []
    posicion_actual  = 0
    precio_entrada   = None
    fecha_entrada    = None
    tam_entrada      = 0.0

    ret_t1 = precios[ticker1].pct_change()
    ret_t2 = precios[ticker2].pct_change()

    for i in range(n):
        fecha       = precios.index[i]
        z           = zscore.iloc[i]
        b           = beta.iloc[i]
        tam         = tamaños.iloc[i] if not np.isnan(tamaños.iloc[i]) else tam_entrada
        en_regimen  = bool(regime[i])

        # PnL diario de posición abierta (independiente del régimen)
        if posicion_actual != 0:
            r1 = ret_t1.iloc[i]
            r2 = ret_t2.iloc[i]
            rd = (r1 - b * r2) if posicion_actual == Señal.COMPRAR_SPREAD else (-r1 + b * r2)
            retornos_diarios.iloc[i] = rd * tam_entrada / p.capital

        if np.isnan(z):
            continue

        # Cierre forzado al perder el régimen
        if posicion_actual != 0 and not en_regimen:
            pnl = (spread.iloc[i] - precio_entrada) * posicion_actual * tam_entrada
            trades.append({
                "fecha_entrada": fecha_entrada,
                "fecha_salida":  fecha,
                "direccion":     "LONG" if posicion_actual == 1 else "SHORT",
                "z_entrada":     float(zscore.loc[fecha_entrada]),
                "z_salida":      float(z),
                "pnl":           float(pnl),
                "duracion_dias": (fecha - fecha_entrada).days,
                "cierre_por":    "RUPTURA_COINT",
            })
            retornos_diarios.iloc[i] -= coste_total * tam_entrada / p.capital
            posicion_actual = 0
            precio_entrada  = None
            tam_entrada     = 0.0
            continue

        if not en_regimen:
            continue

        # Stop-loss
        if posicion_actual != 0 and abs(z) > p.stop_z:
            pnl = (spread.iloc[i] - precio_entrada) * posicion_actual * tam_entrada
            trades.append({
                "fecha_entrada": fecha_entrada,
                "fecha_salida":  fecha,
                "direccion":     "LONG" if posicion_actual == 1 else "SHORT",
                "z_entrada":     float(zscore.loc[fecha_entrada]),
                "z_salida":      float(z),
                "pnl":           float(pnl),
                "duracion_dias": (fecha - fecha_entrada).days,
                "cierre_por":    "STOP",
            })
            retornos_diarios.iloc[i] -= coste_total * tam_entrada / p.capital
            posicion_actual = 0
            precio_entrada  = None
            tam_entrada     = 0.0

        # Cierre por reversión
        elif posicion_actual == Señal.COMPRAR_SPREAD and z >= -p.salida_z:
            pnl = (spread.iloc[i] - precio_entrada) * posicion_actual * tam_entrada
            trades.append({
                "fecha_entrada": fecha_entrada,
                "fecha_salida":  fecha,
                "direccion":     "LONG",
                "z_entrada":     float(zscore.loc[fecha_entrada]),
                "z_salida":      float(z),
                "pnl":           float(pnl),
                "duracion_dias": (fecha - fecha_entrada).days,
                "cierre_por":    "REVERSION",
            })
            retornos_diarios.iloc[i] -= coste_total * tam_entrada / p.capital
            posicion_actual = 0
            precio_entrada  = None
            tam_entrada     = 0.0

        elif posicion_actual == Señal.VENDER_SPREAD and z <= p.salida_z:
            pnl = (spread.iloc[i] - precio_entrada) * posicion_actual * tam_entrada
            trades.append({
                "fecha_entrada": fecha_entrada,
                "fecha_salida":  fecha,
                "direccion":     "SHORT",
                "z_entrada":     float(zscore.loc[fecha_entrada]),
                "z_salida":      float(z),
                "pnl":           float(pnl),
                "duracion_dias": (fecha - fecha_entrada).days,
                "cierre_por":    "REVERSION",
            })
            retornos_diarios.iloc[i] -= coste_total * tam_entrada / p.capital
            posicion_actual = 0
            precio_entrada  = None
            tam_entrada     = 0.0

        # Apertura (sin posición y régimen activo)
        elif posicion_actual == 0:
            if z < -p.entrada_z:
                posicion_actual = Señal.COMPRAR_SPREAD
                precio_entrada  = spread.iloc[i]
                fecha_entrada   = fecha
                tam_entrada     = tam
                retornos_diarios.iloc[i] -= coste_total * tam / p.capital
            elif z > p.entrada_z:
                posicion_actual = Señal.VENDER_SPREAD
                precio_entrada  = spread.iloc[i]
                fecha_entrada   = fecha
                tam_entrada     = tam
                retornos_diarios.iloc[i] -= coste_total * tam / p.capital

    # Cerrar posición abierta al final del período
    if posicion_actual != 0 and precio_entrada is not None:
        pnl = (spread.iloc[-1] - precio_entrada) * posicion_actual * tam_entrada
        trades.append({
            "fecha_entrada": fecha_entrada,
            "fecha_salida":  precios.index[-1],
            "direccion":     "LONG" if posicion_actual == 1 else "SHORT",
            "z_entrada":     float(zscore.loc[fecha_entrada]),
            "z_salida":      float(zscore.iloc[-1]),
            "pnl":           float(pnl),
            "duracion_dias": (precios.index[-1] - fecha_entrada).days,
            "cierre_por":    "FIN_PERIODO",
        })

    # ── 4. Estadísticas del régimen ───────────────────────────────────────────
    regime_series = pd.Series(regime, index=precios.index, name="regimen_activo")
    dias_activos  = int(regime.sum())

    periodos: list[tuple] = []
    en_periodo     = False
    inicio_periodo = None
    for i in range(n):
        if regime[i] and not en_periodo:
            en_periodo     = True
            inicio_periodo = precios.index[i]
        elif not regime[i] and en_periodo:
            en_periodo = False
            periodos.append((inicio_periodo, precios.index[i - 1]))
    if en_periodo:
        periodos.append((inicio_periodo, precios.index[-1]))

    curva_capital = (1 + retornos_diarios).cumprod() * p.capital
    df_trades     = pd.DataFrame(trades) if trades else pd.DataFrame()
    pnl_series    = df_trades["pnl"] if not df_trades.empty else pd.Series(dtype=float)

    # ── 5. Comparación con backtest naive ─────────────────────────────────────
    res_naive  = MotorBacktest(precios, ticker1, ticker2, p).ejecutar()
    m_naive    = res_naive["metricas"]
    pnl_naive  = res_naive["trades"]["pnl"] if not res_naive["trades"].empty else pd.Series(dtype=float)

    metricas = {
        "par":                f"{ticker1}/{ticker2}",
        "half_life":          round(hl, 1),
        "window_usado":       window,
        "capital_inicial":    p.capital,
        "capital_final":      float(curva_capital.iloc[-1]),
        "n_trades":           len(trades),
        "dias_activos":       dias_activos,
        "pct_tiempo_activo":  round(dias_activos / n * 100, 1),
        "n_periodos":         len(periodos),
        "sharpe":             round(sharpe_ratio(retornos_diarios), 3),
        "sortino":            round(sortino_ratio(retornos_diarios), 3),
        "cagr":               round(cagr(retornos_diarios) * 100, 2),
        "mdd":                round(max_drawdown(retornos_diarios) * 100, 2),
        "win_rate":           round(win_rate(pnl_series) * 100, 2) if len(pnl_series) > 0 else 0.0,
        "profit_factor":      round(profit_factor(pnl_series), 3) if len(pnl_series) > 0 else 0.0,
        # columnas naive para comparación
        "n_trades_naive":     m_naive["n_trades"],
        "win_rate_naive":     m_naive["win_rate"],
        "profit_factor_naive": m_naive["profit_factor"],
        "cagr_naive":         m_naive["cagr"],
        "sharpe_naive":       m_naive["sharpe"],
        "sortino_naive":      m_naive["sortino"],
        "mdd_naive":          m_naive["mdd"],
    }

    if verbose:
        _imprimir_reporte_paper(
            metricas, periodos, df_trades,
            str(precios.index[0])[:10],
            str(precios.index[-1])[:10],
        )

    return {
        "retornos":         retornos_diarios,
        "curva_capital":    curva_capital,
        "spread":           spread,
        "zscore":           zscore,
        "beta":             beta,
        "trades":           df_trades,
        "metricas":         metricas,
        "regimen":          regime_series,
        "periodos_activos": periodos,
    }


# ── Función de conveniencia ───────────────────────────────────────────────────

def backtest_completo(
    precios: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    params: ParametrosBacktest | None = None,
    imprimir_reporte: bool = True,
) -> dict:
    """
    Ejecuta backtest + Monte Carlo + Bootstrap Sharpe + test de permutaciones.
    Punto de entrada recomendado para análisis completo de un par.
    """
    params = params or ParametrosBacktest()
    motor  = MotorBacktest(precios, ticker1, ticker2, params)
    res    = motor.ejecutar()

    retornos = res["retornos"]

    mc      = simulacion_monte_carlo(retornos)
    bs      = bootstrap_sharpe(retornos)
    perm    = test_permutaciones(retornos)

    if imprimir_reporte:
        reporte_completo(retornos, res["trades"].get("pnl") if not res["trades"].empty else None,
                         nombre=f"{ticker1}/{ticker2}")
        print(f"\n[Monte Carlo]")
        print(f"  P(ganancia)          : {mc['prob_ganancia']*100:.1f}%")
        print(f"  Capital medio final  : ${mc['capital_medio']:,.0f}")
        print(f"  Capital P5 / P95     : ${mc['percentiles_capital']['p5']:,.0f} / ${mc['percentiles_capital']['p95']:,.0f}")
        print(f"\n[Bootstrap Sharpe {bs['ci_inferior']} – {bs['ci_superior']}]")
        print(f"  Sharpe observado     : {bs['sharpe_observado']}")
        print(f"\n[Test de Permutaciones]")
        print(f"  p-value              : {perm['p_value']} ({'SIGNIFICATIVO' if perm['significativo'] else 'NO SIGNIFICATIVO'})")

    res.update({"monte_carlo": mc, "bootstrap_sharpe": bs, "permutaciones": perm})
    return res
