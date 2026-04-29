"""
backtesting.py — Motor de backtesting parametrizable (SMART Objetivo 2).

Cubre el periodo 2008–2026 con separación in-sample / out-of-sample.
Implementa walk-forward validation para evitar overfitting.

Modelos estadísticos y cuantitativos incluidos:
  - Walk-Forward Validation : ventanas deslizantes de detección + operación
  - Grid Search de parámetros: optimiza umbral de entrada/salida y ventana OU
  - Monte Carlo Simulation   : distribución de posibles resultados futuros
  - Bootstrap Sharpe         : intervalo de confianza del Sharpe Ratio
  - Test de permutaciones    : valida que los retornos no son fruto del azar
"""

import warnings
from dataclasses import dataclass, field

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

warnings.filterwarnings("ignore")

DIAS_ANIO = 252


# ── Parámetros del backtest ───────────────────────────────────────────────────

@dataclass
class ParametrosBacktest:
    entrada_z:      float = 2.0    # umbral de z-score para abrir posición
    salida_z:       float = 0.5    # umbral de z-score para cerrar posición
    stop_z:         float = 3.5    # stop-loss en z-score
    window_zscore:  int | None = None  # None = usar half-life OU automático
    capital:        float = 100_000.0
    fraccion_riesgo: float = 0.10  # fracción del capital por trade
    slippage:       float = 0.0005 # 0.05% por operación por pata
    comision:       float = 0.001  # 0.10% por operación por pata
    usar_log:       bool  = True


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
        spread, beta, alpha = calcular_spread_kalman(
            self.precios, self.t1, self.t2, usar_log=p.usar_log
        )

        # 2. Half-life OU para la ventana del z-score
        hl     = calcular_half_life(spread)
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

        for i in range(len(self.precios)):
            fecha  = self.precios.index[i]
            señal  = señales.iloc[i]
            tam    = tamaños.iloc[i] if not np.isnan(tamaños.iloc[i]) else tam_entrada
            z      = zscore.iloc[i]
            b      = beta.iloc[i]

            if posicion_actual != 0:
                # Retorno diario de la posición abierta
                ret_t1 = precios_t1.pct_change().iloc[i]
                ret_t2 = precios_t2.pct_change().iloc[i]

                if posicion_actual == Señal.COMPRAR_SPREAD:
                    # Long t1, short t2
                    ret_dia = ret_t1 - b * ret_t2
                else:
                    # Short t1, long t2
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
    años_deteccion: int = 3,
    años_operacion: int = 1,
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

        if len(oos) < 60:
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
) -> tuple[ParametrosBacktest, pd.DataFrame]:
    """
    Grid search sobre los parámetros críticos de la estrategia.
    Maximiza la métrica elegida (sharpe | sortino | calmar).

    Grid por defecto:
      entrada_z    : [1.5, 2.0, 2.5]
      salida_z     : [0.25, 0.5]
      window_zscore: [None (OU), 30, 60]
    """
    if grid is None:
        grid = {
            "entrada_z":     [1.5, 2.0, 2.5],
            "salida_z":      [0.25, 0.5],
            "window_zscore": [None, 30, 60],
        }

    resultados = []
    combinaciones = [
        (e, s, w)
        for e in grid["entrada_z"]
        for s in grid["salida_z"]
        for w in grid["window_zscore"]
        if s < e
    ]

    print(f"[INFO] Grid search: {len(combinaciones)} combinaciones...")

    for entrada, salida, window in combinaciones:
        p = ParametrosBacktest(
            entrada_z=entrada, salida_z=salida, window_zscore=window
        )
        motor = MotorBacktest(precios, ticker1, ticker2, p)
        try:
            res = motor.ejecutar()
            m   = res["metricas"]
            resultados.append({
                "entrada_z":    entrada,
                "salida_z":     salida,
                "window_zscore": window,
                "sharpe":       m["sharpe"],
                "sortino":      m["sortino"],
                "calmar":       m["calmar"],
                "mdd":          m["mdd"],
                "n_trades":     m["n_trades"],
            })
        except Exception:
            continue

    df_grid = pd.DataFrame(resultados).sort_values(metrica, ascending=False)
    mejor   = df_grid.iloc[0]

    params_optimos = ParametrosBacktest(
        entrada_z=mejor["entrada_z"],
        salida_z=mejor["salida_z"],
        window_zscore=None if pd.isna(mejor["window_zscore"]) else int(mejor["window_zscore"]),
    )

    print(f"[OK] Mejores parámetros: entrada={mejor['entrada_z']}, "
          f"salida={mejor['salida_z']}, window={mejor['window_zscore']} "
          f"→ {metrica.upper()} = {mejor[metrica]:.3f}")

    return params_optimos, df_grid


# ── Simulación de Monte Carlo ─────────────────────────────────────────────────

def simulacion_monte_carlo(
    retornos: pd.Series,
    n_simulaciones: int = 1000,
    horizonte_dias: int = 252,
    capital_inicial: float = 100_000.0,
    semilla: int = 42,
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
    n_bootstrap: int = 1000,
    nivel_confianza: float = 0.95,
    semilla: int = 42,
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
    n_permutaciones: int = 500,
    semilla: int = 42,
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
