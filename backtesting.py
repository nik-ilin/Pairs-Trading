"""Causal pair backtesting, optimization, and walk-forward validation."""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    BS_N_BOOTSTRAP,
    BS_NIVEL_CONFIANZA,
    CALIBRACION_INICIAL,
    CAPITAL_INICIAL,
    COMISION,
    ENTRADA_Z,
    EXPOSICION_BRUTA_MAX,
    FRACCION_RIESGO,
    GRID_ENTRADA_Z,
    GRID_SALIDA_Z,
    GRID_WINDOW,
    HL_MAX_DIAS,
    MC_HORIZONTE_DIAS,
    MC_N_SIMULACIONES,
    MC_SEMILLA,
    MIN_OBS_OPTIMIZACION,
    SALIDA_Z,
    SLIPPAGE,
    STOP_Z,
    USAR_LOG,
    VENTANA_VOL,
    WF_ANOS_DETECCION,
    WF_ANOS_OPERACION,
)
from metricas import (
    cagr,
    calmar_ratio,
    cvar,
    max_drawdown,
    profit_factor,
    reporte_completo,
    sharpe_ratio,
    sortino_ratio,
    var_historico,
    win_rate,
)
from spread import (
    Señal,
    calcular_half_life,
    calcular_spread_kalman,
    calcular_zscore,
    generar_señales,
    tamaño_posicion_volatilidad,
)

warnings.filterwarnings("ignore")


@dataclass
class ParametrosBacktest:
    entrada_z: float = ENTRADA_Z
    salida_z: float = SALIDA_Z
    stop_z: float = STOP_Z
    window_zscore: int | None = None  # None = half-life OU automático
    capital: float = CAPITAL_INICIAL
    fraccion_riesgo: float = FRACCION_RIESGO
    slippage: float = SLIPPAGE
    comision: float = COMISION
    usar_log: bool = USAR_LOG
    exposicion_bruta_max: float = EXPOSICION_BRUTA_MAX
    calibracion_inicial: int = CALIBRACION_INICIAL
    ventana_vol: int = VENTANA_VOL


class MotorBacktest:
    def __init__(
        self,
        precios: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        params: ParametrosBacktest | None = None,
    ):
        self.precios = precios[[ticker1, ticker2]].dropna()
        self.t1 = ticker1
        self.t2 = ticker2
        self.params = params or ParametrosBacktest()

    def ejecutar(self) -> dict:
        p = self.params

        if len(self.precios) < max(20, p.calibracion_inicial + 2):
            raise ValueError("Datos insuficientes para calibración y evaluación causal.")

        # Calibration is observable but non-tradable; all orders execute one bar later.
        spread, beta = calcular_spread_kalman(self.precios, self.t1, self.t2, usar_log=p.usar_log)

        calibracion = max(20, int(p.calibracion_inicial))
        hl = calcular_half_life(spread.iloc[:calibracion])
        if not np.isfinite(hl):
            hl = float(HL_MAX_DIAS)
        window = p.window_zscore or max(5, int(np.ceil(hl)))

        zscore = calcular_zscore(spread, window=window)
        tamaños = tamaño_posicion_volatilidad(
            spread,
            p.capital,
            ventana_vol=p.ventana_vol,
            fraccion=p.fraccion_riesgo,
            exposicion_bruta_max=p.exposicion_bruta_max,
        )
        listo_desde = max(calibracion, window, p.ventana_vol + 1)
        señales = pd.Series(Señal.NINGUNA, index=zscore.index, name="señal")
        señales.iloc[listo_desde:] = generar_señales(zscore.iloc[listo_desde:], p.entrada_z, p.salida_z, p.stop_z)

        pnl_diario = pd.Series(0.0, index=self.precios.index, name="pnl")
        exposicion = pd.Series(0.0, index=self.precios.index, name="exposicion_bruta")
        posiciones = pd.Series(0, index=self.precios.index, dtype=int, name="posicion")
        trades: list[dict] = []
        precios_t1 = self.precios[self.t1]
        precios_t2 = self.precios[self.t2]
        q1 = q2 = 0.0
        posicion_actual = 0
        trade_actual = None
        equity = float(p.capital)
        tasa_coste = p.slippage + p.comision

        def cerrar(i: int, motivo: str) -> None:
            nonlocal q1, q2, posicion_actual, trade_actual, equity
            coste_t1 = abs(q1) * precios_t1.iloc[i] * tasa_coste
            coste_t2 = abs(q2) * precios_t2.iloc[i] * tasa_coste
            coste = coste_t1 + coste_t2
            pnl_diario.iloc[i] -= coste
            equity -= coste
            trade_actual["coste_salida"] = float(coste)
            trade_actual["coste_salida_t1"] = float(coste_t1)
            trade_actual["coste_salida_t2"] = float(coste_t2)
            trade_actual["pnl"] += -float(coste)
            trade_actual["fecha_salida"] = self.precios.index[i]
            trade_actual["duracion_dias"] = (self.precios.index[i] - trade_actual["fecha_entrada"]).days
            trade_actual["motivo_salida"] = motivo
            trades.append(trade_actual)
            q1 = q2 = 0.0
            posicion_actual = 0
            trade_actual = None

        for i in range(1, len(self.precios)):
            fecha = self.precios.index[i]
            pnl_mercado = q1 * (precios_t1.iloc[i] - precios_t1.iloc[i - 1]) + q2 * (
                precios_t2.iloc[i] - precios_t2.iloc[i - 1]
            )
            pnl_diario.iloc[i] += pnl_mercado
            equity += pnl_mercado
            if trade_actual is not None:
                trade_actual["pnl"] += float(pnl_mercado)

            orden = señales.iloc[i - 1]
            if orden == Señal.CERRAR and posicion_actual != 0:
                cerrar(i, "señal")
            elif (
                orden in (Señal.COMPRAR_SPREAD, Señal.VENDER_SPREAD)
                and posicion_actual == 0
                and i < len(self.precios) - 1
            ):
                gross_obj = tamaños.iloc[i - 1]
                if np.isfinite(gross_obj) and gross_obj > 0:
                    gross = min(float(gross_obj), equity * p.exposicion_bruta_max)
                    b = abs(float(beta.iloc[i - 1]))
                    notional1 = gross / (1.0 + b)
                    notional2 = gross - notional1
                    direccion = int(orden)
                    q1 = direccion * notional1 / precios_t1.iloc[i]
                    q2 = -direccion * notional2 / precios_t2.iloc[i]
                    coste_t1 = notional1 * tasa_coste
                    coste_t2 = notional2 * tasa_coste
                    coste = coste_t1 + coste_t2
                    pnl_diario.iloc[i] -= coste
                    equity -= coste
                    posicion_actual = direccion
                    trade_actual = {
                        "fecha_señal": self.precios.index[i - 1],
                        "fecha_entrada": fecha,
                        "direccion": "LONG" if direccion == 1 else "SHORT",
                        "beta_entrada": float(beta.iloc[i - 1]),
                        "notional_t1": float(notional1),
                        "notional_t2": float(notional2),
                        "q_t1": float(q1),
                        "q_t2": float(q2),
                        "coste_entrada": float(coste),
                        "coste_entrada_t1": float(coste_t1),
                        "coste_entrada_t2": float(coste_t2),
                        "pnl": -float(coste),
                    }

            posiciones.iloc[i] = posicion_actual
            exposicion.iloc[i] = abs(q1) * precios_t1.iloc[i] + abs(q2) * precios_t2.iloc[i]

        if posicion_actual != 0:
            cerrar(len(self.precios) - 1, "fin_periodo")
            posiciones.iloc[-1] = 0
            exposicion.iloc[-1] = 0.0

        curva_capital = p.capital + pnl_diario.cumsum()
        capital_previo = curva_capital.shift(1).fillna(p.capital)
        retornos_diarios = (pnl_diario / capital_previo).rename("retorno")
        df_trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        pnl_trades = df_trades["pnl"] if not df_trades.empty else pd.Series(dtype=float)

        metricas_dict = {
            "par": f"{self.t1}/{self.t2}",
            "half_life": round(hl, 1),
            "window_usado": window,
            "sharpe": round(sharpe_ratio(retornos_diarios), 3),
            "sortino": round(sortino_ratio(retornos_diarios), 3),
            "calmar": round(calmar_ratio(retornos_diarios), 3),
            "mdd": round(max_drawdown(retornos_diarios) * 100, 2),
            "cagr": round(cagr(retornos_diarios) * 100, 2),
            "var_95": round(var_historico(retornos_diarios) * 100, 2),
            "cvar_95": round(cvar(retornos_diarios) * 100, 2),
            "n_trades": len(trades),
            "win_rate": round(win_rate(pnl_trades) * 100, 2) if len(pnl_trades) > 0 else 0.0,
            "profit_factor": round(profit_factor(pnl_trades), 3) if len(pnl_trades) > 0 else 0.0,
            "capital_inicial": p.capital,
            "capital_final": float(curva_capital.iloc[-1]),
        }

        return {
            "retornos": retornos_diarios,
            "curva_capital": curva_capital,
            "spread": spread,
            "zscore": zscore,
            "beta": beta,
            "señales": señales,
            "posiciones": posiciones,
            "exposicion_bruta": exposicion,
            "pnl_diario": pnl_diario,
            "trades": df_trades,
            "metricas": metricas_dict,
        }


def _recortar_resultado(res: dict, indice: pd.Index, capital: float) -> dict:
    salida = dict(res)
    pnl = res["pnl_diario"].loc[indice]
    curva = capital + pnl.cumsum()
    retornos = (pnl / curva.shift(1).fillna(capital)).rename("retorno")
    trades = res["trades"]
    if not trades.empty:
        trades = trades[(trades["fecha_entrada"] >= indice[0]) & (trades["fecha_salida"] <= indice[-1])].copy()
    pnls = trades["pnl"] if not trades.empty else pd.Series(dtype=float)
    salida.update({"pnl_diario": pnl, "retornos": retornos, "curva_capital": curva, "trades": trades})
    salida["metricas"] = {
        **res["metricas"],
        "sharpe": round(sharpe_ratio(retornos), 3),
        "sortino": round(sortino_ratio(retornos), 3),
        "calmar": round(calmar_ratio(retornos), 3),
        "mdd": round(max_drawdown(retornos) * 100, 2),
        "cagr": round(cagr(retornos) * 100, 2),
        "n_trades": len(trades),
        "win_rate": round(win_rate(pnls) * 100, 2) if len(pnls) else 0.0,
        "profit_factor": round(profit_factor(pnls), 3) if len(pnls) else 0.0,
        "capital_inicial": capital,
        "capital_final": float(curva.iloc[-1]),
    }
    return salida


def walk_forward(
    precios: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    params: ParametrosBacktest | None = None,
    años_deteccion: int = WF_ANOS_DETECCION,
    años_operacion: int = WF_ANOS_OPERACION,
    verbose: bool = True,
) -> pd.DataFrame:
    fechas = precios.index
    inicio_año = fechas[0].year
    fin_año = fechas[-1].year
    resultados = []

    año_actual = inicio_año
    while año_actual + años_deteccion + años_operacion <= fin_año + 1:
        f_inicio_det = f"{año_actual}-01-01"
        f_fin_det = f"{año_actual + años_deteccion}-01-01"
        f_fin_op = f"{año_actual + años_deteccion + años_operacion}-01-01"

        train = precios[(precios.index >= f_inicio_det) & (precios.index < f_fin_det)]
        test = precios[(precios.index >= f_fin_det) & (precios.index < f_fin_op)]

        if len(train) < 100 or len(test) < 20:
            año_actual += 1
            continue

        params_ventana = ParametrosBacktest(**vars(params)) if params else None
        if params_ventana is None:
            params_ventana, _ = optimizar_parametros(
                train, ticker1, ticker2, fraccion_is=1.0, min_obs_entrenamiento=100
            )
        params_ventana.calibracion_inicial = len(train)
        res_total = MotorBacktest(pd.concat([train, test]), ticker1, ticker2, params_ventana).ejecutar()
        res = _recortar_resultado(res_total, test.index, params_ventana.capital)
        retornos_oos = res["retornos"]
        m = {
            "sharpe": sharpe_ratio(retornos_oos),
            "sortino": sortino_ratio(retornos_oos),
            "mdd": max_drawdown(retornos_oos) * 100,
            "cagr": cagr(retornos_oos) * 100,
            "n_trades": int(
                (res["trades"].get("fecha_entrada", pd.Series(dtype="datetime64[ns]")) >= test.index[0]).sum()
            ),
            "win_rate": 0.0,
        }

        resultados.append(
            {
                "ventana": f"{año_actual}-{año_actual + años_deteccion} → op. {año_actual + años_deteccion}",
                "sharpe": m["sharpe"],
                "sortino": m["sortino"],
                "mdd": m["mdd"],
                "cagr": m["cagr"],
                "n_trades": m["n_trades"],
                "win_rate": m["win_rate"],
                "inicio_train": train.index[0],
                "fin_train": train.index[-1],
                "inicio_test": test.index[0],
                "fin_test": test.index[-1],
                "retornos_oos": retornos_oos,
            }
        )

        if verbose:
            print(f"  Ventana {resultados[-1]['ventana']} | Sharpe: {m['sharpe']:.2f} | MDD: {m['mdd']:.1f}%")

        año_actual += 1

    return pd.DataFrame(resultados) if resultados else pd.DataFrame()


def optimizar_parametros(
    precios: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    grid: dict | None = None,
    metrica: str = "sharpe",
    fraccion_is: float = 0.7,
    min_obs_entrenamiento: int = MIN_OBS_OPTIMIZACION,
) -> tuple[ParametrosBacktest, pd.DataFrame]:
    if grid is None:
        grid = {
            "entrada_z": GRID_ENTRADA_Z,
            "salida_z": GRID_SALIDA_Z,
            "window_zscore": GRID_WINDOW,
        }

    n_is = int(len(precios) * fraccion_is)
    precios_is = precios.iloc[:n_is]

    if len(precios_is) < min_obs_entrenamiento:
        raise ValueError(
            f"Datos insuficientes para grid search: {len(precios_is)} barras "
            f"(mínimo {min_obs_entrenamiento}). Usa más histórico."
        )

    combinaciones = [
        (e, s, w) for e in grid["entrada_z"] for s in grid["salida_z"] for w in grid["window_zscore"] if s < e
    ]

    print(
        f"[INFO] Grid search: {len(combinaciones)} combinaciones "
        f"(IS: {len(precios_is)} barras | OOS: {len(precios) - n_is} barras)..."
    )

    resultados = []
    for entrada, salida, window in combinaciones:
        p = ParametrosBacktest(entrada_z=entrada, salida_z=salida, window_zscore=window)
        try:
            res = MotorBacktest(precios_is, ticker1, ticker2, p).ejecutar()
            m = res["metricas"]
            resultados.append(
                {
                    "entrada_z": entrada,
                    "salida_z": salida,
                    "window_zscore": window,
                    "sharpe": m["sharpe"],
                    "sortino": m["sortino"],
                    "calmar": m["calmar"],
                    "mdd": m["mdd"],
                    "n_trades": m["n_trades"],
                }
            )
        except Exception:
            continue

    if not resultados:
        raise RuntimeError("Ninguna combinación del grid produjo resultados válidos.")

    df_grid = pd.DataFrame(resultados).sort_values(metrica, ascending=False)
    mejor = df_grid.iloc[0]

    params_optimos = ParametrosBacktest(
        entrada_z=mejor["entrada_z"],
        salida_z=mejor["salida_z"],
        window_zscore=None if pd.isna(mejor["window_zscore"]) else int(mejor["window_zscore"]),
    )
    precios_oos = precios.iloc[n_is:]
    df_grid.attrs["indice_corte"] = n_is
    if len(precios_oos) >= 2:
        params_eval = ParametrosBacktest(**vars(params_optimos))
        params_eval.calibracion_inicial = len(precios_is)
        total = MotorBacktest(precios, ticker1, ticker2, params_eval).ejecutar()
        df_grid.attrs["resultado_oos"] = _recortar_resultado(total, precios_oos.index, params_eval.capital)
    else:
        df_grid.attrs["resultado_oos"] = None

    print(
        f"[OK] Mejores parametros (IS): entrada={mejor['entrada_z']}, "
        f"salida={mejor['salida_z']}, window={mejor['window_zscore']} "
        f"-> {metrica.upper()} = {mejor[metrica]:.3f}"
    )

    return params_optimos, df_grid


def simulacion_monte_carlo(
    retornos: pd.Series,
    n_simulaciones: int = MC_N_SIMULACIONES,
    horizonte_dias: int = MC_HORIZONTE_DIAS,
    capital_inicial: float = CAPITAL_INICIAL,
    semilla: int = MC_SEMILLA,
) -> dict:
    rng = np.random.default_rng(semilla)
    ret_arr = retornos.dropna().values

    finales = np.zeros(n_simulaciones)
    ddrawdwn = np.zeros(n_simulaciones)

    for i in range(n_simulaciones):
        muestra = rng.choice(ret_arr, size=horizonte_dias, replace=True)
        curva = capital_inicial * np.cumprod(1 + muestra)
        finales[i] = curva[-1]
        pico = np.maximum.accumulate(curva)
        ddrawdwn[i] = np.min((curva - pico) / pico)

    percentiles = [5, 25, 50, 75, 95]
    return {
        "capital_final": pd.Series(finales),
        "max_drawdown_sim": pd.Series(ddrawdwn),
        "percentiles_capital": dict(zip([f"p{p}" for p in percentiles], np.percentile(finales, percentiles))),
        "prob_ganancia": float((finales > capital_inicial).mean()),
        "capital_medio": float(np.mean(finales)),
        "n_simulaciones": n_simulaciones,
    }


def bootstrap_sharpe(
    retornos: pd.Series,
    n_bootstrap: int = BS_N_BOOTSTRAP,
    nivel_confianza: float = BS_NIVEL_CONFIANZA,
    semilla: int = MC_SEMILLA,
) -> dict:
    rng = np.random.default_rng(semilla)
    ret_arr = retornos.dropna().values
    sharpes = []

    for _ in range(n_bootstrap):
        muestra = rng.choice(ret_arr, size=len(ret_arr), replace=True)
        serie = pd.Series(muestra)
        sharpes.append(sharpe_ratio(serie))

    alpha = (1 - nivel_confianza) / 2
    return {
        "sharpe_observado": round(sharpe_ratio(retornos), 3),
        "ci_inferior": round(float(np.percentile(sharpes, alpha * 100)), 3),
        "ci_superior": round(float(np.percentile(sharpes, (1 - alpha) * 100)), 3),
        "sharpe_medio": round(float(np.mean(sharpes)), 3),
    }


def backtest_completo(
    precios: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    params: ParametrosBacktest | None = None,
    imprimir_reporte: bool = True,
) -> dict:
    params = params or ParametrosBacktest()
    motor = MotorBacktest(precios, ticker1, ticker2, params)
    res = motor.ejecutar()

    retornos = res["retornos"]

    mc = simulacion_monte_carlo(retornos)
    bs = bootstrap_sharpe(retornos)

    if imprimir_reporte:
        reporte_completo(
            retornos,
            res["trades"].get("pnl") if not res["trades"].empty else None,
            nombre=f"{ticker1}/{ticker2}",
        )
        print("\n[Monte Carlo]")
        print(f"  P(ganancia)          : {mc['prob_ganancia'] * 100:.1f}%")
        print(f"  Capital medio final  : ${mc['capital_medio']:,.0f}")
        p5 = mc["percentiles_capital"]["p5"]
        p95 = mc["percentiles_capital"]["p95"]
        print(f"  Capital P5 / P95     : ${p5:,.0f} / ${p95:,.0f}")
        print(f"\n[Bootstrap Sharpe {bs['ci_inferior']} – {bs['ci_superior']}]")
        print(f"  Sharpe observado     : {bs['sharpe_observado']}")
    res.update({"monte_carlo": mc, "bootstrap_sharpe": bs})
    return res
