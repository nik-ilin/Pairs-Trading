"""Causal spread estimation, signals, and volatility-based sizing."""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from config import (
    CAPITAL_INICIAL,
    ENTRADA_Z,
    FRACCION_RIESGO,
    HL_MAX_DIAS,
    HL_MIN_DIAS,
    HORAS_DIA,
    KALMAN_DELTA,
    KALMAN_VAR_OBS,
    SALIDA_Z,
    STOP_Z,
    VENTANA_VOL,
)


def _bpd(index: pd.DatetimeIndex) -> float:
    if len(index) < 10:
        return 1.0
    n_dias = pd.Series(index).dt.date.nunique()
    return HORAS_DIA if (len(index) / max(n_dias, 1)) > 2 else 1.0


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = KALMAN_DELTA,
    var_obs: float = KALMAN_VAR_OBS,
) -> pd.Series:
    n = len(y)
    y_arr = y.values
    x_arr = x.values

    Vw = delta / (1.0 - delta)
    Ve = var_obs

    # Neutral prior avoids estimating the initial state from future observations.
    theta = np.ones(n)
    P = np.ones(n)
    P[0] = 1.0

    for t in range(1, n):
        theta_pred = theta[t - 1]
        P_pred = P[t - 1] + Vw

        F = x_arr[t]

        innovacion = y_arr[t] - F * theta_pred
        S = F * P_pred * F + Ve

        K = P_pred * F / S

        theta[t] = theta_pred + K * innovacion
        P[t] = (1.0 - K * F) * P_pred

    return pd.Series(theta, index=y.index, name="beta_kalman")


def calcular_half_life(spread: pd.Series) -> float:
    delta_s = spread.diff().dropna()
    s_lag = spread.shift(1).dropna()
    s_lag = s_lag.loc[delta_s.index]

    X = add_constant(s_lag.values)
    ols = OLS(delta_s.values, X).fit()
    b = ols.params[1]

    if b >= 0:
        return np.inf

    bpd = _bpd(spread.index)
    half_life = float(-np.log(2) / b)
    return max(HL_MIN_DIAS * bpd, min(half_life, HL_MAX_DIAS * bpd))


def parametros_ou(spread: pd.Series) -> dict:
    delta_s = spread.diff().dropna()
    s_lag = spread.shift(1).dropna().loc[delta_s.index]

    X = add_constant(s_lag.values)
    ols = OLS(delta_s.values, X).fit()
    a, b = ols.params[0], ols.params[1]

    bpd = _bpd(spread.index)
    kappa = max(-b, 1e-8)
    mu = -a / b if b != 0 else float(spread.mean())
    sigma_res = float(ols.resid.std())
    half_life = float(np.log(2) / kappa)
    half_life = max(HL_MIN_DIAS * bpd, min(half_life, HL_MAX_DIAS * bpd))

    return {"kappa": kappa, "mu": mu, "sigma": sigma_res, "half_life": half_life}


def calcular_spread_kalman(
    precios: pd.DataFrame,
    t1: str,
    t2: str,
    usar_log: bool = True,
) -> tuple[pd.Series, pd.Series]:
    s1 = np.log(precios[t1]) if usar_log else precios[t1]
    s2 = np.log(precios[t2]) if usar_log else precios[t2]

    beta = kalman_hedge_ratio(s1, s2)
    spread = s1 - beta * s2

    return spread, beta


def calcular_zscore(spread: pd.Series, window: int | None = None) -> pd.Series:
    if window is None:
        hl = calcular_half_life(spread)
        window = int(np.ceil(hl))

    media = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    zscore = (spread - media) / std
    return zscore.rename("zscore")


class Señal:
    COMPRAR_SPREAD = +1
    VENDER_SPREAD = -1
    CERRAR = 0
    NINGUNA = 2


def generar_señales(
    zscore: pd.Series,
    entrada: float = ENTRADA_Z,
    salida: float = SALIDA_Z,
    stop: float = STOP_Z,
) -> pd.Series:
    señales = pd.Series(Señal.NINGUNA, index=zscore.index, name="señal")
    posicion = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z):
            continue

        if posicion != 0 and abs(z) > stop:
            señales.iloc[i] = Señal.CERRAR
            posicion = 0

        elif posicion == 1 and z >= -salida:
            señales.iloc[i] = Señal.CERRAR
            posicion = 0

        elif posicion == -1 and z <= +salida:
            señales.iloc[i] = Señal.CERRAR
            posicion = 0

        elif posicion == 0 and z < -entrada:
            señales.iloc[i] = Señal.COMPRAR_SPREAD
            posicion = 1

        elif posicion == 0 and z > +entrada:
            señales.iloc[i] = Señal.VENDER_SPREAD
            posicion = -1

    return señales


def tamaño_posicion_volatilidad(
    spread: pd.Series,
    capital: float = CAPITAL_INICIAL,
    ventana_vol: int | None = None,
    fraccion: float = FRACCION_RIESGO,
    exposicion_bruta_max: float = 1.0,
) -> pd.Series:
    if ventana_vol is None:
        ventana_vol = max(5, int(VENTANA_VOL * _bpd(spread.index)))
    # A size chosen at t may use volatility observed only through t-1.
    vol = spread.diff().rolling(ventana_vol, min_periods=ventana_vol).std().shift(1)
    vol = vol.replace(0, np.nan)
    objetivo = capital * fraccion
    tamaño = (objetivo / vol).clip(upper=capital * exposicion_bruta_max)
    return tamaño.rename("tamaño_posicion")
