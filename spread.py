"""
spread.py — Cálculo del spread, ratio de cobertura dinámico y señales de trading.

Modelos matemáticos implementados:
  - Filtro de Kalman      : hedge ratio dinámico que se adapta en el tiempo.
  - Proceso Ornstein-Uhlenbeck (OU) : estima la velocidad de reversión a la media
                            y calcula la vida media (half-life) del spread.
  - Z-score rolling       : normaliza el spread usando la ventana OU óptima.
  - Señales entrada/salida: basadas en umbrales estadísticos del z-score.

Por qué Kalman en lugar de OLS estático:
  OLS asume una relación constante entre los dos activos. En la práctica, el ratio
  de cobertura (beta) evoluciona con el tiempo debido a cambios estructurales del
  negocio, ciclos económicos y rotaciones sectoriales. El Filtro de Kalman modela
  el beta como un proceso de paseo aleatorio y lo actualiza día a día con cada nueva
  observación, produciendo un spread más estacionario y señales más fiables.

Por qué OU en lugar de ventana fija:
  La ventana del z-score debe reflejar la velocidad real de reversión del spread.
  Si el spread tarda 30 días en volver a la media, usar una ventana de 90 días
  diluye la señal. El proceso OU calcula el half-life empírico del spread para
  fijar la ventana óptima automáticamente.
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from config import (
    KALMAN_DELTA, KALMAN_VAR_OBS,
    HL_MIN_DIAS, HL_MAX_DIAS, HORAS_DIA,
    ENTRADA_Z, SALIDA_Z, STOP_Z,
    VENTANA_VOL, CAPITAL_INICIAL, FRACCION_RIESGO,
)


def _bpd(index: pd.DatetimeIndex) -> float:
    """Barras por día de negociación: HORAS_DIA para horario, 1.0 para diario."""
    if len(index) < 10:
        return 1.0
    n_dias = pd.Series(index).dt.date.nunique()
    return HORAS_DIA if (len(index) / max(n_dias, 1)) > 2 else 1.0


# ── Filtro de Kalman para hedge ratio dinámico ────────────────────────────────

def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = KALMAN_DELTA,
    var_obs: float = KALMAN_VAR_OBS,
) -> pd.Series:
    """
    Estima el ratio de cobertura (beta) dinámico mediante el Filtro de Kalman.
    Estado 1D: solo beta. El z-score rolling se encarga del centrado.

    Modelo de espacio de estados:
      Estado:       θ_t = beta_t  — paseo aleatorio
      Observación:  y_t = x_t * beta_t + ε_t

    Parámetros:
      delta   : velocidad de adaptación del estado (mayor = más reactivo)
      var_obs : varianza del ruido de observación

    Devuelve:
      beta_series : Serie de ratios de cobertura dinámicos
    """
    n = len(y)
    y_arr = y.values
    x_arr = x.values

    Vw = delta / (1.0 - delta)  # escalar
    Ve = var_obs

    theta = np.zeros(n)   # solo beta
    P     = np.ones(n)    # varianza escalar
    P[0]  = 1.0

    # Warmup: OLS sobre los primeros 60 días de negociación
    warmup = int(60 * _bpd(y.index))
    if n >= warmup:
        ols = OLS(y_arr[:warmup], x_arr[:warmup]).fit()
        theta[0] = float(ols.params[0])

    for t in range(1, n):
        theta_pred = theta[t - 1]
        P_pred     = P[t - 1] + Vw

        F = x_arr[t]  # escalar

        innovacion = y_arr[t] - F * theta_pred
        S          = F * P_pred * F + Ve

        K = P_pred * F / S

        theta[t] = theta_pred + K * innovacion
        P[t]     = (1.0 - K * F) * P_pred

    return pd.Series(theta, index=y.index, name="beta_kalman")


# ── Proceso Ornstein-Uhlenbeck — half-life de reversión ──────────────────────

def calcular_half_life(spread: pd.Series) -> float:
    """
    Estima el half-life (vida media) del spread usando el proceso OU discretizado:

      ΔS_t = κ·(μ - S_{t-1})·Δt + σ·ε_t

    En forma de regresión:
      ΔS_t = a + b·S_{t-1} + ε_t
      donde b = -κ·Δt  →  κ = -b
      half-life = ln(2) / κ

    Un half-life de 20 días significa que el spread tarda ~20 días en recorrer
    la mitad del camino hacia su media.
    """
    delta_s = spread.diff().dropna()
    s_lag   = spread.shift(1).dropna()
    s_lag   = s_lag.loc[delta_s.index]

    X    = add_constant(s_lag.values)
    ols  = OLS(delta_s.values, X).fit()
    b    = ols.params[1]

    if b >= 0:
        # Sin reversión a la media → spread no es estacionario
        return np.inf

    bpd = _bpd(spread.index)
    half_life = float(-np.log(2) / b)
    return max(HL_MIN_DIAS * bpd, min(half_life, HL_MAX_DIAS * bpd))


def parametros_ou(spread: pd.Series) -> dict:
    """Devuelve parámetros completos del proceso OU: κ, μ, σ y half-life."""
    delta_s = spread.diff().dropna()
    s_lag   = spread.shift(1).dropna().loc[delta_s.index]

    X   = add_constant(s_lag.values)
    ols = OLS(delta_s.values, X).fit()
    a, b = ols.params[0], ols.params[1]

    bpd       = _bpd(spread.index)
    kappa     = max(-b, 1e-8)
    mu        = -a / b if b != 0 else float(spread.mean())
    sigma_res = float(ols.resid.std())
    half_life = float(np.log(2) / kappa)
    half_life = max(HL_MIN_DIAS * bpd, min(half_life, HL_MAX_DIAS * bpd))

    return {"kappa": kappa, "mu": mu, "sigma": sigma_res, "half_life": half_life}


# ── Cálculo del spread y z-score ─────────────────────────────────────────────

def calcular_spread_kalman(
    precios: pd.DataFrame,
    t1: str,
    t2: str,
    usar_log: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """
    Calcula el spread usando el Filtro de Kalman para el ratio dinámico.

    Devuelve:
      spread      : serie del spread (s1 - beta * s2)
      beta_series : ratio de cobertura dinámico
    """
    s1 = np.log(precios[t1]) if usar_log else precios[t1]
    s2 = np.log(precios[t2]) if usar_log else precios[t2]

    beta   = kalman_hedge_ratio(s1, s2)
    spread = s1 - beta * s2

    return spread, beta


def calcular_zscore(spread: pd.Series, window: int | None = None) -> pd.Series:
    """
    Z-score rolling del spread.
    Si window=None, la ventana se calcula automáticamente con el half-life OU.
    """
    if window is None:
        hl = calcular_half_life(spread)
        window = int(np.ceil(hl))

    media = spread.rolling(window).mean()
    std   = spread.rolling(window).std()
    zscore = (spread - media) / std
    return zscore.rename("zscore")


# ── Generación de señales de trading ─────────────────────────────────────────

class Señal:
    COMPRAR_SPREAD  = +1   # long t1 / short t2
    VENDER_SPREAD   = -1   # short t1 / long t2
    CERRAR          =  0
    NINGUNA         =  2   # sin posición ni señal


def generar_señales(
    zscore: pd.Series,
    entrada: float = ENTRADA_Z,
    salida: float  = SALIDA_Z,
    stop: float    = STOP_Z,
) -> pd.Series:
    """
    Genera señales de posición basadas en umbrales del z-score.

    Lógica:
      - Entrada LONG spread  : zscore < -entrada  (spread barato)
      - Entrada SHORT spread : zscore > +entrada  (spread caro)
      - Cierre de posición   : |zscore| < salida  (reversión completada)
      - Stop-loss            : |zscore| > stop    (spread se aleja más)

    Devuelve Serie con valores {+1, -1, 0, 2}:
      +1 = COMPRAR spread (long t1 / short t2)
      -1 = VENDER spread  (short t1 / long t2)
       0 = CERRAR posición
       2 = SIN SEÑAL (mantener estado actual)
    """
    señales    = pd.Series(Señal.NINGUNA, index=zscore.index, name="señal")
    posicion   = 0  # estado actual: +1, -1 o 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z):
            continue

        # Stop-loss: cerrar si el spread se alejó demasiado
        if posicion != 0 and abs(z) > stop:
            señales.iloc[i] = Señal.CERRAR
            posicion = 0

        # Cierre por reversión
        elif posicion == 1 and z >= -salida:
            señales.iloc[i] = Señal.CERRAR
            posicion = 0

        elif posicion == -1 and z <= +salida:
            señales.iloc[i] = Señal.CERRAR
            posicion = 0

        # Apertura de nuevas posiciones (solo si no hay posición abierta)
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
) -> pd.Series:
    """
    Sizing basado en volatilidad inversa del spread (volatility scaling).

    Tamaño = (capital × fracción) / σ_rolling(spread)

    Principio: cuanto más volátil el spread, menor la posición para mantener
    una exposición al riesgo constante en términos de unidades monetarias.
    """
    if ventana_vol is None:
        ventana_vol = max(5, int(VENTANA_VOL * _bpd(spread.index)))
    vol      = spread.rolling(ventana_vol).std()
    vol      = vol.replace(0, np.nan).ffill()
    fallback = spread.std()
    vol      = vol.fillna(fallback if fallback > 0 else 1.0)
    tamaño   = (capital * fraccion) / vol
    return tamaño.rename("tamaño_posicion")
