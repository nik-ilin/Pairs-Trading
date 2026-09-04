import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import backtesting
from backtesting import MotorBacktest, ParametrosBacktest, optimizar_parametros, walk_forward
from deteccion import estabilidad_rolling
from deteccion import test_engle_granger as engle_granger
from spread import Señal, tamaño_posicion_volatilidad


def precios_sinteticos(n=700, seed=7):
    rng = np.random.default_rng(seed)
    x = 4.5 + np.cumsum(rng.normal(0, 0.008, n))
    ruido = np.zeros(n)
    for i in range(1, n):
        ruido[i] = 0.85 * ruido[i - 1] + rng.normal(0, 0.01)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({"A": np.exp(x + ruido), "B": np.exp(x)}, index=idx)


def test_deteccion_cointegrado_y_no_cointegrado():
    cointegrado = precios_sinteticos(500)
    rng = np.random.default_rng(21)
    no_coint = pd.DataFrame(
        {
            "A": np.exp(4 + np.cumsum(rng.normal(0, 0.015, 500))),
            "B": np.exp(4 + np.cumsum(rng.normal(0, 0.015, 500))),
        },
        index=cointegrado.index,
    )
    assert engle_granger(cointegrado.A, cointegrado.B)["pasa"]
    assert not engle_granger(no_coint.A, no_coint.B)["pasa"]


def test_invariancia_frente_a_datos_futuros():
    precios = precios_sinteticos(500)
    params = ParametrosBacktest(window_zscore=30, calibracion_inicial=60, ventana_vol=20)
    base = MotorBacktest(precios, "A", "B", params).ejecutar()
    alterado = precios.copy()
    corte = 350
    alterado.iloc[corte:, 0] *= np.linspace(1, 3, len(alterado) - corte)
    nuevo = MotorBacktest(alterado, "A", "B", params).ejecutar()
    pdt.assert_series_equal(base["señales"].iloc[:corte], nuevo["señales"].iloc[:corte])
    pdt.assert_series_equal(base["posiciones"].iloc[:corte], nuevo["posiciones"].iloc[:corte])


def test_sizing_lagged_y_acotado():
    rng = np.random.default_rng(4)
    s = pd.Series(np.cumsum(rng.normal(size=100)), index=pd.bdate_range("2020-01-01", periods=100))
    size = tamaño_posicion_volatilidad(s, capital=10_000, ventana_vol=20, fraccion=0.1, exposicion_bruta_max=0.4)
    assert size.iloc[:21].isna().all()
    assert size.dropna().max() <= 4_000


def test_siguiente_bar_costes_y_reconciliacion(monkeypatch):
    precios = precios_sinteticos(140)
    señal_idx = 70
    fecha_señal = precios.index[señal_idx]
    fecha_cierre = precios.index[señal_idx + 5]

    def señales_stub(z, *args):
        out = pd.Series(Señal.NINGUNA, index=z.index)
        out.loc[fecha_señal] = Señal.COMPRAR_SPREAD
        out.loc[fecha_cierre] = Señal.CERRAR
        return out

    monkeypatch.setattr(backtesting, "generar_señales", señales_stub)
    monkeypatch.setattr(
        backtesting,
        "tamaño_posicion_volatilidad",
        lambda spread, *a, **k: pd.Series(5_000.0, index=spread.index),
    )
    p = ParametrosBacktest(
        window_zscore=20,
        calibracion_inicial=30,
        slippage=0.001,
        comision=0.001,
        exposicion_bruta_max=0.5,
    )
    res = MotorBacktest(precios, "A", "B", p).ejecutar()
    trade = res["trades"].iloc[0]
    assert trade.fecha_entrada == precios.index[señal_idx + 1]
    assert trade.fecha_salida == precios.index[señal_idx + 6]
    assert trade.coste_entrada == pytest.approx(5_000 * 0.002)
    assert trade.coste_entrada == pytest.approx(trade.coste_entrada_t1 + trade.coste_entrada_t2)
    assert trade.coste_salida == pytest.approx(trade.coste_salida_t1 + trade.coste_salida_t2)
    assert trade.notional_t1 + trade.notional_t2 == pytest.approx(5_000)
    assert res["exposicion_bruta"].max() <= p.capital * p.exposicion_bruta_max
    assert res["curva_capital"].iloc[-1] - p.capital == pytest.approx(res["trades"].pnl.sum())
    assert res["pnl_diario"].sum() == pytest.approx(res["trades"].pnl.sum())
    assert p.capital * (1 + res["retornos"]).prod() == pytest.approx(res["curva_capital"].iloc[-1])


def test_cierre_forzoso_se_registra(monkeypatch):
    precios = precios_sinteticos(120)
    fecha_señal = precios.index[70]

    def señales_stub(z, *args):
        out = pd.Series(Señal.NINGUNA, index=z.index)
        out.loc[fecha_señal] = Señal.VENDER_SPREAD
        return out

    monkeypatch.setattr(backtesting, "generar_señales", señales_stub)
    monkeypatch.setattr(
        backtesting,
        "tamaño_posicion_volatilidad",
        lambda s, *a, **k: pd.Series(2_000.0, index=s.index),
    )
    res = MotorBacktest(precios, "A", "B", ParametrosBacktest(window_zscore=20, calibracion_inicial=30)).ejecutar()
    assert res["trades"].iloc[-1].motivo_salida == "fin_periodo"
    assert res["posiciones"].iloc[-1] == 0


def test_walk_forward_solo_cose_oos():
    precios = precios_sinteticos(1800)
    wf = walk_forward(
        precios,
        "A",
        "B",
        ParametrosBacktest(window_zscore=30),
        años_deteccion=2,
        años_operacion=1,
        verbose=False,
    )
    assert not wf.empty
    for _, fila in wf.iterrows():
        assert fila.fin_train < fila.inicio_test
        assert fila.retornos_oos.index.min() >= fila.inicio_test
        assert fila.retornos_oos.index.max() <= fila.fin_test


def test_optimizacion_reporta_solo_test_no_tocado():
    precios = precios_sinteticos(1000)
    _, tabla = optimizar_parametros(
        precios,
        "A",
        "B",
        grid={"entrada_z": [1.5], "salida_z": [0.5], "window_zscore": [30]},
        fraccion_is=0.75,
        min_obs_entrenamiento=500,
    )
    corte = tabla.attrs["indice_corte"]
    resultado = tabla.attrs["resultado_oos"]
    assert resultado["retornos"].index.equals(precios.index[corte:])
    assert resultado["curva_capital"].index.equals(precios.index[corte:])


def test_rolling_corto_no_exige_cinco_anos():
    precios = precios_sinteticos(126)
    rolling = estabilidad_rolling(precios, "A", "B", ventana=60)
    assert not rolling.empty
