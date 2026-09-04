import importlib
import sys

import pandas as pd
import pytest


def test_imports_sin_red(monkeypatch):
    import yfinance

    monkeypatch.setattr(yfinance, "download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("red")))
    for nombre in (
        "config",
        "datos",
        "deteccion",
        "spread",
        "metricas",
        "backtesting",
        "automatizacion",
        "evaluacion",
        "main",
    ):
        importlib.import_module(nombre)


def test_cli_help_sin_red(monkeypatch, capsys):
    import main

    monkeypatch.setattr(sys, "argv", ["main.py", "--help"])
    try:
        main.main()
    except SystemExit as exc:
        assert exc.code == 0
    assert "--modo" in capsys.readouterr().out


def test_fallo_yahoo_es_explicito_y_no_usa_cache_distinta(monkeypatch, tmp_path):
    import datos

    monkeypatch.setattr(datos, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(datos.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no devolvió datos"):
        datos.descargar_precios(["AAPL"], "2024-01-01", "2024-02-01")


def test_cache_exacta_no_accede_a_red(monkeypatch, tmp_path):
    import datos

    monkeypatch.setattr(datos, "CACHE_DIR", tmp_path)
    indice = pd.bdate_range("2024-01-01", periods=3)
    esperado = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0]}, index=indice)
    ruta = datos._clave_cache(["AAPL"], "2024-01-01", "2024-02-01", "precios_yahoo")
    esperado.to_pickle(ruta)
    monkeypatch.setattr(datos.yf, "download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("red")))
    obtenido = datos.descargar_precios(["AAPL"], "2024-01-01", "2024-02-01")
    pd.testing.assert_frame_equal(obtenido, esperado)
