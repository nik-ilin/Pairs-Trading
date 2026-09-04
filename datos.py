"""Yahoo Finance access with exact-query local caching."""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import FIN_DEFAULT, INICIO_DEFAULT, MIN_OBS_HISTORICO, MIN_VOLUMEN_DIARIO

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_FUENTE_SP500 = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
_SP500_FALLBACK = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "BRK-B",
    "JPM",
    "V",
    "MA",
    "WMT",
    "PG",
    "KO",
    "PEP",
    "XOM",
    "CVX",
    "JNJ",
    "MRK",
    "UNH",
    "LLY",
]


def mercado_abierto() -> bool:
    return False


def tiempo_hasta_apertura():
    return None


def verificar_horario_mercado(modo: str = "señales") -> bool:
    return True


def _clave_cache(tickers: list[str], inicio: str, fin: str, tipo: str) -> Path:
    tickers_norm = sorted({str(t).upper() for t in tickers})
    digest = hashlib.sha256("|".join(tickers_norm).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{tipo}_{inicio}_{fin}_{digest}.pkl"


def obtener_sp500() -> list[str]:
    try:
        df = pd.read_csv(_FUENTE_SP500)
        col = next(c for c in df.columns if "symbol" in c.lower())
        if len(df) < 400:
            raise ValueError("universo incompleto")
        return df[col].str.replace(".", "-", regex=False).tolist()
    except Exception as exc:
        print(f"[WARN] No se pudo actualizar el S&P 500 ({exc}); usando lista local reducida.")
        return list(_SP500_FALLBACK)


def obtener_sp500_completo() -> pd.DataFrame:
    try:
        df = pd.read_csv(_FUENTE_SP500)
        columnas = {c.lower(): c for c in df.columns}
        simbolo = next(v for k, v in columnas.items() if "symbol" in k)
        sector = next((v for k, v in columnas.items() if "sector" in k), None)
        nombre = next((v for k, v in columnas.items() if k in {"name", "security"}), None)
        return pd.DataFrame(
            {
                "ticker": df[simbolo].str.replace(".", "-", regex=False),
                "nombre": df[nombre] if nombre else df[simbolo],
                "sector": df[sector] if sector else "Desconocido",
                "subsector": "",
            }
        )
    except Exception:
        return pd.DataFrame({"ticker": obtener_sp500(), "nombre": "", "sector": "Desconocido", "subsector": ""})


def filtrar_universo_interactivo() -> list[str]:
    return obtener_sp500()


def _descargar_yahoo(tickers: list[str], inicio: str, fin: str) -> pd.DataFrame:
    try:
        raw = yf.download(
            sorted(set(tickers)),
            start=inicio,
            end=fin,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Fallo de red o proveedor Yahoo Finance: {exc}") from exc
    if raw.empty:
        raise RuntimeError("Yahoo Finance no devolvió datos para el intervalo solicitado.")
    if isinstance(raw.columns, pd.MultiIndex):
        cierre = raw["Close"]
    else:
        cierre = raw[["Close"]].rename(columns={"Close": tickers[0]})
    cierre.index = pd.to_datetime(cierre.index).tz_localize(None)
    cierre = cierre.sort_index()
    if cierre.dropna(how="all").empty:
        raise RuntimeError("Yahoo Finance devolvió una respuesta sin cierres utilizables.")
    return cierre


def descargar_precios(
    tickers: list[str],
    inicio: str = INICIO_DEFAULT,
    fin: str = FIN_DEFAULT,
    forzar_descarga: bool = False,
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("La lista de tickers no puede estar vacía.")
    ruta = _clave_cache(tickers, inicio, fin, "precios_yahoo")
    if ruta.exists() and not forzar_descarga:
        print(f"[CACHÉ] Consulta exacta: {ruta}")
        return pd.read_pickle(ruta)
    df = _descargar_yahoo(tickers, inicio, fin)
    df.to_pickle(ruta)
    print(f"[OK] Datos Yahoo guardados en {ruta}")
    return df


def descargar_ohlcv(
    tickers: list[str],
    inicio: str,
    fin: str,
    forzar_descarga: bool = False,
) -> dict[str, pd.DataFrame]:
    ruta = _clave_cache(tickers, inicio, fin, "ohlcv_yahoo")
    rutas = {k: ruta.with_name(f"{ruta.stem}_{k}.pkl") for k in ("close", "open", "volume")}
    if not forzar_descarga and all(p.exists() for p in rutas.values()):
        print(f"[CACHÉ] Consulta OHLCV exacta: {ruta.stem}")
        return {k: pd.read_pickle(p) for k, p in rutas.items()}
    try:
        raw = yf.download(
            sorted(set(tickers)),
            start=inicio,
            end=fin,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Fallo de red o proveedor Yahoo Finance: {exc}") from exc
    if raw.empty:
        raise RuntimeError("Yahoo Finance no devolvió OHLCV; no se utilizó caché de otra consulta.")
    resultado = {}
    for nombre, campo in (("close", "Close"), ("open", "Open"), ("volume", "Volume")):
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[campo]
        else:
            df = raw[[campo]].rename(columns={campo: tickers[0]})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        resultado[nombre] = df.sort_index()
        df.to_pickle(rutas[nombre])
    return resultado


def descargar_ohlcv_horario(
    tickers: list[str],
    dias_atras: int = 365,
    forzar_descarga: bool = False,
    inicio: str | None = None,
    fin: str | None = None,
) -> dict[str, pd.DataFrame]:
    fin = fin or datetime.now().strftime("%Y-%m-%d")
    inicio = inicio or (datetime.now() - timedelta(days=dias_atras + 30)).strftime("%Y-%m-%d")
    return descargar_ohlcv(tickers, inicio, fin, forzar_descarga)


def filtrar_datos(
    df_close: pd.DataFrame,
    df_volume: pd.DataFrame | None = None,
    min_obs: int = MIN_OBS_HISTORICO,
    min_vol: float = MIN_VOLUMEN_DIARIO,
) -> pd.DataFrame:
    df = df_close.sort_index().dropna(axis=1, thresh=min_obs)
    if df_volume is not None and not df.empty:
        disponibles = [c for c in df.columns if c in df_volume.columns]
        liquidos = df_volume[disponibles].mean()
        df = df[[c for c in disponibles if liquidos.get(c, 0) >= min_vol]]
    return df.ffill().dropna(how="all")
