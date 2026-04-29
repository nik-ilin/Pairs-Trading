"""
datos.py — Capa de descarga y caché de precios históricos.

Responsabilidades:
  - Descargar precios ajustados desde Yahoo Finance (2008-2026).
  - Almacenar en caché local (Parquet) para evitar re-descargas.
  - Proveer funciones de filtrado y limpieza usadas por todos los módulos.
"""

import os
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ── Directorio de caché ──────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

INICIO_DEFAULT = "2008-01-01"
FIN_DEFAULT    = "2026-01-01"
MIN_OBS        = 1260  # ~5 años de datos diarios mínimos por ticker


# ── Universo de activos ──────────────────────────────────────────────────────

def obtener_sp500():
    """Descarga la lista actual del S&P 500 desde Wikipedia."""
    try:
        tabla = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return tabla["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        # Lista de respaldo con los 50 mayores componentes
        return [
            "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","LLY","AVGO","JPM",
            "V","UNH","XOM","TSLA","MA","JNJ","PG","HD","MRK","COST","ABBV","CVX",
            "CRM","BAC","NFLX","KO","PEP","WMT","TMO","LIN","ABT","ORCL","ACN",
            "MCD","PM","CSCO","IBM","TXN","NEE","RTX","QCOM","GE","SPGI","HON",
            "AMAT","CAT","GS","AMGN","ISRG","BKNG",
        ]


def obtener_tickers_url():
    """Descarga tickers ampliados desde GitHub (fallback al S&P 500)."""
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
    try:
        tickers = pd.read_csv(url, header=None)[0].tolist()
        return [t for t in tickers if isinstance(t, str) and 1 <= len(t) <= 5]
    except Exception:
        return obtener_sp500()


# ── Descarga y caché ─────────────────────────────────────────────────────────

def _ruta_cache(nombre: str) -> str:
    return os.path.join(CACHE_DIR, f"{nombre}.parquet")


def descargar_precios(
    tickers: list[str],
    inicio: str = INICIO_DEFAULT,
    fin: str = FIN_DEFAULT,
    forzar_descarga: bool = False,
) -> pd.DataFrame:
    """
    Devuelve DataFrame de precios de cierre ajustados (filas=fechas, columnas=tickers).
    Usa caché local; solo descarga si el archivo no existe o forzar_descarga=True.
    """
    nombre_cache = f"precios_{inicio[:4]}_{fin[:4]}"
    ruta = _ruta_cache(nombre_cache)

    if not forzar_descarga and os.path.exists(ruta):
        print(f"[CACHÉ] Cargando precios desde {ruta}")
        df = pd.read_parquet(ruta)
        # Añadir tickers faltantes si el universo creció
        faltantes = [t for t in tickers if t not in df.columns]
        if faltantes:
            print(f"[INFO] Descargando {len(faltantes)} tickers nuevos...")
            df_nuevo = _descargar_batch(faltantes, inicio, fin)
            df = pd.concat([df, df_nuevo], axis=1)
            df.to_parquet(ruta)
        return df[sorted(df.columns)]

    print(f"[INFO] Descargando precios para {len(tickers)} tickers ({inicio} → {fin})...")
    df = _descargar_batch(tickers, inicio, fin)
    df.to_parquet(ruta)
    print(f"[OK] Precios guardados en {ruta}")
    return df


def _descargar_batch(tickers: list[str], inicio: str, fin: str) -> pd.DataFrame:
    """Descarga en lotes de 100 para evitar límites de rate de yfinance."""
    batch_size = 100
    frames = []
    for i in range(0, len(tickers), batch_size):
        lote = tickers[i : i + batch_size]
        try:
            raw = yf.download(
                lote,
                start=inicio,
                end=fin,
                auto_adjust=True,   # precios ajustados por splits y dividendos
                progress=False,
                threads=True,
            )
            # yfinance devuelve MultiIndex cuando hay >1 ticker
            if isinstance(raw.columns, pd.MultiIndex):
                cierre = raw["Close"]
            else:
                cierre = raw[["Close"]].rename(columns={"Close": lote[0]})
            frames.append(cierre)
        except Exception as e:
            print(f"  [WARN] Error en lote {i}-{i+batch_size}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


# ── Limpieza ─────────────────────────────────────────────────────────────────

def filtrar_datos(df: pd.DataFrame, min_obs: int = MIN_OBS) -> pd.DataFrame:
    """Elimina tickers con datos insuficientes y rellena huecos menores."""
    # Eliminar columnas con demasiados NaN
    df = df.dropna(axis=1, thresh=min_obs)
    # Rellenar huecos cortos (fines de semana, festivos) con forward-fill
    df = df.ffill().bfill()
    return df


def preparar_par(df: pd.DataFrame, t1: str, t2: str) -> pd.DataFrame | None:
    """Extrae y limpia el par (t1, t2); devuelve None si no hay suficientes datos."""
    if t1 not in df.columns or t2 not in df.columns:
        return None
    par = df[[t1, t2]].dropna()
    if len(par) < MIN_OBS:
        return None
    return par


def dividir_muestra(
    df: pd.DataFrame,
    corte: str = "2020-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide en in-sample (detección de pares) y out-of-sample (backtesting).
    Por defecto: 2008-2020 in-sample | 2020-2026 out-of-sample.
    """
    in_sample  = df[df.index < corte]
    out_sample = df[df.index >= corte]
    return in_sample, out_sample
