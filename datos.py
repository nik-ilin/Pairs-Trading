"""
datos.py — Capa de descarga y caché de precios históricos.

Fuentes de datos:
  - yfinance  → histórico diario 2008-2020 (in-sample, detección de pares)
  - Alpaca    → histórico diario 2020-hoy  (out-of-sample, backtest y señales)
  - Alpaca    → intradiario 1Min-1Hour     (señales en tiempo real)

Usar descargar_precios_combinado() como punto de entrada principal.
Las credenciales de Alpaca van en .env (ALPACA_API_KEY, ALPACA_API_SECRET).
"""

import os
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Cargar variables de entorno desde .env ───────────────────────────────────

from config import (
    ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_INTERVALO,
    INICIO_DEFAULT, FIN_DEFAULT, MIN_OBS, CORTE_IN_SAMPLE,
)

# True si las credenciales de Alpaca están configuradas
_ALPACA_DISPONIBLE = bool(ALPACA_API_KEY and ALPACA_API_KEY != "tu_api_key_aqui")

# ── Directorio de caché ──────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


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


def obtener_sp500_completo() -> pd.DataFrame:
    """Descarga el S&P 500 con metadatos de sector desde Wikipedia."""
    try:
        tabla = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tabla["Symbol"] = tabla["Symbol"].str.replace(".", "-", regex=False)
        return tabla[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].rename(
            columns={"Symbol": "ticker", "Security": "nombre",
                     "GICS Sector": "sector", "GICS Sub-Industry": "subsector"}
        )
    except Exception:
        tickers = obtener_sp500()
        return pd.DataFrame({
            "ticker": tickers,
            "nombre": tickers,
            "sector": "N/A",
            "subsector": "N/A",
        })


def filtrar_universo_interactivo() -> list[str]:
    """
    Diálogo interactivo para filtrar el universo S&P 500 por sector antes del scan.
    Permite elegir uno o varios sectores, o saltar el filtro y usar el universo completo.
    Devuelve lista de tickers a analizar.
    """
    print("\n" + "=" * 62)
    print("  FILTRO DE UNIVERSO — S&P 500")
    print("=" * 62)
    print("  Puedes limitar la búsqueda a sectores específicos para")
    print("  reducir el tiempo de cómputo y enfocar los resultados.")
    print()

    sp500_df = obtener_sp500_completo()
    sectores = sorted(sp500_df["sector"].dropna().unique())

    print("  Sectores disponibles:")
    for i, s in enumerate(sectores, 1):
        n = (sp500_df["sector"] == s).sum()
        print(f"    {i:2}. {s:<45} ({n:>3} empresas)")

    print()
    print("  Opciones de selección:")
    print("    Números separados por coma  →  ej: 1,3,7")
    print("    Rango con guión             →  ej: 2-5")
    print("    ENTER sin texto             →  usar todo el S&P 500")
    print()

    entrada = input("  Tu selección: ").strip()

    if not entrada:
        tickers = sp500_df["ticker"].tolist()
        print(f"\n  [OK] Universo completo: {len(tickers)} tickers")
        return tickers

    # Parsear selección con soporte de rangos y listas
    indices: set[int] = set()
    try:
        for parte in entrada.split(","):
            parte = parte.strip()
            if "-" in parte:
                a, b = parte.split("-", 1)
                indices.update(range(int(a) - 1, int(b)))
            else:
                indices.add(int(parte) - 1)
        sectores_sel = [sectores[i] for i in sorted(indices) if 0 <= i < len(sectores)]
    except (ValueError, IndexError):
        print("  [WARN] Selección inválida. Usando universo completo.")
        return sp500_df["ticker"].tolist()

    if not sectores_sel:
        print("  [WARN] Sin sectores válidos. Usando universo completo.")
        return sp500_df["ticker"].tolist()

    filtrado = sp500_df[sp500_df["sector"].isin(sectores_sel)]
    tickers = filtrado["ticker"].tolist()
    print(f"\n  [OK] Sectores: {', '.join(sectores_sel)}")
    print(f"       Total tickers a escanear: {len(tickers)}")
    return tickers


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
    Fuente: yfinance (histórico diario).
    """
    nombre_cache = f"precios_{inicio[:4]}_{fin[:4]}"
    ruta = _ruta_cache(nombre_cache)

    if not forzar_descarga and os.path.exists(ruta):
        print(f"[CACHÉ] Cargando precios desde {ruta}")
        df = pd.read_parquet(ruta)
        faltantes = [t for t in tickers if t not in df.columns]
        if faltantes:
            print(f"[INFO] Descargando {len(faltantes)} tickers nuevos...")
            df_nuevo = _descargar_batch_yfinance(faltantes, inicio, fin)
            df = pd.concat([df, df_nuevo], axis=1)
            df.to_parquet(ruta)
        return df[sorted(df.columns)]

    print(f"[INFO] Descargando precios para {len(tickers)} tickers ({inicio} → {fin})...")
    df = _descargar_batch_yfinance(tickers, inicio, fin)
    df.to_parquet(ruta)
    print(f"[OK] Precios guardados en {ruta}")
    return df


def _descargar_batch_yfinance(tickers: list[str], inicio: str, fin: str) -> pd.DataFrame:
    """Descarga histórico diario en lotes de 100 desde yfinance."""
    batch_size = 100
    frames = []
    for i in range(0, len(tickers), batch_size):
        lote = tickers[i : i + batch_size]
        try:
            raw = yf.download(
                lote,
                start=inicio,
                end=fin,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
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


# ── Alpaca — histórico diario (out-of-sample: 2020-hoy) ──────────────────────

def descargar_precios_alpaca(
    tickers: list[str],
    inicio: str = CORTE_IN_SAMPLE,
    fin: str | None = None,
    forzar_descarga: bool = False,
) -> pd.DataFrame:
    """
    Descarga histórico diario desde Alpaca para el periodo out-of-sample.
    Usado para backtesting (2020-hoy) y señales diarias.

    Si Alpaca no está configurado cae a yfinance automáticamente.
    """
    fin = fin or datetime.now().strftime("%Y-%m-%d")

    if not _ALPACA_DISPONIBLE:
        print("[WARN] Alpaca no configurado. Usando yfinance como fallback.")
        return descargar_precios(tickers, inicio=inicio, fin=fin, forzar_descarga=forzar_descarga)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        print("[WARN] alpaca-py no instalado. Ejecuta: pip install alpaca-py")
        return descargar_precios(tickers, inicio=inicio, fin=fin, forzar_descarga=forzar_descarga)

    nombre_cache = f"alpaca_diario_{inicio[:4]}_{fin[:4]}"
    ruta = _ruta_cache(nombre_cache)

    if not forzar_descarga and os.path.exists(ruta):
        print(f"[CACHÉ] Cargando histórico Alpaca desde {ruta}")
        df = pd.read_parquet(ruta)
        faltantes = [t for t in tickers if t not in df.columns]
        if not faltantes:
            return df[sorted(df.columns)]

    print(f"[ALPACA] Descargando histórico diario para {len(tickers)} tickers ({inicio} → {fin})...")

    client  = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    frames  = []
    # Alpaca tiene límite de símbolos por petición; lotes de 100
    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        lote = tickers[i : i + batch_size]
        try:
            request = StockBarsRequest(
                symbol_or_symbols=lote,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=inicio,
                end=fin,
                adjustment="all",  # ajustado por splits y dividendos
                feed="iex",
            )
            bars = client.get_stock_bars(request).df
            if bars.empty:
                continue
            if isinstance(bars.index, pd.MultiIndex):
                cierre = bars["close"].unstack(level=0)
            else:
                cierre = bars[["close"]].rename(columns={"close": lote[0]})
            cierre.index = pd.to_datetime(cierre.index).tz_localize(None)
            frames.append(cierre)
        except Exception as e:
            print(f"  [WARN] Error Alpaca lote {i}-{i+batch_size}: {e}")

    if not frames:
        print("[WARN] Sin datos de Alpaca. Usando yfinance como fallback.")
        return descargar_precios(tickers, inicio=inicio, fin=fin, forzar_descarga=forzar_descarga)

    df = pd.concat(frames, axis=1).sort_index()
    df.to_parquet(ruta)
    print(f"[OK] Histórico Alpaca guardado en {ruta}")
    return df


def descargar_precios_combinado(
    tickers: list[str],
    forzar_descarga: bool = False,
) -> pd.DataFrame:
    """
    Combina yfinance (2008-2020) y Alpaca (2020-hoy) en un único DataFrame diario.

    Este es el punto de entrada principal para obtener el histórico completo:
      - In-sample  (2008-2020): yfinance — histórico largo y fiable
      - Out-of-sample (2020-hoy): Alpaca — datos recientes consistentes con señales en vivo

    Returns:
        DataFrame con precios de cierre ajustados (filas=fechas, columnas=tickers).
    """
    print("[INFO] Descargando histórico combinado yfinance + Alpaca...")

    # Tramo in-sample: yfinance 2008 → corte
    df_yf = descargar_precios(
        tickers,
        inicio=INICIO_DEFAULT,
        fin=CORTE_IN_SAMPLE,
        forzar_descarga=forzar_descarga,
    )

    # Tramo out-of-sample: Alpaca corte → hoy
    df_alp = descargar_precios_alpaca(
        tickers,
        inicio=CORTE_IN_SAMPLE,
        forzar_descarga=forzar_descarga,
    )

    # Unir verticalmente eliminando solapamientos
    df_yf  = df_yf[df_yf.index  < CORTE_IN_SAMPLE]
    df_alp = df_alp[df_alp.index >= CORTE_IN_SAMPLE]

    df = pd.concat([df_yf, df_alp]).sort_index()

    # Alinear columnas: solo tickers presentes en ambos tramos
    tickers_comunes = [t for t in tickers if t in df.columns]
    df = df[sorted(tickers_comunes)]

    print(f"[OK] Histórico combinado: {df.index[0].date()} → {df.index[-1].date()} "
          f"| {len(df)} días | {len(df.columns)} tickers")
    return df


# ── Alpaca — datos intradiarios ───────────────────────────────────────────────

def descargar_precios_intradiarios(
    tickers: list[str],
    intervalo: str | None = None,
    dias_atras: int = 5,
    forzar_descarga: bool = False,
) -> pd.DataFrame:
    """
    Descarga precios intradiarios desde Alpaca.

    Requiere ALPACA_API_KEY y ALPACA_API_SECRET en .env.
    Si Alpaca no está configurado, cae de vuelta a yfinance con el intervalo más cercano.

    Args:
        tickers:          Lista de tickers del S&P 500.
        intervalo:        '1Min', '5Min', '15Min', '30Min', '1Hour'. Por defecto el del .env.
        dias_atras:       Cuántos días de histórico intradiario descargar.
        forzar_descarga:  Ignora caché y vuelve a descargar.

    Returns:
        DataFrame con precios de cierre (filas=timestamps, columnas=tickers).
    """
    intervalo = intervalo or ALPACA_INTERVALO

    if not _ALPACA_DISPONIBLE:
        print("[WARN] Alpaca no configurado. Usando yfinance como fallback.")
        return _intradiario_yfinance_fallback(tickers, intervalo, dias_atras)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        print("[WARN] alpaca-trade-api no instalado. Ejecuta: pip install alpaca-py")
        print("[WARN] Usando yfinance como fallback.")
        return _intradiario_yfinance_fallback(tickers, intervalo, dias_atras)

    _MAP_TIMEFRAME = {
        "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
    }
    if intervalo not in _MAP_TIMEFRAME:
        raise ValueError(f"Intervalo '{intervalo}' no válido. Opciones: {list(_MAP_TIMEFRAME)}")

    nombre_cache = f"intradiario_{intervalo}_{dias_atras}d"
    ruta = _ruta_cache(nombre_cache)

    if not forzar_descarga and os.path.exists(ruta):
        ts_cache = datetime.fromtimestamp(os.path.getmtime(ruta))
        # Reusar caché si tiene menos de 1 hora
        if datetime.now() - ts_cache < timedelta(hours=1):
            print(f"[CACHÉ] Datos intradiarios recientes ({intervalo}) desde {ruta}")
            return pd.read_parquet(ruta)

    fin_dt    = datetime.now()
    inicio_dt = fin_dt - timedelta(days=dias_atras)

    print(f"[ALPACA] Descargando datos {intervalo} para {len(tickers)} tickers...")

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=_MAP_TIMEFRAME[intervalo],
        start=inicio_dt,
        end=fin_dt,
        feed="iex",  # feed gratuito (IEX); cambiar a "sip" con plan de pago
    )

    try:
        bars = client.get_stock_bars(request).df
    except Exception as e:
        print(f"[ERROR] Alpaca: {e}")
        print("[WARN] Usando yfinance como fallback.")
        return _intradiario_yfinance_fallback(tickers, intervalo, dias_atras)

    # Alpaca devuelve MultiIndex (symbol, timestamp) → pivotar a columnas por ticker
    if isinstance(bars.index, pd.MultiIndex):
        df = bars["close"].unstack(level=0)
    else:
        df = bars[["close"]].rename(columns={"close": tickers[0]})

    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df = df.sort_index()
    df.to_parquet(ruta)
    print(f"[OK] {len(df)} velas {intervalo} guardadas en {ruta}")
    return df


def _intradiario_yfinance_fallback(
    tickers: list[str], intervalo: str, dias_atras: int
) -> pd.DataFrame:
    """Fallback a yfinance cuando Alpaca no está disponible."""
    _MAP_YF = {
        "1Min":  ("1m",  7),
        "5Min":  ("5m",  60),
        "15Min": ("15m", 60),
        "30Min": ("30m", 60),
        "1Hour": ("1h",  730),
    }
    yf_intervalo, max_dias = _MAP_YF.get(intervalo, ("1h", 730))
    dias = min(dias_atras, max_dias)
    fin_dt    = datetime.now()
    inicio_dt = fin_dt - timedelta(days=dias)

    frames = []
    for ticker in tickers:
        try:
            raw = yf.download(
                ticker,
                start=inicio_dt.strftime("%Y-%m-%d"),
                end=fin_dt.strftime("%Y-%m-%d"),
                interval=yf_intervalo,
                auto_adjust=True,
                progress=False,
            )
            if not raw.empty:
                frames.append(raw["Close"].rename(ticker))
        except Exception as e:
            print(f"  [WARN] yfinance fallback error en {ticker}: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


# ── Limpieza ─────────────────────────────────────────────────────────────────

def filtrar_datos(df: pd.DataFrame, min_obs: int = MIN_OBS) -> pd.DataFrame:
    """Elimina tickers con datos insuficientes y rellena huecos menores."""
    # Eliminar columnas con demasiados NaN
    df = df.dropna(axis=1, thresh=min_obs)
    # Solo forward-fill: propagar último precio conocido hacia adelante.
    # bfill está prohibido — usaría precios futuros para rellenar el pasado (look-ahead bias).
    df = df.ffill()
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
    corte: str = CORTE_IN_SAMPLE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide en in-sample (detección de pares) y out-of-sample (backtesting).
    Por defecto: 2008-2020 in-sample (yfinance) | 2020-hoy out-of-sample (Alpaca).
    El corte se configura en config.py → CORTE_IN_SAMPLE.
    """
    in_sample  = df[df.index < corte]
    out_sample = df[df.index >= corte]
    return in_sample, out_sample
