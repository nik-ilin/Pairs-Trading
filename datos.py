"""
datos.py — Capa de descarga y caché de precios históricos.

Fuentes de datos:
  - Alpaca horario  → últimos 12 meses, detección de pares y señales
  - Alpaca diario   → 2020-hoy, backtesting out-of-sample
  - yfinance diario → fallback cuando Alpaca no está disponible

Las credenciales de Alpaca van en .env (ALPACA_API_KEY, ALPACA_API_SECRET).
"""

import os
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

# ── Cargar variables de entorno desde .env ───────────────────────────────────

from config import (
    ALPACA_API_KEY, ALPACA_API_SECRET,
    INICIO_DEFAULT, FIN_DEFAULT, MIN_OBS,
    CORTE_IN_SAMPLE, MIN_VOLUMEN_DIARIO,
    MERCADO_ZONA_HORARIA, MERCADO_APERTURA, MERCADO_CIERRE,
)

# True si las credenciales de Alpaca están configuradas
_ALPACA_DISPONIBLE = bool(ALPACA_API_KEY and ALPACA_API_KEY != "tu_api_key_aqui")

# ── Directorio de caché ──────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Horario de mercado ────────────────────────────────────────────────────────

def mercado_abierto() -> bool:
    """Devuelve True si el mercado americano (NYSE) está abierto ahora mismo."""
    ny = datetime.now(ZoneInfo(MERCADO_ZONA_HORARIA))
    if ny.weekday() >= 5:  # sábado=5, domingo=6
        return False
    apertura = ny.replace(hour=MERCADO_APERTURA[0], minute=MERCADO_APERTURA[1],
                          second=0, microsecond=0)
    cierre   = ny.replace(hour=MERCADO_CIERRE[0],   minute=MERCADO_CIERRE[1],
                          second=0, microsecond=0)
    return apertura <= ny <= cierre


def tiempo_hasta_apertura() -> timedelta | None:
    """
    Devuelve el tiempo que falta para la próxima apertura del mercado.
    Devuelve None si el mercado está abierto ahora mismo.
    """
    if mercado_abierto():
        return None
    ny  = datetime.now(ZoneInfo(MERCADO_ZONA_HORARIA))
    hoy = ny.replace(hour=MERCADO_APERTURA[0], minute=MERCADO_APERTURA[1],
                     second=0, microsecond=0)
    if ny >= hoy:
        # Ya pasó la apertura de hoy → siguiente día laborable
        dias_extra = 1
        if ny.weekday() == 4:  # viernes → lunes
            dias_extra = 3
        elif ny.weekday() == 5:  # sábado → lunes
            dias_extra = 2
        hoy += timedelta(days=dias_extra)
    return hoy - ny


def verificar_horario_mercado(modo: str = "señales") -> bool:
    """
    Comprueba si tiene sentido ejecutar el sistema ahora mismo.

    - modo='señales'   : solo durante horario de mercado
    - modo='validacion': siempre (puede ejecutarse al cierre)
    - modo='deteccion' : siempre (análisis offline)
    """
    if modo == "señales" and not mercado_abierto():
        espera = tiempo_hasta_apertura()
        horas  = int(espera.total_seconds() // 3600)
        mins   = int((espera.total_seconds() % 3600) // 60)
        print(f"[INFO] Mercado cerrado. Próxima apertura en {horas}h {mins}min.")
        return False
    return True


# ── Caché inteligente ─────────────────────────────────────────────────────────

def _cache_vigente(ruta: str, max_horas: float) -> bool:
    """
    Devuelve True si el archivo de caché existe y es más reciente que max_horas.

    TTL por tipo de dato:
      - yfinance histórico  : float('inf')  — nunca refresca (periodo cerrado)
      - Alpaca diario       : 24h           — refresca una vez al día
      - Alpaca horario      : 1h            — refresca cada hora
    """
    if not os.path.exists(ruta):
        return False
    edad_horas = (datetime.now() - datetime.fromtimestamp(
        os.path.getmtime(ruta)
    )).total_seconds() / 3600
    return edad_horas < max_horas


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

    # yfinance histórico: caché permanente (periodo 2008-2020 no cambia)
    if not forzar_descarga and _cache_vigente(ruta, float("inf")):
        print(f"[CACHÉ] Cargando precios históricos desde {ruta}")
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

    # Alpaca diario: refresca una vez al día (TTL 24h)
    if not forzar_descarga and _cache_vigente(ruta, 24):
        print(f"[CACHÉ] Cargando histórico Alpaca diario desde {ruta}")
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




# ── OHLCV — descarga completa (close + open + volume) ────────────────────────

def _extraer_ohlcv_yfinance(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Extrae close, open y volume de un DataFrame MultiIndex de yfinance."""
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"]
        open_  = raw["Open"]
        volume = raw["Volume"]
    else:
        t = tickers[0]
        close  = raw[["Close"]].rename(columns={"Close": t})
        open_  = raw[["Open"]].rename(columns={"Open": t})
        volume = raw[["Volume"]].rename(columns={"Volume": t})
    return {"close": close, "open": open_, "volume": volume}


def _extraer_ohlcv_alpaca(bars: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Extrae close, open y volume de un DataFrame de Alpaca con MultiIndex."""
    if isinstance(bars.index, pd.MultiIndex):
        close  = bars["close"].unstack(level=0)
        open_  = bars["open"].unstack(level=0)
        volume = bars["volume"].unstack(level=0)
    else:
        t = tickers[0]
        close  = bars[["close"]].rename(columns={"close": t})
        open_  = bars[["open"]].rename(columns={"open": t})
        volume = bars[["volume"]].rename(columns={"volume": t})
    for df in (close, open_, volume):
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return {"close": close, "open": open_, "volume": volume}



def descargar_ohlcv_horario(
    tickers: list[str],
    dias_atras: int = 730,
    forzar_descarga: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Descarga barras horarias OHLCV desde Alpaca.
    Usado para detección de pares y generación de señales.

    Returns:
        {'close': df, 'open': df, 'volume': df}  — índice: timestamps horarios
    """
    if not _ALPACA_DISPONIBLE:
        print("[WARN] Alpaca no configurado. Datos horarios no disponibles sin API keys.")
        return {"close": pd.DataFrame(), "open": pd.DataFrame(), "volume": pd.DataFrame()}

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        print("[WARN] alpaca-py no instalado: pip install alpaca-py")
        return {"close": pd.DataFrame(), "open": pd.DataFrame(), "volume": pd.DataFrame()}

    nombre_cache = f"ohlcv_horario_{dias_atras}d"
    ruta = _ruta_cache(nombre_cache)

    # OHLCV horario: refresca cada hora solo si el mercado está abierto
    ruta_close = ruta.replace(".parquet", "_close.parquet")
    if not forzar_descarga and _cache_vigente(ruta_close, 1 if mercado_abierto() else float("inf")):
        print(f"[CACHÉ] OHLCV horario desde {ruta}")
        return {c: pd.read_parquet(ruta.replace(".parquet", f"_{c}.parquet"))
                for c in ("close", "open", "volume")}

    fin_dt    = datetime.now()
    inicio_dt = fin_dt - timedelta(days=dias_atras)
    client    = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    resultados: dict[str, list] = {"close": [], "open": [], "volume": []}

    for i in range(0, len(tickers), 100):
        lote = tickers[i : i + 100]
        try:
            request = StockBarsRequest(
                symbol_or_symbols=lote,
                timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                start=inicio_dt,
                end=fin_dt,
                adjustment="all",
                feed="iex",
            )
            bars = client.get_stock_bars(request).df
            if bars.empty:
                continue
            ohlcv = _extraer_ohlcv_alpaca(bars, lote)
            for campo in resultados:
                resultados[campo].append(ohlcv[campo])
        except Exception as e:
            print(f"  [WARN] Alpaca horario lote {i}: {e}")

    if not any(resultados["close"]):
        return {"close": pd.DataFrame(), "open": pd.DataFrame(), "volume": pd.DataFrame()}

    resultado_final = {
        campo: pd.concat(frames, axis=1).sort_index()
        for campo, frames in resultados.items()
        if frames
    }

    for campo, df in resultado_final.items():
        df.to_parquet(ruta.replace(".parquet", f"_{campo}.parquet"))

    print(f"[OK] OHLCV horario: {len(resultado_final['close'])} barras | {len(tickers)} tickers")
    return resultado_final


def descargar_ohlcv_diario(
    tickers: list[str],
    inicio: str = CORTE_IN_SAMPLE,
    fin: str | None = None,
    forzar_descarga: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Descarga barras diarias OHLCV desde Alpaca (2020-hoy).
    Usado para validación diaria de pares activos.

    Returns:
        {'close': df, 'open': df, 'volume': df}  — índice: fechas diarias
    """
    fin = fin or datetime.now().strftime("%Y-%m-%d")

    if not _ALPACA_DISPONIBLE:
        print("[WARN] Alpaca no configurado. Usando yfinance como fallback.")
        return _ohlcv_diario_yfinance(tickers, inicio, fin)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        print("[WARN] alpaca-py no instalado: pip install alpaca-py")
        return _ohlcv_diario_yfinance(tickers, inicio, fin)

    nombre_cache = f"ohlcv_diario_{inicio[:4]}_{fin[:4]}"
    ruta_base = _ruta_cache(nombre_cache)

    # OHLCV diario: refresca una vez al día (TTL 24h)
    if not forzar_descarga and _cache_vigente(ruta_base.replace(".parquet", "_close.parquet"), 24):
        print(f"[CACHÉ] OHLCV diario desde {ruta_base}")
        return {c: pd.read_parquet(ruta_base.replace(".parquet", f"_{c}.parquet"))
                for c in ("close", "open", "volume")}

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    resultados: dict[str, list] = {"close": [], "open": [], "volume": []}

    for i in range(0, len(tickers), 100):
        lote = tickers[i : i + 100]
        try:
            request = StockBarsRequest(
                symbol_or_symbols=lote,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=inicio,
                end=fin,
                adjustment="all",
                feed="iex",
            )
            bars = client.get_stock_bars(request).df
            if bars.empty:
                continue
            ohlcv = _extraer_ohlcv_alpaca(bars, lote)
            for campo in resultados:
                resultados[campo].append(ohlcv[campo])
        except Exception as e:
            print(f"  [WARN] Alpaca diario lote {i}: {e}")

    if not any(resultados["close"]):
        print("[WARN] Sin datos Alpaca. Usando yfinance.")
        return _ohlcv_diario_yfinance(tickers, inicio, fin)

    resultado_final = {
        campo: pd.concat(frames, axis=1).sort_index()
        for campo, frames in resultados.items()
        if frames
    }

    for campo, df in resultado_final.items():
        df.to_parquet(ruta_base.replace(".parquet", f"_{campo}.parquet"))

    print(f"[OK] OHLCV diario Alpaca: {len(resultado_final['close'])} días | {len(tickers)} tickers")
    return resultado_final


def _ohlcv_diario_yfinance(tickers: list[str], inicio: str, fin: str) -> dict[str, pd.DataFrame]:
    """Fallback: descarga OHLCV diario desde yfinance."""
    resultados: dict[str, list] = {"close": [], "open": [], "volume": []}
    for i in range(0, len(tickers), 100):
        lote = tickers[i : i + 100]
        try:
            raw = yf.download(lote, start=inicio, end=fin, auto_adjust=True,
                              progress=False, threads=True)
            ohlcv = _extraer_ohlcv_yfinance(raw, lote)
            for campo in resultados:
                resultados[campo].append(ohlcv[campo])
        except Exception as e:
            print(f"  [WARN] yfinance OHLCV lote {i}: {e}")
    return {
        campo: pd.concat(frames, axis=1).sort_index()
        for campo, frames in resultados.items() if frames
    }


# ── Limpieza ─────────────────────────────────────────────────────────────────

def filtrar_datos(
    df_close: pd.DataFrame,
    df_volume: pd.DataFrame | None = None,
    min_obs: int = MIN_OBS,
    min_vol: float = MIN_VOLUMEN_DIARIO,
) -> pd.DataFrame:
    """
    Limpia y filtra el DataFrame de precios de cierre.

    Pasos:
      1. Elimina tickers con observaciones insuficientes
      2. Elimina tickers con volumen medio insuficiente (si se pasa df_volume)
      3. Forward-fill para huecos de festivos y fines de semana

    Nota: no se filtran outliers de precio — un movimiento extremo puede ser
    real (OPA, earnings, crisis) y eliminarlos introduciría sesgo.
    """
    # 1. Mínimo de observaciones
    df = df_close.dropna(axis=1, thresh=min_obs)

    # 2. Filtro de liquidez
    if df_volume is not None:
        vol_comun = [t for t in df.columns if t in df_volume.columns]
        vol_medio = df_volume[vol_comun].mean()
        tickers_liquidos = vol_medio[vol_medio >= min_vol].index.tolist()
        eliminados = len(df.columns) - len(tickers_liquidos)
        if eliminados > 0:
            print(f"[INFO] Tickers eliminados por baja liquidez: {eliminados}")
        df = df[tickers_liquidos]

    # 3. Forward-fill para huecos de festivos y fines de semana (nunca bfill)
    df = df.ffill()

    print(f"[OK] filtrar_datos: {len(df.columns)} tickers válidos | {len(df)} barras")
    return df


