"""
config.py — Configuración centralizada del sistema de Pairs Trading.

Todos los parámetros configurables del algoritmo están aquí.
Las API keys van en .env (nunca en este archivo).
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARN] python-dotenv no instalado. Ejecuta: pip install python-dotenv")

# ── Mercado ───────────────────────────────────────────────────────────────────
MERCADO_ZONA_HORARIA = "America/New_York"
MERCADO_APERTURA     = (9, 30)    # hora, minuto
MERCADO_CIERRE       = (16, 0)

# ── Alpaca ────────────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")

# ── Procesamiento de datos ────────────────────────────────────────────────────
MIN_VOLUMEN_DIARIO  = 500_000   # volumen medio diario mínimo para considerar un ticker líquido

# ── Datos históricos ──────────────────────────────────────────────────────────
INICIO_DEFAULT    = "2020-01-01"   # inicio del histórico Alpaca (out-of-sample)
FIN_DEFAULT       = "2026-01-01"
MIN_OBS           = 1260           # ~5 años de datos diarios mínimos por ticker

# ── Calendario ────────────────────────────────────────────────────────────────
DIAS_ANIO         = 252            # días de negociación por año
HORAS_ANIO        = 1638           # horas de negociación por año (252 × 6.5h)
HORAS_DIA         = 6.5            # horas de mercado abierto por día (9:30-16:00 NY)

# ── Observaciones mínimas por frecuencia ─────────────────────────────────────
MIN_OBS           = 1260           # ~5 años de datos diarios
MIN_OBS_HORARIO   = 1638           # ~1 año de datos horarios (252 × 6.5h)

# ── Detección de pares (deteccion.py) ────────────────────────────────────────
UMBRAL_EG            = 0.05        # p-value máximo para pasar el pre-filtro Engle-Granger
MIN_SCORE_JOHANSEN   = 1.0         # score mínimo de Johansen (traza / valor crítico)
VENTANA_ROLLING      = 819         # barras para ventana rolling (6 meses horario = 1638//2)
VENTANA_COINT_ACTIVA = 252         # barras diarias recientes para verificar cointegración activa

# ── Filtro de Kalman (spread.py) ──────────────────────────────────────────────
KALMAN_DELTA      = 1e-4           # velocidad de adaptación del estado (mayor = más reactivo)
KALMAN_VAR_OBS    = 1e-3           # varianza del ruido de observación

# ── Proceso Ornstein-Uhlenbeck (spread.py) ────────────────────────────────────
HL_MIN_DIAS       = 5              # half-life mínimo permitido (días)
HL_MAX_DIAS       = 252            # half-life máximo permitido (días)

# ── Señales de trading (spread.py) ───────────────────────────────────────────
ENTRADA_Z         = 2.0            # umbral z-score para abrir posición
SALIDA_Z          = 0.5            # umbral z-score para cerrar posición
STOP_Z            = 3.5            # stop-loss en z-score
VENTANA_VOL       = 20             # ventana rolling para calcular volatilidad del spread

# ── Backtesting (backtesting.py) ──────────────────────────────────────────────
CAPITAL_INICIAL   = 100_000.0      # capital inicial de la simulación (€/$)
FRACCION_RIESGO   = 0.10           # fracción del capital por trade (10%)
SLIPPAGE          = 0.0005         # 0.05% por operación por pata
COMISION          = 0.001          # 0.10% por operación por pata
USAR_LOG          = True           # usar log-precios para el spread

# ── Grid search (backtesting.py) ─────────────────────────────────────────────
GRID_ENTRADA_Z    = [1.5, 2.0, 2.5]
GRID_SALIDA_Z     = [0.25, 0.5]
GRID_WINDOW       = [None, 30, 60]  # None = ventana OU automática

# ── Walk-forward (backtesting.py) ─────────────────────────────────────────────
WF_ANOS_DETECCION = 3              # años de ventana de detección
WF_ANOS_OPERACION = 1              # años de ventana de operación

# ── Monte Carlo (backtesting.py) ──────────────────────────────────────────────
MC_N_SIMULACIONES = 1000
MC_HORIZONTE_DIAS = 252            # días hacia adelante a simular
MC_SEMILLA        = 42

# ── Bootstrap Sharpe (backtesting.py) ────────────────────────────────────────
BS_N_BOOTSTRAP    = 1000
BS_NIVEL_CONFIANZA = 0.95

# ── Test de permutaciones (backtesting.py) ────────────────────────────────────
PERM_N            = 500
PERM_SEMILLA      = 42

# ── Métricas y objetivos SMART (metricas.py) ─────────────────────────────────
TASA_LIBRE_RIESGO = 0.0            # tasa libre de riesgo anual (0 = efectivo)
VAR_NIVEL         = 0.05           # nivel de confianza para VaR/CVaR (5%)
OBJETIVO_SHARPE   = 1.0            # objetivo mínimo de Sharpe Ratio
OBJETIVO_MDD      = -0.15          # objetivo máximo de drawdown (-15%)
