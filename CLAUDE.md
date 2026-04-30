# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the system

```bash
# Primera ejecución: scan semanal + backtest de los 5 mejores pares
python main.py --modo full

# Scan semanal: detectar pares cointegrados con datos horarios de Alpaca (12 meses)
python main.py --modo scan

# Pipeline diario (modo por defecto): verificar pares activos + señales del día
python main.py --modo diario

# Backtesting de un par específico con grid search, walk-forward y gráficos
python main.py --modo backtest --par AAPL MSFT --optimizar --walk-forward --graficos

# Informe visual completo de un par
python main.py --modo evaluar --par KO PEP
```

## Dependencies

```bash
pip install alpaca-py yfinance pandas numpy matplotlib statsmodels scipy python-dotenv
```

Credenciales de Alpaca en `.env` (nunca en el código):
```
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
```

## Architecture

**8-module system** with clear separation of concerns. Data flows strictly top-to-bottom:

```
datos.py → deteccion.py → spread.py → backtesting.py → metricas.py
                                    ↓
                           automatizacion.py (pipelines)
                                    ↓
                           evaluacion.py (charts)
                                    ↓
                              main.py (CLI)
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `datos.py` | Alpaca (primary) + yfinance (fallback); smart cache with TTL per data type; S&P 500 universe; market hours guard |
| `deteccion.py` | Engle-Granger pre-filter → Johansen validator; uses hourly data (12 months); exports `pares_cointegrados.csv` |
| `spread.py` | Kalman Filter for dynamic hedge ratio (warmup=390 bars); OU half-life in hourly bars; entry/exit signals; volatility-scaled sizing |
| `backtesting.py` | Full backtest engine (daily data); walk-forward validation; grid search with IS/OOS split; Monte Carlo; permutation test |
| `metricas.py` | Pure functions: Sharpe, Sortino, Calmar, Omega, MDD, VaR, CVaR, profit factor |
| `automatizacion.py` | Two pipelines: daily (verify active pairs + signals) and weekly (full scan) |
| `evaluacion.py` | Presentation-quality dark-theme charts saved to `graficos/` |
| `main.py` | CLI with `--modo {scan,diario,backtest,evaluar,full}` |
| `config.py` | All algorithm parameters centralized; API keys via `.env` |

### Data sources by use case

| Use case | Source | Frequency |
|---|---|---|
| Pair detection | Alpaca | Hourly — last 12 months |
| Daily signals | Alpaca | Hourly — last 12 months |
| Backtesting | Alpaca | Daily — 2020 to present |
| Fallback (no API keys) | yfinance | Daily |

### Two-pipeline automation

- **Daily pipeline** (run at market open 9:30 EST): verify cointegration of active pairs → generate today's signals
- **Weekly pipeline** (run on weekends): full scan of all ~125,000 S&P 500 pairs, update `pares_cointegrados.csv`

### Key mathematical models

- **Kalman Filter** (not static OLS) for hedge ratio — adapts to structural changes; warmup = 390 bars (= 60 trading days at hourly frequency)
- **OU process half-life in bars** (not days) for z-score window — clamped to [32.5, 1638] hourly bars
- **Engle-Granger as pre-filter + Johansen as validator** — EG takes ~0.001s/pair, full scan ~3 min
- **CVaR / Expected Shortfall** instead of VaR alone
- **Walk-forward validation** — IS/OOS grid search split at 70%/30%

### In-sample / out-of-sample split

- **Detection**: hourly data, last 12 months (current cointegration, not historical)
- **Backtesting**: Alpaca daily data from 2020-01-01 to present (out-of-sample only)

### Generated outputs

- `cache/` — Parquet price files (auto-created)
- `graficos/` — PNG charts per pair (auto-created)
- `pares_cointegrados.csv` — validated pairs from weekly scan
- `señales_diarias.csv` — today's trading signals
- `estado_posiciones.json` — open positions state

### Code conventions

- All comments and print messages written in **Spanish**
- SMART objective targets: Sharpe > 1.0, Max Drawdown < 15%
- No `bfill()` anywhere — only `ffill()` to prevent look-ahead bias
- Liquidity filter: minimum 500,000 average daily volume
