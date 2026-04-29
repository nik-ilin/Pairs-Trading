# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the system

```bash
# Pipeline completo (primera ejecución recomendada)
python main.py --modo full

# Detectar pares cointegrados (in-sample 2008-2020)
python main.py --modo scan

# Backtesting de un par específico con gráficos
python main.py --modo backtest --par AAPL MSFT --optimizar --walk-forward --graficos

# Señales del día para los pares validados
python main.py --modo señales

# Informe visual completo de un par
python main.py --modo evaluar --par KO PEP
```

The legacy interactive script still works standalone:
```bash
python cointegración.py
```

## Dependencies

```bash
pip install yfinance pandas numpy matplotlib statsmodels scipy
```

## Architecture

**8-module system** with clear separation of concerns. Data flows strictly top-to-bottom:

```
datos.py → deteccion.py → spread.py → backtesting.py → metricas.py
                                    ↓
                           automatizacion.py (daily)
                                    ↓
                           evaluacion.py (charts)
                                    ↓
                              main.py (CLI)
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `datos.py` | yfinance download with local Parquet cache; S&P 500 universe; in-sample/out-of-sample split at 2020-01-01 |
| `deteccion.py` | Engle-Granger pre-filter (fast O(n)) → Johansen validator; exports `pares_cointegrados.csv` |
| `spread.py` | Kalman Filter for dynamic hedge ratio; Ornstein-Uhlenbeck half-life for adaptive z-score window; entry/exit signal generation; volatility-scaled position sizing |
| `backtesting.py` | Full backtest engine; walk-forward validation; grid search; Monte Carlo simulation; permutation test for statistical significance |
| `metricas.py` | Pure functions: Sharpe, Sortino, Calmar, Omega, MDD, VaR, CVaR, profit factor |
| `automatizacion.py` | Daily pipeline: ADF stationarity check, Bollinger bands, volatility regime, cointegration break alert, position state via JSON |
| `evaluacion.py` | 11 presentation-quality dark-theme charts saved to `graficos/` |
| `main.py` | CLI with `--modo {scan,backtest,evaluar,señales,full}` |

### Key mathematical models

- **Kalman Filter** (not static OLS) for hedge ratio — adapts to structural changes over time
- **OU process half-life** (not fixed window) for z-score normalization — empirically calibrated to the spread's mean-reversion speed
- **Engle-Granger as pre-filter + Johansen as validator** — reduces scan time ~10× while maintaining statistical rigor
- **CVaR / Expected Shortfall** instead of VaR alone — captures tail risk more faithfully
- **Walk-forward validation** (not single backtest) — detects strategy degradation over time

### In-sample / out-of-sample split

- **In-sample 2008–2020**: cointegration detection only. Never used for backtesting.
- **Out-of-sample 2020–2026**: all backtesting and performance metrics. Prevents data snooping.

### Generated outputs

- `cache/` — Parquet price files (auto-created)
- `graficos/` — 11 PNG charts per pair (auto-created)
- `pares_cointegrados.csv` — validated pairs from scan
- `señales_diarias.csv` — today's trading signals
- `estado_posiciones.json` — open positions state

### Code conventions

- All comments and print messages written in **Spanish**
- SMART objective targets: Sharpe > 1.0, Max Drawdown < 15%
