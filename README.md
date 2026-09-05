# Pairs Trading Research

An educational university project for studying equity pairs trading with daily Yahoo Finance data. It discovers candidate pairs, estimates a dynamic hedge ratio, creates causal trading signals, and evaluates them with a two-leg backtest.

This repository is research software. It does not connect to a broker, place orders, run as a server, or guarantee profitable results. A Spanish translation is available at [docs/README.es.md](docs/README.es.md).

## Method

1. Engle-Granger screens candidate pairs. Benjamini-Hochberg controls the false-discovery rate across that screening family, then Johansen supplies a second diagnostic.
2. A recursive Kalman filter estimates the hedge ratio without future initialization.
3. Ornstein-Uhlenbeck half-life is calibrated before the evaluated period. Rolling z-scores and lagged volatility produce signals and capped sizing.
4. A close-derived signal executes on the next daily bar. Gross notional is divided between both legs using `abs(beta)`.
5. Grid search selects parameters on training data only. Walk-forward validation calibrates each training window and stitches only the following out-of-sample returns.

Transaction costs are charged separately to each leg on entry and exit. Defaults are 5 basis points of slippage plus 10 basis points of commission per leg per transaction. Gross exposure is capped at 1.0 times current equity. Any open position is closed and recorded on the final bar.

## Architecture

| Module | Responsibility |
|---|---|
| `datos.py` | All Yahoo Finance access and exact-query caching. |
| `deteccion.py` | Cointegration screening, FDR control, and rolling diagnostics. |
| `spread.py` | Kalman hedge ratio, OU half-life, z-score, signals, and sizing. |
| `backtesting.py` | Holdings, costs, ledger, optimization, and walk-forward evaluation. |
| `metricas.py` | Performance and risk metrics. |
| `automatizacion.py` | Manual latest-signal snapshot and scan helpers. |
| `evaluacion.py` | Reproducible local charts. |
| `main.py` | Public command-line interface. |
| `config.py` | Central research parameters. |

## Installation

Python 3.11 or 3.12 is supported. Python 3.12 is used for clean-release validation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --help
```

Yahoo Finance is the sole market-data provider and requires no API key. Network failures for new queries raise an error; a cache is reused only when tickers and date bounds match exactly.

## Quickstart

```bash
python main.py --modo backtest --par AAPL MSFT --inicio 2020-01-01 --fin 2026-09-04
```

Supported modes are `backtest`, `diario`, `evaluar`, `scan`, and `full`. The pair is supplied with `--par TICKER1 TICKER2`; backtests also accept `--optimizar`, `--walk-forward`, and `--graficos`. Use `python main.py --help` for the maintained syntax.

`diario` is a manual latest-signal snapshot; it does not execute an order. `scan` explores the current index universe, while `full` combines that scan with selected backtests. Both can request hundreds of symbols and are intentionally not smoke tests.

## Verified example

The release check uses AAPL/MSFT adjusted daily closes requested from 2020-01-01 through the exclusive end date 2026-09-04. It uses the defaults in `config.py`: 60-bar non-trading calibration, automatic OU z-score window, entry/exit/stop z-scores of 1.5/0.5/3.5, 20-bar lagged volatility, 10% risk fraction, and 1.0x gross-exposure cap. The numerical result is printed by the command rather than presented as a stable benchmark because Yahoo may revise historical data and dependency versions can alter estimates.

## Tests and quality checks

```bash
python -m pip install -r requirements-dev.txt
ruff format --check .
ruff check .
python -m pytest -q
python main.py --help
```

Tests use synthetic data and do not require network access. They cover pair detection, future-data invariance, next-bar execution, exposure caps, costs, ledger reconciliation, walk-forward separation, context-specific rolling minima, imports, and CLI help.

## Limitations

- Current S&P 500 membership creates survivorship bias.
- Yahoo Finance is free and convenient, but is not an institutional feed and may revise, omit, or delay data.
- Benjamini-Hochberg covers the Engle-Granger scan only; later pair selection and repeated experiments still create multiple-testing and data-snooping risk.
- Hedge ratios, half-lives, thresholds, and cost assumptions are uncertain and may fail after structural breaks.
- Daily-bar execution omits realistic borrow availability, bid-ask dynamics, market impact, margin calls, taxes, and intraday path dependence.
- Bootstrap and Monte Carlo outputs are descriptive scenarios, not evidence of future profitability.

## Financial disclaimer

This software and its outputs are for education and research only, not investment advice or an offer to trade. Historical and simulated performance does not predict future results. You are responsible for independent validation and for any financial decisions. See [DISCLAIMER.md](DISCLAIMER.md).

## License

MIT. See [LICENSE](LICENSE).
