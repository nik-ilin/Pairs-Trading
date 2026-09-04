# AGENTS.md

Read this before working in the repository. This is a university research project for public GitHub publication, not a live-trading product.

## Scope

- Study statistical-arbitrage pairs trading on S&P 500 equities.
- Yahoo Finance is the sole market-data provider and all network access belongs in `datos.py`.
- Do not add brokers, order execution, messaging, model services, servers, schedulers, or production-readiness claims.
- `automatizacion.py` is a manual latest-signal snapshot demonstration. A reported signal is not an order and is executable no earlier than the following daily bar.
- Preserve the flat module structure and CLI unless a task explicitly requires an incompatible change.
- Public documentation, comments, and docstrings use concise professional English. Stable Spanish function names and CLI output remain for compatibility.

## Commands

Supported Python: 3.11 or 3.12. The clean release environment is verified with Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python main.py --help
python main.py --modo backtest --par AAPL MSFT --inicio 2020-01-01 --fin 2026-09-04
python main.py --modo backtest --par AAPL MSFT --optimizar --walk-forward --graficos
python main.py --modo diario --par AAPL MSFT
python main.py --modo evaluar --par AAPL MSFT
python main.py --modo scan
python -m pytest -q
```

Never run the full S&P 500 scan as a routine smoke test. Use one pair and an explicit bounded date range.

## Modules

| File | Responsibility |
|---|---|
| `config.py` | Central thresholds, horizons, exposure cap, and calibration length. |
| `datos.py` | Yahoo Finance, exact-query pandas cache, universe, and cleaning. |
| `deteccion.py` | Engle–Granger, Benjamini–Hochberg, Johansen, and rolling diagnostics. |
| `spread.py` | Causal Kalman beta, OU, z-score, signals, and lagged-volatility sizing. |
| `backtesting.py` | Holdings/accounting, grid search, walk-forward, Monte Carlo, bootstrap. |
| `metricas.py` | Pure performance and risk metrics. |
| `automatizacion.py` | Manual recent snapshot plus scan helpers; no external notifications. |
| `evaluacion.py` | Local Matplotlib charts. |
| `main.py` | Public CLI. |
| `tests/` | Offline synthetic regression suite. |

## Quantitative invariants

- No `bfill`, centered rolling windows, full-sample initialization, or future-derived fallback values.
- A change after cutoff T must not change signals or positions before T.
- Closing-price signals execute no earlier than the next bar.
- Initial calibration is non-tradable. OU and other fitted values for a test period use prior training data or lagged rolling/expanding history.
- Volatility sizing is lagged, remains unavailable during warm-up, and is capped by `EXPOSICION_BRUTA_MAX`.
- Define both signed leg holdings. Allocate gross notional between legs using `abs(beta)` and charge costs on each leg at every rebalance/open/close.
- Force-close and ledger any terminal position. Total daily PnL, trade-ledger PnL, equity change, win rate, and profit factor must reconcile.
- Optimization selects parameters only on train. Final reported metrics use untouched test only.
- Each walk-forward train precedes its test; only OOS returns are stitched/reported.
- Do not restore the invalid fixed-return permutation p-value.
- Full discovery/backtests may require `MIN_OBS_HISTORICO` (~1,260 daily bars); optimization training uses its explicit three-year minimum. Rolling and latest-signal checks use their own minima; a 126-bar test must never inherit 1,260.
- Full scans apply Benjamini–Hochberg to EG p-values before Johansen confirmation.

## Data/cache rules

- Public download functions accept explicit `inicio` and `fin`.
- Cache identity includes exact date bounds and normalized tickers.
- An exact cache may be reused deterministically. On a new-query network failure, raise a clear error; never silently substitute a stale/different cache.
- `filtrar_datos` defaults to the historical minimum, so callers with rolling/snapshot contexts must pass the relevant minimum or avoid the historical filter.
- Current S&P 500 membership introduces survivorship bias; retain this disclosure.

## Generated state

- `cache/`, `graficos/`, `señales_diarias.csv`, and `estado_posiciones.json` are local runtime artifacts.
- `pares_cointegrados.csv` is ignored operational scan output and must not be committed.
- Use synthetic data and monkeypatch network access in automated tests.

## Verification

After changes, run the complete offline suite, CLI help, and syntax compilation. For release checks, perform only a bounded Yahoo request such as AAPL/MSFT with explicit dates. Report commands and failures honestly, including environment/provider limitations.

## Methodological limits

Yahoo data is not institutional-grade. Current-index membership creates survivorship bias. Multiple downstream choices still create data-snooping risk beyond BH. Cointegration can break structurally. The simulator does not fully model borrow availability, margin, taxes, market impact, or intraday execution. Bootstrap and Monte Carlo outputs are descriptive, not guarantees of significance or future profit.
