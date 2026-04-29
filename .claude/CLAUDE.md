# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the script

```bash
python cointegración.py
```

The script is fully interactive — it prompts for fundamental filters at runtime, then downloads data and outputs results.

## Dependencies

```bash
pip install yfinance pandas numpy matplotlib statsmodels
```

## Architecture

Single script (`cointegración.py`) implementing a two-stage pipeline:

1. **Fundamental filtering** (`motor_busqueda_interactivo`) — prompts the user for up to 6 filters (sector, price, dividend yield, P/E, beta, profit margin), then scans the first 300 tickers from a GitHub-hosted US stock list via `yf.Ticker(t).info`. Matching tickers are passed to stage 2.

2. **Cointegration analysis** (`evaluar_cointegracion`) — downloads daily Close prices from 2020 to present for all filtered tickers, tests every pairwise combination using a 252-day rolling Johansen trace test (`coint_johansen` from `statsmodels`), ranks pairs by `trace / critical_value`, and plots the rolling cointegration history for the best pair.

### Key design notes

- The Johansen test uses `det_order=0` (no deterministic trend) and `k_ar_diff=1` (one lag in the VECM).
- Cointegration is assessed on log prices (`np.log`), standard practice for price series.
- The ticker scan is capped at 300 to keep runtime reasonable; the rolling plot window is fixed at 252 trading days (~1 year).
- The fallback ticker list in `obtener_gran_lista_tickers` activates only if the GitHub CSV download fails.
