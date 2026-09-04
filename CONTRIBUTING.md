# Contributing

Contributions should preserve this small, reproducible university research project.

1. Create a focused branch and exclude unrelated changes.
2. Keep Yahoo Finance access inside `datos.py`; do not add brokers, execution, messaging, or server dependencies.
3. Preserve causal timing, train/test separation, exposure limits, and ledger reconciliation.
4. Add synthetic offline tests for behavioral changes.
5. Run `ruff format .`, `ruff check .`, and `python -m pytest -q`.

Document assumptions and limitations plainly. Do not claim statistical proof or future profitability from a backtest.
