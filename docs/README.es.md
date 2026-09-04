# Investigación de pairs trading

Proyecto universitario educativo para estudiar pairs trading con datos diarios de Yahoo Finance. Detecta pares candidatos, estima un ratio de cobertura dinámico, genera señales causales y las evalúa mediante un backtest de dos patas.

No es un sistema de trading en vivo: no se conecta a brokers, no coloca órdenes y no garantiza rentabilidad. La documentación principal y los comandos mantenidos están en [README.md](../README.md).

## Instalación y prueba rápida

Requiere Python 3.11 o 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --modo backtest --par AAPL MSFT --inicio 2020-01-01 --fin 2026-09-04
```

Para las pruebas y controles de estilo:

```bash
python -m pip install -r requirements-dev.txt
ruff format --check .
ruff check .
python -m pytest -q
```

## Supuestos y limitaciones

Las señales calculadas al cierre se ejecutan en la barra siguiente. Los costes se aplican por separado a cada pata, la exposición bruta está limitada y cualquier posición restante se cierra al final del backtest.

El universo actual del S&P 500 introduce sesgo de supervivencia. Yahoo Finance no es una fuente institucional. El control Benjamini-Hochberg no elimina todo el riesgo de múltiples pruebas. Los parámetros y relaciones de cointegración son inciertos y pueden sufrir rupturas estructurales. La simulación no reproduce completamente préstamo de acciones, horquillas, impacto de mercado, margen, impuestos ni ejecución intradía.

Este repositorio es exclusivamente educativo y de investigación; no constituye asesoramiento financiero. El rendimiento histórico o simulado no garantiza resultados futuros. Consulte [DISCLAIMER.md](../DISCLAIMER.md).
