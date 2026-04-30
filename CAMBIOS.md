# Cambios respecto al proyecto original

---

## 1. Fuente de datos — Alpaca en lugar de yfinance

**Original:** todo el sistema usaba yfinance como única fuente de datos.

**Nuevo:** Alpaca es la fuente principal, yfinance actúa como fallback.

| Uso | Fuente | Frecuencia |
|---|---|---|
| Detección de pares | Alpaca | Horaria — últimos 12 meses |
| Señales diarias | Alpaca | Horaria — últimos 12 meses |
| Backtesting | Alpaca | Diaria — 2020 hasta hoy |
| Fallback (sin API keys) | yfinance | Diaria |

**Por qué:** el backtesting y las señales en vivo deben usar la misma fuente. Si el backtest usa yfinance y las señales usan Alpaca, los resultados del backtest no son representativos del comportamiento real de la estrategia. Con Alpaca en ambos lados, lo que funciona en el backtest funciona igual en producción.

**Variables descargadas:** `close`, `open` y `volume`. Se descarta `high`, `low`, `vwap` y `trade_count` — no aportan para una estrategia de pairs trading con stop-loss por z-score.

---

## 2. Caché inteligente con TTL por tipo de dato

**Original:** si el archivo de caché existía, siempre se usaba sin importar su antigüedad.

**Nuevo:** cada tipo de dato tiene un tiempo de vida (TTL) distinto según con qué frecuencia cambia:

| Dato | TTL |
|---|---|
| Alpaca diario (backtesting) | 24 horas |
| Alpaca horario — mercado abierto | 1 hora |
| Alpaca horario — mercado cerrado | No expira |
| yfinance histórico (fallback) | No expira |

El caché horario no expira fuera de horario de mercado porque descargar datos cuando el mercado está cerrado no añade información nueva.

---

## 3. Limpieza de datos

### Eliminación de look-ahead bias

**Original:**
```python
df = df.ffill().bfill()
```
El `bfill()` rellena huecos hacia atrás usando precios futuros para completar el pasado. En un backtest esto implica que el modelo "ve" información que no habría tenido disponible en ese momento, inflando artificialmente los resultados.

**Nuevo:**
```python
df = df.ffill()
```
Solo se rellena hacia adelante: los días sin mercado mantienen el precio del cierre anterior. El índice temporal queda regular sin introducir sesgo.

### Filtro de liquidez

Se elimina cualquier ticker cuyo volumen medio diario sea inferior a 500.000 acciones antes de entrar al pipeline de detección. Tickers ilíquidos generan spreads ruidosos y son difíciles de ejecutar en real.

**Decisión de diseño:** no se filtran movimientos extremos de precio. Una subida o bajada brusca puede ser real (OPA, earnings, crisis) y eliminarla introduciría sesgo. Se prefiere asumir un falso positivo ocasional antes que descartar información real.

---

## 4. Control de horario de mercado

**Original:** el sistema podía ejecutarse en cualquier momento.

**Nuevo:** tres funciones en `datos.py` controlan cuándo tiene sentido ejecutar cada parte del sistema:

- `mercado_abierto()` — devuelve `True` si NYSE está abierto (lunes-viernes 9:30-16:00 NY)
- `tiempo_hasta_apertura()` — devuelve el tiempo hasta la próxima apertura
- `verificar_horario_mercado(modo)` — bloquea la ejecución de señales fuera de horario

---

## 5. Configuración centralizada

**Original:** todos los parámetros del algoritmo estaban hardcodeados en cada módulo por separado. Cambiar el umbral de entrada al z-score, por ejemplo, requería editar varios archivos.

**Nuevo:** `config.py` centraliza todos los parámetros del sistema. Las API keys se leen desde `.env` y nunca se escriben en el código.

Parámetros principales:

| Categoría | Ejemplos |
|---|---|
| Mercado | `MERCADO_APERTURA`, `MERCADO_CIERRE`, `MERCADO_ZONA_HORARIA` |
| Datos | `MIN_OBS`, `MIN_OBS_HORARIO`, `MIN_VOLUMEN_DIARIO` |
| Detección | `UMBRAL_EG`, `MIN_SCORE_JOHANSEN`, `VENTANA_ROLLING` |
| Kalman | `KALMAN_DELTA`, `KALMAN_VAR_OBS` |
| Señales | `ENTRADA_Z`, `SALIDA_Z`, `STOP_Z` |
| Backtesting | `CAPITAL_INICIAL`, `FRACCION_RIESGO`, `SLIPPAGE`, `COMISION` |
| Validación estadística | `MC_N_SIMULACIONES`, `BS_N_BOOTSTRAP`, `PERM_N` |

---

## 6. Detección de pares — cointegración actual, no histórica

**Original:** la detección buscaba pares que hubieran estado cointegrados en el histórico 2008-2020. El resultado era una lista de relaciones que en algún momento del pasado funcionaron.

**Nuevo:** la detección usa datos horarios de los últimos 12 meses (≈ 1638 barras) y busca pares cointegrados **ahora**. Un par que estuvo cointegrado entre 2012 y 2015 no es relevante si hoy no lo está.

**Pipeline:** Engle-Granger como cribado rápido → Johansen como validación robusta. Sin pre-filtros adicionales — la cointegración no requiere que los activos sean del mismo sector ni que tengan alta correlación de retornos.

**Coste computacional:** ~125.000 pares para el S&P 500 completo. EG tarda ~0.001s por par, por lo que el scan completo toma ~3 minutos. Asumible para una ejecución semanal.

---

## 7. Automatización — pipeline diario y pipeline semanal

**Original:** un único pipeline diario sin distinción entre buscar pares nuevos y verificar pares activos.

**Nuevo:** dos pipelines separados con responsabilidades y frecuencias distintas:

### Pipeline diario — apertura de mercado (9:30 EST)

1. **Verificación de cointegración activa** — para cada par con posición abierta, se ejecuta un test EG sobre las barras más recientes. Si la cointegración se ha roto, la señal pasa a `SUSPENDIDO` y la posición debe cerrarse.
2. **Generación de señales** — para los pares que siguen cointegrados, calcula el z-score actual y emite la señal del día (LONG / SHORT / CERRAR / HOLD).

### Pipeline semanal — fin de semana

1. Comprueba si `pares_cointegrados.csv` tiene más de 7 días de antigüedad.
2. Descarga datos horarios de los últimos 12 meses para todo el S&P 500.
3. Ejecuta el scan completo EG + Johansen.
4. Actualiza el CSV con los pares actualmente cointegrados.

**Por qué frecuencias distintas:** la cointegración entre dos activos no cambia de un día para otro — escanear 125.000 pares cada día sería innecesario y costoso. En cambio, verificar si un par activo sigue siendo válido antes de operar es crítico y barato (pocos pares, un test rápido).

---

## 8. Spread y Filtro de Kalman — adaptación a datos horarios

### Half-life en barras, no en días

El proceso Ornstein-Uhlenbeck estima cuánto tarda el spread en volver a su media. Este tiempo (half-life) se usaba como ventana del z-score.

**Original:** el half-life se expresaba en días y se clampeaba entre 5 y 252 días.

**Nuevo:** se expresa en barras y se clampea entre los equivalentes horarios:
- Mínimo: 5 días × 6.5 h/día = 32.5 barras
- Máximo: 252 días × 6.5 h/día = 1638 barras

### Warm-up del Filtro de Kalman

El Kalman necesita un periodo inicial para estimar el hedge ratio antes de que sus estimaciones sean fiables.

**Original:** warm-up de 60 barras — equivale a 3 meses con datos diarios, pero solo a 9 días con datos horarios.

**Nuevo:** warm-up de `60 × 6.5 = 390 barras` — siempre equivale a 60 días de negociación independientemente de la frecuencia.

### Ventana de volatilidad para el sizing

**Original:** ventana de 20 barras para calcular la volatilidad del spread — correcto para datos diarios (20 días), pero con datos horarios equivale a solo 3 días.

**Nuevo:** `20 días × 6.5 h/día = 130 barras` — siempre representa 20 días de negociación.

---

## 9. Backtesting — corrección de bugs

### Cálculo de retornos dentro del bucle (bug de rendimiento)

**Original:** en cada iteración del bucle principal se llamaba a `pct_change()` sobre la serie completa de precios. Con N barras, esto ejecutaba N × N operaciones en lugar de N.

**Nuevo:** los retornos se precalculan una sola vez antes de entrar al bucle. El tiempo de ejecución pasa de cuadrático a lineal.

### Grid search — separación in-sample / out-of-sample

**Original:** el grid search optimizaba los parámetros (umbral de entrada, umbral de salida, ventana del z-score) sobre todos los datos disponibles y luego evaluaba el resultado sobre esos mismos datos. Esto garantiza encontrar los mejores parámetros para el pasado, pero no dice nada sobre el futuro.

**Nuevo:** los datos se dividen en dos partes:
- **In-sample (70%):** se usa para buscar los mejores parámetros.
- **Out-of-sample (30%):** se usa para evaluar si esos parámetros funcionan en datos que el optimizador no ha visto.

Si el rendimiento en out-of-sample es significativamente peor que en in-sample, es señal de overfitting.

### Parámetros desde config

Todos los valores antes hardcodeados (capital inicial, número de simulaciones Monte Carlo, semillas aleatorias, ventanas del walk-forward, etc.) se leen ahora desde `config.py`.

---

## 10. Orquestador — main.py

**Original:** cinco modos mezclando conceptos distintos (`scan`, `señales`, `backtest`, `evaluar`, `full`), con la detección usando datos yfinance y el modo `señales` como punto de entrada diario principal.

**Nuevo:** cinco modos alineados con la nueva arquitectura de dos pipelines:

| Modo | Descripción |
|---|---|
| `scan` | Pipeline semanal — descarga datos horarios de Alpaca y escanea todos los pares del S&P 500 |
| `diario` | Pipeline diario (modo por defecto) — verifica cointegración activa y genera señales del día |
| `backtest` | Backtesting con datos diarios de Alpaca, con opciones de grid search y walk-forward |
| `evaluar` | Genera los gráficos de análisis completos para un par específico |
| `full` | Scan + backtest de los 5 mejores pares, sin interacción |

**Cambios estructurales:**
- El modo por defecto pasa de `scan` a `diario` — lo que se ejecuta cada día en producción, no el scan semanal.
- Se elimina `dividir_muestra()` del flujo de backtesting: la división temporal está integrada en la lógica de descarga (Alpaca desde 2020).
- `evaluacion.py` se importa de forma diferida dentro de un `try/except` — los gráficos son opcionales y no bloquean el flujo principal si el módulo no está disponible.

---

## 11. Nuevos archivos

| Archivo | Descripción |
|---|---|
| `config.py` | Configuración centralizada de todos los parámetros del sistema |
| `.env` | API keys de Alpaca — nunca se sube a GitHub |
| `requirements.txt` | Dependencias del proyecto con versiones mínimas |
| `.gitignore` | Excluye `.env`, caché y archivos generados del repositorio |
