# Sistema de Arbitraje Estadístico — Pairs Trading (S&P 500)

Modelo cuantitativo de trading automatizado basado en la cointegración estadística entre pares de acciones del mercado estadounidense. El sistema detecta pares cointegrados, genera señales de entrada y salida, valida la estrategia con backtesting riguroso y se controla en tiempo real desde un bot de Telegram.

---

## Tabla de contenidos

1. [Descripción](#descripción)
2. [Fundamentos matemáticos](#fundamentos-matemáticos)
3. [Arquitectura](#arquitectura)
4. [Instalación](#instalación)
5. [Configuración](#configuración)
6. [Uso — CLI](#uso--cli)
7. [Bot de Telegram](#bot-de-telegram)
8. [Sistema de fuentes de datos](#sistema-de-fuentes-de-datos)
9. [Métricas de evaluación](#métricas-de-evaluación)
10. [Gráficos generados](#gráficos-generados)
11. [Objetivos SMART](#objetivos-smart)

---

## Descripción

El **arbitraje estadístico por pares** (pairs trading) explota la relación histórica de largo plazo entre dos activos. Cuando el precio relativo se aleja de su equilibrio estadístico, el modelo toma posiciones opuestas (comprar el barato, vender el caro) esperando que la relación se restablezca.

| Fase | Datos | Propósito |
|---|---|---|
| Detección de pares | Alpaca horario — últimos 12 meses | Cointegración actual, no histórica |
| Señales diarias | Alpaca horario — últimos 12 meses | Kalman + z-score + ADF en tiempo real |
| Backtesting | Alpaca diario — 2020 a hoy | Validación out-of-sample honesta |

---

## Fundamentos matemáticos

### 1. Detección de cointegración

#### Test de Engle-Granger (pre-filtro rápido)
Dado un par $(S_1, S_2)$, se ajusta la regresión:

$$\log S_{1,t} = \alpha + \beta \cdot \log S_{2,t} + \varepsilon_t$$

Si los residuos $\varepsilon_t$ son estacionarios (test ADF: $p < 0.05$), el par está cointegrado. Coste: ~0.001 s/par → escaneo de 126.000 pares en ~3 min.

#### Test de Johansen (validador robusto)
Evalúa el rango de cointegración sin asumir dirección de causalidad. Estadístico de traza al 95%:

$$\text{Score} = \frac{\lambda_{traza}}{\text{Valor Crítico}_{95\%}}$$

Score > 1 confirma cointegración. Los pares se ordenan por score descendente.

---

### 2. Ratio de cobertura dinámico — Filtro de Kalman

El ratio $\beta_t$ no es constante. El Filtro de Kalman lo actualiza en cada barra:

**Estado:** $\theta_t = [\beta_t,\ \alpha_t]^\top$, modelado como paseo aleatorio.

**Ganancia de Kalman:**
$$K_t = P_{t|t-1} \mathbf{F}_t \cdot (V_e + \mathbf{F}_t^\top P_{t|t-1} \mathbf{F}_t)^{-1}$$
$$\theta_t = \theta_{t-1} + K_t(y_t - \mathbf{F}_t^\top \theta_{t-1})$$

El spread resultante es más estacionario que con OLS estático. Warmup: 390 barras (60 días hábiles en horario).

---

### 3. Proceso Ornstein-Uhlenbeck — Velocidad de reversión

El spread se modela como:

$$dS_t = \kappa(\mu - S_t)\,dt + \sigma\,dW_t$$

**Half-life** (barras para recorrer la mitad del camino a la media):

$$\text{Half-life} = \frac{-\ln 2}{b}, \quad \Delta S_t = a + b \cdot S_{t-1} + \varepsilon_t$$

El half-life se usa automáticamente como ventana del z-score. Rango válido: 32.5–1638 barras horarias.

---

### 4. Z-score y señales de trading

$$Z_t = \frac{S_t - \mu_{rolling}(S, \text{HL})}{\sigma_{rolling}(S, \text{HL})}$$

| Condición | Acción | Razonamiento |
|---|---|---|
| $Z_t < -2.0$ | **LONG spread** | $S_1$ infravalorado relativo a $S_2$ |
| $Z_t > +2.0$ | **SHORT spread** | $S_1$ sobrevalorado relativo a $S_2$ |
| $\|Z_t\| < 0.5$ | **CERRAR** | Reversión completada |
| $\|Z_t\| > 3.5$ | **STOP-LOSS** | Ruptura de cointegración |

---

### 5. Dimensionado — Volatility Scaling

$$N_t = \frac{\text{Capital} \times f}{\sigma_{20d}(S_t)}$$

Con $f = 10\%$ de fracción de riesgo. Mantiene exposición constante independientemente del régimen de volatilidad.

---

## Arquitectura

```
Finanzas/
├── config.py          → Parámetros centralizados; API keys vía .env
├── datos.py           → Descarga Alpaca/yfinance; caché Parquet; fallback automático
├── deteccion.py       → Engle-Granger pre-filtro + Johansen validador
├── spread.py          → Kalman, proceso OU, z-score, señales de trading
├── backtesting.py     → Motor backtest; walk-forward; grid search; Monte Carlo
├── metricas.py        → Sharpe, Sortino, Calmar, Omega, MDD, VaR, CVaR
├── automatizacion.py  → Pipeline diario y semanal; gestión de posiciones
├── evaluacion.py      → Gráficos dark-theme guardados en graficos/
├── main.py            → Orquestador CLI (--modo scan/diario/backtest/evaluar/full)
├── bot_telegram.py    → Bot de Telegram (espejo completo del CLI)
├── cache/             → Parquet con precios (TTL automático por tipo de dato)
├── graficos/          → PNG exportados por evaluacion.py
├── pares_cointegrados.csv  → Resultado del scan semanal
├── señales_diarias.csv     → Señales del pipeline diario
└── estado_posiciones.json  → Posiciones abiertas persistidas
```

**Flujo de datos:**

```
datos.py → deteccion.py → spread.py → backtesting.py → metricas.py
                                    ↓
                           automatizacion.py
                            ↙           ↘
                     main.py (CLI)   bot_telegram.py (Telegram)
                                    ↓
                           evaluacion.py (gráficos)
```

---

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 2. Instalar dependencias
pip install alpaca-py yfinance pandas numpy matplotlib statsmodels scipy \
            python-dotenv python-telegram-bot
```

> **`python-dotenv`** es requerido para cargar las credenciales del `.env`.  
> **`alpaca-py`** es opcional: si no está instalado o las credenciales fallan, el sistema usa `yfinance` automáticamente.  
> **`python-telegram-bot`** solo es necesario si vas a usar el bot de Telegram.

---

## Configuración

### Archivo `.env` (en la raíz del proyecto)

```env
# Alpaca (opcional — sin él el sistema usa yfinance automáticamente)
ALPACA_API_KEY=tu_api_key_aqui
ALPACA_API_SECRET=tu_api_secret_aqui

# Telegram (necesario para bot_telegram.py)
TELEGRAM_BOT_TOKEN=123456789:ABCdef-GHIjkl...
TELEGRAM_CHAT_ID=987654321
```

> El `.env` nunca se sube al repositorio (está en `.gitignore`).

### Sistema de fallback automático

El sistema tolera la ausencia total de credenciales de Alpaca:

| Situación | Comportamiento |
|---|---|
| Sin `.env` o `python-dotenv` no instalado | Avisa y continúa; keys quedan vacías → yfinance |
| `alpaca-py` no instalado | Detectado al arrancar → fallback a yfinance |
| Credenciales inválidas / error de autenticación | `try/except` global → fallback a yfinance |
| Rate limit o error de red durante la descarga | `try/except` global → fallback a yfinance |
| Todo OK con Alpaca | Usa Alpaca (datos horarios, mayor resolución) |

---

## Uso — CLI

### Pipeline completo (recomendado la primera vez)
```bash
python main.py --modo full
```
Ejecuta scan + backtest de los 5 mejores pares de forma no interactiva.

### Scan semanal — detectar pares cointegrados
```bash
python main.py --modo scan
```
Descarga datos horarios de ~500 tickers (12 meses) y escanea ~126.000 pares.  
**Duración estimada:** 15–35 min (descarga) + 3–6 min (escaneo).  
Al terminar muestra un menú interactivo para elegir qué pares guardar.

### Pipeline diario — señales del día
```bash
# Top 20 pares (default)
python main.py --modo diario

# Top 50 pares
python main.py --modo diario --top-n 50
```
Lee `pares_cointegrados.csv` (sin re-escanear), descarga datos recientes y genera `señales_diarias.csv`.

### Backtest de un par específico
```bash
# Con parámetros por defecto
python main.py --modo backtest --par KO PEP

# Con optimización + walk-forward + gráficos
python main.py --modo backtest --par AAPL MSFT --optimizar --walk-forward --graficos
```

### Backtest interactivo desde el CSV
```bash
# Menú de selección con los top 10 pares guardados
python main.py --modo backtest

# Top 30 en el menú
python main.py --modo backtest --top-n 30
```

### Informe visual completo
```bash
python main.py --modo evaluar --par KO PEP
```
Genera todos los gráficos en `graficos/` y muestra métricas completas.

### Argumentos disponibles

| Argumento | Valores | Default | Descripción |
|---|---|---|---|
| `--modo` | `scan`, `diario`, `backtest`, `evaluar`, `full` | `diario` | Modo de ejecución |
| `--par` | `TICKER1 TICKER2` | — | Par específico |
| `--top-n` | entero | `10` | Número de pares a evaluar |
| `--optimizar` | flag | — | Grid search de parámetros |
| `--walk-forward` | flag | — | Validación walk-forward |
| `--graficos` | flag | — | Genera y guarda gráficos PNG |

---

## Bot de Telegram

El bot de Telegram espeja completamente el CLI: cada modo del sistema tiene su comando equivalente, con salida formateada en Markdown y envío automático de gráficos PNG.

### Paso 1 — Crear el bot con BotFather

1. Abre Telegram y busca **@BotFather**
2. Escribe `/newbot`
3. Elige un nombre visible (ej: `Pairs Trading Bot`)
4. Elige un username que termine en `bot` (ej: `mis_pares_bot`)
5. BotFather te dará el **token** → cópialo

### Paso 2 — Obtener tu Chat ID

1. Busca en Telegram **@userinfobot**
2. Escribe `/start`
3. Te devolverá tu ID numérico (ej: `987654321`)

> El `TELEGRAM_CHAT_ID` restringe el bot para que solo responda a ti. Sin él, cualquiera que encuentre el bot puede usarlo.

### Paso 3 — Configurar `.env`

```env
TELEGRAM_BOT_TOKEN=123456789:ABCdef-GHIjkl...
TELEGRAM_CHAT_ID=987654321
```

### Paso 4 — Arrancar el bot

```bash
python bot_telegram.py
```

Abre Telegram, busca tu bot por su username y escribe `/start`.

### Paso 5 (opcional) — Ejecutar en background

```bash
# macOS/Linux — persiste al cerrar la terminal
nohup python bot_telegram.py > bot.log 2>&1 &

# Ver log en tiempo real
tail -f bot.log

# Detener
kill $(pgrep -f bot_telegram.py)
```

---

### Comandos del bot

| Comando | Equivalente CLI | Descripción |
|---|---|---|
| `/start` | — | Bienvenida y menú de ayuda |
| `/ayuda` | — | Lista de comandos |
| `/diario [N]` | `--modo diario --top-n N` | Pipeline diario para top N pares (default 20) |
| `/senales` | — | Señales del último `/diario` guardadas en CSV |
| `/estado` | — | Posiciones abiertas con dirección, fecha y z-score de entrada |
| `/pares [N]` | — | Top N pares del CSV ordenados por score (default 10) |
| `/backtest T1 T2` | `--modo backtest --par T1 T2` | Backtest 2020–hoy con todas las métricas |
| `/evaluar T1 T2` | `--modo evaluar --par T1 T2` | Backtest + genera y envía todos los gráficos PNG |
| `/scan` | `--modo scan` | Scan completo S&P 500 en background (~20-35 min) |

### Ejemplos de uso

```
/diario 50           → Señales para los 50 mejores pares
/pares 20            → Top 20 pares del CSV
/backtest KO PEP     → Backtest Coca-Cola vs PepsiCo
/evaluar AAPL MSFT   → Informe completo + gráficos enviados por Telegram
/estado              → Posiciones abiertas actuales
/scan                → Iniciar scan completo (avisa cuando termina)
```

### Ejemplo de respuesta — `/diario 50`

```
✅ Pipeline diario — 2026-04-30 09:35

📊 Evaluados: 50 | ▲▼ Entradas: 2 | ✕ Cierres: 1 | — Hold: 47 | ⚠ Susp: 0

🔔 Señales de entrada:
▲ `NRG/DIS`  Z=+2.14 β=0.923 HL=89b Vol=NORMAL
▼ `MSCI/NRG` Z=-2.31 β=1.102 HL=64b Vol=BAJA

✕ Cierres recomendados:
✕ `COF/TPL` Z=+0.43
```

---

## Sistema de fuentes de datos

| Uso | Fuente primaria | Fallback | Frecuencia | TTL caché |
|---|---|---|---|---|
| Detección de pares | Alpaca | yfinance diario | Horario — 12 meses | 1h (mercado abierto) / ∞ |
| Señales diarias | Alpaca | yfinance diario | Horario — 12 meses | 1h (mercado abierto) / ∞ |
| Backtesting | Alpaca | yfinance diario | Diario — 2020–hoy | 24h |
| yfinance OHLCV fallback | — | yfinance diario | Diario — 13 meses* | 24h |
| Lista S&P 500 | GitHub CSV | Wikipedia | — | — |
| Sectores S&P 500 | GitHub CSV | Wikipedia / yfinance | — | 30 días |

> *El fallback de yfinance descarga 395 días (365 + 30 de buffer para festivos) para garantizar siempre ≥252 barras hábiles.

---

## Métricas de evaluación

### Rendimiento

| Métrica | Fórmula | Objetivo |
|---|---|---|
| **CAGR** | $(V_f / V_0)^{1/n} - 1$ | Máximo |
| **Sharpe Ratio** | $\bar{r}_e / \sigma_r \cdot \sqrt{252}$ | **> 1.0** |
| **Sortino Ratio** | $\bar{r}_e / \sigma_{down} \cdot \sqrt{252}$ | > 1.5 |
| **Calmar Ratio** | CAGR / \|MDD\| | > 0.5 |
| **Omega Ratio** | Σ ganancias / Σ pérdidas | > 1.5 |
| **Profit Factor** | Beneficio bruto / Pérdida bruta | > 1.3 |

### Riesgo

| Métrica | Descripción | Objetivo |
|---|---|---|
| **Máx. Drawdown** | Mayor caída pico-a-valle | **< 15%** |
| **VaR 95%** | Pérdida máxima en el 95% de días | Referencia |
| **CVaR 95%** | Pérdida media en el peor 5% de casos | < VaR × 1.5 |

> El **CVaR** (Expected Shortfall) captura el comportamiento de cola: dos estrategias con el mismo VaR pueden tener CVaR muy diferentes.

### Validación estadística

| Test | Propósito |
|---|---|
| **Bootstrap Sharpe** | Intervalo de confianza del Sharpe (1000 remuestras) |
| **Test de permutaciones** | Verifica que los retornos no son aleatorios ($p < 0.05$) |
| **Monte Carlo** | Distribución de resultados futuros (1000 trayectorias, 1 año) |
| **ADF en tiempo real** | Confirma estacionariedad del spread antes de operar |
| **Johansen rolling** | Detecta rupturas de cointegración en tiempo real |

---

## Gráficos generados

Guardados en `graficos/` con estilo dark profesional. Generados con `--modo evaluar` o `/evaluar T1 T2` en el bot.

| Archivo | Descripción |
|---|---|
| `01_curva_capital_T1_T2.png` | Curva de capital + drawdown superpuesto (límite 15% marcado) |
| `02_spread_zscore_T1_T2.png` | Spread y z-score con marcas de entrada/salida/stop |
| `03_monte_carlo_T1_T2.png` | 1000 trayectorias MC + distribución del capital final |
| `04_panel_metricas_T1_T2.png` | Panel resumen: todas las métricas con semáforos SMART |
| `07_rolling_cointegracion_T1_T2.png` | Estabilidad temporal de la cointegración (Johansen rolling) |

---

## Objetivos SMART

### Objetivo 1 — Detección de pares cointegrados ✓
- **S**: `deteccion.py` + `datos.py` — universo S&P 500 completo (~500 tickers, ~126.000 pares)
- **M**: CSV con score Johansen, p-value EG y número de observaciones por par
- **A**: Pipeline EG (pre-filtro ~0.001s/par) + Johansen (validador robusto)
- **R**: Base de toda la estrategia; se actualiza semanalmente
- **T**: Datos horarios de los últimos 12 meses (cointegración actual, no histórica)

### Objetivo 2 — Backtesting riguroso ✓
- **S**: `backtesting.py` — walk-forward, grid search IS/OOS 70/30, Monte Carlo, permutaciones
- **M**: Sharpe > 1.0 y MDD < 15% con semáforos visuales en panel de métricas
- **A**: Parámetros configurables: z-scores de entrada/salida, ventana OU, slippage, comisión
- **R**: Valida la viabilidad antes de operar en vivo (split estricto IS/OOS)
- **T**: Out-of-sample 2020–hoy (datos nunca vistos en la detección)

### Objetivo 3 — Automatización con fallback ✓
- **S**: `automatizacion.py` + `datos.py` — señales reproducibles con doble fuente de datos
- **M**: Pipeline diario genera `señales_diarias.csv` con z-score, ADF, régimen de volatilidad y alertas de ruptura de cointegración
- **A**: Alpaca primario + yfinance fallback automático ante cualquier error (credenciales, red, paquete)
- **R**: Operación continua independiente de la disponibilidad de Alpaca
- **T**: Señales disponibles cada día hábil en la apertura del mercado

### Objetivo 4 — Control desde Telegram ✓
- **S**: `bot_telegram.py` — espejo completo del CLI accesible desde cualquier dispositivo
- **M**: 9 comandos con respuesta en Markdown + envío automático de gráficos PNG
- **A**: `python bot_telegram.py` + token de BotFather + chat_id personal en `.env`
- **R**: Control total del sistema desde móvil sin acceso a terminal
- **T**: Disponible 24/7 mientras el proceso Python esté en ejecución
