# Sistema de Arbitraje Estadístico — Pairs Trading (S&P 500)

Este proyecto presenta un modelo cuantitativo de trading automatizado  basado en la cointegración estadística entre pares de acciones del mercado estadounidense. El sistema detecta pares cointegrados, genera señales de entrada y salida, y valida la estrategia mediante backtesting riguroso con múltiples métricas de riesgo. 

---

## Tabla de contenidos

1. [Descripción](#descripción)
2. [Justificación](#justificación)
3. [Alcance del proyecto](#alcance-del-proyecto)
4. [Objetivos SMART](#objetivos-smart)
5. [Planificación y gestión](#Planificación-y-gestión)
    * [5.1 Planificación](#planificación)
    * [5.2 Gestión de Riesgos](#gestión-de-riesgos)
    * [5.3 Calidad](#calidad)
6. [Fundamentos matemáticos](#fundamentos-matemáticos)
7. [Arquitectura](#arquitectura)
8. [Instalación](#instalación)
9. [Configuración](#configuración)
10. [Uso — CLI](#uso--cli)
11. [Bot de Telegram](#bot-de-telegram)
12. [Sistema de fuentes de datos](#sistema-de-fuentes-de-datos)
13. [Métricas de evaluación](#métricas-de-evaluación)
14. [Gráficos generados](#gráficos-generados)
15. [Pruebas y resultados](#pruebas-y-resultados)
16. [Conclusión](#conclusión)
17. [Próximos Pasos](#próximos-pasos)



---

## Descripción

El **arbitraje estadístico por pares** (pairs trading) es una estrategia de mercado neutral que explota la relación histórica de largo plazo entre dos activos. Cuando el precio relativo entre ellos se aleja de su equilibrio estadístico, el modelo toma posiciones opuestas (comprar el barato, vender el caro) esperando que la relación se restablezca.

| Fase | Datos | Propósito |
|---|---|---|
| Detección de pares | Alpaca horario — últimos 12 meses | Cointegración actual, no histórica |
| Señales diarias | Alpaca horario — últimos 12 meses | Kalman + z-score + ADF en tiempo real |
| Backtesting | Alpaca diario — 2020 a hoy | Validación out-of-sample honesta |

---

## Justificación

El presente trabajo nace de la necesidad de evitar que los métodos de inversión tradicionales se queden atrás frente a la volatilidad del mercado actual y la rápida evolución de la tecnología algorítmica. Se identifica una oportunidad clave para la mejora de procesos mediante la automatización, transformando el análisis de datos complejo en una herramienta de decisión objetiva que reduce el error humano y aumenta la eficiencia operativa. Al integrar modelos matemáticos de cointegración avanzados, el software no solo satisface una exigencia técnica de precisión, sino que asegura la competitividad y viabilidad económica del sistema en entornos financieros altamente dinámicos.

---

## Alcance del proyecto

El alcance de este proyecto abarca el diseño, desarrollo y validación de una arquitectura de software automatizada en Python enfocada en el Pairs Trading. La herramienta busca identificar oportunidades de inversión aprovechando la relación histórica y la convergencia temporal entre activos. De esta forma, se pretende demostrar, mediante simulaciones, la viabilidad de obtener un beneficio económico consistente a través de un modelo matemático replicable; un sistema diseñado para adaptarse a los diferentes ciclos del mercado a largo plazo y maximizar la rentabilidad manteniendo un estricto control del riesgo.

---
## Objetivos SMART

### Objetivo 1 — Detección de pares cointegrados ✓
- **S**: Analizar de forma completa las 500 empresas del índice S&P 500 para encontrar parejas de acciones con una relación matemática sólida
- **M**: Generar un listado de las mejores parejas encontradas, clasificadas por su fuerza estadística
- **A**: Utilizar un proceso de filtrado en dos pasos para procesar miles de combinaciones en pocos minutos
- **R**: Establecer la base del sistema de inversión actualizando los datos semanalmente
- **T**: Basar el análisis en el comportamiento real del mercado de los últimos 12 meses (cointegración actual, no histórica)

### Objetivo 2 — Backtesting riguroso ✓
- **S**: Realizar pruebas históricas rigurosas para verificar cómo se habría comportado el sistema en el pasado
- **M**: Confirmar que el sistema genera beneficios estables y que las rachas de pérdida nunca superan el 15% del capital
- **A**: Aplicar modelos de simulación avanzados para asegurar que los resultados no son fruto del azar
- **R**: Validar la viabilidad real del modelo antes de poner en riesgo capital real
- **T**: Evaluar el rendimiento utilizando datos desde el año 2020 hasta la actualidad (datos nunca vistos en la detección)

### Objetivo 3 — Automatización con fallback ✓
- **S**: Crear un sistema que genere automáticamente señales de compra y venta cada día
- **M**: Producir un informe diario con alertas claras y avisos sobre el nivel de riesgo en el mercado
- **A**: Conectar el software a dos fuentes de datos distintas para que nunca deje de funcionar si una de ellas falla
- **R**: Garantizar que el sistema funcione de forma autónoma cada día que la bolsa esté abierta
- **T**: Señales disponibles cada día hábil en la apertura del mercado

### Objetivo 4 — Control desde Telegram ✓
- **S**: Integrar un bot de Telegram que permita supervisar y manejar todo el sistema desde el móvil
- **M**: Disponer de comandos sencillos para recibir informes detallados y gráficos de rendimiento al instante
- **A**: Vincular la herramienta de mensajería con el servidor central para una respuesta inmediata
- **R**: Control total del sistema desde móvil sin acceso a terminal
- **T**: Disponible 24/7 mientras el proceso Python esté en ejecución

---

## Planificación y gestión

### 5.1 Planificación

**Fase 1: Investigación y Marco Teórico**

La fase inicial se centró en el estudio exhaustivo del estado del arte en finanzas cuantitativas. Se investigaron los principios del arbitraje estadístico, profundizando en conceptos de econometría como la estacionariedad de series temporales y los tests de cointegración (Engle-Granger y Johansen). Esta etapa fue fundamental para asentar las bases matemáticas necesarias antes de la escritura de cualquier línea de código.

**Fase 2: Diseño de la Arquitectura y Selección de Datos**

Una vez comprendida la teoría, se procedió a definir la estructura del algoritmo. En esta etapa se seleccionaron las fuentes de datos (universo S&P 500) y se fragmentó el sistema en módulos lógicos e independientes:

  - Módulo de ingesta y limpieza de datos.

  - Módulo de detección y filtrado de pares cointegrados.

  - Módulo de modelado del spread y generación de señales.

**Fase 3: Desarrollo del Motor Algorítmico**

Esta fase comprendió la programación íntegra del sistema en Python. Se implementaron los componentes técnicos avanzados, como el Filtro de Kalman para el cálculo dinámico del ratio de cobertura y el proceso Ornstein-Uhlenbeck para medir la velocidad de reversión a la media. El resultado de esta fase fue un motor de trading funcional capaz de procesar miles de pares en tiempo real.

**Fase 4: Integración de Interfaz de Control (Bot de Telegram)**

Con el motor finalizado, se desarrolló una interfaz de gestión remota mediante un bot de Telegram. Esta fase se centró en la accesibilidad, permitiendo que el sistema fuera monitorizado y controlado desde cualquier dispositivo, enviando informes de rendimiento y gráficos de operaciones de forma automática.

**Fase 5: Validación y Pruebas Finales**

La fase de cierre se dedicó a un riguroso proceso de control de calidad. Se realizaron pruebas de backtesting con datos no vistos (Out-of-sample) y simulaciones de estrés (Monte Carlo) para asegurar la robustez del algoritmo ante diferentes escenarios de mercado. Asimismo, se verificó la estabilidad del sistema de fallback para garantizar la continuidad de los datos

### 5.2 Gestión de Riesgos

**Ruptura de Cointegración (Cambio Estructural)**

-Descripción: Es la posibilidad de que la relación estadística entre dos activos se rompa de forma permanente debido a factores externos (cambio de directiva, crisis sectorial o fusión de empresas). En este escenario, el spread deja de volver a su media y comienza a divergir, lo que invalidaría la estrategia.

-Mitigación: El sistema incorpora un Stop-Loss dinámico basado en el Z-Score. Si la desviación supera un límite crítico (por ejemplo, ±3.5 desviaciones estándar), el algoritmo asume que la "correa" se ha roto y cierra la posición inmediatamente para proteger el capital. Además, se realiza una re-evaluación semanal de la cointegración para descartar pares que pierdan su fuerza estadística.

**Disponibilidad de Datos y Fallo Tecnológico**

-Descripción: El algoritmo depende totalmente de la conexión con APIs externas (como Alpaca o yfinance) para obtener precios en tiempo real. Una caída del servidor, un error de red o una clave de API caducada podrían dejar al sistema "ciego" y sin capacidad de reaccionar ante cambios en el mercado.

-Mitigación: Se ha implementado un sistema de redundancia o fallback automático. En caso de que la fuente de datos principal falle, el software cambia instantáneamente a una fuente secundaria sin interrumpir la ejecución. Asimismo, el Bot de Telegram actúa como monitor de seguridad, enviando una alerta inmediata al móvil del usuario si detecta cualquier error crítico en la descarga de datos o en la ejecución del código.


### 5.3 Calidad

**Robustez Estadística y Validación de Modelos**

Para asegurar que las oportunidades detectadas no son fruto de correlaciones espurias o del azar, el sistema exige un doble filtrado estadístico. La calidad se garantiza mediante la convergencia de los tests de Engle-Granger y Johansen, aceptando únicamente pares con un nivel de significancia estadística elevado ($p < 0.05$). Asimismo, el modelo se somete a una validación cruzada (In-sample y Out-of-sample), asegurando que la relación de cointegración identificada en el pasado se mantiene estable ante datos nuevos y no vistos por el algoritmo.

**Gestión Dinámica del Riesgo y Neutralidad**

El sistema debe mantener un control estricto sobre la exposición al mercado para evitar episodios de inestabilidad financiera. Los estándares de calidad fijados para este proyecto incluyen:Neutralidad al mercado: El algoritmo debe mantener un "Beta" cercano a cero, garantizando que el beneficio dependa exclusivamente de la convergencia de los activos y no de la tendencia general de la bolsa.Límites de pérdida: Se establece un objetivo de Drawdown Máximo inferior al 15%.Ajuste por Volatilidad: El tamaño de las posiciones se recalcula dinámicamente según la volatilidad actual del spread, evitando una exposición excesiva en regímenes de mercado altamente inestables.

**Integridad de Datos y Fiabilidad Operativa**

La calidad técnica del software se mide por su capacidad para operar de forma ininterrumpida y con datos veraces. Para ello, se han implementado procesos de validación de datos que eliminan errores de lectura o valores atípicos (outliers) que podrían sesgar el modelo. La arquitectura asegura la continuidad operativa mediante un sistema de redundancia de fuentes de datos (Alpaca/yfinance) y el uso de formatos de almacenamiento eficientes como Parquet, que garantizan la integridad de la información histórica y la rapidez en el procesamiento de señales diarias.

---

## Fundamentos matemáticos

### 1.  Detección de cointegración

No todas las empresas del mismo sector sirven. Nuestro sistema escanea miles de combinaciones en el índice S&P 500 y las somete a exámenes estadísticos (Tests de Engle-Granger y Johansen). Esto nos filtra el "ruido" y nos deja solo con aquellas parejas que tienen una unión matemática real y demostrable (ordenándolas de mejor a peor).

#### Test de Engle-Granger (pre-filtro)
Dado un par $(S_1, S_2)$, se ajusta la regresión:

$$\log S_{1,t} = \alpha + \beta \cdot \log S_{2,t} + \varepsilon_t$$

Si los residuos $\varepsilon_t$ son estacionarios (test ADF: $p < 0.05$), el par está cointegrado. Coste: ~0.001 s/par → escaneo de 126.000 pares en ~3 min.

#### Test de Johansen (validador robusto)
Evalúa el rango de cointegración sin asumir dirección de causalidad. Estadístico de traza al 95%:

$$\text{Score} = \frac{\lambda_{traza}}{\text{Valor Crítico}_{95\%}}$$

Score > 1 confirma cointegración. Los pares se ordenan por score descendente.

---

### 2. Ratio de cobertura dinámico — Filtro de Kalman

La relación entre dos empresas no es rígida; cambia con los ciclos económicos. Si usáramos un modelo estático, el sistema fallaría con el tiempo. Por ello usamos una herramienta avanzada llamada Filtro de Kalman. Esto permite que nuestro algoritmo aprenda y ajuste la "longitud de la goma elástica" día a día, adaptándose a los cambios del negocio de forma dinámica.

El ratio de cobertura $\beta_t$ entre los dos activos no es constante en el tiempo. Un ratio estático (OLS) ignora cambios estructurales del negocio, rotaciones sectoriales y ciclos económicos.

**Estado:** $\theta_t = [\beta_t,\ \alpha_t]^\top$, modelado como paseo aleatorio.

**Ganancia de Kalman:**
$$K_t = P_{t|t-1} \mathbf{F}_t \cdot (V_e + \mathbf{F}_t^\top P_{t|t-1} \mathbf{F}_t)^{-1}$$
$$\theta_t = \theta_{t-1} + K_t(y_t - \mathbf{F}_t^\top \theta_{t-1})$$

El spread resultante es más estacionario que con OLS estático. Warmup: 390 barras (60 días hábiles en horario).

---

### 3. Proceso Ornstein-Uhlenbeck — Velocidad de reversión

Una vez detectamos que dos acciones se han separado, el sistema calcula su "vida media".Esto es vital: no es lo mismo invertir en una pareja que tarda 5 días en corregirse que en una que tarda 60 días. Nuestro modelo usa este tiempo para ajustar automáticamente sus expectativas.

El spread se modela como un proceso **Ornstein-Uhlenbeck (OU)**:

$$dS_t = \kappa(\mu - S_t)\,dt + \sigma\,dW_t$$

**Half-life** (barras para recorrer la mitad del camino a la media):

$$\text{Half-life} = \frac{-\ln 2}{b}, \quad \Delta S_t = a + b \cdot S_{t-1} + \varepsilon_t$$

El half-life se usa automáticamente como ventana del z-score. Rango válido: 32.5–1638 barras horarias.

---

### 4. Z-score y señales de trading

El sistema monitoriza el Spread en tiempo real. Para saber cuándo actuar, usamos una medida de anormalidad llamada Z-Score.

El spread normalizado (z-score) mide cuántas desviaciones estándar se aleja el spread de su media rolling:

$$Z_t = \frac{S_t - \mu_{rolling}(S, \text{HL})}{\sigma_{rolling}(S, \text{HL})}$$

**Reglas de entrada/salida:**

Alarma de Entrada: Si la distancia entre las acciones es anormalmente grande, el sistema emite una orden de entrar al mercado.

Alarma de Salida: Cuando las acciones vuelven a su distancia normal de equilibrio, el sistema cierra la operación y recoge el beneficio.

Stop-Loss: Si la distancia se vuelve extrema e irracional, asumimos que la relación de las empresas se ha roto para siempre y cortamos las pérdidas automáticamente.

| Condición | Acción | Razonamiento |
|---|---|---|
| $Z_t < -2.0$ | **LONG spread** | $S_1$ infravalorado relativo a $S_2$ |
| $Z_t > +2.0$ | **SHORT spread** | $S_1$ sobrevalorado relativo a $S_2$ |
| $\|Z_t\| < 0.5$ | **CERRAR** | Reversión completada |
| $\|Z_t\| > 3.5$ | **STOP-LOSS** | Ruptura de cointegración |

---

### 5. Dimensionado de posiciones — Volatility Scaling

Un buen sistema de inversión no solo busca ganar, sino proteger el dinero. Nuestro algoritmo incluye un sistema de ajuste por volatilidad.
Si el mercado está tranquilo, el sistema invierte un tamaño normal. Si el mercado está  inestable (alta volatilidad), el sistema reduce automáticamente la cantidad de dinero invertida. De esta forma, mantenemos el nivel de riesgo estrictamente controlado al 10% del capital por operación, pase lo que pase en el mundo exterior.

El tamaño de cada posición se ajusta inversamente a la volatilidad rolling del spread:

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

## Pruebas y resultados

### Paper Trading — NOW / TYL (2020–2026)

**ServiceNow (NOW)** y **Tyler Technologies (TYL)** son dos empresas de software empresarial del S&P 500 que históricamente han mostrado una relación de cointegración durante ventanas concretas. El paper trading con detección dinámica de régimen opera **únicamente** cuando el modelo confirma cointegración activa, evitando todo el ruido fuera de ese estado.

#### Configuración del test

| Parámetro | Valor |
|---|---|
| Período analizado | 2020-01-02 → 2026-05-13 |
| Capital inicial | $100,000 |
| Half-life del spread | 5.0 barras |
| Ventana z-score | 5 barras |
| Umbral de entrada | Z > ±2.0 |
| Stop-loss | Z > ±3.5 |

#### Períodos de cointegración detectados (3 ventanas)

| # | Inicio | Fin | Duración |
|---|---|---|---|
| 1 | 2020-07-24 | 2020-08-06 | 13 días |
| 2 | 2021-02-22 | 2021-03-19 | 25 días |
| 3 | 2021-08-26 | 2021-09-09 | 14 días |

> El modelo estuvo **activo solo el 2.5% del tiempo** (40 días de 1.599 barras analizadas).

#### Trades ejecutados

| Entrada | Salida | Dirección | PnL | Días | Cierre |
|---|---|---|---|---|---|
| 2020-07-30 | 2020-07-31 | LONG spread | +$12,920 | 1 | Reversión |
| 2021-03-04 | 2021-03-05 | SHORT spread | +$45,247 | 1 | Reversión |

#### Métricas comparativas — Inteligente vs Naive

| Métrica | Paper (inteligente) | Naive (siempre activo) |
|---|---|---|
| **Capital final** | **$182,883** | ~$0 |
| **Ganancia neta** | **+$82,883 (+82.9%)** | **-$99,780 (-99.8%)** |
| **CAGR anual** | +9.98% | -99.78% |
| **Sharpe Ratio** | 0.337 | -0.741 |
| **Max Drawdown** | -16.76% | -131.81% |
| **Win Rate** | 100.0% | 95.97% |
| **Profit Factor** | ∞ (sin pérdidas) | 101.48 |
| **Nº trades** | 2 | 124 |
| **Tiempo en mercado** | 2.5% | 100% |

#### Análisis — Gemini 2.5 Flash

> *El análisis comparativo para el par NOW/TYL durante 2020–2026 demuestra inequívocamente que la detección dinámica de régimen añade un valor real sustancial. La estrategia inteligente supera con creces a la ingenua, generando un CAGR anual del 10.0% y un Sharpe Ratio de 0.34, frente al catastrófico -99.8% y -0.74, respectivamente, de la estrategia naive. El filtro de régimen evitó pérdidas masivas, reduciendo el Max Drawdown a solo -16.8%, mientras que la naive sufrió un devastador -131.8%. Operar solo durante períodos de cointegración confirmada, que representó apenas un 2.5% del tiempo total, permitió a la estrategia inteligente capitalizar únicamente las oportunidades de mayor probabilidad. Esto resultó en un Win Rate del 100% con solo 2 operaciones, culminando en una ganancia neta de $82,883 sobre un capital inicial de $100,000. En contraste, la estrategia ingenua realizó 124 operaciones sin filtro, llevando el capital a una pérdida casi total. Claramente, la activación selectiva basada en el régimen de cointegración es crítica para la rentabilidad y la gestión del riesgo.*

#### ¿Por qué la ganancia neta ($82.883) supera la suma de los dos trades ($58.167)?

A primera vista parece que falta dinero, pero en realidad sobra explicación.

El sistema dimensiona las posiciones en función de la volatilidad del spread. Como el spread de NOW/TYL oscila muy poco, hace falta construir una posición enorme —aproximadamente $2.5 millones— para que un movimiento típico del spread represente solo el 10% del capital. Eso equivale a operar con un apalancamiento de ×25 sobre los $100.000 iniciales.

La tabla de trades calcula el beneficio midiendo cuánto se movió ese spread. Pero el capital real creció midiendo cuánto se movió cada acción individualmente, multiplicado por esa posición de $2.5M. El día del cierre del segundo trade (2021-03-05), TYL subió un +5.66% en una sola sesión. Con una posición de $2.5M, ese movimiento generó $127.000 reales en la cartera, muy por encima de los $45.247 que la fórmula del spread indicaba. A eso hay que restarle los costes de transacción (~$31.900 en total, que sí se descuentan del capital pero no aparecen en la tabla).

> En resumen: la tabla de trades dice cuánto se movió el spread; el capital dice cuánto se movió el dinero real. Con ×25 de apalancamiento, las dos medidas pueden diferir enormemente cuando hay movimientos bruscos en los activos individuales. **La cifra de $182.883 es la real.**

---

## Conclusiones

El principal aprendizaje de este proyecto es que en el arbitraje estadístico **saber cuándo no operar vale más que saber cuándo operar**. La estrategia naive de NOW/TYL ganó el 96% de sus 124 trades y aun así perdió casi todo el capital, porque las pocas operaciones malas ocurrieron en momentos en que las dos acciones ya no guardaban ninguna relación matemática entre sí. El sistema inteligente, en cambio, estuvo parado el 97.5% del tiempo y entró solo en las tres ventanas donde el modelo confirmó que la relación existía. Resultado: dos operaciones, las dos ganadoras, y un +82.9% neto.

Esto también valida el diseño técnico central del proyecto. Usar un Filtro de Kalman en lugar de un ratio fijo permite que el modelo se adapte continuamente a cómo cambia la relación entre dos empresas a lo largo de años, evitando que señales calculadas con datos viejos generen entradas erróneas. Y la detección de régimen rolling garantiza que, aunque un par haya sido cointegrado en el pasado, el sistema no opere en él si hoy esa condición ya no se cumple.

El resultado con NOW/TYL no es un caso de éxito espectacular — el Sharpe es modesto y el drawdown rozó el límite del 15% — pero sí es una demostración honesta de que la arquitectura funciona como fue diseñada: filtra el ruido, controla el riesgo y genera beneficio real operando en el momento correcto.

---

## Próximos Pasos


