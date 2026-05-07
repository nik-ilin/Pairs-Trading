# Sistema de Arbitraje Estadístico — Pairs Trading (2008–2026)

Este proyecto presenta un modelo cuantitativo de trading automatizado  basado en la cointegración estadística entre pares de acciones del mercado estadounidense. El sistema detecta pares cointegrados, genera señales de entrada y salida, y valida la estrategia mediante backtesting riguroso con múltiples métricas de riesgo. 

---

## Tabla de contenidos

1. [Descripción del proyecto](#descripción-del-proyecto)
2. [Alcance del proyecto](#alcance-del-proyecto)
3. [Fundamentos matemáticos](#fundamentos-matemáticos)
4. [Arquitectura del sistema](#arquitectura-del-sistema)
5. [Instalación](#instalación)
6. [Uso](#uso)
7. [Métricas de evaluación](#métricas-de-evaluación)
8. [Gráficos generados](#gráficos-generados)
9. [Objetivos SMART](#objetivos-smart)

---

## Descripción del proyecto

El **arbitraje estadístico por pares** (pairs trading) es una estrategia de mercado neutral que explota la relación histórica de largo plazo entre dos activos. Cuando el precio relativo entre ellos se aleja de su equilibrio estadístico, el modelo toma posiciones opuestas (comprar el barato, vender el caro) esperando que la relación se restablezca.

El sistema opera en dos fases:

| Fase | Periodo | Propósito |
|---|---|---|
| In-sample  | 2008–2020 | Detectar pares cointegrados |
| Out-of-sample | 2020–2026 | Backtesting y validación honesta |

---

## Alcance del proyecto

El alcance de este proyecto abarca el diseño, desarrollo y validación de una arquitectura de software automatizada en Python enfocada en el Pairs Trading. La herramienta busca identificar oportunidades de inversión aprovechando la relación histórica y la convergencia temporal entre activos. De esta forma, se pretende demostrar, mediante simulaciones, la viabilidad de obtener un beneficio económico consistente a través de un modelo matemático replicable; un sistema diseñado para adaptarse a los diferentes ciclos del mercado a largo plazo y maximizar la rentabilidad manteniendo un estricto control del riesgo.

---

## Fundamentos matemáticos

### 1. Detección de cointegración

No todas las empresas del mismo sector sirven. Nuestro sistema escanea miles de combinaciones en el índice S&P 500 y las somete a exámenes estadísticos (Tests de Engle-Granger y Johansen). Esto nos filtra el "ruido" y nos deja solo con aquellas parejas que tienen una unión matemática real y demostrable (ordenándolas de mejor a peor).

#### Test de Engle-Granger (pre-filtro)
Dado un par $(S_1, S_2)$, se ajusta la regresión:

$$\log S_{1,t} = \alpha + \beta \cdot \log S_{2,t} + \varepsilon_t$$

Si los residuos $\varepsilon_t$ son estacionarios (test ADF: $p < 0.05$), el par está cointegrado. Este test es computacionalmente barato y se usa como cribado inicial para reducir el universo de búsqueda $\approx 10\times$.

#### Test de Johansen (validador)
El test de traza de Johansen evalúa el rango de cointegración del sistema bivariante sin asumir una dirección de causalidad. Utiliza el estadístico de traza $\lambda_{traza}$ comparado con el valor crítico al 95%:

$$\text{Score} = \frac{\lambda_{traza}}{\text{Valor Crítico}_{95\%}}$$

Un score $> 1$ confirma la cointegración. Los pares se ordenan por score descendente.

---

### 2. Ratio de cobertura dinámico — Filtro de Kalman

La relación entre dos empresas no es rígida; cambia con los ciclos económicos. Si usáramos un modelo estático, el sistema fallaría con el tiempo. Por ello usamos una herramienta avanzada llamada Filtro de Kalman. Esto permite que nuestro algoritmo aprenda y ajuste la "longitud de la goma elástica" día a día, adaptándose a los cambios del negocio de forma dinámica.

El ratio de cobertura $\beta_t$ entre los dos activos no es constante en el tiempo. Un ratio estático (OLS) ignora cambios estructurales del negocio, rotaciones sectoriales y ciclos económicos.

El **Filtro de Kalman** modela $\beta_t$ como un proceso de paseo aleatorio y lo actualiza en cada nueva observación:

**Modelo de espacio de estados:**
$$\theta_t = \begin{bmatrix} \beta_t \\ \alpha_t \end{bmatrix}, \quad \theta_t = \theta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, V_w)$$

**Ecuación de observación:**
$$y_t = \mathbf{F}_t^\top \theta_t + \varepsilon_t, \quad \mathbf{F}_t = [x_t, 1]^\top, \quad \varepsilon_t \sim \mathcal{N}(0, V_e)$$

**Actualización (Ganancia de Kalman):**
$$K_t = P_{t|t-1} \mathbf{F}_t \cdot (V_e + \mathbf{F}_t^\top P_{t|t-1} \mathbf{F}_t)^{-1}$$
$$\theta_t = \theta_{t-1} + K_t(y_t - \mathbf{F}_t^\top \theta_{t-1})$$

El spread resultante es más estacionario que con OLS estático, lo que produce señales más fiables.

---

### 3. Proceso Ornstein-Uhlenbeck — Velocidad de reversión

Una vez detectamos que dos acciones se han separado, el sistema calcula su "vida media".Esto es vital: no es lo mismo invertir en una pareja que tarda 5 días en corregirse que en una que tarda 60 días. Nuestro modelo usa este tiempo para ajustar automáticamente sus expectativas.

El spread se modela como un proceso **Ornstein-Uhlenbeck (OU)**:

$$dS_t = \kappa(\mu - S_t)\,dt + \sigma\,dW_t$$

Donde:
- $\kappa$ = velocidad de reversión a la media
- $\mu$ = nivel de equilibrio del spread
- $\sigma$ = volatilidad del spread

En forma discreta (estimable por OLS):

$$\Delta S_t = a + b \cdot S_{t-1} + \varepsilon_t, \quad b = -\kappa \cdot \Delta t$$

La **vida media (half-life)** mide cuántos días tarda el spread en recorrer la mitad del camino hacia su media:

$$\text{Half-life} = \frac{\ln 2}{\kappa} = \frac{-\ln 2}{b}$$

Esta vida media se usa automáticamente como ventana del z-score en lugar de un valor fijo arbitrario. Un half-life de 20 días $\Rightarrow$ ventana de 20 días.

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
| $Z_t < -2.0\sigma$ | **LONG spread** (comprar $S_1$, vender $S_2$) | Spread barato: $S_1$ infravalorado relativo a $S_2$ |
| $Z_t > +2.0\sigma$ | **SHORT spread** (vender $S_1$, comprar $S_2$) | Spread caro: $S_1$ sobrevalorado relativo a $S_2$ |
| $|Z_t| < 0.5\sigma$ | **CERRAR posición** | Reversión completada |
| $|Z_t| > 3.5\sigma$ | **STOP-LOSS** | Ruptura de la relación de cointegración |

---

### 5. Dimensionado de posiciones — Volatility Scaling

Un buen sistema de inversión no solo busca ganar, sino proteger el dinero. Nuestro algoritmo incluye un sistema de ajuste por volatilidad.
Si el mercado está tranquilo, el sistema invierte un tamaño normal. Si el mercado está  inestable (alta volatilidad), el sistema reduce automáticamente la cantidad de dinero invertida. De esta forma, mantenemos el nivel de riesgo estrictamente controlado al 10% del capital por operación, pase lo que pase en el mundo exterior.

El tamaño de cada posición se ajusta inversamente a la volatilidad rolling del spread:

$$N_t = \frac{\text{Capital} \times f}{\sigma_{20d}(S_t)}$$

Donde $f = 10\%$ es la fracción de riesgo por trade. Esto mantiene una exposición al riesgo **constante en unidades monetarias** independientemente del régimen de volatilidad.

---

## Arquitectura del sistema

```
Finanzas/
├── datos.py          # Descarga y caché de precios (yfinance, Parquet)
├── deteccion.py      # Engle-Granger pre-filtro + Johansen validador
├── spread.py         # Filtro de Kalman, proceso OU, z-score, señales
├── backtesting.py    # Motor de backtesting, Walk-Forward, Monte Carlo
├── metricas.py       # Sharpe, Sortino, Calmar, Omega, VaR, CVaR, MDD
├── automatizacion.py # Pipeline diario de señales + ADF en tiempo real
├── evaluacion.py     # 11 tipos de gráficos para presentación
├── main.py           # Orquestador CLI
├── cache/            # Precios en Parquet (generado automáticamente)
├── graficos/         # Gráficos PNG exportados (generado automáticamente)
├── pares_cointegrados.csv  # Resultado del scan
└── señales_diarias.csv     # Señales del día
```

**Flujo de datos:**

```
datos.py → deteccion.py → spread.py → backtesting.py → metricas.py
                                    ↓
                           automatizacion.py (modo diario)
                                    ↓
                           evaluacion.py (gráficos)
```

---

## Instalación

```bash
pip install yfinance pandas numpy matplotlib statsmodels scipy
```

---

## Uso

### Pipeline completo (recomendado la primera vez)
```bash
python main.py --modo full
```

### Detectar pares cointegrados en el universo S&P 500
```bash
python main.py --modo scan
```

### Backtesting de un par específico con optimización y gráficos
```bash
python main.py --modo backtest --par AAPL MSFT --optimizar --walk-forward --graficos
```

### Backtesting de los 10 mejores pares detectados
```bash
python main.py --modo backtest --top-n 10 --graficos
```

### Generar informe visual completo de un par
```bash
python main.py --modo evaluar --par KO PEP
```

### Señales del día (ejecutar al cierre del mercado, 22:05 UTC)
```bash
python main.py --modo señales
```

---

## Métricas de evaluación

### Métricas de rendimiento

| Métrica | Fórmula | Objetivo |
|---|---|---|
| **CAGR** | $(V_f / V_0)^{1/n} - 1$ | Máximo posible |
| **Sharpe Ratio** | $\bar{r}_e / \sigma_r \cdot \sqrt{252}$ | **> 1.0** |
| **Sortino Ratio** | $\bar{r}_e / \sigma_{down} \cdot \sqrt{252}$ | > 1.5 |
| **Calmar Ratio** | $\text{CAGR} / |\text{MDD}|$ | > 0.5 |
| **Omega Ratio** | $\sum \text{ganancias} / \sum \text{pérdidas}$ | > 1.5 |

### Métricas de riesgo

| Métrica | Descripción | Objetivo |
|---|---|---|
| **Máx. Drawdown (MDD)** | Mayor caída pico-a-valle | **< 15%** |
| **VaR 95%** | Pérdida máxima en el 95% de los días | Referencia |
| **CVaR 95%** | Pérdida media en el peor 5% de casos | < VaR × 1.5 |

> El **CVaR** (también llamado *Expected Shortfall*) es superior al VaR porque captura el comportamiento de la cola de la distribución. Dos estrategias con el mismo VaR pueden tener CVaR muy diferentes.

### Validación estadística

| Test | Propósito |
|---|---|
| **Bootstrap Sharpe** | Intervalo de confianza del Sharpe (1000 remuestras) |
| **Test de permutaciones** | Verifica que los retornos no son aleatorios ($p < 0.05$) |
| **Monte Carlo** | Distribución de resultados futuros (1000 trayectorias) |
| **ADF en tiempo real** | Confirma estacionariedad del spread antes de operar |
| **Johansen rolling** | Detecta rupturas de cointegración en tiempo real |

---

## Gráficos generados

Todos los gráficos se guardan en `graficos/` con estilo dark profesional.

| Archivo | Descripción |
|---|---|
| `01_curva_capital_*.png` | Curva de capital con drawdown superpuesto y límite 15% |
| `02_spread_zscore_*.png` | Spread y z-score con marcas de entrada/salida |
| `03_rolling_sharpe_*.png` | Sharpe Ratio rodante (ventana 252 días) |
| `04_distribucion_retornos_*.png` | Histograma de retornos con distribución normal y VaR/CVaR |
| `05_heatmap_mensual_*.png` | Heatmap de retornos mensuales por año |
| `06_monte_carlo_*.png` | Abanico de trayectorias MC + distribución del capital final |
| `07_rolling_cointegracion_*.png` | Estabilidad temporal de la cointegración (Johansen) |
| `08_precio_relativo_*.png` | Precios normalizados y beta dinámico (Kalman) |
| `09_walk_forward_*.png` | Sharpe Ratio por ventana walk-forward |
| `10_analisis_trades_*.png` | Duración y PnL de cada trade |
| `11_panel_metricas_*.png` | Panel resumen con todas las métricas y semáforos SMART |

---

## Objetivos SMART

### Objetivo 1 — Detección de pares cointegrados ✓
- **S**: `deteccion.py` + `datos.py` — universo S&P 500, periodos 2008–2026
- **M**: CSV con pares validados, ordenados por score Johansen
- **A**: Pipeline EG (pre-filtro) + Johansen (validador)
- **R**: Base de toda la estrategia
- **T**: Datos desde 2008-01-01

### Objetivo 2 — Backtesting parametrizable ✓
- **S**: `backtesting.py` — motor walk-forward, grid search, Monte Carlo
- **M**: Validación Sharpe > 1.0 y MDD < 15% con semáforos visuales
- **A**: Parámetros configurables: entrada/salida z-score, ventana OU, slippage
- **R**: Valida la viabilidad antes de operar en vivo
- **T**: Out-of-sample 2020–2026

### Objetivo 3 — Automatización y sostenibilidad ✓
- **S**: `automatizacion.py` — señales reproducibles con tests estadísticos diarios
- **M**: Señales CSV + JSON con z-score, ADF, régimen de volatilidad, alerta de ruptura
- **A**: `python main.py --modo señales` (programable con cron/scheduler)
- **R**: Operación continua a corto, medio y largo plazo
- **T**: Pipeline listo para producción
