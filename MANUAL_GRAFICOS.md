# Manual de Interpretación de Gráficos
## Sistema de Arbitraje Estadístico — Pairs Trading

Este manual explica cómo leer cada uno de los 5 gráficos que genera el sistema, qué conclusiones extraer de ellos y qué señales de alerta buscar.

Los gráficos se guardan en la carpeta `graficos/` con fondo oscuro profesional.

---

## Gráfico 01 — Curva de Capital y Drawdown

**Archivo:** `01_curva_capital_TICKER1_TICKER2.png`

### Qué muestra

| Panel | Contenido |
|---|---|
| **Superior (3/4)** | Evolución del capital en dólares a lo largo del tiempo |
| **Inferior (1/4)** | Drawdown: caída porcentual desde el máximo histórico hasta el mínimo local |

**Elementos del panel superior:**
- Línea azul = curva de capital (comienza en $100,000 por defecto)
- Zona verde translúcida = periodo por encima del capital inicial (ganando dinero)
- Zona roja translúcida = periodo por debajo del capital inicial (perdiendo dinero)
- Línea gris punteada = nivel de capital inicial ($100,000)

**Elementos del panel inferior:**
- Área roja = profundidad del drawdown en ese momento (negativo siempre)
- Línea amarilla punteada = límite objetivo del 15% de caída máxima (SMART)

### Cómo leerlo

1. **Tendencia general:** La línea azul debe subir de forma sostenida. Si cae por debajo del nivel inicial durante periodos largos, la estrategia no está funcionando.

2. **Drawdown:** Un drawdown de –10% significa que desde el pico anterior, el capital ha caído un 10%. Cuanto más tiempo pase en zona roja, peor.

3. **Objetivo SMART:** La línea amarilla punteada marca el –15%. Si el drawdown supera esa línea, la estrategia viola el criterio de riesgo establecido.

### Conclusiones que debes extraer

| Observación | Conclusión |
|---|---|
| Línea azul con tendencia alcista sostenida | Estrategia rentable a largo plazo |
| Zona verde dominante | Capital por encima del inicial la mayor parte del tiempo |
| Drawdown siempre por encima de –15% | Riesgo controlado — SMART objetivo cumplido |
| Caídas bruscas y recuperación rápida | Drawdowns pasajeros, reversión funciona |
| Drawdown supera –15% y tarda en recuperarse | Riesgo elevado, posible ruptura de cointegración |
| Capital por debajo del inicial más del 30% del tiempo | Estrategia marginal en ese par |

---

## Gráfico 02 — Spread y Z-score con Señales de Trading

**Archivo:** `02_spread_zscore_TICKER1_TICKER2.png`

### Qué muestra

| Panel | Contenido |
|---|---|
| **Superior (1/3)** | Spread crudo: diferencia logarítmica entre los dos activos, ajustada por el ratio de cobertura dinámico (Kalman) |
| **Inferior (2/3)** | Z-score del spread: cuántas desviaciones estándar se aleja el spread de su media reciente |

**Elementos del panel inferior:**
- Línea azul = z-score a lo largo del tiempo
- Líneas rojas punteadas en ±2σ = umbrales de entrada
- Líneas grises punteadas en ±0.5σ = umbrales de cierre
- Triángulos verdes (▲) = entrada LONG spread (comprar T1, vender T2)
- Triángulos rojos (▼) = entrada SHORT spread (vender T1, comprar T2)
- Cruces amarillas (✕) = cierre de posición
- Zonas sombreadas = spread fuera de los umbrales de entrada

### Cómo leerlo

**El spread** es la diferencia ajustada entre los dos activos. Si el par está cointegrado, el spread debe oscilar alrededor de cero sin alejarse indefinidamente.

**El z-score** normaliza el spread para comparar con desviaciónes históricas:
- z = 0 → spread en su media histórica (no hay oportunidad)
- z = +2 → spread 2 desviaciones estándar por encima (T1 caro relativo a T2 → SHORT)
- z = –2 → spread 2 desviaciones estándar por debajo (T1 barato relativo a T2 → LONG)

**Las señales** deben aparecer en los extremos del z-score y los cierres deben ocurrir cuando el spread vuelve al centro.

### Conclusiones que debes extraer

| Observación | Conclusión |
|---|---|
| Spread oscila cerca de cero con reversiones claras | Par bien cointegrado, spread estacionario |
| Señales ▲ y ▼ en los picos y valles del z-score | El modelo captura las desviaciones correctamente |
| Cierres (✕) cerca del cero | Las posiciones se mantienen el tiempo justo para aprovechar la reversión |
| Spread con tendencia sin oscilaciones | El par puede haber perdido la cointegración — no operar |
| z-score supera ±3.5 (stop-loss) | Evento extremo o ruptura del par — el sistema para la posición automáticamente |
| Pocos trades con señales esparcidas | Normal; el modelo sólo entra cuando la oportunidad es estadísticamente significativa |

---

## Gráfico 03 — Simulación Monte Carlo

**Archivo:** `03_monte_carlo_TICKER1_TICKER2.png`

### Qué muestra

| Panel | Contenido |
|---|---|
| **Izquierdo** | 200 trayectorias simuladas del capital durante el próximo año (1 año hacia adelante) |
| **Derecho** | Distribución del capital final tras 1,000 simulaciones |

**Panel izquierdo:**
- Líneas azules translúcidas = trayectorias individuales (muestra del camino posible)
- Línea roja = percentil 5 (peor 5% de escenarios)
- Línea amarilla inferior = percentil 25
- Línea azul gruesa = percentil 50 (escenario mediano)
- Línea amarilla superior = percentil 75
- Línea verde = percentil 95 (mejor 5% de escenarios)
- Línea gris punteada = capital inicial ($100,000)

**Panel derecho:**
- Histograma azul = distribución del capital final en las 1,000 simulaciones
- Línea gris = capital inicial
- P(ganancia) en el título = % de simulaciones que terminaron con ganancia

### Cómo leerlo

El Monte Carlo muestra el **rango de resultados posibles** basándose en la distribución histórica de retornos del backtest. No predice el futuro exacto, sino que cuantifica la incertidumbre.

**Lo más importante:**
1. ¿La mediana (P50) supera el capital inicial? → La estrategia es rentable en el escenario típico
2. ¿El P5 preserva al menos el 80% del capital? → En el peor 5% de casos, las pérdidas son manejables
3. ¿P(ganancia) > 60%? → Más probable ganar que perder en el horizonte de un año

### Conclusiones que debes extraer

| Observación | Conclusión |
|---|---|
| P50 > $100,000 y distribución sesgada hacia la derecha | Expectativa positiva — estrategia rentable |
| P5 > $85,000 | Protección razonable incluso en escenarios adversos |
| P(ganancia) > 65% | Alta probabilidad de resultado positivo |
| Distribución muy dispersa (P5 a P95 muy separados) | Alta incertidumbre — estrategia volátil |
| P50 < $100,000 | Estrategia no rentable en expectativa — revisar el par |
| P5 < $70,000 | Riesgo de cola excesivo — reducir tamaño de posición |

---

## Gráfico 04 — Cointegración Rolling (Johansen)

**Archivo:** `04_rolling_cointegracion_TICKER1_TICKER2.png`

### Qué muestra

Evolución temporal de la relación de cointegración entre los dos activos, calculada en ventanas deslizantes de 252 días (~1 año).

**Elementos:**
- Línea azul = estadístico de traza de Johansen (calculado cada día sobre los últimos 252 días)
- Línea roja punteada = valor crítico al 95% de confianza (≈ 15.4 para el test estándar)
- Zona verde = cuando la traza supera el crítico → par **cointegrado** en esa ventana
- Zona roja = cuando la traza no supera el crítico → par **NO cointegrado** en esa ventana

### Cómo leerlo

El estadístico de traza de Johansen mide la fuerza de la relación de largo plazo entre los dos activos:
- **Traza > Crítico (zona verde):** Los precios están ligados estadísticamente. El spread tiende a volver a la media. Es seguro operar.
- **Traza < Crítico (zona roja):** La relación se ha debilitado o roto. El spread puede alejarse indefinidamente. No operar en esta zona.

### Conclusiones que debes extraer

| Observación | Conclusión |
|---|---|
| Zona verde dominante (>80% del tiempo) | Par muy estable y fiable para trading |
| Pequeñas caídas breves a zona roja y rápida recuperación | Par robusto; rupturas momentáneas, no estructurales |
| Traza muy por encima del crítico (ratio > 1.5) | Cointegración fuerte — mayor confianza en las señales |
| Largos periodos en zona roja (>6 meses) | Ruptura estructural — el par puede no ser operable en ese régimen |
| Traza cayendo de forma sostenida hacia el crítico | Señal temprana de deterioro — monitorizar o reducir exposición |
| Patrón cíclico verde/rojo regular | La cointegración es estacional; el sistema de alerta de `automatizacion.py` gestionará esto |

---

## Gráfico 05 — Panel de Métricas

**Archivo:** `05_panel_metricas_TICKER1_TICKER2.png`

### Qué muestra

Tabla resumen con todas las métricas de rendimiento y riesgo, organizada en tres categorías. Las marcas ✓ y ✗ indican si la estrategia cumple los objetivos SMART.

**Secciones de la tabla:**

| Sección | Métricas incluidas |
|---|---|
| **RENDIMIENTO** | CAGR, Sharpe, Sortino, Calmar |
| **RIESGO** | Máx. Drawdown, VaR 95%, CVaR 95% |
| **TRADES** | N° trades, Win Rate, Profit Factor |
| **VALIDACIÓN** | IC Bootstrap Sharpe, p-value permutaciones |

### Glosario de métricas

| Métrica | Qué mide | Objetivo |
|---|---|---|
| **CAGR** | Rentabilidad anual compuesta | Máximo posible |
| **Sharpe Ratio** | Retorno por unidad de riesgo total | **> 1.0** ✓ |
| **Sortino Ratio** | Retorno por unidad de riesgo a la baja | > 1.5 |
| **Calmar Ratio** | CAGR dividido entre el máximo drawdown | > 0.5 |
| **Máx. Drawdown** | Mayor caída desde pico hasta valle | **< –15%** ✓ |
| **VaR 95%** | Pérdida que no se supera el 95% de los días | Referencia |
| **CVaR 95%** | Pérdida media en el peor 5% de días | < VaR × 1.5 |
| **Win Rate** | % de trades con ganancia | > 50% |
| **Profit Factor** | Suma ganancias / suma pérdidas | > 1.3 |
| **IC Sharpe (95%)** | Intervalo de confianza del Sharpe (Bootstrap) | Ambos extremos > 0 |
| **p-value permutaciones** | Probabilidad de que los retornos sean aleatorios | < 0.05 |

### Cómo leer las marcas ✓ / ✗

- **✓ Sharpe > 1.0:** La estrategia genera suficiente retorno ajustado por riesgo para ser viable
- **✗ Sharpe < 1.0:** El retorno no compensa el riesgo tomado — revisar parámetros o descartar el par
- **✓ MDD > –15%:** El capital nunca cayó más del 15% desde su máximo — riesgo controlado
- **✗ MDD < –15%:** El drawdown excedió el límite — estrategia demasiado arriesgada

### Validación estadística

| Resultado | Interpretación |
|---|---|
| IC Bootstrap con ambos extremos > 0 | El Sharpe es estadísticamente positivo (no es un artefacto) |
| IC Bootstrap cruza el cero | Incertidumbre alta — el rendimiento puede ser aleatorio |
| p-value permutaciones < 0.05 | Los retornos **no** son aleatorios — hay señal real |
| p-value permutaciones > 0.05 | No se puede descartar que los retornos sean aleatorios — precaución |

---

## Resumen: Flujo de decisión

Para evaluar un par, lee los gráficos en este orden:

```
04 → ¿El par ha estado cointegrado establemente?
            │ NO → Descartar el par
            │ SÍ ↓
02 → ¿Las señales aparecen en los extremos del z-score?
            │ NO → Revisar parámetros de entrada
            │ SÍ ↓
05 → ¿Sharpe > 1 y MDD < 15%? ¿p-value < 0.05?
            │ NO → Par marginal, no operar
            │ SÍ ↓
01 → ¿La curva de capital sube de forma sostenida?
            │ NO → Periodo de datos insuficiente o régimen adverso
            │ SÍ ↓
03 → ¿P50 > capital inicial y P5 > 85%?
            │ NO → Ajustar tamaño de posición
            │ SÍ ↓
         PAR VIABLE — operar con monitorización diaria
```

---

## Señales de alerta global (red flags)

Si observas cualquiera de estas situaciones en los gráficos, **no operes ese par**:

1. **Gráfico 04:** Traza por debajo del crítico en los últimos 6 meses
2. **Gráfico 02:** Spread con tendencia clara (no oscila alrededor de cero)
3. **Gráfico 05:** p-value de permutaciones > 0.05
4. **Gráfico 05:** IC Bootstrap Sharpe con extremo inferior < 0
5. **Gráfico 01:** Drawdown supera –15% y tarda más de 3 meses en recuperarse
6. **Gráfico 03:** P(ganancia) < 50% o P5 < $80,000

El módulo `automatizacion.py` ejecuta estas verificaciones automáticamente cada día y marca el par como `SUSPENDIDO` si detecta una ruptura de cointegración en tiempo real.
