import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import time
import itertools
import warnings

# Ignorar warnings de yfinance y pandas para mantener la consola limpia
warnings.filterwarnings("ignore")

def obtener_gran_lista_tickers():
    """Descarga tickers de las bolsas de EE.UU."""
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
    try:
        return pd.read_csv(url, header=None)[0].tolist()
    except:
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "CSCO", "TXN", "IBM", "ORCL", "ADP", "ADEA"]

def motor_busqueda_interactivo():
    """Terminal interactiva para definir filtros y generar el dataset."""
    print("\n" + "="*50)
    print("   MOTOR DE BÚSQUEDA & PAIRS TRADING PRO")
    print("="*50)
    
    try:
        f_sector = input("1. Sector (ej. Technology, Energy) [Enter para saltar]: ") or None
        f_precio = input("2. Precio Máximo (ej. 500) [Enter para saltar]: ")
        f_div = input("3. Mínimo Dividendo % (ej. 1.5) [Enter para saltar]: ")
        f_per = input("4. Máximo PER (ej. 30) [Enter para saltar]: ")
        f_beta = input("5. Máximo Beta/Riesgo (ej. 1.2) [Enter para saltar]: ")
        f_margen = input("6. Mínimo Margen % (ej. 15) [Enter para saltar]: ")
        
        filtros = {
            'sector': f_sector,
            'precio_max': float(f_precio) if f_precio else None,
            'min_div': float(f_div) if f_div else None,
            'max_per': float(f_per) if f_per else None,
            'max_beta': float(f_beta) if f_beta else None,
            'min_margen': float(f_margen) if f_margen else None
        }
    except ValueError:
        print("\n[!] Error: Ingresa solo números en los campos numéricos.")
        return []

    tickers = obtener_gran_lista_tickers()
    resultados = []
    
    # Aumenté un poco el rango para asegurar que haya suficientes pares
    print(f"\n[INFO] Escaneando mercado con tus filtros... (Analizando primeros 300 tickers)")
    
    for t in tickers[:300]: 
        try:
            asset = yf.Ticker(t)
            info = asset.info
            
            div_raw = info.get('dividendYield')
            div_final = (div_raw * 100) if div_raw is not None else 0
            
            datos = {
                "Ticker": t,
                "Precio": info.get('currentPrice', 0),
                "Sector": info.get('sector', 'N/A'),
                "PER": info.get('trailingPE'),
                "Div_%": div_final,
                "Beta_Riesgo": info.get('beta'),
                "Margen_%": (info.get('profitMargins', 0) or 0) * 100
            }

            cumple = True
            if filtros['sector'] and datos['Sector'] != filtros['sector']: cumple = False
            if filtros['precio_max'] and datos['Precio'] > filtros['precio_max']: cumple = False
            if filtros['min_div'] and datos['Div_%'] < filtros['min_div']: cumple = False
            if filtros['max_per'] and (datos['PER'] is None or datos['PER'] > filtros['max_per']): cumple = False
            if filtros['min_margen'] and datos['Margen_%'] < filtros['min_margen']: cumple = False
            if filtros['max_beta'] and (datos['Beta_Riesgo'] is None or datos['Beta_Riesgo'] > filtros['max_beta']): cumple = False

            if cumple:
                print(f"  -> MATCH FUNDAMENTAL: {t}")
                resultados.append(t)

            time.sleep(0.05) 
        except:
            continue

    return resultados

def test_johansen_actual(subset):
    """Ejecuta Johansen y devuelve el Trace y el Valor Crítico al 95%"""
    res = coint_johansen(subset, det_order=0, k_ar_diff=1)
    return res.lr1[0], res.cvt[0, 1]

def evaluar_cointegracion(tickers_filtrados):
    if len(tickers_filtrados) < 2:
        print("\n[!] No hay suficientes activos (mínimo 2) para buscar cointegración.")
        return

    print(f"\n[INFO] Descargando histórico (2020-Hoy) para {len(tickers_filtrados)} activos...")
    data = yf.download(tickers_filtrados, start="2020-01-01")['Close']
    
    # Generar todas las combinaciones de pares posibles
    pares = list(itertools.combinations(tickers_filtrados, 2))
    print(f"[INFO] Evaluando {len(pares)} pares posibles en la ventana actual (últimos 252 días)...")
    
    pares_cointegrados = []
    
    for t1, t2 in pares:
        # Extraer datos del par y limpiar NaNs comunes
        par_data = data[[t1, t2]].dropna()
        if len(par_data) < 252:
            continue
            
        log_par = np.log(par_data)
        # Evaluar solo el último año (ventana actual)
        subset_actual = log_par.iloc[-252:]
        
        try:
            trace, crit = test_johansen_actual(subset_actual)
            # Si trace > crit, están cointegrados. Guardamos la "puntuación"
            if trace > crit:
                score = trace / crit # Cuanto mayor sea esto, más fuerte es la cointegración
                pares_cointegrados.append({'Par': (t1, t2), 'Score': score, 'Trace': trace, 'Crit': crit})
        except:
            continue # Ignorar si la matriz da error (LinAlgError)

    if not pares_cointegrados:
        print("\n[!] Ninguno de los pares generados se encuentra cointegrado actualmente.")
        return

    # Ordenar para obtener el mejor
    pares_cointegrados.sort(key=lambda x: x['Score'], reverse=True)
    mejor_par = pares_cointegrados[0]
    t1_ganador, t2_ganador = mejor_par['Par']
    
    print("\n" + "="*50)
    print(f"🏆 MEJOR PAR COINTEGRADO: {t1_ganador} vs {t2_ganador} 🏆")
    print(f"Estadístico de Traza Actual: {mejor_par['Trace']:.2f} (Crítico: {mejor_par['Crit']:.2f})")
    print("="*50)
    
    # --- PROYECCIÓN DEL GRÁFICO ROLLING PARA EL GANADOR ---
    print(f"\n[INFO] Calculando el histórico de cointegración (Rolling) para el gráfico...")
    log_ganador = np.log(data[[t1_ganador, t2_ganador]].dropna())
    window = 252
    johansen_results = []

    for i in range(window, len(log_ganador)):
        subset = log_ganador.iloc[i-window:i]
        try:
            trace, crit = test_johansen_actual(subset)
            johansen_results.append({'Date': log_ganador.index[i], 'Trace_Stat': trace, 'Crit_95': crit})
        except:
            pass

    results_df = pd.DataFrame(johansen_results).set_index('Date')

    print("\n[OK] ¡Proceso completado! Cierra el gráfico para terminar.")
    
    # Graficar
    plt.figure(figsize=(12,6))
    plt.plot(results_df['Trace_Stat'], label='Estadístico de Traza (Johansen)')
    plt.axhline(y=results_df['Crit_95'].iloc[0], color='r', linestyle='--', label='Valor Crítico 95%')
    plt.fill_between(results_df.index, results_df['Trace_Stat'], results_df['Crit_95'], 
                     where=(results_df['Trace_Stat'] > results_df['Crit_95']), color='green', alpha=0.3, label='Zona Cointegrada')
    plt.title(f'Rolling Cointegration: {t1_ganador} vs {t2_ganador}')
    plt.ylabel('Estadístico de Traza')
    plt.xlabel('Fecha')
    plt.legend()
    plt.show()

# Ejecutar el Pipeline completo
if __name__ == "__main__":
    activos_filtrados = motor_busqueda_interactivo()
    evaluar_cointegracion(activos_filtrados)