"""
analista_ia.py — Análisis narrativo con Gemini AI para el sistema de Pairs Trading.

Genera párrafos de análisis en español después de los resultados numéricos,
actuando como un analista cuantitativo senior que explica y justifica los resultados.
"""

import logging

import pandas as pd

_cliente_ok: bool = False

try:
    from google import genai
    from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_ACTIVO
    if GEMINI_ACTIVO:
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
        _cliente_ok = True
except Exception as _e:
    logging.getLogger(__name__).warning(f"[GeminiAI] No disponible: {_e}")
    GEMINI_MODEL = "gemini-2.0-flash"


def _construir_prompt_backtest(resultado: dict, t1: str, t2: str) -> str:
    """Construye el prompt para analizar los resultados de un backtest."""
    m    = resultado.get("metricas", {})
    mc   = resultado.get("monte_carlo", {})
    perm = resultado.get("permutaciones", {})
    df_t = resultado.get("trades", pd.DataFrame())

    sharpe    = m.get("sharpe", 0)
    mdd       = m.get("mdd", 0)
    cagr      = m.get("cagr", 0)
    sortino   = m.get("sortino", 0)
    n_trades  = m.get("n_trades", 0)
    win_rate  = m.get("win_rate", 0)
    pf        = m.get("profit_factor", 0)
    half_life = m.get("half_life", 0)
    var95     = m.get("var_95", 0)

    prob_mc  = mc.get("prob_ganancia", 0) * 100
    cap_p50  = mc.get("percentiles_capital", {}).get("p50", 0)

    p_value  = perm.get("p_value", 1.0)
    sig      = perm.get("significativo", False)

    dur_media = 0
    n_long = n_short = 0
    if not df_t.empty:
        dur_media = df_t["duracion_dias"].mean() if "duracion_dias" in df_t else 0
        n_long    = (df_t["direccion"] == "LONG").sum()  if "direccion" in df_t else 0
        n_short   = (df_t["direccion"] == "SHORT").sum() if "direccion" in df_t else 0

    sig_txt = "SIGNIFICATIVO ✓" if sig else "NO SIGNIFICATIVO ✗"

    return (
        "Eres un analista cuantitativo senior. Explica los resultados "
        "del siguiente backtest de arbitraje estadístico a un cliente "
        "inversor. Usa lenguaje claro y profesional en español. Justifica "
        "los números con frases como \"debido a\", \"lo que indica\", "
        "\"la estrategia se benefició de\", \"el modelo detectó\". "
        "Máximo 200 palabras. Un solo párrafo, sin bullets ni títulos.\n\n"
        f"PAR ANALIZADO: {t1}/{t2}\n"
        "PERÍODO: 2020–2026\n\n"
        "RENDIMIENTO:\n"
        f"  CAGR anual: {cagr:.1f}%\n"
        f"  Sharpe Ratio: {sharpe:.2f} (mínimo aceptable: 1.0)\n"
        f"  Sortino Ratio: {sortino:.2f}\n"
        f"  Profit Factor: {pf:.2f}\n"
        f"  Win Rate: {win_rate:.1f}%\n\n"
        "RIESGO:\n"
        f"  Máximo Drawdown: {mdd:.1f}%\n"
        f"  VaR 95%: {var95:.2f}%\n\n"
        "OPERACIONES:\n"
        f"  Total: {n_trades} trades\n"
        f"  LONG: {n_long} | SHORT: {n_short}\n"
        f"  Duración media: {dur_media:.0f} días\n"
        f"  Half-life del spread: {half_life:.0f} barras\n\n"
        "VALIDACIÓN ESTADÍSTICA:\n"
        f"  Monte Carlo P(ganancia en 1 año): {prob_mc:.0f}%\n"
        f"  Capital mediana simulado (1 año): ${cap_p50:,.0f}\n"
        f"  Test de permutaciones: {sig_txt} (p-valor = {p_value:.3f})\n\n"
        "Redacta ahora el párrafo de análisis para el cliente:"
    )


def _construir_prompt_señales(df_señales: pd.DataFrame, fecha: str) -> str | None:
    """Construye el prompt para analizar las señales del pipeline diario."""
    activas = df_señales[df_señales["señal"].isin(["LONG_SPREAD", "SHORT_SPREAD"])]
    cierres = df_señales[df_señales["señal"] == "CERRAR"]

    if activas.empty and cierres.empty:
        return None

    iconos = {"LONG_SPREAD": "▲ COMPRA SPREAD", "SHORT_SPREAD": "▼ VENDE SPREAD"}

    lineas_activas = []
    for _, r in activas.iterrows():
        icono    = iconos.get(r["señal"], r["señal"])
        z        = r.get("z_score", 0)
        hl       = r.get("half_life_bars", 0)
        vol      = r.get("regimen_vol", "?")
        madurez  = r.get("madurez_estado", "?")
        tendencia = r.get("madurez_tendencia", "")
        lineas_activas.append(
            f"- {r['par']}: {icono} | Z-score={z:.2f} ({abs(z):.1f} desv. estándar "
            f"del equilibrio) | Half-life={hl:.0f} barras | Volatilidad={vol} | "
            f"Cointegración: {madurez} {tendencia}"
        )

    lineas_cierres = []
    for _, r in cierres.iterrows():
        z = r.get("z_score", 0)
        lineas_cierres.append(
            f"- {r['par']}: CIERRE | Z-score={z:.2f} (spread revirtió al equilibrio)"
        )

    total  = len(df_señales)
    n_act  = len(activas)
    n_cie  = len(cierres)
    cierres_txt = "\n".join(lineas_cierres) if lineas_cierres else "Ninguno"

    return (
        f"Eres un analista cuantitativo. Son las 9:30 EST del {fecha}. "
        "El sistema ha identificado las siguientes oportunidades de "
        "arbitraje estadístico. Para cada señal, explica en UNA frase "
        "por qué se genera y qué espera el modelo que ocurra. Termina "
        "con una frase de contexto general sobre el día. Máximo 150 "
        "palabras en español. Sin bullets — texto corrido.\n\n"
        "SEÑALES DE ENTRADA HOY:\n"
        f"{chr(10).join(lineas_activas)}\n\n"
        "CIERRES DETECTADOS:\n"
        f"{cierres_txt}\n\n"
        f"CONTEXTO: De {total} pares evaluados, {n_act} generan entrada "
        f"y {n_cie} se cierran hoy."
    )


def _construir_prompt_paper(resultado: dict, t1: str, t2: str) -> str:
    """Construye el prompt para comparar paper trading inteligente vs naive."""
    m = resultado.get("metricas", {})

    cagr        = m.get("cagr", 0)
    cagr_naive  = m.get("cagr_naive", 0)
    sharpe      = m.get("sharpe", 0)
    sharpe_naive = m.get("sharpe_naive", 0)
    mdd         = m.get("mdd", 0)
    mdd_naive   = m.get("mdd_naive", 0)
    win_rate    = m.get("win_rate", 0)
    win_rate_naive = m.get("win_rate_naive", 0)
    pf          = m.get("profit_factor", 0)
    pf_naive    = m.get("profit_factor_naive", 0)
    n_trades    = m.get("n_trades", 0)
    n_trades_naive = m.get("n_trades_naive", 0)
    capital_final  = m.get("capital_final", 0)
    capital_inicial = m.get("capital_inicial", 100_000)
    pct_activo  = m.get("pct_tiempo_activo", 0)
    n_periodos  = len(resultado.get("periodos_activos", []))

    ganancia = capital_final - capital_inicial
    ganancia_pct = (ganancia / capital_inicial * 100) if capital_inicial else 0

    return (
        "Eres un analista cuantitativo senior. Compara los resultados del paper trading "
        "inteligente (solo opera cuando el modelo detecta cointegración activa) frente a la "
        "estrategia ingenua (opera siempre, sin filtro de régimen) para el par "
        f"{t1}/{t2}. Explica en español si la detección dinámica de régimen añade valor "
        "real y por qué. Usa frases como \"la estrategia inteligente supera\", \"el filtro "
        "de régimen evitó\", \"operar solo durante períodos de cointegración confirmada\". "
        "Máximo 180 palabras. Un párrafo, sin bullets ni títulos.\n\n"
        f"PAR: {t1}/{t2}\n"
        "PERÍODO: 2020–2026\n"
        f"TIEMPO EN MERCADO: {pct_activo:.1f}% del histórico ({n_periodos} períodos cointegrados)\n\n"
        "RESULTADOS COMPARADOS:\n"
        f"{'':26} {'PAPEL (inteligente)':>20}   {'NAIVE (siempre activo)':>22}\n"
        f"{'CAGR anual:':<26} {cagr:>19.1f}%   {cagr_naive:>21.1f}%\n"
        f"{'Sharpe Ratio:':<26} {sharpe:>20.2f}   {sharpe_naive:>22.2f}\n"
        f"{'Max Drawdown:':<26} {mdd:>19.1f}%   {mdd_naive:>21.1f}%\n"
        f"{'Win Rate:':<26} {win_rate:>19.1f}%   {win_rate_naive:>21.1f}%\n"
        f"{'Profit Factor:':<26} {pf:>20.2f}   {pf_naive:>22.2f}\n"
        f"{'Nº trades:':<26} {n_trades:>20}   {n_trades_naive:>22}\n"
        f"{'Capital final:':<26} ${capital_final:>18,.0f}   (inicio: ${capital_inicial:,.0f})\n"
        f"{'Ganancia neta:':<26} ${ganancia:>+18,.0f}   ({ganancia_pct:+.1f}%)\n\n"
        "Redacta ahora el análisis comparativo para el cliente:"
    )


def analizar_backtest(resultado: dict, t1: str, t2: str) -> str | None:
    """
    Genera un párrafo narrativo de análisis del backtest usando Gemini AI.
    Devuelve None si Gemini no está disponible o si ocurre algún error.
    """
    if not _cliente_ok:
        return None
    prompt = _construir_prompt_backtest(resultado, t1, t2)
    try:
        response = _genai_client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logging.getLogger(__name__).warning(f"[GeminiAI] Error en analizar_backtest: {e}")
        return None


def analizar_señales(df_señales: pd.DataFrame, fecha: str = "") -> str | None:
    """
    Genera un párrafo narrativo de contexto para las señales del día usando Gemini AI.
    Devuelve None si Gemini no está disponible, sin señales accionables, o si hay error.
    """
    if not _cliente_ok:
        return None
    prompt = _construir_prompt_señales(df_señales, fecha)
    if prompt is None:
        return None
    try:
        response = _genai_client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logging.getLogger(__name__).warning(f"[GeminiAI] Error en analizar_señales: {e}")
        return None


def analizar_paper(resultado: dict, t1: str, t2: str) -> str | None:
    """
    Genera un análisis comparativo del paper trading inteligente vs naive usando Gemini AI.
    Devuelve None si Gemini no está disponible o si ocurre algún error.
    """
    if not _cliente_ok:
        return None
    prompt = _construir_prompt_paper(resultado, t1, t2)
    try:
        response = _genai_client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logging.getLogger(__name__).warning(f"[GeminiAI] Error en analizar_paper: {e}")
        return None
        return None
