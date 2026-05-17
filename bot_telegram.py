"""
bot_telegram.py — Bot de Telegram para el sistema de Pairs Trading.

Espeja todos los modos del CLI en comandos de Telegram:

  /scan           — Scan S&P 500 + filtro SMART (~30-45 min)
  /full [N]       — Pipeline completo: scan + backtest top N pares
  /diario [N]     — Pipeline diario de señales (top N pares, default 20)
  /senal T1 T2    — Señal detallada de un par (Z-score, beta Kalman, OU)
  /senales        — Señales del último pipeline guardadas en CSV
  /estado         — Posiciones abiertas (estado_posiciones.json)
  /pares [N]      — Top N pares cointegrados del CSV (default 10)
  /backtest T1 T2 [opt] [wf] — Backtest out-of-sample (opt=grid search, wf=walk-forward)
  /paper T1 T2    — Paper trading histórico con detección dinámica de régimen
  /evaluar T1 T2  — Informe completo + gráficos PNG enviados por Telegram
  /ayuda          — Lista de todos los comandos
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from analista_ia import analizar_backtest, analizar_señales, analizar_paper

logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


# ── Autorización ──────────────────────────────────────────────────────────────

def _autorizado(update: Update) -> bool:
    """Solo responde al chat_id configurado en .env. Sin restricción si está vacío."""
    if not TELEGRAM_CHAT_ID:
        return True
    return str(update.effective_chat.id) == str(TELEGRAM_CHAT_ID)


async def _rechazar(update: Update) -> None:
    logger.warning(f"Acceso denegado: chat_id={update.effective_chat.id}")
    await update.message.reply_text("⛔ No autorizado.")


# ── Helpers de formato ────────────────────────────────────────────────────────

def _fmt(v, pct: bool = False, dec: int = 2) -> str:
    """Formatea un valor numérico; devuelve 'N/A' si es None o NaN."""
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    return f"{v * 100:.{dec}f}%" if pct else f"{v:.{dec}f}"


def _ok(condicion: bool) -> str:
    return "✅" if condicion else "⚠️"


def _truncar(texto: str, limite: int = 4000) -> str:
    """Telegram limita los mensajes a 4096 caracteres."""
    if len(texto) <= limite:
        return texto
    return texto[:limite - 50] + "\n\n_... (mensaje truncado)_"


# ── /start y /ayuda ───────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _autorizado(update):
        return await _rechazar(update)

    texto = (
        "🤖 *Bot de Pairs Trading — S&P 500*\n\n"
        "*Comandos disponibles:*\n\n"
        "`/scan` — Scan S&P 500 + filtro SMART (~30-45 min)\n"
        "`/full [N]` — Pipeline completo: scan + backtest top N pares\n"
        "`/diario [N]` — Señales del día (top N pares)\n"
        "`/senal T1 T2` — Señal detallada (Z-score, beta, OU, madurez)\n"
        "`/senales` — Última ejecución guardada en CSV\n"
        "`/backtest T1 T2 [opt] [wf]` — Backtest 2020–hoy\n"
        "    `opt` = grid search · `wf` = walk-forward\n"
        "`/paper T1 T2` — Paper trading con detección de régimen\n"
        "`/evaluar T1 T2` — Informe completo + gráficos PNG\n"
        "`/pares [N]` — Top N pares cointegrados del CSV\n"
        "`/estado` — Posiciones abiertas actuales\n"
        "`/ayuda` — Este menú\n\n"
        "_Datos: Alpaca (horario/diario) con fallback automático a yfinance._\n"
        "_Objetivos SMART: Sharpe ≥ 1.0 | MDD ≤ 15%_"
    )
    await update.message.reply_text(texto, parse_mode=ParseMode.MARKDOWN)


async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


# ── /senales ──────────────────────────────────────────────────────────────────

async def cmd_senales(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _autorizado(update):
        return await _rechazar(update)

    ruta = BASE_DIR / "señales_diarias.csv"
    if not ruta.exists():
        await update.message.reply_text(
            "⚠️ No hay señales guardadas. Ejecuta `/diario` primero.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    df      = pd.read_csv(ruta)
    activas = df[df["señal"].isin(["LONG_SPREAD", "SHORT_SPREAD"])]
    hold    = df[df["señal"] == "HOLD"]
    cerrar  = df[df["señal"] == "CERRAR"]
    susp    = df[df["señal"] == "SUSPENDIDO"]

    fecha = datetime.today().strftime("%Y-%m-%d")
    lineas = [
        f"📊 *Señales guardadas — {fecha}*",
        f"▲▼ Entradas: {len(activas)} | ✕ Cierres: {len(cerrar)} | — Hold: {len(hold)} | ⚠ Susp: {len(susp)}",
    ]

    if not activas.empty:
        lineas.append("\n*🔔 Entradas activas:*")
        for _, r in activas.iterrows():
            icono = "▲" if r["señal"] == "LONG_SPREAD" else "▼"
            lineas.append(
                f"{icono} `{r['par']}` Z={r.get('z_score', 0):+.2f} "
                f"β={r.get('beta_kalman', 0):.3f} HL={r.get('half_life_bars', 0):.0f}b"
            )

    if not cerrar.empty:
        lineas.append("\n*✕ Cierres recomendados:*")
        for _, r in cerrar.iterrows():
            lineas.append(f"✕ `{r['par']}` Z={r.get('z_score', 0):+.2f}")

    if len(activas) == 0 and len(cerrar) == 0:
        lineas.append("\n_Sin señales de acción hoy._")

    await update.message.reply_text(
        _truncar("\n".join(lineas)), parse_mode=ParseMode.MARKDOWN
    )


# ── /estado ───────────────────────────────────────────────────────────────────

async def cmd_estado(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _autorizado(update):
        return await _rechazar(update)

    ruta = BASE_DIR / "estado_posiciones.json"
    if not ruta.exists():
        await update.message.reply_text("📭 Sin posiciones abiertas.")
        return

    with open(ruta) as f:
        estado = json.load(f)

    if not estado:
        await update.message.reply_text("📭 Sin posiciones abiertas.")
        return

    lineas = [f"📂 *Posiciones abiertas: {len(estado)}*", ""]
    for par, info in estado.items():
        dir_txt = "▲ LONG" if info.get("direccion") == "LONG_SPREAD" else "▼ SHORT"
        lineas.append(
            f"`{par}` — {dir_txt}\n"
            f"  Entrada: {info.get('fecha_entrada', '?')} | "
            f"Z: {info.get('z_entrada', 0):+.2f} | "
            f"β: {info.get('beta_entrada', 0):.4f}"
        )

    await update.message.reply_text(
        _truncar("\n".join(lineas)), parse_mode=ParseMode.MARKDOWN
    )


# ── /pares ────────────────────────────────────────────────────────────────────

async def cmd_pares(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _autorizado(update):
        return await _rechazar(update)

    n = 10
    if context.args:
        try:
            n = max(1, min(int(context.args[0]), 50))
        except ValueError:
            pass

    ruta = BASE_DIR / "pares_cointegrados.csv"
    if not ruta.exists():
        await update.message.reply_text(
            "⚠️ No hay pares guardados. Ejecuta `/scan` primero.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    df_full = pd.read_csv(ruta)
    df      = df_full.head(n)
    total   = len(df_full)

    tiene_smart = "sharpe" in df_full.columns

    if tiene_smart:
        lineas = [
            f"*Top {n} de {total:,} pares (filtro SMART aplicado)*",
            "```",
            f"{'#':>3}  {'Par':<14} {'Score':>6} {'Sharpe':>7} {'MDD%':>6}",
            "─" * 42,
        ]
        for i, r in df.iterrows():
            par = f"{r['ticker1']}/{r['ticker2']}"
            lineas.append(
                f"{i+1:>3}. {par:<13} {r['score']:>6.4f} "
                f"{r.get('sharpe', 0):>+7.2f} {r.get('mdd', 0):>5.1f}%"
            )
    else:
        lineas = [
            f"*Top {n} de {total:,} pares cointegrados*",
            "```",
            f"{'#':>3}  {'Par':<14} {'Score':>7} {'p-EG':>7}",
            "─" * 36,
        ]
        for i, r in df.iterrows():
            par = f"{r['ticker1']}/{r['ticker2']}"
            lineas.append(f"{i+1:>3}. {par:<13} {r['score']:>7.4f} {r['p_value_eg']:>7.4f}")

    lineas.append("```")
    lineas.append("_Score Johansen: mayor = cointegración más fuerte (>1 válido)_")

    await update.message.reply_text(
        _truncar("\n".join(lineas)), parse_mode=ParseMode.MARKDOWN
    )


# ── /senal ────────────────────────────────────────────────────────────────────

async def cmd_senal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Señal detallada de un par: Z-score, beta Kalman, OU, madurez de cointegración."""
    if not _autorizado(update):
        return await _rechazar(update)

    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Uso: `/senal TICKER1 TICKER2`\nEjemplo: `/senal KO PEP`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    t1     = context.args[0].upper()
    t2     = context.args[1].upper()
    nombre = f"{t1}/{t2}"
    msg    = await update.message.reply_text(
        f"⏳ Analizando señal de `{nombre}`...",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        from datos import descargar_ohlcv_horario
        from automatizacion import evaluar_par

        def _run():
            ohlcv    = descargar_ohlcv_horario([t1, t2], dias_atras=365)
            df_close = ohlcv.get("close", pd.DataFrame())
            if df_close.empty or t1 not in df_close.columns or t2 not in df_close.columns:
                return None
            return evaluar_par(df_close, t1, t2)

        resultado = await asyncio.to_thread(_run)

        if resultado is None:
            await msg.edit_text(
                f"❌ Sin datos para `{nombre}`.", parse_mode=ParseMode.MARKDOWN
            )
            return

        _ICONOS_SEÑAL = {
            "LONG_SPREAD":  "▲ LONG SPREAD",
            "SHORT_SPREAD": "▼ SHORT SPREAD",
            "CERRAR":       "✕ CERRAR",
            "HOLD":         "— HOLD",
            "SUSPENDIDO":   "⚠️ SUSPENDIDO",
        }
        _ICONOS_MAD = {
            "RECIENTE": "◈", "CONSOLIDADA": "◉", "MADURA": "◎",
            "AGOTADA": "○", "INESTABLE": "◌",
        }

        señal_txt = _ICONOS_SEÑAL.get(resultado["señal"], resultado["señal"])
        estado_m  = resultado.get("madurez_estado", "?")
        icono_m   = _ICONOS_MAD.get(estado_m, "?")
        coint_ok  = resultado.get("coint_activa", False)

        texto = (
            f"📊 *{nombre}* — Señal detallada\n\n"
            f"*Señal:*       {señal_txt}\n"
            f"*Z-score:*     `{resultado['z_score']:+.4f}`\n"
            f"*Beta Kalman:* `{resultado['beta_kalman']:.4f}`\n"
            f"*Half-life:*   `{resultado.get('half_life_bars', 0):.1f}` barras\n"
            f"*Ventana z:*   `{resultado.get('window_zscore', 0)}` barras\n"
            f"*Vol régimen:* `{resultado.get('regimen_vol', '?')}`\n"
            f"*ADF p-val:*   `{resultado.get('adf_p_value', 1):.4f}` "
            f"({'✓ estacionario' if resultado.get('spread_estac') else '✗'})\n"
            f"*EG p-val:*    `{resultado.get('p_value_eg', 1):.4f}` "
            f"({'✓ cointegrado' if coint_ok else '✗ RUPTURA'})\n\n"
            f"*Precios:* {t1} `${resultado.get('precio_t1', 0):.2f}` | "
            f"{t2} `${resultado.get('precio_t2', 0):.2f}`\n\n"
            f"*Cointegración:* {icono_m} *{estado_m}* "
            f"{resultado.get('madurez_tendencia', '')}\n"
            f"_{resultado.get('madurez_descripcion', '')}_"
        )
        if resultado.get("alerta"):
            texto += f"\n\n⚠️ *{resultado['alerta']}*"

        await msg.edit_text(_truncar(texto), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.exception("Error en cmd_senal")
        await msg.edit_text(f"❌ Error: `{e}`", parse_mode=ParseMode.MARKDOWN)


# ── /diario ───────────────────────────────────────────────────────────────────

async def cmd_diario(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _autorizado(update):
        return await _rechazar(update)

    top_n = 20
    if context.args:
        try:
            top_n = max(1, min(int(context.args[0]), 100))
        except ValueError:
            pass

    msg = await update.message.reply_text(
        f"⏳ Ejecutando pipeline diario para top *{top_n}* pares...",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        from automatizacion import ejecutar_pipeline_diario, actualizar_estado

        df = await asyncio.to_thread(
            ejecutar_pipeline_diario, top_n=top_n, guardar=True, verbose=False
        )

        if df.empty:
            await msg.edit_text(
                "❌ Sin señales. Verifica credenciales o ejecuta `/scan` primero.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        activas = df[df["señal"].isin(["LONG_SPREAD", "SHORT_SPREAD"])]
        hold    = df[df["señal"] == "HOLD"]
        cerrar  = df[df["señal"] == "CERRAR"]
        susp    = df[df["señal"] == "SUSPENDIDO"]

        await asyncio.to_thread(actualizar_estado, df)

        fecha  = datetime.today().strftime("%Y-%m-%d %H:%M")
        lineas = [
            f"✅ *Pipeline diario — {fecha}*",
            "",
            f"📊 Evaluados: {len(df)} | ▲▼ Entradas: {len(activas)} | "
            f"✕ Cierres: {len(cerrar)} | — Hold: {len(hold)} | ⚠ Susp: {len(susp)}",
        ]

        if not activas.empty:
            lineas.append("\n*🔔 Señales de entrada:*")
            for _, r in activas.iterrows():
                icono = "▲" if r["señal"] == "LONG_SPREAD" else "▼"
                lineas.append(
                    f"{icono} `{r['par']}` Z={r.get('z_score', 0):+.2f} "
                    f"β={r.get('beta_kalman', 0):.3f} HL={r.get('half_life_bars', 0):.0f}b "
                    f"Vol={r.get('regimen_vol', '?')}"
                )

        if not cerrar.empty:
            lineas.append("\n*✕ Cierres recomendados:*")
            for _, r in cerrar.iterrows():
                lineas.append(f"✕ `{r['par']}` Z={r.get('z_score', 0):+.2f}")

        if len(activas) == 0 and len(cerrar) == 0:
            lineas.append("\n_Sin señales de acción hoy. Todos los pares en HOLD._")

        await msg.edit_text(
            _truncar("\n".join(lineas)), parse_mode=ParseMode.MARKDOWN
        )

        # Análisis IA
        if not activas.empty or not cerrar.empty:
            try:
                fecha_hoy = datetime.today().strftime("%Y-%m-%d")
                narrativa = await asyncio.to_thread(analizar_señales, df, fecha_hoy)
                if narrativa:
                    ia_msg = (
                        "━━━━━━━━━━━━━━━━━━━━━━\n"
                        "🤖 <b>Contexto de las señales</b>\n\n"
                        f"{narrativa}\n\n"
                        f"<i>Gemini 2.5 Flash · {fecha_hoy}</i>"
                    )
                    await update.message.reply_text(ia_msg, parse_mode=ParseMode.HTML)
            except Exception:
                pass

    except Exception as e:
        logger.exception("Error en cmd_diario")
        await msg.edit_text(f"❌ Error en pipeline diario:\n`{e}`", parse_mode=ParseMode.MARKDOWN)


# ── /backtest ─────────────────────────────────────────────────────────────────

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Backtest out-of-sample (2020–hoy).
    Flags opcionales: opt = grid search, wf = walk-forward validation.
    Uso: /backtest KO PEP [opt] [wf]
    """
    if not _autorizado(update):
        return await _rechazar(update)

    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Uso: `/backtest TICKER1 TICKER2 [opt] [wf]`\n\n"
            "  `opt` — optimizar parámetros con grid search\n"
            "  `wf`  — incluir walk-forward validation\n\n"
            "Ejemplo: `/backtest KO PEP opt wf`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    t1    = context.args[0].upper()
    t2    = context.args[1].upper()
    flags = {a.lower() for a in context.args[2:]}
    optimizar = "opt" in flags or "optimizar" in flags
    walk_fwd  = "wf"  in flags or "walkforward" in flags

    desc_flags = ""
    if optimizar: desc_flags += " · grid search"
    if walk_fwd:  desc_flags += " · walk-forward"

    msg = await update.message.reply_text(
        f"⏳ Backtest `{t1}/{t2}` (2020–hoy){desc_flags}...\n"
        "_Esto puede tardar varios minutos._",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        from datos import descargar_precios_alpaca, filtrar_datos
        from backtesting import (
            backtest_completo, ParametrosBacktest,
            optimizar_parametros, walk_forward,
        )
        from deteccion import estabilidad_rolling, diagnostico_madurez_cointegracion

        def _run():
            precios = descargar_precios_alpaca([t1, t2])
            precios = filtrar_datos(precios)
            if t1 not in precios.columns or t2 not in precios.columns:
                return None

            params     = ParametrosBacktest()
            param_info = ""
            if optimizar:
                params, grid_df = optimizar_parametros(precios[[t1, t2]], t1, t2)
                if not grid_df.empty:
                    param_info = (
                        f"\n📐 *Parámetros optimizados:*\n"
                        f"  entrada\\_z=`{params.entrada_z}` | "
                        f"salida\\_z=`{params.salida_z}`"
                    )

            resultado = backtest_completo(
                precios[[t1, t2]], t1, t2, params, imprimir_reporte=False
            )

            wf_df = None
            if walk_fwd:
                wf_df = walk_forward(precios[[t1, t2]], t1, t2, params, verbose=False)

            rolling = estabilidad_rolling(precios[[t1, t2]], t1, t2)
            madurez = diagnostico_madurez_cointegracion(rolling)

            return resultado, wf_df, madurez, param_info

        ret = await asyncio.to_thread(_run)

        if ret is None:
            await msg.edit_text(
                f"❌ Datos insuficientes para `{t1}/{t2}`.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        resultado, wf_df, madurez, param_info = ret
        m      = resultado["metricas"]
        sharpe = m.get("sharpe")
        mdd    = m.get("mdd")

        texto = (
            f"📈 *Backtest {t1}/{t2}* (2020–hoy){desc_flags}\n\n"
            f"{_ok(sharpe and sharpe > 1.0)} Sharpe:        `{_fmt(sharpe)}`  _(objetivo >1.0)_\n"
            f"{_ok(mdd and abs(mdd) < 15.0)} Max Drawdown: `{_fmt(mdd)}%`  _(objetivo <15%)_\n\n"
            f"📊 CAGR:          `{_fmt(m.get('cagr'))}%`\n"
            f"📊 Sortino:       `{_fmt(m.get('sortino'))}`\n"
            f"📊 Calmar:        `{_fmt(m.get('calmar'))}`\n"
            f"📊 Profit Factor: `{_fmt(m.get('profit_factor'))}`\n"
            f"📊 Trades:        `{m.get('n_trades', 'N/A')}`\n"
            f"📊 Win Rate:      `{_fmt(m.get('win_rate'))}%`\n"
            f"📊 VaR 95%:       `{_fmt(m.get('var_95'))}%`\n"
            f"📊 CVaR 95%:      `{_fmt(m.get('cvar_95'))}%`\n"
        )

        if param_info:
            texto += param_info + "\n"

        if wf_df is not None and not wf_df.empty:
            texto += f"\n📅 *Walk-Forward ({len(wf_df)} ventanas):*\n```\n"
            texto += f"{'Ventana':<34} {'Sharpe':>7} {'MDD%':>6}\n"
            texto += "─" * 50 + "\n"
            for _, r in wf_df.iterrows():
                texto += (
                    f"{str(r['ventana']):<34} "
                    f"{r['sharpe']:>7.2f} {r['mdd']:>5.1f}%\n"
                )
            texto += "```\n"

        estado_m = madurez.get("estado", "?")
        icono_m  = {
            "RECIENTE": "◈", "CONSOLIDADA": "◉", "MADURA": "◎",
            "AGOTADA": "○", "INESTABLE": "◌",
        }.get(estado_m, "?")
        texto += (
            f"\n*Cointegración:* {icono_m} {estado_m} "
            f"{madurez.get('tendencia', '')}\n"
            f"_{madurez.get('descripcion', '')}_\n"
        )
        if not walk_fwd:
            texto += f"\n_Para gráficos: `/evaluar {t1} {t2}`_"

        await msg.edit_text(_truncar(texto), parse_mode=ParseMode.MARKDOWN)

        # Análisis IA
        try:
            narrativa = await asyncio.to_thread(analizar_backtest, resultado, t1, t2)
            if narrativa:
                fecha_str = datetime.today().strftime("%d/%m/%Y")
                ia_msg = (
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "🤖 <b>Análisis del sistema</b>\n\n"
                    f"{narrativa}\n\n"
                    f"<i>Gemini 2.5 Flash · {fecha_str}</i>"
                )
                await update.message.reply_text(ia_msg, parse_mode=ParseMode.HTML)
        except Exception:
            pass

    except Exception as e:
        logger.exception("Error en cmd_backtest")
        await msg.edit_text(f"❌ Error en backtest:\n`{e}`", parse_mode=ParseMode.MARKDOWN)


# ── /evaluar ──────────────────────────────────────────────────────────────────

async def cmd_evaluar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _autorizado(update):
        return await _rechazar(update)

    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Uso: `/evaluar TICKER1 TICKER2`\nEjemplo: `/evaluar KO PEP`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    t1     = context.args[0].upper()
    t2     = context.args[1].upper()
    nombre = f"{t1}/{t2}"
    msg    = await update.message.reply_text(
        f"⏳ Generando informe completo de `{nombre}`...\n_Esto puede tardar 1-2 minutos._",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        from datos import descargar_precios_alpaca, filtrar_datos, descargar_ohlcv_horario
        from backtesting import backtest_completo, ParametrosBacktest
        from deteccion import estabilidad_rolling
        from evaluacion import generar_informe_completo, grafico_rolling_cointegracion

        def _run():
            precios = descargar_precios_alpaca([t1, t2])
            precios = filtrar_datos(precios)
            if t1 not in precios.columns or t2 not in precios.columns:
                return None
            params    = ParametrosBacktest()
            resultado = backtest_completo(
                precios[[t1, t2]], t1, t2, params, imprimir_reporte=False
            )
            ohlcv   = descargar_ohlcv_horario([t1, t2], dias_atras=365)
            rolling = estabilidad_rolling(ohlcv["close"], t1, t2)
            grafico_rolling_cointegracion(rolling, nombre_par=nombre)
            generar_informe_completo(resultado, nombre_par=nombre)
            return resultado

        resultado = await asyncio.to_thread(_run)

        if resultado is None:
            await msg.edit_text(
                f"❌ Datos insuficientes para `{nombre}`.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        m      = resultado["metricas"]
        sharpe = m.get("sharpe")
        mdd    = m.get("mdd")
        cagr   = m.get("cagr")

        slug         = nombre.replace("/", "_")
        graficos_dir = BASE_DIR / "graficos"
        pngs         = sorted(graficos_dir.glob(f"*_{slug}.png"))

        resumen = (
            f"📈 *{nombre}* — Informe completo\n\n"
            f"{_ok(sharpe and sharpe > 1.0)} Sharpe: `{_fmt(sharpe)}`  |  "
            f"{_ok(mdd and abs(mdd) < 15.0)} MDD: `{_fmt(mdd)}%`\n"
            f"📊 CAGR: `{_fmt(cagr)}%` | Trades: `{m.get('n_trades', 'N/A')}` | "
            f"Win Rate: `{_fmt(m.get('win_rate'))}%`\n\n"
            f"_Enviando {len(pngs)} gráficos..._"
        )
        await msg.edit_text(resumen, parse_mode=ParseMode.MARKDOWN)

        for png in pngs:
            with open(png, "rb") as f:
                await update.message.reply_photo(photo=f, caption=png.stem.replace("_", " "))

        # Análisis IA
        try:
            narrativa = await asyncio.to_thread(analizar_backtest, resultado, t1, t2)
            if narrativa:
                fecha_str = datetime.today().strftime("%d/%m/%Y")
                ia_msg = (
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "🤖 <b>Análisis del sistema</b>\n\n"
                    f"{narrativa}\n\n"
                    f"<i>Gemini 2.5 Flash · {fecha_str}</i>"
                )
                await update.message.reply_text(ia_msg, parse_mode=ParseMode.HTML)
        except Exception:
            pass

    except Exception as e:
        logger.exception("Error en cmd_evaluar")
        await msg.edit_text(f"❌ Error generando informe:\n`{e}`", parse_mode=ParseMode.MARKDOWN)


# ── /paper ────────────────────────────────────────────────────────────────────

async def cmd_paper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _autorizado(update):
        return await _rechazar(update)

    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Uso: `/paper TICKER1 TICKER2`\nEjemplo: `/paper KO PEP`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    t1  = context.args[0].upper()
    t2  = context.args[1].upper()
    msg = await update.message.reply_text(
        f"⏳ Paper trading `{t1}/{t2}` con detección de régimen...\n"
        "_Esto puede tardar 2-3 minutos._",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        from datos import descargar_precios_alpaca, filtrar_datos
        from backtesting import paper_trading_historico

        def _run():
            precios = descargar_precios_alpaca([t1, t2])
            precios = filtrar_datos(precios)
            if precios.empty or t1 not in precios.columns or t2 not in precios.columns:
                return None
            return paper_trading_historico(precios[[t1, t2]], t1, t2, verbose=False)

        resultado = await asyncio.to_thread(_run)

        if resultado is None:
            await msg.edit_text(
                f"❌ Datos insuficientes para `{t1}/{t2}`.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        m             = resultado.get("metricas", {})
        n_periodos    = len(resultado.get("periodos_activos", []))
        capital_final = m.get("capital_final", 0)
        capital_ini   = m.get("capital_inicial", 100_000)
        ganancia      = capital_final - capital_ini

        texto = (
            f"📊 *Paper Trading {t1}/{t2}*\n\n"
            f"Períodos cointegrados: *{n_periodos}* | "
            f"Tiempo activo: *{m.get('pct_tiempo_activo', 0):.1f}%*\n\n"
            f"```\n"
            f"{'Métrica':<22} {'Paper':>9} {'Naive':>9}\n"
            f"{'─'*42}\n"
            f"{'Trades':<22} {m.get('n_trades', 0):>9} {m.get('n_trades_naive', 0):>9}\n"
            f"{'Win rate':<22} {m.get('win_rate', 0):>8.1f}% {m.get('win_rate_naive', 0):>8.1f}%\n"
            f"{'Profit factor':<22} {m.get('profit_factor', 0):>9.3f} {m.get('profit_factor_naive', 0):>9.3f}\n"
            f"{'CAGR':<22} {m.get('cagr', 0):>+8.2f}% {m.get('cagr_naive', 0):>+8.2f}%\n"
            f"{'Sharpe':<22} {m.get('sharpe', 0):>9.3f} {m.get('sharpe_naive', 0):>9.3f}\n"
            f"{'Max Drawdown':<22} {m.get('mdd', 0):>+8.2f}% {m.get('mdd_naive', 0):>+8.2f}%\n"
            f"```\n\n"
            f"Capital final: *${capital_final:,.0f}* "
            f"(`{ganancia:+,.0f}` {ganancia / capital_ini * 100:+.1f}%)"
        )
        await msg.edit_text(texto, parse_mode=ParseMode.MARKDOWN)

        # Análisis IA
        try:
            narrativa = await asyncio.to_thread(analizar_paper, resultado, t1, t2)
            if narrativa:
                fecha_str = datetime.today().strftime("%d/%m/%Y")
                ia_msg = (
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "🤖 <b>Análisis del sistema</b>\n\n"
                    f"{narrativa}\n\n"
                    f"<i>Gemini 2.5 Flash · {fecha_str}</i>"
                )
                await update.message.reply_text(ia_msg, parse_mode=ParseMode.HTML)
        except Exception:
            pass

    except Exception as e:
        logger.exception("Error en cmd_paper")
        await msg.edit_text(f"❌ Error en paper trading:\n`{e}`", parse_mode=ParseMode.MARKDOWN)


# ── /scan ─────────────────────────────────────────────────────────────────────

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Scan completo S&P 500 con filtro SMART automático.
    Equivale a: python main.py --modo scan
    """
    if not _autorizado(update):
        return await _rechazar(update)

    msg = await update.message.reply_text(
        "⏳ *Scan del S&P 500 iniciado*\n\n"
        "Etapas:\n"
        "  1️⃣ Descarga de datos horarios (~10-30 min)\n"
        "  2️⃣ Escaneo EG + Johansen (~3-6 min)\n"
        "  3️⃣ Filtro SMART: backtest rápido (Sharpe≥1.0, MDD≤15%)\n\n"
        "Recibirás los resultados cuando termine. ☕",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        from automatizacion import ejecutar_scan_semanal, filtrar_por_backtest_smart
        from datos import descargar_precios_alpaca, filtrar_datos
        from deteccion import guardar_pares

        # Etapas 1 + 2: scan horario
        pares = await asyncio.to_thread(ejecutar_scan_semanal, forzar=True, verbose=False)

        if pares.empty:
            await msg.edit_text("❌ Scan completado: no se encontraron pares cointegrados.")
            return

        await msg.edit_text(
            f"✅ Scan EG+Johansen: *{len(pares):,}* pares cointegrados\n"
            f"⏳ Aplicando filtro SMART (backtest rápido)...",
            parse_mode=ParseMode.MARKDOWN,
        )

        # Etapa 3: filtro SMART
        tickers = list(set(pares["ticker1"].tolist() + pares["ticker2"].tolist()))
        precios = await asyncio.to_thread(
            lambda: filtrar_datos(descargar_precios_alpaca(tickers))
        )
        pares_smart, todos = await asyncio.to_thread(
            filtrar_por_backtest_smart, pares, precios, verbose=False
        )

        n_pasan = len(pares_smart)
        n_total = len(todos)
        fecha   = datetime.today().strftime("%Y-%m-%d %H:%M")

        if not pares_smart.empty:
            await asyncio.to_thread(guardar_pares, pares_smart)

        lineas = [
            f"✅ *Scan completado — {fecha}*",
            "",
            f"🔍 Cointegrados (EG+Johansen): *{len(pares):,}*",
            f"{'✅' if n_pasan > 0 else '⚠️'} Pasan filtro SMART: *{n_pasan}* de {n_total}",
            "",
        ]

        if not pares_smart.empty:
            top5 = pares_smart.head(5)
            lineas += [
                "*Top 5 pares (filtro SMART):*",
                "```",
                f"{'Par':<14} {'Score':>6} {'Sharpe':>7} {'MDD%':>6}",
                "─" * 38,
            ]
            for _, r in top5.iterrows():
                par = f"{r['ticker1']}/{r['ticker2']}"
                lineas.append(
                    f"{par:<14} {r.get('score', 0):>6.3f} "
                    f"{r.get('sharpe', 0):>+7.2f} {r.get('mdd', 0):>5.1f}%"
                )
            lineas.append("```")
            lineas.append(f"*{n_pasan}* pares guardados en CSV (filtro SMART).")
        else:
            top5 = pares.head(5)
            lineas += [
                "⚠️ *Ningún par pasó el filtro SMART.*",
                "Se conservan los pares cointegrados sin filtrar.",
                "",
                "*Top 5 cointegrados (sin filtro):*",
                "```",
                f"{'Par':<14} {'Score':>7} {'p-EG':>7}",
                "─" * 32,
            ]
            for _, r in top5.iterrows():
                par = f"{r['ticker1']}/{r['ticker2']}"
                lineas.append(f"{par:<14} {r['score']:>7.4f} {r['p_value_eg']:>7.4f}")
            lineas.append("```")

        lineas.append("Usa `/pares 20` para ver más o `/diario` para señales.")

        await msg.edit_text("\n".join(lineas), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.exception("Error en cmd_scan")
        await msg.edit_text(f"❌ Error durante el scan:\n`{e}`", parse_mode=ParseMode.MARKDOWN)


# ── /full ─────────────────────────────────────────────────────────────────────

async def cmd_full(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Pipeline completo: scan + filtro SMART + backtest optimizado top N pares.
    Equivale a: python main.py --modo full
    Uso: /full [N]  (default N=5)
    """
    if not _autorizado(update):
        return await _rechazar(update)

    n_top = 5
    if context.args:
        try:
            n_top = max(1, min(int(context.args[0]), 10))
        except ValueError:
            pass

    msg = await update.message.reply_text(
        f"⏳ *Pipeline completo — top {n_top} pares*\n\n"
        "Etapas:\n"
        "  1️⃣ Scan S&P 500 (horario, 12 meses)\n"
        "  2️⃣ Filtro SMART (backtest rápido)\n"
        f"  3️⃣ Backtest optimizado de los {n_top} mejores pares\n\n"
        "⏱ Esto puede tardar *30-60 minutos*. ☕☕",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        from automatizacion import ejecutar_scan_semanal, filtrar_por_backtest_smart
        from datos import descargar_precios_alpaca, filtrar_datos
        from backtesting import backtest_completo, ParametrosBacktest, optimizar_parametros
        from deteccion import guardar_pares

        # Etapa 1: scan horario
        pares = await asyncio.to_thread(ejecutar_scan_semanal, forzar=True, verbose=False)

        if pares.empty:
            await msg.edit_text("❌ Scan completado: no se encontraron pares cointegrados.")
            return

        await msg.edit_text(
            f"✅ Scan: *{len(pares):,}* pares cointegrados\n"
            f"⏳ Aplicando filtro SMART...",
            parse_mode=ParseMode.MARKDOWN,
        )

        # Etapa 2: filtro SMART
        tickers = list(set(pares["ticker1"].tolist() + pares["ticker2"].tolist()))
        precios = await asyncio.to_thread(
            lambda: filtrar_datos(descargar_precios_alpaca(tickers))
        )
        pares_smart, todos = await asyncio.to_thread(
            filtrar_por_backtest_smart, pares, precios, verbose=False
        )

        n_smart     = len(pares_smart)
        fuente      = pares_smart if not pares_smart.empty else pares
        top_pares   = fuente.head(n_top)

        if not pares_smart.empty:
            await asyncio.to_thread(guardar_pares, pares_smart)

        tickers_bt = list(set(
            top_pares["ticker1"].tolist() + top_pares["ticker2"].tolist()
        ))

        await msg.edit_text(
            f"✅ Filtro SMART: *{n_smart}* pares pasan\n"
            f"⏳ Backtest optimizado de top *{len(top_pares)}* pares...",
            parse_mode=ParseMode.MARKDOWN,
        )

        # Etapa 3: backtest con grid search
        precios_bt = await asyncio.to_thread(
            lambda: filtrar_datos(descargar_precios_alpaca(tickers_bt))
        )

        resultados_bt = []
        for _, row in top_pares.iterrows():
            t1, t2 = row["ticker1"], row["ticker2"]
            if t1 not in precios_bt.columns or t2 not in precios_bt.columns:
                resultados_bt.append((f"{t1}/{t2}", None))
                continue

            def _bt(t1=t1, t2=t2):
                params, _ = optimizar_parametros(precios_bt[[t1, t2]], t1, t2)
                return backtest_completo(
                    precios_bt[[t1, t2]], t1, t2, params, imprimir_reporte=False
                )

            try:
                res = await asyncio.to_thread(_bt)
                resultados_bt.append((f"{t1}/{t2}", res["metricas"]))
            except Exception:
                resultados_bt.append((f"{t1}/{t2}", None))

        # Reporte final
        fecha  = datetime.today().strftime("%Y-%m-%d %H:%M")
        lineas = [
            f"✅ *Pipeline completo — {fecha}*",
            "",
            f"Cointegrados: *{len(pares):,}* → SMART: *{n_smart}* → "
            f"Backtestados: *{len(resultados_bt)}*",
            "",
            "*Resultados (backtest optimizado):*",
            "```",
            f"{'Par':<14} {'Sharpe':>7} {'MDD%':>6} {'CAGR%':>6} {'WR%':>5}",
            "─" * 44,
        ]

        for nombre, m in resultados_bt:
            if m is None:
                lineas.append(f"{nombre:<14}  ERROR")
                continue
            pasa = m.get("sharpe", 0) >= 1.0 and m.get("mdd", 100) <= 15.0
            marca = "✓" if pasa else "·"
            lineas.append(
                f"{marca} {nombre:<13} {m.get('sharpe', 0):>+7.2f} "
                f"{m.get('mdd', 0):>5.1f}% {m.get('cagr', 0):>+5.1f}% "
                f"{m.get('win_rate', 0):>4.0f}%"
            )
        lineas.append("```")
        lineas.append("✓ = pasa objetivos SMART (Sharpe≥1.0, MDD≤15%)")
        lineas.append("Pares guardados. Usa `/diario` para señales.")

        await msg.edit_text(_truncar("\n".join(lineas)), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.exception("Error en cmd_full")
        await msg.edit_text(
            f"❌ Error en pipeline completo:\n`{e}`", parse_mode=ParseMode.MARKDOWN
        )


# ── Setup y arranque ──────────────────────────────────────────────────────────

async def _registrar_comandos(app: Application) -> None:
    """Registra los comandos en el menú desplegable de Telegram."""
    await app.bot.set_my_commands([
        BotCommand("scan",      "Scan S&P500 + filtro SMART (~30-45 min)"),
        BotCommand("full",      "Pipeline completo: scan + backtest [N]"),
        BotCommand("diario",    "Señales del día [N pares]"),
        BotCommand("senal",     "Señal detallada de un par: T1 T2"),
        BotCommand("senales",   "Última ejecución guardada en CSV"),
        BotCommand("backtest",  "Backtest 2020-hoy: T1 T2 [opt] [wf]"),
        BotCommand("paper",     "Paper trading con régimen: T1 T2"),
        BotCommand("evaluar",   "Informe completo + gráficos: T1 T2"),
        BotCommand("pares",     "Top N pares cointegrados [N]"),
        BotCommand("estado",    "Posiciones abiertas actuales"),
        BotCommand("ayuda",     "Lista de todos los comandos"),
        BotCommand("start",     "Bienvenida"),
    ])


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit(
            "[ERROR] TELEGRAM_BOT_TOKEN no configurado.\n"
            "Añádelo al archivo .env: TELEGRAM_BOT_TOKEN=tu_token_aqui"
        )

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(_registrar_comandos)
        .build()
    )

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("ayuda",    cmd_ayuda))
    app.add_handler(CommandHandler("scan",     cmd_scan))
    app.add_handler(CommandHandler("full",     cmd_full))
    app.add_handler(CommandHandler("diario",   cmd_diario))
    app.add_handler(CommandHandler("senal",    cmd_senal))
    app.add_handler(CommandHandler("senales",  cmd_senales))
    app.add_handler(CommandHandler("estado",   cmd_estado))
    app.add_handler(CommandHandler("pares",    cmd_pares))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("paper",    cmd_paper))
    app.add_handler(CommandHandler("evaluar",  cmd_evaluar))

    logger.info("✅ Bot iniciado con polling. Ctrl+C para detener.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
