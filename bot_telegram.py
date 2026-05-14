"""
bot_telegram.py — Bot de Telegram para el sistema de Pairs Trading.

Espeja los modos del CLI en comandos de Telegram, con salida formateada
y envío automático de gráficos PNG generados por evaluacion.py.

Comandos:
  /start          — Bienvenida y menú de ayuda
  /ayuda          — Lista de comandos
  /diario [N]     — Pipeline diario de señales (top N pares, default 20)
  /senales        — Señales del último pipeline guardadas en CSV
  /estado         — Posiciones abiertas (estado_posiciones.json)
  /pares [N]      — Top N pares cointegrados del CSV (default 10)
  /backtest T1 T2 — Backtest out-of-sample de un par específico
  /evaluar T1 T2  — Informe completo + gráficos enviados por Telegram
  /scan           — Scan completo S&P 500 (~20-35 min, en background)
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin GUI — necesario para generar gráficos fuera del hilo principal

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
        "`/diario [N]` — Señales del día (top N pares)\n"
        "`/senales` — Última ejecución guardada\n"
        "`/estado` — Posiciones abiertas actuales\n"
        "`/pares [N]` — Top N pares cointegrados\n"
        "`/backtest T1 T2` — Backtest de un par\n"
        "`/evaluar T1 T2` — Informe completo + gráficos\n"
        "`/scan` — Scan completo S&P 500 (~20 min)\n"
        "`/ayuda` — Este menú\n\n"
        "_Fuente de datos: Alpaca (horario) con fallback automático a yfinance (diario)._"
    )
    await update.message.reply_text(texto, parse_mode=ParseMode.MARKDOWN)


async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


# ── /senales ─────────────────────────────────────────────────────────────────

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
    lineas.append("_p-EG: p-value Engle-Granger (<0.05 = cointegrado)_")

    await update.message.reply_text(
        _truncar("\n".join(lineas)), parse_mode=ParseMode.MARKDOWN
    )


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
            lineas.append("\n_Sin señales de acción hoy. Todos los pares en HOLD o evaluados._")

        await msg.edit_text(
            _truncar("\n".join(lineas)), parse_mode=ParseMode.MARKDOWN
        )

        # ── Análisis IA (solo si hay señales accionables) ────────
        if not activas.empty or not cerrar.empty:
            try:
                fecha_hoy = datetime.today().strftime("%Y-%m-%d")
                narrativa = await asyncio.to_thread(
                    analizar_señales, df, fecha_hoy
                )
                if narrativa:
                    ia_msg = (
                        "━━━━━━━━━━━━━━━━━━━━━━\n"
                        "🤖 <b>Contexto de las señales</b>\n\n"
                        f"{narrativa}\n\n"
                        f"<i>Gemini 2.0 Flash · {fecha_hoy}</i>"
                    )
                    await update.message.reply_text(
                        ia_msg, parse_mode=ParseMode.HTML
                    )
            except Exception:
                pass

    except Exception as e:
        logger.exception("Error en cmd_diario")
        await msg.edit_text(f"❌ Error en pipeline diario:\n`{e}`", parse_mode=ParseMode.MARKDOWN)


# ── /backtest ─────────────────────────────────────────────────────────────────

async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _autorizado(update):
        return await _rechazar(update)

    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Uso: `/backtest TICKER1 TICKER2`\nEjemplo: `/backtest KO PEP`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    t1  = context.args[0].upper()
    t2  = context.args[1].upper()
    msg = await update.message.reply_text(
        f"⏳ Ejecutando backtest de `{t1}/{t2}` (2020–hoy)...",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        from datos import descargar_precios_alpaca, filtrar_datos
        from backtesting import backtest_completo, ParametrosBacktest

        def _run():
            precios = descargar_precios_alpaca([t1, t2])
            precios = filtrar_datos(precios)
            if t1 not in precios.columns or t2 not in precios.columns:
                return None
            params = ParametrosBacktest()
            return backtest_completo(precios[[t1, t2]], t1, t2, params, imprimir_reporte=False)

        resultado = await asyncio.to_thread(_run)

        if resultado is None:
            await msg.edit_text(
                f"❌ Datos insuficientes para `{t1}/{t2}`.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        m      = resultado["metricas"]
        sharpe = m.get("sharpe")
        mdd    = m.get("mdd")

        texto = (
            f"📈 *Backtest {t1}/{t2}* (2020–hoy)\n\n"
            f"{_ok(sharpe and sharpe > 1.0)} Sharpe Ratio:    `{_fmt(sharpe)}`  _(objetivo >1.0)_\n"
            f"{_ok(mdd and abs(mdd) < 15.0)} Max Drawdown:   `{_fmt(mdd)}%`  _(objetivo <15%)_\n\n"
            f"📊 CAGR:          `{_fmt(m.get('cagr'))}%`\n"
            f"📊 Sortino:       `{_fmt(m.get('sortino'))}`\n"
            f"📊 Calmar:        `{_fmt(m.get('calmar'))}`\n"
            f"📊 Profit Factor: `{_fmt(m.get('profit_factor'))}`\n"
            f"📊 Trades:        `{m.get('n_trades', 'N/A')}`\n"
            f"📊 Win Rate:      `{_fmt(m.get('win_rate'))}%`\n"
            f"📊 VaR 95%:       `{_fmt(m.get('var_95'))}%`\n"
            f"📊 CVaR 95%:      `{_fmt(m.get('cvar_95'))}%`\n\n"
            f"_Para gráficos completos usa `/evaluar {t1} {t2}`_"
        )
        await msg.edit_text(texto, parse_mode=ParseMode.MARKDOWN)

        # ── Análisis IA ──────────────────────────────────────────
        try:
            narrativa = await asyncio.to_thread(
                analizar_backtest, resultado, t1, t2
            )
            if narrativa:
                fecha_str = datetime.today().strftime("%d/%m/%Y")
                ia_msg = (
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "🤖 <b>Análisis del sistema</b>\n\n"
                    f"{narrativa}\n\n"
                    f"<i>Gemini 2.0 Flash · {fecha_str}</i>"
                )
                await update.message.reply_text(
                    ia_msg, parse_mode=ParseMode.HTML
                )
        except Exception:
            pass   # La narrativa IA es opcional — nunca bloquea

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

        # Buscar los PNGs generados: naming es XX_tipo_T1_T2.png
        slug         = nombre.replace("/", "_")
        graficos_dir = BASE_DIR / "graficos"
        pngs         = sorted(graficos_dir.glob(f"*_{slug}.png"))

        resumen = (
            f"📈 *{nombre}* — Informe completo generado\n\n"
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

        # ── Análisis IA ──────────────────────────────────────────
        try:
            narrativa = await asyncio.to_thread(
                analizar_backtest, resultado, t1, t2
            )
            if narrativa:
                fecha_str = datetime.today().strftime("%d/%m/%Y")
                ia_msg = (
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "🤖 <b>Análisis del sistema</b>\n\n"
                    f"{narrativa}\n\n"
                    f"<i>Gemini 2.0 Flash · {fecha_str}</i>"
                )
                await update.message.reply_text(
                    ia_msg, parse_mode=ParseMode.HTML
                )
        except Exception:
            pass   # La narrativa IA es opcional — nunca bloquea

    except Exception as e:
        logger.exception("Error en cmd_evaluar")
        await msg.edit_text(f"❌ Error generando informe:\n`{e}`", parse_mode=ParseMode.MARKDOWN)


# ── /paper ───────────────────────────────────────────────────────────────────

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

        # ── Análisis IA ──────────────────────────────────────────
        try:
            narrativa = await asyncio.to_thread(
                analizar_paper, resultado, t1, t2
            )
            if narrativa:
                fecha_str = datetime.today().strftime("%d/%m/%Y")
                ia_msg = (
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "🤖 <b>Análisis del sistema</b>\n\n"
                    f"{narrativa}\n\n"
                    f"<i>Gemini 2.5 Flash · {fecha_str}</i>"
                )
                await update.message.reply_text(
                    ia_msg, parse_mode=ParseMode.HTML
                )
        except Exception:
            pass   # La narrativa IA es opcional — nunca bloquea

    except Exception as e:
        logger.exception("Error en cmd_paper")
        await msg.edit_text(f"❌ Error en paper trading:\n`{e}`", parse_mode=ParseMode.MARKDOWN)


# ── /scan ─────────────────────────────────────────────────────────────────────

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _autorizado(update):
        return await _rechazar(update)

    msg = await update.message.reply_text(
        "⏳ *Scan del S&P 500 iniciado*\n\n"
        "Etapas:\n"
        "  1. Descarga de datos horarios (~10-30 min)\n"
        "  2. Escaneo de ~126.000 pares (~3-6 min)\n\n"
        "Recibirás un mensaje cuando termine. ☕",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        from automatizacion import ejecutar_scan_semanal

        pares = await asyncio.to_thread(ejecutar_scan_semanal, forzar=True, verbose=False)

        if pares.empty:
            await msg.edit_text("❌ Scan completado: no se encontraron pares cointegrados.")
            return

        top5   = pares.head(5)
        fecha  = datetime.today().strftime("%Y-%m-%d %H:%M")
        lineas = [
            f"✅ *Scan completado — {fecha}*",
            f"🔍 Pares cointegrados encontrados: *{len(pares):,}*",
            "",
            "*Top 5 pares por score Johansen:*",
            "```",
            f"{'Par':<14} {'Score':>7} {'p-EG':>7}",
            "─" * 32,
        ]
        for _, r in top5.iterrows():
            par = f"{r['ticker1']}/{r['ticker2']}"
            lineas.append(f"{par:<14} {r['score']:>7.4f} {r['p_value_eg']:>7.4f}")
        lineas.append("```")
        lineas.append("Usa `/pares 20` para ver más, `/diario` para señales.")

        await msg.edit_text("\n".join(lineas), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.exception("Error en cmd_scan")
        await msg.edit_text(f"❌ Error durante el scan:\n`{e}`", parse_mode=ParseMode.MARKDOWN)


# ── Setup y arranque ──────────────────────────────────────────────────────────

async def _registrar_comandos(app: Application) -> None:
    """Registra los comandos en el menú desplegable de Telegram."""
    await app.bot.set_my_commands([
        BotCommand("diario",   "Pipeline diario de señales [N pares]"),
        BotCommand("senales",  "Última ejecución guardada"),
        BotCommand("estado",   "Posiciones abiertas actuales"),
        BotCommand("pares",    "Top N pares cointegrados [N]"),
        BotCommand("backtest", "Backtest de un par: T1 T2"),
        BotCommand("paper",    "Paper trading con detección de régimen: T1 T2"),
        BotCommand("evaluar",  "Informe completo con gráficos: T1 T2"),
        BotCommand("scan",     "Scan completo S&P 500 (~20-35 min)"),
        BotCommand("ayuda",    "Lista de todos los comandos"),
        BotCommand("start",    "Bienvenida"),
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

    app.add_handler(CommandHandler("start",               cmd_start))
    app.add_handler(CommandHandler("ayuda",               cmd_ayuda))
    app.add_handler(CommandHandler("senales", cmd_senales))
    app.add_handler(CommandHandler("estado",              cmd_estado))
    app.add_handler(CommandHandler("pares",               cmd_pares))
    app.add_handler(CommandHandler("diario",              cmd_diario))
    app.add_handler(CommandHandler("backtest",            cmd_backtest))
    app.add_handler(CommandHandler("paper",               cmd_paper))
    app.add_handler(CommandHandler("evaluar",             cmd_evaluar))
    app.add_handler(CommandHandler("scan",                cmd_scan))

    logger.info("✅ Bot iniciado con polling. Ctrl+C para detener.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
