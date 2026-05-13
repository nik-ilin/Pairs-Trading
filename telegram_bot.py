"""
telegram_bot.py — Bot de Telegram para el sistema de pairs trading.

Comandos disponibles desde el chat:
  /start           — Muestra los comandos disponibles
  /ayuda           — Idem /start
  /diario          — Pipeline diario: señales para todos los pares del CSV
  /diario T1 T2    — Señal del día para un par específico
  /senal T1 T2     — Señal + métricas detalladas de un par (datos horarios)
  /pares [n]       — Lista los top N pares cointegrados guardados
  /estado          — Posiciones abiertas actualmente
  /backtest T1 T2  — Backtesting out-of-sample 2020→hoy
  /paper T1 T2     — Paper trading histórico con detección dinámica de régimen
  /scan            — Scan semanal completo (operación lenta, ~5-10 min)

Requisitos:
  pip install python-telegram-bot>=20.0
  TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env

Ejecución:
  python telegram_bot.py
"""

import asyncio
import io
import warnings
import urllib.request
import urllib.parse
from contextlib import redirect_stdout
from datetime import datetime
from functools import wraps

import pandas as pd

try:
    from telegram import Update
    from telegram.ext import (
        Application, CommandHandler, ContextTypes, MessageHandler, filters,
    )
    from telegram.constants import ParseMode
    _TELEGRAM_OK = True
except ImportError:
    _TELEGRAM_OK = False

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    CAPITAL_INICIAL,
)

warnings.filterwarnings("ignore")

_MAX_MSG = 4000   # margen bajo el límite de 4096 chars de Telegram


# ── Utilidades base ───────────────────────────────────────────────────────────

def _escapar_html(texto: str) -> str:
    """Escapa caracteres especiales HTML para Telegram."""
    return texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _dividir_texto(texto: str, maximo: int = _MAX_MSG) -> list[str]:
    """Divide un texto largo en trozos <= maximo chars, cortando en saltos de línea."""
    if len(texto) <= maximo:
        return [texto]
    trozos = []
    while texto:
        if len(texto) <= maximo:
            trozos.append(texto)
            break
        corte = texto.rfind("\n", 0, maximo)
        if corte == -1:
            corte = maximo
        trozos.append(texto[:corte])
        texto = texto[corte:].lstrip("\n")
    return trozos


def _mono(texto: str) -> str:
    """Envuelve texto en bloque <pre><code> (monospace en Telegram HTML)."""
    return f"<pre><code>{_escapar_html(texto)}</code></pre>"


def _capturar_stdout(func, *args, **kwargs):
    """Ejecuta func capturando su stdout. Devuelve (salida: str, resultado)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            resultado = func(*args, **kwargs)
        except Exception as e:
            resultado = None
            print(f"[ERROR] {e}")
    return buf.getvalue(), resultado


def _enviar_http(token: str, chat_id: str, mensaje: str, html: bool = False) -> None:
    """Envío HTTP directo a la API de Telegram (no requiere asyncio)."""
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for trozo in _dividir_texto(mensaje):
        datos: dict = {"chat_id": chat_id, "text": trozo}
        if html:
            datos["parse_mode"] = "HTML"
        data = urllib.parse.urlencode(datos).encode()
        try:
            urllib.request.urlopen(url, data, timeout=10)
        except Exception as e:
            print(f"[TELEGRAM] Error al enviar notificación: {e}")


# ── Autorización ──────────────────────────────────────────────────────────────

def _solo_autorizado(handler):
    """Decorator: rechaza mensajes de chat_ids no configurados en TELEGRAM_CHAT_ID."""
    @wraps(handler)
    async def _wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if TELEGRAM_CHAT_ID and str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
            await update.message.reply_text("⛔ No autorizado.")
            return
        return await handler(update, context)
    return _wrapper


async def _responder(update: Update, texto: str, mono: bool = True) -> None:
    """Envía uno o varios mensajes al chat. Si mono=True, usa bloque de código."""
    if not texto.strip():
        texto = "(sin salida)"
    for trozo in _dividir_texto(texto):
        contenido = _mono(trozo) if mono else trozo
        modo = ParseMode.HTML if mono else None
        await update.message.reply_text(contenido, parse_mode=modo)


# ── Handlers de comandos ──────────────────────────────────────────────────────

@_solo_autorizado
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra la lista de comandos disponibles."""
    msg = (
        "📈 <b>Pairs Trading Bot</b>\n\n"
        "Comandos disponibles:\n\n"
        "<code>/diario</code>         — señales del día (todos los pares)\n"
        "<code>/diario T1 T2</code>   — señal de un par específico\n"
        "<code>/senal T1 T2</code>    — señal + métricas detalladas\n"
        "<code>/pares [n]</code>      — top N pares cointegrados\n"
        "<code>/estado</code>         — posiciones abiertas\n"
        "<code>/backtest T1 T2</code> — backtesting 2020 → hoy\n"
        "<code>/paper T1 T2</code>    — paper trading con detección de régimen\n"
        "<code>/scan</code>           — scan semanal completo (~5-10 min)\n"
        "<code>/ayuda</code>          — este mensaje\n\n"
        f"Capital base: <b>${CAPITAL_INICIAL:,.0f}</b>"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


@_solo_autorizado
async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


@_solo_autorizado
async def cmd_diario(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Pipeline diario. Opcionalmente acepta dos tickers como argumentos."""
    await update.message.reply_text("⏳ Ejecutando pipeline diario...")

    args = context.args

    def _run():
        from automatizacion import ejecutar_pipeline_diario, imprimir_resumen_diario
        pares_lista = None
        if args and len(args) >= 2:
            pares_lista = [{"ticker1": args[0].upper(), "ticker2": args[1].upper()}]
        salida1, df   = _capturar_stdout(ejecutar_pipeline_diario,
                                         pares_lista=pares_lista, verbose=True)
        salida2, _    = (_capturar_stdout(imprimir_resumen_diario, df)
                         if df is not None and not df.empty else ("", None))
        return salida1 + salida2, df

    loop = asyncio.get_running_loop()
    salida, df = await loop.run_in_executor(None, _run)
    await _responder(update, salida or "Pipeline completado.")

    if df is not None and not df.empty:
        activas = df[df["señal"].isin(["LONG_SPREAD", "SHORT_SPREAD"])]
        if not activas.empty:
            lineas = ["<b>🔔 Señales activas:</b>"]
            for _, r in activas.iterrows():
                icono = "▲ LONG" if r["señal"] == "LONG_SPREAD" else "▼ SHORT"
                madurez = r.get("madurez_estado", "")
                lineas.append(
                    f"  <code>{r['par']:<12}</code> {icono} | "
                    f"Z={r['z_score']:+.2f} | "
                    f"HL={r.get('half_life_bars', 0):.0f}b | "
                    f"{madurez}"
                )
            await update.message.reply_text("\n".join(lineas), parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text("— Sin nuevas entradas hoy.", parse_mode=ParseMode.HTML)


@_solo_autorizado
async def cmd_senal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Señal detallada + métricas OU para un par específico."""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Uso: /senal TICKER1 TICKER2\nEj: /senal KO PEP")
        return

    t1, t2 = context.args[0].upper(), context.args[1].upper()
    await update.message.reply_text(f"⏳ Analizando <code>{t1}/{t2}</code>...",
                                    parse_mode=ParseMode.HTML)

    def _run():
        from datos import descargar_ohlcv_horario
        from automatizacion import evaluar_par
        ohlcv    = descargar_ohlcv_horario([t1, t2], dias_atras=365)
        df_close = ohlcv.get("close", pd.DataFrame())
        if df_close.empty or t1 not in df_close.columns or t2 not in df_close.columns:
            return None
        return evaluar_par(df_close, t1, t2)

    loop = asyncio.get_running_loop()
    resultado = await loop.run_in_executor(None, _run)

    if resultado is None:
        await update.message.reply_text(f"❌ Sin datos para {t1}/{t2}.")
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

    msg = (
        f"<b>📊 {t1}/{t2}</b>\n\n"
        f"<b>Señal:</b>       {señal_txt}\n"
        f"<b>Z-score:</b>     {resultado['z_score']:+.4f}\n"
        f"<b>Beta Kalman:</b> {resultado['beta_kalman']:.4f}\n"
        f"<b>Half-life:</b>   {resultado.get('half_life_bars', 0):.1f} barras\n"
        f"<b>Ventana z:</b>   {resultado.get('window_zscore', 0)} barras\n"
        f"<b>Vol régimen:</b> {resultado.get('regimen_vol', '?')}\n"
        f"<b>ADF p-val:</b>   {resultado.get('adf_p_value', 1):.4f} "
        f"({'✓ estacionario' if resultado.get('spread_estac') else '✗'})\n"
        f"<b>EG p-val:</b>    {resultado.get('p_value_eg', 1):.4f} "
        f"({'✓ cointegrado' if coint_ok else '✗ RUPTURA'})\n\n"
        f"<b>Precios:</b>  {t1} = ${resultado.get('precio_t1', 0):.2f}   "
        f"{t2} = ${resultado.get('precio_t2', 0):.2f}\n\n"
        f"<b>Cointegración:</b> {icono_m} <b>{estado_m}</b> "
        f"{resultado.get('madurez_tendencia', '')}\n"
        f"<i>{resultado.get('madurez_descripcion', '')}</i>"
    )
    if resultado.get("alerta"):
        msg += f"\n\n⚠️ <b>{_escapar_html(resultado['alerta'])}</b>"

    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


@_solo_autorizado
async def cmd_pares(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra los top N pares del CSV (por defecto top 10)."""
    n = 10
    if context.args:
        try:
            n = int(context.args[0])
        except ValueError:
            pass

    def _run():
        from deteccion import top_pares
        try:
            return top_pares(n=n)
        except FileNotFoundError:
            return None

    loop = asyncio.get_running_loop()
    df = await loop.run_in_executor(None, _run)

    if df is None or df.empty:
        await update.message.reply_text(
            "❌ No hay pares guardados. Ejecuta /scan primero."
        )
        return

    lineas = [f"<b>📋 Top {len(df)} pares cointegrados</b>\n"]
    lineas.append(f"<code>{'#':>3}  {'Par':<14}  {'Score':>6}  {'p-EG':>6}</code>")
    lineas.append(f"<code>{'─'*3}  {'─'*14}  {'─'*6}  {'─'*6}</code>")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        par = f"{row['ticker1']}/{row['ticker2']}"
        lineas.append(
            f"<code>{i:>3}  {par:<14}  {row['score']:>6.3f}  {row['p_value_eg']:>6.4f}</code>"
        )
    await update.message.reply_text("\n".join(lineas), parse_mode=ParseMode.HTML)


@_solo_autorizado
async def cmd_estado(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra las posiciones abiertas según estado_posiciones.json."""
    def _run():
        from automatizacion import cargar_estado
        return cargar_estado()

    loop = asyncio.get_running_loop()
    estado = await loop.run_in_executor(None, _run)

    if not estado:
        await update.message.reply_text("📭 No hay posiciones abiertas.")
        return

    lineas = [f"<b>📂 Posiciones abiertas ({len(estado)})</b>\n"]
    for par, info in estado.items():
        icono = "▲" if "LONG" in info.get("direccion", "") else "▼"
        lineas.append(
            f"<code>{icono} {par}</code>\n"
            f"   Dir.: {info.get('direccion', '?')}\n"
            f"   Entrada: {info.get('fecha_entrada', '?')}\n"
            f"   Z entrada: {info.get('z_entrada', 0):+.2f}\n"
            f"   Beta: {info.get('beta_entrada', 0):.4f}\n"
        )
    await update.message.reply_text("\n".join(lineas), parse_mode=ParseMode.HTML)


@_solo_autorizado
async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Backtesting out-of-sample completo de un par (datos diarios desde 2020)."""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Uso: /backtest TICKER1 TICKER2\nEj: /backtest KO PEP"
        )
        return

    t1, t2 = context.args[0].upper(), context.args[1].upper()
    await update.message.reply_text(
        f"⏳ Backtesting <code>{t1}/{t2}</code> — puede tardar 1-2 min...",
        parse_mode=ParseMode.HTML,
    )

    def _run():
        from datos import descargar_precios_alpaca, filtrar_datos
        from backtesting import backtest_completo, ParametrosBacktest
        from deteccion import estabilidad_rolling, diagnostico_madurez_cointegracion

        precios = descargar_precios_alpaca([t1, t2])
        precios = filtrar_datos(precios)
        if precios.empty or t1 not in precios.columns or t2 not in precios.columns:
            return None, None, None

        rolling  = estabilidad_rolling(precios[[t1, t2]], t1, t2)
        madurez  = diagnostico_madurez_cointegracion(rolling)
        salida, resultado = _capturar_stdout(
            backtest_completo, precios[[t1, t2]], t1, t2,
            ParametrosBacktest(), imprimir_reporte=True,
        )
        return salida, resultado, madurez

    loop = asyncio.get_running_loop()
    salida, resultado, madurez = await loop.run_in_executor(None, _run)

    if salida is None:
        await update.message.reply_text(f"❌ Sin datos para {t1}/{t2}.")
        return

    await _responder(update, salida or "Backtest completado.")

    if resultado and madurez:
        m        = resultado.get("metricas", {})
        estado_m = madurez.get("estado", "?")
        icono_m  = {"RECIENTE": "◈", "CONSOLIDADA": "◉", "MADURA": "◎",
                    "AGOTADA": "○", "INESTABLE": "◌"}.get(estado_m, "?")

        retorno_total = float((resultado["curva_capital"].iloc[-1] / CAPITAL_INICIAL - 1) * 100)
        msg = (
            f"<b>📊 Resumen {t1}/{t2}</b>\n\n"
            f"<code>{'Métrica':<22} {'Valor':>10}</code>\n"
            f"<code>{'─'*34}</code>\n"
            f"<code>{'Retorno total':<22} {retorno_total:>+9.1f}%</code>\n"
            f"<code>{'CAGR':<22} {m.get('cagr', 0):>+9.2f}%</code>\n"
            f"<code>{'Sharpe':<22} {m.get('sharpe', 0):>10.3f}</code>\n"
            f"<code>{'Sortino':<22} {m.get('sortino', 0):>10.3f}</code>\n"
            f"<code>{'Max Drawdown':<22} {m.get('mdd', 0):>+9.2f}%</code>\n"
            f"<code>{'Win rate':<22} {m.get('win_rate', 0):>9.1f}%</code>\n"
            f"<code>{'Profit factor':<22} {m.get('profit_factor', 0):>10.3f}</code>\n"
            f"<code>{'Trades':<22} {m.get('n_trades', 0):>10}</code>\n\n"
            f"<b>Cointegración:</b> {icono_m} {estado_m} "
            f"{madurez.get('tendencia', '')}\n"
            f"<i>{madurez.get('descripcion', '')}</i>"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


@_solo_autorizado
async def cmd_paper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Paper trading histórico con detección dinámica del régimen de cointegración."""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Uso: /paper TICKER1 TICKER2\nEj: /paper KO PEP"
        )
        return

    t1, t2 = context.args[0].upper(), context.args[1].upper()
    await update.message.reply_text(
        f"⏳ Paper trading <code>{t1}/{t2}</code> con detección de régimen — puede tardar 2-3 min...",
        parse_mode=ParseMode.HTML,
    )

    def _run():
        from datos import descargar_precios_alpaca, filtrar_datos
        from backtesting import paper_trading_historico

        precios = descargar_precios_alpaca([t1, t2])
        precios = filtrar_datos(precios)
        if precios.empty or t1 not in precios.columns or t2 not in precios.columns:
            return None, None
        salida, resultado = _capturar_stdout(
            paper_trading_historico, precios[[t1, t2]], t1, t2, verbose=True
        )
        return salida, resultado

    loop = asyncio.get_running_loop()
    salida, resultado = await loop.run_in_executor(None, _run)

    if salida is None:
        await update.message.reply_text(f"❌ Sin datos para {t1}/{t2}.")
        return

    await _responder(update, salida or "Paper trading completado.")

    if resultado and isinstance(resultado, dict):
        m      = resultado.get("metricas", {})
        n_per  = m.get("n_periodos", 0)
        n_per_list = len(resultado.get("periodos_activos", []))
        capital_final = m.get("capital_final", CAPITAL_INICIAL)
        ganancia = capital_final - CAPITAL_INICIAL

        msg = (
            f"<b>📊 Paper Trading {t1}/{t2}</b>\n\n"
            f"Períodos cointegrados detectados: <b>{n_per_list}</b>\n"
            f"Tiempo operando: <b>{m.get('pct_tiempo_activo', 0):.1f}%</b> del histórico\n\n"
            f"<code>{'Métrica':<22} {'Paper':>9} {'Naive':>9}</code>\n"
            f"<code>{'─'*42}</code>\n"
            f"<code>{'Trades':<22} {m.get('n_trades', 0):>9} {m.get('n_trades_naive', 0):>9}</code>\n"
            f"<code>{'Win rate':<22} {m.get('win_rate', 0):>8.1f}% {m.get('win_rate_naive', 0):>8.1f}%</code>\n"
            f"<code>{'Profit factor':<22} {m.get('profit_factor', 0):>9.3f} {m.get('profit_factor_naive', 0):>9.3f}</code>\n"
            f"<code>{'CAGR':<22} {m.get('cagr', 0):>+8.2f}% {m.get('cagr_naive', 0):>+8.2f}%</code>\n"
            f"<code>{'Sharpe':<22} {m.get('sharpe', 0):>9.3f} {m.get('sharpe_naive', 0):>9.3f}</code>\n"
            f"<code>{'Max Drawdown':<22} {m.get('mdd', 0):>+8.2f}% {m.get('mdd_naive', 0):>+8.2f}%</code>\n\n"
            f"Capital final: <b>${capital_final:,.0f}</b> "
            f"(<code>{ganancia:+,.0f}</code> {ganancia/CAPITAL_INICIAL*100:+.1f}%)"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


@_solo_autorizado
async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan semanal completo sobre el universo configurado. Operación lenta (~5-10 min)."""
    await update.message.reply_text(
        "⏳ Iniciando scan semanal completo...\n"
        "⚠️ Esta operación puede tardar <b>5-10 minutos</b>.\n"
        "Recibirás los resultados cuando termine.",
        parse_mode=ParseMode.HTML,
    )

    def _run():
        from automatizacion import ejecutar_scan_semanal
        salida, pares = _capturar_stdout(ejecutar_scan_semanal, forzar=True, verbose=True)
        return salida, pares

    loop = asyncio.get_running_loop()
    salida, pares = await loop.run_in_executor(None, _run)

    await _responder(update, salida or "Scan completado.")

    if pares is not None and not pares.empty:
        n = min(10, len(pares))
        lineas = [f"<b>✅ Scan completado — {len(pares)} pares encontrados (top {n})</b>\n"]
        for i, (_, row) in enumerate(pares.head(n).iterrows(), 1):
            par = f"{row['ticker1']}/{row['ticker2']}"
            lineas.append(
                f"<code>{i:>2}. {par:<14} score={row['score']:.3f}  "
                f"p={row['p_value_eg']:.4f}</code>"
            )
        await update.message.reply_text("\n".join(lineas), parse_mode=ParseMode.HTML)
    elif pares is not None:
        await update.message.reply_text("⚠️ No se encontraron pares cointegrados.")


@_solo_autorizado
async def cmd_desconocido(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Responde a comandos no reconocidos."""
    await update.message.reply_text(
        "❓ Comando no reconocido. Usa /ayuda para ver los comandos disponibles."
    )


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Registra errores no capturados del bot."""
    print(f"[BOT ERROR] {context.error}")
    if isinstance(update, Update) and update.message:
        try:
            await update.message.reply_text(
                f"❌ Error interno: {_escapar_html(str(context.error))}",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass


# ── API pública de notificaciones ─────────────────────────────────────────────

def notificar(mensaje: str, html: bool = True) -> None:
    """
    Envía un mensaje proactivo al chat configurado en TELEGRAM_CHAT_ID.

    Llamada síncrona — se puede usar desde cualquier módulo:
      from telegram_bot import notificar
      notificar("🔔 Nueva señal: <b>KO/PEP</b> LONG SPREAD")

    Si TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no están configurados, no hace nada.
    """
    _enviar_http(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, mensaje, html=html)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Arranca el bot en modo polling (long-polling)."""
    if not _TELEGRAM_OK:
        print("[ERROR] python-telegram-bot no está instalado.")
        print("        Instala con: pip install 'python-telegram-bot>=20.0'")
        return

    if not TELEGRAM_BOT_TOKEN:
        print("[ERROR] TELEGRAM_BOT_TOKEN no configurado en .env")
        return

    print(f"[BOT] Iniciando... {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if TELEGRAM_CHAT_ID:
        print(f"[BOT] Solo autorizado para chat_id: {TELEGRAM_CHAT_ID}")
    else:
        print("[WARN] TELEGRAM_CHAT_ID no configurado — cualquier usuario puede usar el bot")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("ayuda",    cmd_ayuda))
    app.add_handler(CommandHandler("diario",   cmd_diario))
    app.add_handler(CommandHandler("senal",    cmd_senal))
    app.add_handler(CommandHandler("pares",    cmd_pares))
    app.add_handler(CommandHandler("estado",   cmd_estado))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("paper",    cmd_paper))
    app.add_handler(CommandHandler("scan",     cmd_scan))
    app.add_handler(MessageHandler(filters.COMMAND, cmd_desconocido))
    app.add_error_handler(_error_handler)

    print("[BOT] Escuchando comandos de Telegram... (Ctrl+C para detener)\n")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
