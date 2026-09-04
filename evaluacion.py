"""Local Matplotlib charts for backtest diagnostics."""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from metricas import (
    DIAS_ANIO,
    curva_drawdown,
    cvar,
    sharpe_ratio,
    var_historico,
)

warnings.filterwarnings("ignore")

GRAFICOS_DIR = os.path.join(os.path.dirname(__file__), "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "text.color": "#c9d1d9",
        "grid.color": "#21262d",
        "grid.linewidth": 0.8,
        "font.family": "monospace",
        "figure.dpi": 120,
    }
)

COLOR_VERDE = "#3fb950"
COLOR_ROJO = "#f85149"
COLOR_AZUL = "#58a6ff"
COLOR_AMARILLO = "#d29922"
COLOR_GRIS = "#8b949e"


def _guardar(nombre: str) -> None:
    ruta = os.path.join(GRAFICOS_DIR, nombre)
    plt.savefig(ruta, bbox_inches="tight", facecolor=plt.rcParams["figure.facecolor"])
    print(f"[OK] Gráfico guardado: {ruta}")


def grafico_curva_capital(
    retornos: pd.Series,
    capital_inicial: float = 100_000.0,
    nombre_par: str = "Estrategia",
    guardar: bool = True,
) -> None:
    curva = (1 + retornos).cumprod() * capital_inicial
    dd = curva_drawdown(retornos)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    fig.suptitle(f"Curva de Capital - {nombre_par}", fontsize=14, color="#e6edf3")

    ax1.plot(curva.index, curva.values, color=COLOR_AZUL, linewidth=1.5, label="Capital")
    ax1.fill_between(
        curva.index,
        capital_inicial,
        curva.values,
        where=(curva.values >= capital_inicial),
        alpha=0.15,
        color=COLOR_VERDE,
    )
    ax1.fill_between(
        curva.index,
        capital_inicial,
        curva.values,
        where=(curva.values < capital_inicial),
        alpha=0.15,
        color=COLOR_ROJO,
    )
    ax1.axhline(capital_inicial, color=COLOR_GRIS, linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_ylabel("Capital (USD)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)

    ax2.fill_between(dd.index, dd.values * 100, 0, where=(dd.values < 0), alpha=0.7, color=COLOR_ROJO)
    ax2.plot(dd.index, dd.values * 100, color=COLOR_ROJO, linewidth=0.8)
    ax2.axhline(-15, color=COLOR_AMARILLO, linestyle="--", linewidth=0.8, alpha=0.8, label="Límite MDD 15%")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(min(dd.values * 100) * 1.1, 2)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    if guardar:
        _guardar(f"01_curva_capital_{nombre_par.replace('/', '_')}.png")
    plt.show()


def grafico_spread_zscore(
    spread: pd.Series,
    zscore: pd.Series,
    señales: pd.Series,
    nombre_par: str = "Par",
    entrada_z: float = 2.0,
    salida_z: float = 0.5,
    guardar: bool = True,
) -> None:
    from spread import Señal

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 2]}, sharex=True)
    fig.suptitle(f"Spread y Z-score - {nombre_par}", fontsize=14, color="#e6edf3")

    ax1.plot(spread.index, spread.values, color=COLOR_AZUL, linewidth=1.0)
    ax1.set_ylabel("Spread (log)")
    ax1.grid(True, alpha=0.4)

    ax2.plot(zscore.index, zscore.values, color=COLOR_AZUL, linewidth=1.0, label="Z-score")
    ax2.axhline(0, color=COLOR_GRIS, linewidth=0.8, linestyle="--")
    ax2.axhline(
        +entrada_z,
        color=COLOR_ROJO,
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
        label=f"+{entrada_z}σ entrada",
    )
    ax2.axhline(
        -entrada_z,
        color=COLOR_VERDE,
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
        label=f"-{entrada_z}σ entrada",
    )
    ax2.axhline(+salida_z, color=COLOR_GRIS, linewidth=0.6, linestyle=":", alpha=0.5)
    ax2.axhline(-salida_z, color=COLOR_GRIS, linewidth=0.6, linestyle=":", alpha=0.5)

    longs = señales[señales == Señal.COMPRAR_SPREAD]
    shorts = señales[señales == Señal.VENDER_SPREAD]
    cierres = señales[señales == Señal.CERRAR]

    ax2.scatter(
        longs.index,
        zscore.loc[longs.index],
        color=COLOR_VERDE,
        marker="^",
        s=60,
        zorder=5,
        label="Long spread",
    )
    ax2.scatter(
        shorts.index,
        zscore.loc[shorts.index],
        color=COLOR_ROJO,
        marker="v",
        s=60,
        zorder=5,
        label="Short spread",
    )
    ax2.scatter(
        cierres.index,
        zscore.loc[cierres.index],
        color=COLOR_AMARILLO,
        marker="x",
        s=40,
        zorder=5,
        label="Cierre",
    )

    ax2.fill_between(
        zscore.index,
        entrada_z,
        zscore.values,
        where=(zscore.values > entrada_z),
        alpha=0.1,
        color=COLOR_ROJO,
    )
    ax2.fill_between(
        zscore.index,
        -entrada_z,
        zscore.values,
        where=(zscore.values < -entrada_z),
        alpha=0.1,
        color=COLOR_VERDE,
    )

    ax2.set_ylabel("Z-score")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    if guardar:
        _guardar(f"02_spread_zscore_{nombre_par.replace('/', '_')}.png")
    plt.show()


def grafico_rolling_sharpe(
    retornos: pd.Series,
    window: int = DIAS_ANIO,
    nombre_par: str = "Estrategia",
    guardar: bool = True,
) -> None:
    rolling_sharpe = retornos.rolling(window).apply(lambda r: sharpe_ratio(pd.Series(r)), raw=False)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Rolling Sharpe Ratio ({window}d) - {nombre_par}", fontsize=14, color="#e6edf3")

    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color=COLOR_AZUL, linewidth=1.2)
    ax.axhline(1.0, color=COLOR_VERDE, linestyle="--", linewidth=1.0, label="Objetivo Sharpe = 1")
    ax.axhline(0.0, color=COLOR_GRIS, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.fill_between(
        rolling_sharpe.index,
        1.0,
        rolling_sharpe.values,
        where=(rolling_sharpe.values >= 1.0),
        alpha=0.15,
        color=COLOR_VERDE,
    )
    ax.fill_between(
        rolling_sharpe.index,
        0.0,
        rolling_sharpe.values,
        where=(rolling_sharpe.values < 0.0),
        alpha=0.15,
        color=COLOR_ROJO,
    )

    ax.set_ylabel("Sharpe Ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if guardar:
        _guardar(f"03_rolling_sharpe_{nombre_par.replace('/', '_')}.png")
    plt.show()


def grafico_distribucion_retornos(
    retornos: pd.Series,
    nombre_par: str = "Estrategia",
    guardar: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Distribución de Retornos Diarios - {nombre_par}", fontsize=14, color="#e6edf3")

    ret = retornos.dropna()
    ax.hist(ret * 100, bins=80, color=COLOR_AZUL, alpha=0.7, density=True, label="Retornos")

    mu, sigma = ret.mean() * 100, ret.std() * 100
    x = np.linspace(ret.min() * 100, ret.max() * 100, 200)
    normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, normal, color=COLOR_AMARILLO, linewidth=1.5, label="Normal teórica")

    var95 = var_historico(ret) * 100
    cvar95 = cvar(ret) * 100
    ax.axvline(var95, color=COLOR_ROJO, linewidth=1.2, linestyle="--", label=f"VaR 95%: {var95:.2f}%")
    ax.axvline(cvar95, color=COLOR_ROJO, linewidth=0.8, linestyle=":", label=f"CVaR 95%: {cvar95:.2f}%")
    ax.axvline(0, color=COLOR_GRIS, linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Retorno diario (%)")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if guardar:
        _guardar(f"04_distribucion_retornos_{nombre_par.replace('/', '_')}.png")
    plt.show()


def grafico_heatmap_mensual(
    retornos: pd.Series,
    nombre_par: str = "Estrategia",
    guardar: bool = True,
) -> None:
    ret_mensual = retornos.resample("ME").apply(lambda r: (1 + r).prod() - 1) * 100

    pivot = pd.DataFrame(
        {
            "año": ret_mensual.index.year,
            "mes": ret_mensual.index.month,
            "ret": ret_mensual.values,
        }
    ).pivot(index="año", columns="mes", values="ret")
    pivot.columns = [
        "Ene",
        "Feb",
        "Mar",
        "Abr",
        "May",
        "Jun",
        "Jul",
        "Ago",
        "Sep",
        "Oct",
        "Nov",
        "Dic",
    ]

    fig, ax = plt.subplots(figsize=(16, max(4, len(pivot) * 0.5 + 2)))
    fig.suptitle(f"Retornos Mensuales (%) - {nombre_par}", fontsize=14, color="#e6edf3")

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(val) > vmax * 0.6 else "black",
                )

    plt.colorbar(im, ax=ax, label="Retorno (%)", shrink=0.6)
    plt.tight_layout()
    if guardar:
        _guardar(f"05_heatmap_mensual_{nombre_par.replace('/', '_')}.png")
    plt.show()


def grafico_monte_carlo(
    mc_resultado: dict,
    retornos_reales: pd.Series,
    capital_inicial: float = 100_000.0,
    nombre_par: str = "Estrategia",
    n_trayectorias_mostrar: int = 200,
    guardar: bool = True,
) -> None:
    rng = np.random.default_rng(99)
    ret_arr = retornos_reales.dropna().values
    n_dias = DIAS_ANIO

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Simulación Monte Carlo - {nombre_par}", fontsize=14, color="#e6edf3")

    for _ in range(n_trayectorias_mostrar):
        muestra = rng.choice(ret_arr, size=n_dias, replace=True)
        curva = capital_inicial * np.cumprod(1 + muestra)
        ax1.plot(curva, alpha=0.06, color=COLOR_AZUL, linewidth=0.5)

    todas = []
    for _ in range(1000):
        m = rng.choice(ret_arr, size=n_dias, replace=True)
        todas.append(capital_inicial * np.cumprod(1 + m))
    todas = np.array(todas)

    for p, col, lbl in [
        (5, COLOR_ROJO, "P5"),
        (25, COLOR_AMARILLO, "P25"),
        (50, COLOR_AZUL, "P50"),
        (75, COLOR_AMARILLO, "P75"),
        (95, COLOR_VERDE, "P95"),
    ]:
        ax1.plot(np.percentile(todas, p, axis=0), color=col, linewidth=1.5, label=lbl)

    ax1.axhline(capital_inicial, color=COLOR_GRIS, linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_xlabel("Días")
    ax1.set_ylabel("Capital (USD)")
    ax1.set_title("Trayectorias simuladas (1 año)", color="#c9d1d9")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.4)

    finales = mc_resultado["capital_final"].values
    ax2.hist(finales, bins=60, color=COLOR_AZUL, alpha=0.7, density=True)
    ax2.axvline(capital_inicial, color=COLOR_GRIS, linewidth=1.0, linestyle="--", label="Capital inicial")
    ax2.axvline(np.percentile(finales, 5), color=COLOR_ROJO, linewidth=1.2, linestyle="--", label="P5")
    ax2.axvline(np.percentile(finales, 50), color=COLOR_AZUL, linewidth=1.5, linestyle="-", label="Mediana")
    ax2.axvline(np.percentile(finales, 95), color=COLOR_VERDE, linewidth=1.2, linestyle="--", label="P95")

    prob = mc_resultado["prob_ganancia"]
    ax2.set_title(f"Capital final - P(ganancia)={prob * 100:.1f}%", color="#c9d1d9")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.set_ylabel("Densidad")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    if guardar:
        _guardar(f"03_monte_carlo_{nombre_par.replace('/', '_')}.png")
    plt.show()


def grafico_rolling_cointegracion(
    rolling_df: pd.DataFrame,
    nombre_par: str = "Par",
    guardar: bool = True,
) -> None:
    if rolling_df.empty:
        print("[WARN] Sin datos de cointegración rolling.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Cointegración Rolling (Johansen 252d) - {nombre_par}", fontsize=14, color="#e6edf3")

    ax.plot(
        rolling_df.index,
        rolling_df["traza"],
        color=COLOR_AZUL,
        linewidth=1.2,
        label="Estadístico de Traza",
    )
    ax.plot(
        rolling_df.index,
        rolling_df["critico"],
        color=COLOR_ROJO,
        linewidth=1.0,
        linestyle="--",
        label="Valor Crítico 95%",
    )
    ax.fill_between(
        rolling_df.index,
        rolling_df["traza"],
        rolling_df["critico"],
        where=(rolling_df["traza"] > rolling_df["critico"]),
        alpha=0.25,
        color=COLOR_VERDE,
        label="Zona cointegrada",
    )
    ax.fill_between(
        rolling_df.index,
        rolling_df["traza"],
        rolling_df["critico"],
        where=(rolling_df["traza"] <= rolling_df["critico"]),
        alpha=0.15,
        color=COLOR_ROJO,
    )

    ax.set_ylabel("Estadístico de Traza")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if guardar:
        _guardar(f"04_rolling_cointegracion_{nombre_par.replace('/', '_')}.png")
    plt.show()


def grafico_precio_relativo(
    precios: pd.DataFrame,
    t1: str,
    t2: str,
    beta: pd.Series,
    guardar: bool = True,
) -> None:
    nombre_par = f"{t1}/{t2}"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Precios y Ratio de Cobertura - {nombre_par}", fontsize=14, color="#e6edf3")

    p1 = precios[t1] / precios[t1].iloc[0] * 100
    p2 = precios[t2] / precios[t2].iloc[0] * 100
    ax1.plot(p1.index, p1.values, color=COLOR_AZUL, linewidth=1.2, label=t1)
    ax1.plot(p2.index, p2.values, color=COLOR_VERDE, linewidth=1.2, label=t2)
    ax1.set_ylabel("Precio normalizado (base 100)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)

    ax2.plot(beta.index, beta.values, color=COLOR_AMARILLO, linewidth=1.0, label="Beta Kalman")
    ax2.axhline(beta.mean(), color=COLOR_GRIS, linestyle="--", linewidth=0.8, alpha=0.6, label="Beta media")
    ax2.set_ylabel("Ratio de cobertura (-)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    if guardar:
        _guardar(f"08_precio_relativo_{nombre_par.replace('/', '_')}.png")
    plt.show()


def grafico_walk_forward(
    wf_df: pd.DataFrame,
    nombre_par: str = "Estrategia",
    guardar: bool = True,
) -> None:
    if wf_df.empty:
        print("[WARN] Sin datos de walk-forward.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Walk-Forward Validation - {nombre_par}", fontsize=14, color="#e6edf3")

    colores = [COLOR_VERDE if s >= 1.0 else COLOR_ROJO for s in wf_df["sharpe"]]
    ax.bar(range(len(wf_df)), wf_df["sharpe"], color=colores, alpha=0.8, edgecolor="#21262d")
    ax.axhline(1.0, color=COLOR_AMARILLO, linestyle="--", linewidth=1.2, label="Objetivo Sharpe = 1")
    ax.axhline(0.0, color=COLOR_GRIS, linestyle="-", linewidth=0.6, alpha=0.5)
    ax.set_xticks(range(len(wf_df)))
    ax.set_xticklabels(wf_df["ventana"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4, axis="y")
    plt.tight_layout()
    if guardar:
        _guardar(f"09_walk_forward_{nombre_par.replace('/', '_')}.png")
    plt.show()


def grafico_duracion_trades(
    trades: pd.DataFrame,
    nombre_par: str = "Estrategia",
    guardar: bool = True,
) -> None:
    if trades.empty or "duracion_dias" not in trades.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Análisis de Trades - {nombre_par}", fontsize=14, color="#e6edf3")

    ax1.hist(trades["duracion_dias"], bins=30, color=COLOR_AZUL, alpha=0.8, edgecolor="#21262d")
    ax1.axvline(
        trades["duracion_dias"].mean(),
        color=COLOR_AMARILLO,
        linewidth=1.2,
        linestyle="--",
        label=f"Media: {trades['duracion_dias'].mean():.0f}d",
    )
    ax1.set_xlabel("Duración (días)")
    ax1.set_ylabel("N° trades")
    ax1.set_title("Duración de trades")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.4)

    colores_pnl = [COLOR_VERDE if p > 0 else COLOR_ROJO for p in trades["pnl"]]
    ax2.bar(range(len(trades)), trades["pnl"], color=colores_pnl, alpha=0.8)
    ax2.axhline(0, color=COLOR_GRIS, linewidth=0.8)
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("PnL (USD)")
    ax2.set_title(f"PnL por trade | Win Rate: {(trades['pnl'] > 0).mean() * 100:.1f}%")
    ax2.grid(True, alpha=0.4, axis="y")

    plt.tight_layout()
    if guardar:
        _guardar(f"10_analisis_trades_{nombre_par.replace('/', '_')}.png")
    plt.show()


def panel_metricas(
    metricas: dict,
    bootstrap_sharpe: dict | None = None,
    nombre_par: str = "Estrategia",
    guardar: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f"Panel de Métricas - {nombre_par}", fontsize=14, color="#e6edf3")
    ax.axis("off")

    filas = [
        ("RENDIMIENTO", "", ""),
        ("CAGR", f"{metricas.get('cagr', 0):.2f}%", ""),
        (
            "Sharpe Ratio",
            f"{metricas.get('sharpe', 0):.3f}",
            "OK" if metricas.get("sharpe", 0) > 1 else "FAIL",
        ),
        ("Sortino Ratio", f"{metricas.get('sortino', 0):.3f}", ""),
        ("Calmar Ratio", f"{metricas.get('calmar', 0):.3f}", ""),
        ("RIESGO", "", ""),
        (
            "Máx. Drawdown",
            f"{metricas.get('mdd', 0):.2f}%",
            "OK" if metricas.get("mdd", 0) > -15 else "FAIL",
        ),
        ("VaR 95%", f"{metricas.get('var_95', 0):.2f}%", ""),
        ("CVaR 95%", f"{metricas.get('cvar_95', 0):.2f}%", ""),
        ("TRADES", "", ""),
        ("N° Trades", f"{metricas.get('n_trades', 0)}", ""),
        ("Win Rate", f"{metricas.get('win_rate', 0):.1f}%", ""),
        ("Profit Factor", f"{metricas.get('profit_factor', 0):.3f}", ""),
    ]

    if bootstrap_sharpe:
        filas += [
            ("VALIDACIÓN ESTADÍSTICA", "", ""),
            (
                "IC Sharpe (95%)",
                f"[{bootstrap_sharpe['ci_inferior']}, {bootstrap_sharpe['ci_superior']}]",
                "",
            ),
        ]
    colores_fila = []
    for f in filas:
        if f[1] == "":
            colores_fila.append(["#21262d"] * 3)
        else:
            colores_fila.append(["#161b22"] * 3)

    tabla = ax.table(
        cellText=filas,
        colLabels=["Métrica", "Valor", "Estado"],
        cellLoc="center",
        loc="center",
        cellColours=colores_fila,
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.8)

    for (i, j), celda in tabla.get_celld().items():
        celda.set_edgecolor("#30363d")
        celda.set_text_props(color="#c9d1d9")
        if i == 0:
            celda.set_facecolor("#21262d")
            celda.set_text_props(color="#58a6ff", weight="bold")

    plt.tight_layout()
    if guardar:
        _guardar(f"05_panel_metricas_{nombre_par.replace('/', '_')}.png")
    plt.show()


def generar_informe_completo(resultado_backtest: dict, nombre_par: str, guardar: bool = True) -> None:
    ret = resultado_backtest["retornos"]
    spread = resultado_backtest["spread"]
    zscore = resultado_backtest["zscore"]
    señales = resultado_backtest["señales"]
    metricas = resultado_backtest["metricas"]
    mc = resultado_backtest.get("monte_carlo", {})
    bs = resultado_backtest.get("bootstrap_sharpe", None)

    print(f"\n[INFO] Generando informe para {nombre_par}...")

    grafico_curva_capital(ret, nombre_par=nombre_par, guardar=guardar)
    grafico_spread_zscore(spread, zscore, señales, nombre_par=nombre_par, guardar=guardar)
    chart_count = 3
    if mc:
        grafico_monte_carlo(mc, ret, nombre_par=nombre_par, guardar=guardar)
        chart_count += 1
    panel_metricas(metricas, bs, nombre_par=nombre_par, guardar=guardar)

    print(f"[OK] Informe generado en {GRAFICOS_DIR}/ ({chart_count} gráficos)")
