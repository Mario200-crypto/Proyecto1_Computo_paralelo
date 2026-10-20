#!/usr/bin/env python3
"""Figuras y cifras del reporte (entrega/reporte/), calculadas SOLO con los datos oficiales.

Fuentes (no se modifican):
    results/official/oficial_20261020/mediciones.csv   2,320 corridas (t_total medido en C++ con omp_get_wtime)
    data/200000_data.csv                                solo para reconstruir el costo de cada región de P2

Todas las aceleraciones son razones de MEDIAS de 10 repeticiones (no medias de cocientes individuales):
    S  = media T_serial(N) / media T_version(N, ...)
    A  = media T_serial(N) / media T_P2(N, k, 1 hilo)                 (ganancia algorítmica de P2)
    Pi = media T_X(N, ..., 1 hilo) / media T_X(N, ..., T hilos)       (ganancia de los hilos, misma estrategia)

Uso, desde la raíz del repositorio:
    .venv/bin/python entrega/reporte/generar_figuras.py
Escribe entrega/reporte/figuras/*.pdf y entrega/reporte/cifras_verificadas.txt
"""

import collections
import csv
import heapq
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "results" / "official" / "oficial_20261020"
OUT = Path(__file__).resolve().parent / "figuras"

SIZES = [20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000]
THREADS = [1, 6, 12, 24]
KS = [2, 4, 8, 16, 32, 64]
N_MAX, T_FIX, K_FIX = 200000, 12, 64

# Colores (validados con el script de la guía de visualización: contraste, separación con daltonismo).
P1_C = "#2a78d6"                     # P1: azul, marcador círculo
P2_C = "#c4501f"                     # P2: naranja oscuro, marcador cuadrado
K_RAMP = {2: "#f0a074", 8: "#df7740", 64: "#c4501f"}   # rampa ordinal de un solo tono para k
SERIAL_C = "#52514e"
REF_C = "#898781"                    # líneas de referencia
GRID_C = "#e1e0d9"
AXIS_C = "#c3c2b7"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.8,
    "axes.edgecolor": AXIS_C, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": AXIS_C, "ytick.color": AXIS_C, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 5, "lines.markeredgecolor": "white", "lines.markeredgewidth": 0.9,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

FULL_W, HALF_W = 6.7, 3.3   # pulgadas (ancho de texto del reporte y media columna)


# ---------------------------------------------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------------------------------------------

def load():
    rows = list(csv.DictReader(open(RUN / "mediciones.csv")))
    refs = {int(r["n"]): r for r in csv.DictReader(open(RUN / "referencias.csv"))}
    keys = ("label_hash", "count_hash", "core1", "core2", "noise")
    mismatches = sum(1 for r in rows if any(r[c] != refs[int(r["n"])][c] for c in keys))
    groups = collections.defaultdict(list)
    first = {}
    for r in rows:
        key = (r["mode"], int(r["n"]), int(r["k"]) if r["k"] else None, int(r["threads"]))
        groups[key].append(float(r["t_total"]))
        first.setdefault(key, r)
    return rows, mismatches, groups, first


ROWS, MISMATCHES, G, FIRST = load()


def mean(mode, n, k=None, t=1):
    return st.mean(G[(mode, n, k, t)])


def sd(mode, n, k=None, t=1):
    return st.stdev(G[(mode, n, k, t)])


def serial(n):
    return mean("serial", n)


# Costo de cada región de P2 (n_propios x L), reconstruido con la MISMA regla que src/dbscan.cpp (dbscan_p2):
# region_rows, strip_index, alcance m = eps + 1e-9 (M + eps). Se comprueba contra pairs_step1 del CSV.
def region_costs(n, k, eps=0.03):
    pts = np.loadtxt(REPO / "data" / f"{n}_data.csv", delimiter=",")
    x, y = pts[:, 0], pts[:, 1]
    rows = max(r for r in range(1, k + 1) if r * r <= k and k % r == 0)
    cols = k // rows
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    w, h = (xmax - xmin) / cols, (ymax - ymin) / rows
    big = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
    m = eps + 1e-9 * (big + eps)

    def strip(v, lo, width, count):
        q = (v - lo) / width
        idx = np.where(q >= count, count - 1, np.floor(q)).astype(np.int64)
        return np.where(q >= 0, idx, 0)

    own = strip(y, ymin, h, rows) * cols + strip(x, xmin, w, cols)
    clo, chi = strip(x - m, xmin, w, cols), strip(x + m, xmin, w, cols)
    rlo, rhi = strip(y - m, ymin, h, rows), strip(y + m, ymin, h, rows)
    costs = []
    for r in range(k):
        owned = int((own == r).sum())
        if owned == 0:
            costs.append(0)
            continue
        row, col = r // cols, r % cols
        halo = int(((own != r) & (clo <= col) & (col <= chi) & (rlo <= row) & (row <= rhi)).sum())
        costs.append(owned * (owned + halo))
    return np.array(costs, dtype=np.int64)


def lpt_bound(costs, threads):
    """Cota de Pi si el tiempo fuera proporcional a los pares y los hilos idénticos: tareas en orden descendente,
    cada una al hilo que se libera primero (lo que hace schedule(dynamic,1) con las regiones ordenadas)."""
    loads = [0] * threads
    for c in sorted(costs, reverse=True):
        heapq.heappush(loads, heapq.heappop(loads) + int(c))
    return costs.sum() / max(loads)


# ---------------------------------------------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------------------------------------------

def kfmt(n):
    return f"{n // 1000}"


def end_label(ax, x, y, text, dy=0, color=INK2):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", ha="left",
                fontsize=7.8, color=color)


def fig_speedup_vs_n():
    xs = [n / 1000 for n in SIZES]
    s1 = [serial(n) / mean("p1", n, None, T_FIX) for n in SIZES]
    s2 = [serial(n) / mean("p2", n, K_FIX, T_FIX) for n in SIZES]
    fig, ax = plt.subplots(figsize=(FULL_W, 2.2))
    ax.axhline(1.0, color=SERIAL_C, lw=1.0, zorder=1)
    ax.axhline(1.5, color=REF_C, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.plot(xs, s2, "-s", color=P2_C, label=f"P2 espacial (k = {K_FIX}, {T_FIX} hilos)", zorder=3)
    ax.plot(xs, s1, "-o", color=P1_C, label=f"P1 por puntos ({T_FIX} hilos)", zorder=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 4, 8, 16, 32, 64]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["1", "1.5", "2", "4", "8", "16", "32", "64"])
    ax.set_ylim(0.85, 80)
    ax.set_xticks(xs)
    ax.set_xticklabels([kfmt(n) for n in SIZES])
    ax.set_xlim(12, 228)
    ax.set_xlabel("Número de puntos N (miles)")
    ax.set_ylabel("Aceleración respecto de Serial\n(escala logarítmica)")
    end_label(ax, xs[-1], s2[-1], f"{s2[-1]:.1f}×")
    end_label(ax, xs[-1], s1[-1], f"{s1[-1]:.1f}×")
    ax.text(14, 1.0, "Serial = 1", va="bottom", ha="left", fontsize=7.5, color=SERIAL_C)
    ax.text(14, 1.5, "mínimo exigido: 1.5×", va="bottom", ha="left", fontsize=7.5, color=REF_C)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.64), ncol=2)
    fig.savefig(OUT / "fig_speedup_vs_N.pdf")
    plt.close(fig)
    return s1, s2


def fig_tiempo_vs_n():
    xs = [n / 1000 for n in SIZES]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.15))
    series = [("serial", None, 1, SERIAL_C, "D", "Serial"),
              ("p1", None, T_FIX, P1_C, "o", f"P1 ({T_FIX} hilos)"),
              ("p2", K_FIX, T_FIX, P2_C, "s", f"P2 (k = {K_FIX}, {T_FIX} hilos)")]
    for mode, k, t, color, marker, label in series:
        ys = [mean(mode, n, k, t) for n in SIZES]
        es = [sd(mode, n, k, t) for n in SIZES]
        for ax in ((a, b) if mode != "serial" else (a,)):
            ax.errorbar(xs, ys, yerr=es, fmt="-" + marker, color=color, ecolor=color, elinewidth=0.8, capsize=0,
                        label=label, zorder=3)
    a.set_ylim(0, 9.6)
    b.set_ylim(0, 1.2)
    a.set_ylabel("Tiempo medio t_total (s)")
    a.set_title("(a) Las tres versiones", loc="left", color=INK)
    b.set_title("(b) Solo las versiones paralelas (otra escala)", loc="left", color=INK)
    for ax in (a, b):
        ax.set_xticks([20, 80, 120, 160, 200])
        ax.set_xlabel("Número de puntos N (miles)")
    a.legend(loc="upper left")
    end_label(a, xs[-1], serial(N_MAX), f"{serial(N_MAX):.2f} s")
    end_label(b, xs[-1], mean("p1", N_MAX, None, T_FIX), f"{mean('p1', N_MAX, None, T_FIX):.2f} s")
    end_label(b, xs[-1], mean("p2", N_MAX, K_FIX, T_FIX), f"{mean('p2', N_MAX, K_FIX, T_FIX):.2f} s")
    for ax in (a, b):
        ax.set_xlim(10, 232)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig_tiempo_vs_N.pdf")
    plt.close(fig)


def fig_hilos(costs):
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.25), sharey=True)
    for ax in (a, b):
        ax.axvspan(12, 25.5, color="#f0efec", zorder=0, lw=0)
        ax.plot([1, 13], [1, 13], color=REF_C, lw=1.0, ls=(0, (4, 3)), zorder=1)
        if ax is a:
            ax.text(9.0, 10.6, "ideal: Π = T", fontsize=7.5, color=REF_C, ha="right", va="center")
            ax.text(24.8, 12.7, "más hilos que\nnúcleos (12)", fontsize=7.2, color=INK2, ha="right", va="top")
        ax.set_xticks(THREADS)
        ax.set_xlim(0, 25.5)
        ax.set_ylim(0, 13)
        ax.set_xlabel("Número de hilos de OpenMP T")
    p1 = [mean("p1", N_MAX, None, 1) / mean("p1", N_MAX, None, t) for t in THREADS]
    p2 = [mean("p2", N_MAX, K_FIX, 1) / mean("p2", N_MAX, K_FIX, t) for t in THREADS]
    a.plot(THREADS, p1, "-o", color=P1_C, label="P1", zorder=3)
    a.plot(THREADS, p2, "-s", color=P2_C, label=f"P2 (k = {K_FIX})", zorder=3)
    end_label(a, 24, p1[-1], f"{p1[-1]:.2f}")
    end_label(a, 24, p2[-1], f"{p2[-1]:.2f}")
    a.set_ylabel("Aceleración paralela Π\n(misma versión con 1 hilo)")
    a.set_title(f"(a) P1 y P2, N = {N_MAX // 1000} mil", loc="left", color=INK)
    a.legend(loc="lower right")
    bounds = {}
    for k in (64, 8, 2):
        pi = [mean("p2", N_MAX, k, 1) / mean("p2", N_MAX, k, t) for t in THREADS]
        bound = [lpt_bound(costs[k], t) for t in THREADS]
        bounds[k] = bound
        b.plot(THREADS, bound, color=K_RAMP[k], lw=0.9, ls=(0, (1, 1.6)), zorder=2)
        b.plot(THREADS, pi, "-s", color=K_RAMP[k], label=f"P2, k = {k}", zorder=3)
    b.plot([], [], color=INK2, lw=0.9, ls=(0, (1, 1.6)), label="cota (región más costosa)")
    b.set_title(f"(b) P2 según k, N = {N_MAX // 1000} mil", loc="left", color=INK)
    b.legend(loc="upper left", ncol=2, columnspacing=0.9, handlelength=1.6, frameon=True, facecolor="white",
             edgecolor="none", framealpha=1)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_escalamiento_hilos.pdf")
    plt.close(fig)
    return p1, p2, bounds


def fig_speedup_vs_k():
    s = [serial(N_MAX) / mean("p2", N_MAX, k, T_FIX) for k in KS]
    s_p1 = serial(N_MAX) / mean("p1", N_MAX, None, T_FIX)
    fig, ax = plt.subplots(figsize=(HALF_W, 2.35))
    xs = np.log2(KS)
    ax.axhline(s_p1, color=P1_C, lw=1.4, zorder=2, label=f"P1, {T_FIX} hilos")
    ax.text(xs[-1] + 0.3, s_p1 - 1.0, f"P1: {s_p1:.1f}×", fontsize=7.5, color=INK2, va="top", ha="right")
    ax.plot(xs, s, "-s", color=P2_C, zorder=3, label=f"P2, {T_FIX} hilos")
    for i, offset in ((0, (8, -6)), (1, (9, -7)), (5, (0, 9))):
        ax.annotate(f"{s[i]:.1f}×", (xs[i], s[i]), xytext=offset, textcoords="offset points",
                    ha="center" if offset[0] == 0 else "left", va="center", fontsize=7.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
    ax.set_ylim(0, 56)
    ax.set_xlabel("Número de regiones k (escala log₂)")
    ax.set_ylabel("Aceleración respecto de Serial")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "fig_speedup_vs_k.pdf")
    plt.close(fig)
    return s


def fig_barras():
    sizes = [20000, 120000, 200000]
    ks = [2, 8, 64]
    fig, ax = plt.subplots(figsize=(HALF_W, 2.35))
    width = 0.24
    for gi, n in enumerate(sizes):
        for j, k in enumerate(ks):
            val = serial(n) / mean("p2", n, k, T_FIX)
            x = gi + (j - 1) * (width + 0.02)
            ax.bar(x, val, width=width, color=K_RAMP[k], label=f"P2, k = {k}" if gi == 0 else None, zorder=3)
            ax.text(x, val + 0.8, f"{val:.1f}", ha="center", va="bottom", fontsize=6.8, color=INK2, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
        sp1 = serial(n) / mean("p1", n, None, T_FIX)
        ax.plot([gi - 0.42, gi + 0.42], [sp1, sp1], color=P1_C, lw=1.6, zorder=4,
                label=f"P1 ({T_FIX} hilos)" if gi == 0 else None, solid_capstyle="butt")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"N = {n // 1000} mil" for n in sizes])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 54)
    ax.set_ylabel("Aceleración respecto de Serial")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=0.8, handlelength=1.4,
              borderaxespad=0.3)
    fig.savefig(OUT / "fig_barras_N_k.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    costs = {k: region_costs(N_MAX, k) for k in KS}
    lines = []
    put = lines.append
    put(f"corridas: {len(ROWS)}; configuraciones: {len(G)}; repeticiones por configuración: "
        f"{sorted({len(v) for v in G.values()})}; corridas con clases/conteos distintos de serial: {MISMATCHES}")
    for k in KS:
        pairs_csv = int(FIRST[("p2", N_MAX, k, 1)]["pairs_step1"])
        put(f"k={k}: pares reconstruidos {costs[k].sum()} vs pairs_step1 del CSV {pairs_csv} -> "
            f"{'IGUALES' if costs[k].sum() == pairs_csv else 'DISTINTOS'}")

    s1, s2 = fig_speedup_vs_n()
    fig_tiempo_vs_n()
    p1, p2, bounds = fig_hilos(costs)
    sk = fig_speedup_vs_k()
    fig_barras()

    put("\n== Serial, P1 (12 hilos), P2 (k=64, 12 hilos): media ± desv. est. (s) y aceleración ==")
    for n, a, b in zip(SIZES, s1, s2):
        put(f"N={n:>6}: serial {serial(n):.4f} ± {sd('serial', n):.4f} | P1 {mean('p1', n, None, 12):.4f} ± "
            f"{sd('p1', n, None, 12):.4f} -> {a:.2f}x | P2 {mean('p2', n, 64, 12):.4f} ± {sd('p2', n, 64, 12):.4f}"
            f" -> {b:.2f}x | P1/P2 = {mean('p1', n, None, 12) / mean('p2', n, 64, 12):.2f}")
    put(f"\n== N = {N_MAX}: escalamiento con hilos ==")
    for i, t in enumerate(THREADS):
        m1, m2 = mean("p1", N_MAX, None, t), mean("p2", N_MAX, 64, t)
        put(f"T={t:>2}: P1 {m1:.4f} ± {sd('p1', N_MAX, None, t):.4f} (CV {100 * sd('p1', N_MAX, None, t) / m1:.1f}%)"
            f" S={serial(N_MAX) / m1:.2f} Pi={p1[i]:.2f} ef={p1[i] / t:.2f} | P2 k=64 {m2:.4f} ± "
            f"{sd('p2', N_MAX, 64, t):.4f} (CV {100 * sd('p2', N_MAX, 64, t) / m2:.1f}%) S={serial(N_MAX) / m2:.2f}"
            f" Pi={p2[i]:.2f} ef={p2[i] / t:.2f}")
    put(f"\n== N = {N_MAX}, {T_FIX} hilos: descomposición S = A x Pi por k ==")
    for k, s in zip(KS, sk):
        t1, t12 = mean("p2", N_MAX, k, 1), mean("p2", N_MAX, k, 12)
        share = float(FIRST[("p2", N_MAX, k, 1)]["max_task_share"])
        nonempty = FIRST[("p2", N_MAX, k, 1)]["tiles_nonempty"]
        put(f"k={k:>2}: N2/pares={N_MAX ** 2 / costs[k].sum():.2f} A={serial(N_MAX) / t1:.2f} Pi={t1 / t12:.2f} "
            f"1/max_task_share={1 / share:.2f} cotaLPT(6,12,24)=" +
            ",".join(f"{lpt_bound(costs[k], t):.2f}" for t in (6, 12, 24)) +
            f" S={s:.2f} regiones_no_vacias={nonempty} T_P2(1)={t1:.4f} T_P2(12)={t12:.4f}")
    put("\n== Pi de P2 por k y hilos (N = 200000) ==")
    for k in KS:
        put(f"k={k:>2}: " + "  ".join(f"T={t}: {mean('p2', N_MAX, k, 1) / mean('p2', N_MAX, k, t):.2f}"
                                       for t in THREADS))
    put("\n== P2 más lento que P1 con los mismos hilos ==")
    slower = [(n, k, t) for n in SIZES for k in KS for t in THREADS
              if mean("p2", n, k, t) > mean("p1", n, None, t)]
    put(f"{len(slower)} de {len(SIZES) * len(KS) * len(THREADS)} combinaciones (N, k, T); valores de k: "
        f"{sorted({k for _, k, _ in slower})}; hilos: {sorted({t for _, _, t in slower})}")
    put("\n== Variabilidad ==")
    cvs = sorted(100 * st.stdev(v) / st.mean(v) for v in G.values())
    put(f"CV mediano {st.median(cvs):.2f}%; configuraciones con CV > 5%: {sum(c > 5 for c in cvs)} de {len(cvs)};"
        f" CV máximo {cvs[-1]:.1f}%")
    put("\n== Mejores configuraciones ==")
    best1 = max(((serial(n) / mean("p1", n, None, t), n, t) for n in SIZES for t in THREADS))
    best2 = max(((serial(n) / mean("p2", n, k, t), n, k, t) for n in SIZES for k in KS for t in THREADS))
    put(f"P1: {best1[0]:.2f}x (N={best1[1]}, T={best1[2]}); P2: {best2[0]:.2f}x (N={best2[1]}, k={best2[2]}, "
        f"T={best2[3]})")
    put("\n== Fases de P2 (N = 200000, k = 64): media en ms ==")
    for t in THREADS:
        rs = [r for r in ROWS if r["mode"] == "p2" and int(r["n"]) == N_MAX and r["k"] == "64" and int(r["threads"]) == t]
        put(f"T={t:>2}: " + " ".join(f"{c}={1e3 * st.mean(float(r[c]) for r in rs):.2f}"
                                     for c in ("t_build", "t_step1", "t_step2", "t_total")))
    (OUT.parent / "cifras_verificadas.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
