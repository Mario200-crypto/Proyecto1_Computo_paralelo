#!/usr/bin/env python3
"""Gráficas del reporte (versión 2): pocas figuras, cada una responde una pregunta, todo en español.

Uso: python scripts/graficas_reporte.py [--run-dir results/official/oficial_20261020]

Solo LEE datos ya medidos: <run-dir>/mediciones.csv y referencias.csv (benchmark oficial), <run-dir>/analisis/
anomalias.csv y, para un apéndice separado, results/sensibilidad/sensibilidad.jsonl (prueba complementaria). No ejecuta
el programa, no modifica datos y no toca las gráficas anteriores (analisis/graficas/). Escribe:
    <run-dir>/analisis/graficas_v2/fig1 ... fig6 (*.png)
    <run-dir>/analisis/graficas_v2/apendice/extra1 ... extra3 (*.png)
    <run-dir>/analisis/graficas_v2/README_GRAFICAS.md (qué muestra cada figura, conclusiones y recomendación editorial)
Todos los números del README se calculan aquí a partir de las mediciones.
"""

import argparse
import csv
import json
import statistics as st
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Estilo (guía de visualización): colores por entidad, iguales en todas las figuras; rampas azules validadas para
# variables ordenadas (N, hilos, k); texto en tinta neutra; líneas de 2 px; un solo eje por panel.
C_SERIAL, C_P1, C_P2, C_MARIO = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
RAMP3 = ["#86b6ef", "#3987e5", "#1c5cab"]
RAMP4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK2, GRID, SURFACE, REF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#7a7974"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10})


def miles(n):
    return f"{n:,}".replace(",", " ")


def hilos(t):
    return "1 hilo" if t == 1 else f"{t} hilos"


def new_figure(title, subtitle, size=(7.2, 4.6), panels=1):
    """Figura con título y subtítulo arriba a la izquierda. El subtítulo se ajusta al ancho de la figura y el margen
    superior de los ejes se calcula con su número de líneas, para que nada se encime ni se recorte."""
    fig, axes = plt.subplots(1, panels, figsize=size)
    fig.patch.set_facecolor(SURFACE)
    lines = []
    for paragraph in subtitle.split("\n"):
        lines += textwrap.wrap(paragraph, int(size[0] * 15.5)) or [""]
    height = size[1]
    fig.text(0.012, 1 - 0.10 / height, title, ha="left", va="top", fontsize=12, color=INK)
    fig.text(0.012, 1 - 0.40 / height, "\n".join(lines), ha="left", va="top", fontsize=9, color=INK2, linespacing=1.3)
    fig.axes_top = 1 - (0.40 + 0.18 * len(lines) + 0.15) / height
    return fig, (axes if panels > 1 else [axes])


def style(ax, xlabel, ylabel, panel_title=None):
    if panel_title:
        ax.set_title(panel_title, loc="left", color=INK, fontsize=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def series(ax, xs, ys, color, label, end_label=None):
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=label)
    if end_label:
        ax.annotate(end_label, (xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=9)


def reference(ax, xs, ys, label):
    ax.plot(xs, ys, color=REF, linewidth=1.5, linestyle="--", label=label)


def n_axis(ax, sizes):
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n // 1000) for n in sizes])


def k_axis(ax, ks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()


def save(fig, path, top=None):
    fig.tight_layout(rect=(0, 0, 1, fig.axes_top))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def load_official(run_dir):
    """Media, desviación estándar y CV de t_total por configuración; exige las 2,320 corridas con hashes de serial."""
    refs = {int(r["n"]): r for r in csv.DictReader((run_dir / "referencias.csv").open())}
    groups = {}
    for r in csv.DictReader((run_dir / "mediciones.csv").open()):
        n = int(r["n"])
        if any(r[h] != refs[n][h] for h in ("label_hash", "count_hash", "core1", "core2", "noise")):
            sys.exit(f"corrida distinta de serial: {r}")
        key = (r["mode"], n, int(r["k"]) if r["k"] else None, int(r["threads"]))
        groups.setdefault(key, []).append(float(r["t_total"]))
    if len(groups) != 232 or any(len(v) != 10 for v in groups.values()) or sum(map(len, groups.values())) != 2320:
        sys.exit("el benchmark oficial no está completo (se esperan 232 configuraciones x 10 repeticiones)")
    stats = {k: {"mean": st.mean(v), "sd": st.stdev(v), "cv": 100 * st.stdev(v) / st.mean(v)} for k, v in groups.items()}
    return stats


def main():
    parser = argparse.ArgumentParser(description="Gráficas del reporte (v2) a partir del benchmark oficial.")
    parser.add_argument("--run-dir", default=str(REPO / "results" / "official" / "oficial_20261020"))
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out = run_dir / "analisis" / "graficas_v2"
    extra = out / "apendice"
    extra.mkdir(parents=True, exist_ok=True)

    S = load_official(run_dir)
    sizes = sorted({k[1] for k in S})
    threads = sorted({k[3] for k in S if k[0] == "p1"})
    ks = sorted({k[2] for k in S if k[0] == "p2"})
    T = lambda mode, n, k=None, t=1: S[(mode, n, k, t)]["mean"]
    ts = {n: T("serial", n) for n in sizes}
    sp = lambda mode, n, k=None, t=1: ts[n] / T(mode, n, k, t)
    best_p1 = {n: min(threads, key=lambda t: T("p1", n, None, t)) for n in sizes}
    best_p2 = {n: min(((k, t) for k in ks for t in threads), key=lambda c: T("p2", n, *c)) for n in sizes}
    best_k_at = {(n, t): min(ks, key=lambda k: T("p2", n, k, t)) for n in sizes for t in threads}
    big, small, mid = max(sizes), min(sizes), 80000
    reps_n = [n for n in (small, mid, big) if n in sizes]

    # ---- Figura 1: tiempo contra N ----
    fig, (ax,) = new_figure("Fig. 1 · Tiempo oficial contra número de puntos",
                            "Media de 10 repeticiones de t_total (construcción + paso 1 + paso 2). Eje y en escala "
                            "logarítmica.\nMejor P1 y mejor P2 = configuración más rápida para cada N.")
    series(ax, sizes, [ts[n] for n in sizes], C_SERIAL, "Serial", "Serial")
    series(ax, sizes, [T("p1", n, None, best_p1[n]) for n in sizes], C_P1, "Mejor P1", "Mejor P1")
    series(ax, sizes, [T("p2", n, *best_p2[n]) for n in sizes], C_P2, "Mejor P2 espacial", "Mejor P2")
    ax.set_yscale("log")
    n_axis(ax, sizes)
    style(ax, "N (miles de puntos)", "tiempo medio (s, escala log)")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    save(fig, out / "fig1_tiempo_vs_N.png", top=0.82)

    # ---- Figura 2: speedup de las mejores configuraciones contra N ----
    fig, (ax,) = new_figure("Fig. 2 · Aceleración de la mejor configuración de cada versión",
                            "speedup = T_serial / T_versión (medias de 10 repeticiones). Línea punteada: mínimo de 1.5× "
                            "que exige el enunciado.")
    y1 = [sp("p1", n, None, best_p1[n]) for n in sizes]
    y2 = [sp("p2", n, *best_p2[n]) for n in sizes]
    series(ax, sizes, y2, C_P2, "Mejor P2 espacial", f"Mejor P2: {y2[-1]:.1f}×")
    series(ax, sizes, y1, C_P1, "Mejor P1", f"Mejor P1: {y1[-1]:.1f}×")
    reference(ax, [sizes[0], sizes[-1]], [1.5, 1.5], "umbral del enunciado (1.5×)")
    ax.set_ylim(0, max(y2) * 1.12)
    n_axis(ax, sizes)
    style(ax, "N (miles de puntos)", "speedup respecto de serial")
    ax.legend(frameon=False, loc="center right", labelcolor=INK)
    save(fig, out / "fig2_speedup_mejores_vs_N.png")

    # ---- Figura 3: P1 contra hilos ----
    eff6 = st.mean(sp("p1", n, None, 6) / 6 for n in sizes if n >= mid)
    eff12 = st.mean(sp("p1", n, None, 12) / 12 for n in sizes if n >= mid)
    fig, (ax,) = new_figure("Fig. 3 · P1: aceleración con OpenMP según el número de hilos",
                            f"Tres tamaños representativos. Eficiencia media (N ≥ 80 000): {eff6:.2f} con 6 hilos y "
                            f"{eff12:.2f} con 12 (N = 80 000 y 200 000 casi coinciden).\nEl Mac tiene 12 núcleos (8 de rendimiento y 4 de "
                            "eficiencia).")
    for color, n in zip(RAMP3, reps_n):
        series(ax, threads, [sp("p1", n, None, t) for t in threads], color, f"N = {miles(n)}")
    reference(ax, threads, threads, "ideal (speedup = hilos)")
    ax.axvline(12, color=GRID, linewidth=1.5, linestyle=":")
    ax.annotate("12 núcleos", (12, 22.5), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xticks(threads)
    ax.set_ylim(0, 25)
    style(ax, "hilos", "speedup = T_serial / T_P1")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    save(fig, out / "fig3_p1_speedup_vs_hilos.png", top=0.82)

    # ---- Figura 4: P2 contra hilos (N = 200 000) ----
    k4 = [k for k in (2, 8, 32, 64) if k in ks]
    fig, (ax,) = new_figure(f"Fig. 4 · P2 espacial: aceleración según los hilos (N = {miles(big)})",
                            "Cada línea es un número de regiones k. El speedup total incluye la reducción de pares por "
                            "las regiones\ny el efecto de OpenMP (descomposición en el apéndice).")
    for color, k in zip(RAMP4, k4):
        series(ax, threads, [sp("p2", big, k, t) for t in threads], color, f"k = {k}", f"k = {k}")
    ax.set_xticks(threads)
    style(ax, "hilos", "speedup = T_serial / T_P2")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    save(fig, out / "fig4_p2_speedup_vs_hilos.png", top=0.82)

    # ---- Figura 5: P2 contra k (N = 200 000) ----
    t3 = [t for t in (1, 12, 24) if t in threads]
    best_p1_big = sp("p1", big, None, best_p1[big])
    fig, (ax,) = new_figure(f"Fig. 5 · P2 espacial: aceleración según el número de regiones k (N = {miles(big)})",
                            f"Línea punteada: mejor P1 ({best_p1_big:.1f}×, {hilos(best_p1[big])}). Con 1 hilo, la "
                            "ganancia viene solo de comparar menos pares; 12 y 24 hilos casi coinciden.")
    for color, t in zip(RAMP3, t3):
        series(ax, ks, [sp("p2", big, k, t) for k in ks], color, hilos(t))
    reference(ax, [ks[0], ks[-1]], [best_p1_big, best_p1_big], f"mejor P1 ({best_p1_big:.1f}×)")
    k_axis(ax, ks)
    style(ax, "k (regiones del plano)", "speedup = T_serial / T_P2")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    save(fig, out / "fig5_p2_speedup_vs_k.png")

    # ---- Figura 6: comparación justa P1 contra P2 con los mismos hilos ----
    fig, axes = new_figure("Fig. 6 · ¿Cuánto más rápido es P2 que P1 con el mismo número de hilos?",
                           "Razón T_P1 / T_P2 (> 1: P2 más rápido). (a) P2 con su mejor k para cada N y número de hilos. "
                           f"(b) N = {miles(big)}, 12 hilos,\npara cada k: con pocas regiones (k ≤ 4) P2 es más lento que P1.",
                           size=(11, 4.6), panels=2)
    ax = axes[0]
    for color, t in zip(RAMP3, [t for t in threads if t != 1]):
        series(ax, sizes, [T("p1", n, None, t) / T("p2", n, best_k_at[(n, t)], t) for n in sizes], color, hilos(t))
    reference(ax, [sizes[0], sizes[-1]], [1, 1], "mismo tiempo")
    n_axis(ax, sizes)
    ax.set_ylim(0, None)
    style(ax, "N (miles de puntos)", "T_P1 / T_P2 (mejor k)", "(a) P2 con su mejor k")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.18), ncol=2, labelcolor=INK)
    ax = axes[1]
    ratio_k = [T("p1", big, None, 12) / T("p2", big, k, 12) for k in ks]
    series(ax, ks, ratio_k, C_P2, "P2 contra P1 (12 hilos)")
    for k, r in zip(ks, ratio_k):
        ax.annotate(f"{r:.2f}", (k, r), xytext=(0, 8 if r >= 1.2 else -14), textcoords="offset points", ha="center",
                    color=INK, fontsize=8)
    reference(ax, [ks[0], ks[-1]], [1, 1], "mismo tiempo")
    k_axis(ax, ks)
    ax.set_ylim(-0.4, max(ratio_k) * 1.15)            # margen bajo 0 para la etiqueta del valor más pequeño
    ax.set_yticks(range(0, int(max(ratio_k)) + 1))
    style(ax, "k (regiones del plano)", "T_P1 / T_P2", f"(b) según k, N = {miles(big)}, 12 hilos")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    save(fig, out / "fig6_p1_vs_p2_mismos_hilos.png", top=0.80)

    # ---- Apéndice 1: S_total = A x Pi ----
    t_mid = 12
    A = [ts[big] / T("p2", big, k, 1) for k in ks]
    Pi = [T("p2", big, k, 1) / T("p2", big, k, t_mid) for k in ks]
    Stot = [sp("p2", big, k, t_mid) for k in ks]
    pairs, share, share_by_k = {}, {}, {}
    for r in csv.DictReader((run_dir / "mediciones.csv").open()):
        if r["mode"] == "p2" and int(r["n"]) == big and int(r["threads"]) == t_mid:
            pairs[int(r["k"])] = int(r["pairs_step1"])
            share[int(r["k"])] = float(r["max_task_share"])
            share_by_k[int(r["k"])] = float(r["max_task_share"])
    fig, axes = new_figure(f"Apéndice 1 · P2 espacial: S_total = A × Π (N = {miles(big)}, {hilos(t_mid)})",
                           "A = T_serial / T_P2(1 hilo): ganancia por comparar menos pares. Π = T_P2(1 hilo) / "
                           f"T_P2({t_mid} hilos): ganancia de OpenMP,\nlimitada por la región más costosa "
                           "(1 / max_task_share). Por eso S_total puede superar el número de núcleos.",
                           size=(11.5, 4.3), panels=3)
    series(axes[0], ks, Stot, C_P2, "S_total")
    series(axes[1], ks, A, C_P2, "A (medido)")
    reference(axes[1], ks, [big * big / pairs[k] for k in ks], "N² / pares evaluados")
    series(axes[2], ks, Pi, C_P2, "Π (medido)")
    reference(axes[2], ks, [1 / share[k] for k in ks], "límite 1 / max_task_share")
    for ax, title in zip(axes, ("S_total = T_serial / T_P2", "A = ganancia algorítmica", "Π = ganancia de OpenMP")):
        k_axis(ax, ks)
        style(ax, "k (regiones)", "factor", title)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK)
    axes[2].legend(frameon=False, fontsize=8, labelcolor=INK)
    save(fig, extra / "extra1_descomposicion_A_Pi.png", top=0.80)

    # ---- Apéndice 2: sensibilidad (prueba complementaria; NO es parte del benchmark oficial) ----
    sens_path = REPO / "results" / "sensibilidad" / "sensibilidad.jsonl"
    sens = {}
    for line in sens_path.open():
        r = json.loads(line)
        sens.setdefault((r["tag"], r["label"]), []).append(r["metrics"]["t_total"])
    dists = [("uniforme", "todo el plano"), ("desbalanceada", "un grupo con 80 %"), ("concentrada", "99.5 % en una región")]
    labels = [("p1:12", "P1 (12 hilos)", C_P1), ("p2:64:12", "P2 por bloques, Mario (k=64, 12 hilos)", C_MARIO),
              ("p2s:64:12", "P2 espacial (k=64, 12 hilos)", C_P2)]
    sens_table = {}
    for n_s in (20000, 40000):
        for d, _ in dists:
            tag = f"sensibilidad {d} {n_s}"
            t_ser = st.mean(sens[(tag, "serial")])
            sens_table[(d, n_s)] = {lab: t_ser / st.mean(sens[(tag, lab)]) for lab, _, _ in labels}
    fig, (ax,) = new_figure("Apéndice 2 · Prueba complementaria: sensibilidad a la distribución de los puntos",
                            "Datos sintéticos de 40 000 puntos (results/sensibilidad/), NO del benchmark oficial. Eje y "
                            "logarítmico. El P2 espacial depende\nde cómo se reparten los puntos entre regiones; P1 y "
                            "los bloques reparten por índice y no cambian.", size=(8.6, 4.8))
    width = 0.26
    for j, (lab, name, color) in enumerate(labels):
        xs = [i + (j - 1) * (width + 0.02) for i in range(len(dists))]
        vals = [sens_table[(d, 40000)][lab] for d, _ in dists]
        ax.bar(xs, vals, width=width, color=color, label=name, edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.1f}×", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", color=INK,
                        fontsize=8)
    ax.axhline(1, color=REF, linewidth=1.5, linestyle="--")
    ax.annotate("1× = igual que serial", (-0.5, 1), xytext=(2, 4), textcoords="offset points", color=INK2, fontsize=8,
                ha="left")
    ax.set_yscale("log")
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels([f"{d}\n({desc})" for d, desc in dists])
    style(ax, "distribución de los datos", "speedup respecto de serial (escala log)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    save(fig, extra / "extra2_sensibilidad_distribucion.png", top=0.82)

    # ---- Apéndice 3: variabilidad experimental ----
    anomalies = list(csv.DictReader((run_dir / "analisis" / "anomalias.csv").open()))
    anomaly_keys = {(a["mode"], int(a["n"]), int(a["k"]) if a["k"] else None, int(a["threads"])) for a in anomalies}
    fig, (ax,) = new_figure("Apéndice 3 · Variabilidad entre las 10 repeticiones de cada configuración",
                            "Coeficiente de variación (CV = desv. est. / media) de t_total, un punto por configuración. "
                            "Eje y logarítmico.\nLos círculos marcan las configuraciones con alguna repetición anómala.")
    offsets = {"serial": -0.25, "p1": 0.0, "p2": 0.25}
    colors = {"serial": C_SERIAL, "p1": C_P1, "p2": C_P2}
    names = {"serial": "Serial", "p1": "P1", "p2": "P2 espacial"}
    for mode in ("serial", "p1", "p2"):
        pts = [(sizes.index(k[1]) + offsets[mode], v["cv"]) for k, v in S.items() if k[0] == mode]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=18, color=colors[mode], label=names[mode],
                   edgecolors=SURFACE, linewidths=0.6)
    for key in anomaly_keys:
        ax.scatter([sizes.index(key[1]) + offsets[key[0]]], [S[key]["cv"]], s=150, facecolors="none",
                   edgecolors=INK, linewidths=1.2)
    ax.axhline(5, color=REF, linewidth=1.5, linestyle="--")
    ax.annotate("CV = 5 %", (len(sizes) - 0.6, 5), xytext=(0, 4), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(n // 1000) for n in sizes])
    style(ax, "N (miles de puntos)", "CV de t_total (%)")
    ax.legend(frameon=False, loc="upper center", ncol=3, labelcolor=INK)
    save(fig, extra / "extra3_variabilidad.png", top=0.82)

    # ---- README_GRAFICAS.md (números calculados aquí) ----
    cvs = {m: sorted(v["cv"] for k, v in S.items() if k[0] == m) for m in ("serial", "p1", "p2")}
    all_cv = sorted(v["cv"] for v in S.values())
    p90 = lambda xs: xs[int(round(0.9 * (len(xs) - 1)))]
    slower = [(n, k, t) for n in sizes for k in ks for t in threads if t > 1 and T("p1", n, None, t) < T("p2", n, k, t)]
    slower_ks = sorted({k for _, k, _ in slower})
    ratio_a = {t: [T("p1", n, None, t) / T("p2", n, best_k_at[(n, t)], t) for n in sizes] for t in threads if t > 1}
    k_used = sorted({best_k_at[(n, t)] for n in sizes for t in threads if t > 1})
    ratio = lambda n, k, t: T("p1", n, None, t) / T("p2", n, k, t)
    r_mid = [ratio(n, k, t) for n in sizes for k in ks if 8 <= k <= 16 for t in threads if t > 1]
    r_hi = [ratio(n, k, t) for n in sizes for k in ks if k >= 32 for t in threads if t > 1]
    high_cv = [k for k, v in S.items() if v["cv"] > 5]
    high_by_n = {n: sum(1 for k in high_cv if k[1] == n) for n in sizes}
    big_cv = {m: [v["cv"] for k, v in S.items() if k[0] == m and k[1] >= 120000] for m in ("serial", "p1", "p2")}
    max_cv_rest = max(v["cv"] for k, v in S.items() if k not in anomaly_keys)
    share_small = {k: 1 / share_by_k[k] for k in (2, 4) if k in share_by_k}
    i64 = ks.index(64) if 64 in ks else -1
    md = [
        "# Gráficas del reporte (versión 2)", "",
        f"Generadas con `scripts/graficas_reporte.py` a partir de `{run_dir.relative_to(REPO)}/mediciones.csv` (2,320 corridas "
        "oficiales, 232 configuraciones × 10 repeticiones, todas idénticas a serial en clases y conteos). No se volvió a "
        "ejecutar nada ni se modificó ningún dato. Valores = medias de 10 repeticiones de `t_total` "
        "(`omp_get_wtime()`, sin lectura ni escritura de CSV). Las figuras anteriores siguen en `../graficas/`.", "",
        "## Figuras principales", "",
        "### Fig. 1 · Tiempo oficial contra número de puntos — `fig1_tiempo_vs_N.png`",
        "**Pregunta:** ¿cómo se comparan los tiempos de serial y de las mejores configuraciones paralelas?",
        f"**Muestra:** serial, mejor P1 y mejor P2 por N (eje y logarítmico, porque los tiempos van de {T('p2', small, *best_p2[small]) * 1000:.1f} ms "
        f"a {ts[big]:.2f} s). **Conclusión:** a {miles(big)} puntos, serial tarda {ts[big]:.2f} s, el mejor P1 "
        f"{T('p1', big, None, best_p1[big]):.3f} s y el mejor P2 {T('p2', big, *best_p2[big]):.3f} s; las tres curvas crecen "
        "aproximadamente como N² y conservan su separación.",
        f"*Caption sugerido:* Tiempo oficial medio (10 repeticiones) contra N; escala logarítmica. La variabilidad es pequeña "
        f"(CV mediano {st.median(all_cv):.2f} %), por eso no se dibujan barras de error.", "",
        "### Fig. 2 · Aceleración de la mejor configuración — `fig2_speedup_mejores_vs_N.png`",
        "**Pregunta:** ¿cómo evoluciona la aceleración con el tamaño del problema?",
        f"**Conclusión:** el mejor P1 pasa de {y1[0]:.1f}× ({miles(small)}) a {y1[-1]:.1f}× ({miles(big)}); el mejor P2 de "
        f"{y2[0]:.1f}× a {y2[-1]:.1f}× (máximo {max(y2):.1f}×). Ambas superan con holgura el mínimo de 1.5× del enunciado; "
        "las dos crecen con N porque los costos fijos (crear hilos, construir regiones) pesan menos en problemas grandes.",
        "*Caption sugerido:* Speedup respecto de serial de la configuración más rápida de cada versión para cada N "
        "(mejor P1: " + ", ".join(sorted({hilos(best_p1[n]) for n in sizes})) + "; mejor P2: "
        + ", ".join(sorted({f'k = {best_p2[n][0]} con {hilos(best_p2[n][1])}' for n in sizes})) + ").", "",
        "### Fig. 3 · P1 contra hilos — `fig3_p1_speedup_vs_hilos.png`",
        "**Pregunta:** ¿cómo escala P1 con OpenMP?",
        f"**Conclusión:** con 6 hilos P1 es casi ideal (eficiencia {eff6:.2f}); con 12 la eficiencia baja a {eff12:.2f} porque "
        "4 de los 12 núcleos son de eficiencia (más lentos), `schedule(static)` les da la misma cantidad de puntos y, durante "
        "las mediciones, el proceso del sistema `BTLEServer` ocupaba un núcleo; con 24 "
        f"hilos apenas mejora ({sp('p1', big, None, 24):.1f}× a {miles(big)}): no hay más núcleos. A {miles(small)} puntos "
        "la escala es peor porque el trabajo total es pequeño.",
        "*Caption sugerido:* Speedup de P1 contra el número de hilos para tres tamaños; línea punteada: speedup ideal.", "",
        "### Fig. 5 · P2 contra k — `fig5_p2_speedup_vs_k.png`",
        "**Pregunta:** ¿cómo cambia P2 al variar el número de regiones?",
        f"**Conclusión:** con 1 hilo, P2 ya es {sp('p2', big, ks[0], 1):.1f}× (k={ks[0]}) a {sp('p2', big, ks[-1], 1):.1f}× "
        f"(k={ks[-1]}) más rápido que serial, solo por comparar menos pares; con 12 hilos llega a {sp('p2', big, ks[-1], 12):.1f}× "
        f"(k={ks[-1]}). Con k = 2 y 4 queda por debajo del mejor P1 ({best_p1_big:.1f}×): hay solo 2 o 4 tareas y la más "
        "cargada tiene cerca de la mitad del trabajo, así que los hilos aceleran a lo más "
        + " y ".join(f"{v:.2f}× (k={k})" for k, v in share_small.items()) + " (1 / max_task_share).",
        f"*Caption sugerido:* Speedup de P2 contra k a N = {miles(big)} para 1, 12 y 24 hilos; línea punteada: mejor P1.", "",
        "### Fig. 6 · P1 contra P2 con los mismos hilos — `fig6_p1_vs_p2_mismos_hilos.png`",
        "**Pregunta:** ¿cuándo y cuánto mejora P2 sobre P1 de forma comparable?",
        "**Opción elegida: A (razón T_P1 / T_P2 con el mismo número de hilos),** más un panel (b) por k. Es la comparación "
        "más justa: ambas versiones usan los mismos recursos y la razón no depende del tiempo serial (la opción B repetiría "
        "la Fig. 2). El panel (b) responde *cuándo*: muestra el valor de k a partir del cual P2 supera a P1.",
        f"**Conclusión:** con su mejor k (k = {', '.join(map(str, k_used))}), P2 es entre "
        f"{min(min(v) for v in ratio_a.values()):.1f}× y {max(max(v) for v in ratio_a.values()):.1f}× más rápido que P1 con "
        f"los mismos hilos, en todos los tamaños. Con k = 8 y 16 la ventaja es menor ({min(r_mid):.1f}× a {max(r_mid):.1f}×) "
        f"y con k = 32 y 64 va de {min(r_hi):.1f}× a {max(r_hi):.1f}×. Con k = {' y '.join(map(str, slower_ks))} P2 es más lento que P1 "
        f"({len(slower)} de {len(sizes) * len(ks) * (len(threads) - 1)} comparaciones con 6, 12 o 24 hilos).",
        "*Caption sugerido:* (a) Razón de tiempos T_P1 / T_P2 con 6, 12 y 24 hilos, P2 con su mejor k; (b) la misma razón "
        f"para cada k con N = {miles(big)} y 12 hilos. Valores > 1: P2 más rápido.", "",
        "### Fig. 4 · P2 contra hilos — `fig4_p2_speedup_vs_hilos.png` (apoyo)",
        "**Pregunta:** ¿cómo escala P2 con los hilos y cómo influye k?",
        f"**Conclusión:** con k = 2 el speedup no pasa de {max(sp('p2', big, 2, t) for t in threads):.1f}× aunque se agreguen "
        f"hilos (solo hay 2 tareas); con k = 64 llega a {max(sp('p2', big, 64, t) for t in threads):.1f}×. De 12 a 24 hilos "
        "no hay mejora: solo hay 12 núcleos y la región más cargada limita a Π (ver Apéndice 1).", "",
        "## Apéndice (`apendice/`)", "",
        f"- **Apéndice 1 · `extra1_descomposicion_A_Pi.png`** — S_total = A × Π a N = {miles(big)} y 12 hilos. Con k = 64: "
        f"A = {A[i64]:.2f} (≈ N²/pares = {big * big / pairs[64]:.2f}), Π = {Pi[i64]:.2f} (límite 1/max_task_share = "
        f"{1 / share[64]:.2f}), S_total = {Stot[i64]:.1f}×. Explica por qué P2 supera 12× con 12 núcleos: menos trabajo y "
        "además paralelismo. Es la figura clave para defender el resultado.",
        "- **Apéndice 2 · `extra2_sensibilidad_distribucion.png`** — prueba complementaria con datos sintéticos (no oficial). "
        "A 40 000 puntos: P2 espacial " + ", ".join(f"{d} {sens_table[(d, 40000)]['p2s:64:12']:.1f}×" for d, _ in dists)
        + "; P1 " + ", ".join(f"{sens_table[(d, 40000)]['p1:12']:.1f}×" for d, _ in dists) + "; P2 por bloques (Mario) "
        + ", ".join(f"{sens_table[(d, 40000)]['p2:64:12']:.1f}×" for d, _ in dists)
        + ". Las regiones dividen la caja en partes de igual tamaño, no de igual número de puntos.",
        f"- **Apéndice 3 · `extra3_variabilidad.png`** — CV de cada configuración. CV mediano {st.median(all_cv):.2f} %, "
        f"percentil 90 {p90(all_cv):.2f} %, máximo {max(all_cv):.1f} %. {len(high_cv)} de {len(all_cv)} configuraciones superan "
        "5 % (" + ", ".join(f"{c} con N = {miles(n)}" for n, c in high_by_n.items() if c) + "); casi todas tienen muchos hilos "
        "y tiempos cortos. Con N ≥ 120 000 el CV mediano es " + ", ".join(f"{st.median(big_cv[m]):.2f} % ({names[m]})" for m in big_cv)
        + f" y el máximo {max(max(v) for v in big_cv.values()):.1f} %. P1 es la versión más variable.", "",
        "### Variabilidad por versión", "", "| Versión | CV mediano | CV percentil 90 | CV máximo | configuraciones con CV > 5 % |",
        "|---|---|---|---|---|"]
    md += [f"| {names[m]} | {st.median(v):.2f} % | {p90(v):.2f} % | {max(v):.1f} % | {sum(c > 5 for c in v)} de {len(v)} |"
           for m, v in cvs.items()]
    md += ["", "### Las 3 anomalías detectadas (repeticiones > 1.25× o < 0.8× la mediana de su configuración)", "",
           "| Versión | N | k | hilos | repetición | t_total (s) | mediana (s) | relación |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {names[a['mode']]} | {miles(int(a['n']))} | {a['k']} | {a['threads']} | {a['rep']} | {float(a['t_total_s']):.4f} | "
           f"{float(a['mediana_s']):.4f} | {float(a['relacion']):.2f} |" for a in anomalies]
    md += ["", "**Impacto:** las tres son corridas de " + miles(small) + " puntos que duran entre "
           f"{min(float(a['mediana_s']) for a in anomalies) * 1000:.0f} y {max(float(a['t_total_s']) for a in anomalies) * 1000:.0f} ms, "
           "donde una interrupción del sistema de unos milisegundos pesa mucho. No se borraron: están incluidas en las medias. "
           "No cambian ninguna conclusión: cada una afecta una de diez repeticiones de una configuración pequeña, y las "
           f"diferencias que sostienen el análisis (P2 entre {min(min(v) for v in ratio_a.values()):.1f}× y "
           f"{max(max(v) for v in ratio_a.values()):.1f}× más rápido que P1 con su mejor k; la mejor configuración de cada "
           f"versión, al menos {min(y1):.1f}× más rápida que serial) son mucho mayores que la variabilidad (CV ≤ {max_cv_rest:.1f} % en todas "
           "las demás configuraciones).", "",
           "## Recomendación editorial", "",
           "**Reporte principal (6 figuras):** Fig. 1 (tiempos), Fig. 2 (speedup contra N), Fig. 3 (P1 contra hilos), "
           "Fig. 5 (P2 contra k), Fig. 6 (P1 contra P2 con los mismos hilos) y Apéndice 1 (S_total = A × Π), que conviene "
           "subir al cuerpo porque es la explicación de por qué P2 supera 12×.", "",
           "**Apéndice / respaldo:** Fig. 4 (redundante con la Fig. 5, que ya muestra los hilos y además k), Apéndice 2 "
           "(sensibilidad, para la sección de limitaciones) y Apéndice 3 (variabilidad, para la sección de método).", "",
           "**Mensaje de cada figura:**", "",
           "| Figura | Mensaje |", "|---|---|",
           "| Fig. 1 | Las tres versiones crecen como N²; la paralela más rápida es un orden de magnitud más rápida que serial. |",
           f"| Fig. 2 | Ambas versiones superan 1.5×; P1 llega a ≈{y1[-1]:.0f}× y P2 a ≈{y2[-1]:.0f}×. |",
           "| Fig. 3 | P1 escala casi idealmente hasta 6 hilos; los núcleos de eficiencia (y un núcleo ocupado) limitan el resto. |",
           "| Fig. 5 | P2 mejora al aumentar k; con k ≤ 4 no supera a P1. |",
           f"| Fig. 6 | Con los mismos hilos, P2 supera a P1 desde k = 8 ({min(r_mid):.1f}–{max(r_mid):.1f}×) y es "
           f"{min(r_hi):.1f}–{max(r_hi):.1f}× más rápido con k = 32–64; con k ≤ 4 es más lento. |",
           "| Apéndice 1 | S_total = A × Π: menos pares (A) y OpenMP (Π); Π lo limita la región más cargada. |",
           "| Apéndice 2 | La ventaja de P2 depende de la distribución de los datos; P1 y los bloques no. |",
           f"| Apéndice 3 | Mediciones estables (CV mediano {st.median(all_cv):.2f} %); 3 anomalías aisladas en N = 20 000, "
           "sin efecto en las conclusiones. |", "",
           "**Gráficas anteriores (`../graficas/`) que conviene descartar:** las seis quedan sustituidas. `tiempo_vs_N` fija "
           "P1 en 12 hilos en lugar de usar su mejor configuración (Fig. 1); `mejor_p1_vs_p2` → Fig. 2; "
           "`speedup_p1_vs_hilos` tiene cuatro tamaños casi superpuestos (Fig. 3 usa tres); `speedup_p2_vs_hilos` → Fig. 4; "
           "`speedup_p2_vs_k` tiene cuatro curvas de hilos casi iguales (Fig. 5 usa 1, 12 y 24 y añade la referencia de "
           "P1); `factores_A_Pi` → Apéndice 1 (mismo contenido, mejor explicado).", "",
           "**Qué figura responde cada punto de la rúbrica:**", "",
           "| Pregunta | Figuras |", "|---|---|",
           "| Comparación con el baseline serial | Fig. 1, Fig. 2 |",
           "| Speedup | Fig. 2 (resumen), Fig. 3 (P1), Fig. 5 (P2) |",
           "| Evolución con N | Fig. 1, Fig. 2, Fig. 6 (a) |",
           "| Evolución con hilos | Fig. 3 (P1), Fig. 4 (P2) |",
           "| Evolución con k | Fig. 5, Fig. 6 (b) |",
           "| Por qué P2 fue mejor (o equivalente) que P1 | Fig. 6 y Apéndice 1; límites en Apéndice 2 |", ""]
    (out / "README_GRAFICAS.md").write_text("\n".join(md))
    print(f"escrito: {out.relative_to(REPO)} ({len(list(out.glob('*.png')))} figuras principales, "
          f"{len(list(extra.glob('*.png')))} de apéndice, README_GRAFICAS.md)")


if __name__ == "__main__":
    main()
