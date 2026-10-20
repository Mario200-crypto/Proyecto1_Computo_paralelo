#!/usr/bin/env python3
"""Gráficas del reporte (versión 3): pocas figuras, configuraciones fijas, todo en español.

Uso: python scripts/graficas_v3.py [--run-dir results/official/oficial_20261020]

Solo LEE <run-dir>/mediciones.csv (las 2,320 mediciones oficiales). No ejecuta el programa, no modifica datos y no toca
las versiones anteriores (analisis/graficas/ y analisis/graficas_v2/). Escribe en <run-dir>/analisis/graficas_v3/:
    fig1_tiempo_vs_N.png, fig2_speedup_vs_N.png, fig3_aceleracion_paralela_vs_hilos.png, fig4_p2_tiempo_vs_k.png,
    fig5_barras_200k.png (opcional) y README.md (pies de figura e interpretación; todos los números se calculan aquí).

Configuraciones fijas (no se elige la mejor por N): Serial con 1 hilo, P1 con 12 hilos y P2 espacial con 12 hilos y
k = 64. Cada valor es la media de las 10 repeticiones de t_total. Todo speedup es una razón de medias
(media del tiempo de referencia / media del tiempo de la versión), nunca la media de razones individuales.
"""

import argparse
import csv
import hashlib
import io
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

NS = (20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000)
HILOS = (1, 6, 12, 24)
KS = (2, 4, 8, 16, 32, 64)
N_GRANDE = 200000
REPS = 10
TOTAL = 2320

# Configuraciones fijas: (modo, k, hilos).
SERIAL = ("serial", None, 1)
P1 = ("p1", None, 12)
P2 = ("p2", 64, 12)

# Estilo: un color por versión, igual en todas las figuras (paleta validada en la v2); texto en tinta neutra.
C_SERIAL, C_P1, C_P2 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, FONDO, ZONA = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#f1f0ec"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})
DPI = 300


def miles(n):
    return f"{n:,}".replace(",", " ")


# ------------------------------------------------------------------------------------------------ lectura y verificación

def leer(path):
    """Lee el CSV una sola vez: el SHA-256 y las filas salen de los mismos bytes."""
    datos = path.read_bytes()
    filas = list(csv.DictReader(io.StringIO(datos.decode("utf-8"))))
    return hashlib.sha256(datos).hexdigest(), filas


def verificar(filas):
    """Comprueba el benchmark completo antes de graficar. Regresa {(modo, n, k, hilos): [t_total de las 10 reps]}."""
    fallas = []
    if len(filas) != TOTAL:
        fallas.append(f"se esperaban {TOTAL} mediciones y hay {len(filas)}")

    reps = defaultdict(list)
    tiempos = defaultdict(list)
    for f in filas:
        clave = (f["mode"], int(f["n"]), int(f["k"]) if f["k"] else None, int(f["threads"]))
        reps[clave].append(int(f["rep"]))
        tiempos[clave].append(float(f["t_total"]))

    esperadas = ({("serial", n, None, 1) for n in NS}
                 | {("p1", n, None, t) for n in NS for t in HILOS}
                 | {("p2", n, k, t) for n in NS for k in KS for t in HILOS})
    if set(reps) != esperadas:
        fallas.append(f"configuraciones distintas a las esperadas: faltan {len(esperadas - set(reps))}, "
                      f"sobran {len(set(reps) - esperadas)}")
    incompletas = [c for c, r in reps.items() if sorted(r) != list(range(1, REPS + 1))]
    if incompletas:
        fallas.append(f"{len(incompletas)} configuraciones sin exactamente las repeticiones 1..{REPS}")

    # Cada corrida debe dar clases y conteos idénticos a la versión serial con el mismo N.
    ref = {}
    for f in filas:
        if f["mode"] == "serial":
            ref.setdefault(int(f["n"]), (f["label_hash"], f["count_hash"]))
    distintas = sum(1 for f in filas if (f["label_hash"], f["count_hash"]) != ref.get(int(f["n"])))
    if distintas:
        fallas.append(f"{distintas} corridas con resultados distintos a serial")

    if fallas:
        sys.exit("ERROR: " + "; ".join(fallas))
    print(f"verificado: {len(filas)} mediciones, {len(reps)} configuraciones × {REPS} repeticiones, "
          "todas idénticas a serial")
    return tiempos


# ------------------------------------------------------------------------------------------------ estilo

def figura(titulo, nota, paneles=1, tam=(6.5, 3.6)):
    """Figura con un título corto y, como mucho, una línea de nota con la configuración (no párrafos)."""
    fig, ejes = plt.subplots(1, paneles, figsize=tam)
    fig.patch.set_facecolor(FONDO)
    fig.text(0.012, 0.985, titulo, ha="left", va="top", fontsize=12, color=INK)
    if nota:
        fig.text(0.012, 0.985 - 0.30 / tam[1], nota, ha="left", va="top", fontsize=9, color=INK2)
    fig.techo = 1 - (0.50 if nota else 0.32) / tam[1]
    return fig, (list(ejes) if paneles > 1 else [ejes])


def ejes(ax, xlabel, ylabel, titulo=None):
    if titulo:
        ax.set_title(titulo, loc="left", color=INK, fontsize=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_facecolor(FONDO)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRID)


def linea(ax, xs, ys, color, etiqueta=None, dy=0):
    """Línea de 2 pt con marcadores; la etiqueta va a la derecha del último punto (sin leyenda)."""
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor=FONDO, markeredgewidth=1.2, zorder=3)
    if etiqueta:
        ax.annotate(etiqueta, (xs[-1], ys[-1]), xytext=(7, dy), textcoords="offset points",
                    ha="left", va="center", fontsize=9, color=INK)


def guardar(fig, ruta):
    fig.tight_layout(rect=(0, 0, 1, fig.techo))
    fig.savefig(ruta, dpi=DPI, facecolor=FONDO)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ figuras

def fig1(m, out):
    x = [n / 1000 for n in NS]
    ser = [m[("serial", n, None, 1)] for n in NS]
    p1 = [m[("p1", n, None, 12)] for n in NS]
    p2 = [m[("p2", n, 64, 12)] for n in NS]
    fig, (a, b) = figura("Tiempo de ejecución contra N",
                         "Serial: 1 hilo · P1: 12 hilos · P2: 12 hilos, k = 64 · media de 10 repeticiones",
                         paneles=2, tam=(6.5, 3.4))
    ejes(a, "Número de puntos N (miles)", "Tiempo medio (s)", "(a) Las tres versiones")
    ejes(b, "Número de puntos N (miles)", "Tiempo medio (s)", "(b) Acercamiento: P1 y P2")
    linea(a, x, ser, C_SERIAL, "Serial")
    linea(a, x, p1, C_P1, "P1")
    linea(a, x, p2, C_P2, "P2")
    linea(b, x, p1, C_P1, "P1")
    linea(b, x, p2, C_P2, "P2")
    a.set_ylim(0, max(ser) * 1.08)
    b.set_ylim(0, max(p1) * 1.12)
    for ax in (a, b):
        ax.set_xlim(0, 228)
        ax.set_xticks([0, 40, 80, 120, 160, 200])
    guardar(fig, out / "fig1_tiempo_vs_N.png")


def fig2(m, out):
    x = [n / 1000 for n in NS]
    s1 = [m[("serial", n, None, 1)] / m[("p1", n, None, 12)] for n in NS]
    s2 = [m[("serial", n, None, 1)] / m[("p2", n, 64, 12)] for n in NS]
    fig, (ax,) = figura("Speedup respecto de Serial contra N",
                        "P1: 12 hilos · P2: 12 hilos, k = 64 · Serial con el mismo N")
    ejes(ax, "Número de puntos N (miles)", "Speedup (×)")
    linea(ax, x, s2, C_P2, f"P2: {s2[-1]:.1f}×")
    linea(ax, x, s1, C_P1, f"P1: {s1[-1]:.1f}×")
    ax.set_ylim(0, max(s2) * 1.12)
    ax.set_xlim(0, 228)
    ax.set_xticks([0, 40, 80, 120, 160, 200])
    guardar(fig, out / "fig2_speedup_vs_N.png")
    return s1, s2


def fig3(m, out):
    n = N_GRANDE
    a1 = [m[("p1", n, None, 1)] / m[("p1", n, None, t)] for t in HILOS]
    a2 = [m[("p2", n, 64, 1)] / m[("p2", n, 64, t)] for t in HILOS]
    fig, (ax,) = figura("Aceleración al paralelizar contra número de hilos",
                        f"N = {miles(n)} · cada versión respecto de sí misma con 1 hilo (P2 con k = 64)")
    ejes(ax, "Hilos", "Aceleración paralela (×)")
    tope = max(a1 + a2) * 1.22
    # Zona discreta: más hilos que núcleos físicos.
    ax.axvspan(12, 28, color=ZONA, zorder=0, linewidth=0)
    ax.axvline(12, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.text(11.6, tope * 0.03, "12 núcleos físicos\n(8 de rendimiento + 4 de eficiencia)", ha="right", va="bottom",
            fontsize=8, color=INK2)
    ax.text(12.6, tope * 0.03, "sobresuscripción (más hilos que núcleos)", ha="left", va="bottom", fontsize=8,
            color=INK2)
    linea(ax, HILOS, a1, C_P1, "P1")
    linea(ax, HILOS, a2, C_P2, "P2")
    for t, v1, v2 in zip(HILOS, a1, a2):
        if t == 12:
            ax.annotate(f"{v1:.1f}×", (t, v1), xytext=(6, 7), textcoords="offset points", ha="left",
                        va="bottom", fontsize=9, color=INK)
            ax.annotate(f"{v2:.1f}×", (t, v2), xytext=(6, -7), textcoords="offset points", ha="left",
                        va="top", fontsize=9, color=INK)
    ax.set_xlim(0, 28)
    ax.set_xticks(HILOS)
    ax.set_ylim(0, tope)
    guardar(fig, out / "fig3_aceleracion_paralela_vs_hilos.png")
    return a1, a2


def fig4(m, out):
    n = N_GRANDE
    t2 = [m[("p2", n, k, 12)] for k in KS]
    t1 = m[("p1", n, None, 12)]
    pos = list(range(len(KS)))
    fig, (ax,) = figura("Tiempo de P2 según el número de regiones k",
                        f"N = {miles(n)} · 12 hilos · media de 10 repeticiones")
    ejes(ax, "Número de regiones k", "Tiempo medio (s)")
    ax.axhline(t1, color=C_P1, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"P1 (12 hilos): {t1:.2f} s", (pos[-1], t1), xytext=(0, 5), textcoords="offset points",
                ha="right", va="bottom", fontsize=9, color=INK)
    linea(ax, pos, t2, C_P2)
    ax.annotate("P2", (pos[0], t2[0]), xytext=(-7, -7), textcoords="offset points", ha="right", va="top",
                fontsize=9, color=INK)
    for p, v in zip(pos, t2):
        ax.annotate(f"{v:.2f} s", (p, v), xytext=(5, 5), textcoords="offset points", ha="left", va="bottom",
                    fontsize=8, color=INK2)
    ax.set_xticks(pos, [str(k) for k in KS])
    ax.set_xlim(-0.5, len(KS) - 0.4)
    ax.set_ylim(0, max(t2) * 1.12)
    guardar(fig, out / "fig4_p2_tiempo_vs_k.png")
    return t2, t1


def fig5(m, out):
    n = N_GRANDE
    nombres = ["Serial", "P1 (12 hilos)", "P2 (12 hilos, k = 64)"]
    valores = [m[("serial", n, None, 1)], m[("p1", n, None, 12)], m[("p2", n, 64, 12)]]
    fig, (ax,) = figura(f"Tiempo medio con N = {miles(n)}", None, tam=(6.5, 2.4))
    ejes(ax, "Tiempo medio (s)", "")
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ys = [2, 1, 0]
    ax.barh(ys, valores, height=0.55, color=[C_SERIAL, C_P1, C_P2], zorder=2)
    for y, v in zip(ys, valores):
        ax.annotate(f"{v:.3f} s", (v, y), xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=9, color=INK)
    ax.set_yticks(ys, nombres)
    ax.set_xlim(0, max(valores) * 1.15)
    guardar(fig, out / "fig5_barras_200k.png")
    return valores


# ------------------------------------------------------------------------------------------------ README

def readme(out, sha, m, cv_max, s1, s2, a1, a2, t2, t1, v5, k64_mejor):
    ns, ng = miles(NS[0]), miles(N_GRANDE)
    alg = m[("serial", N_GRANDE, None, 1)] / m[("p2", N_GRANDE, 64, 1)]
    cruce = next(k for k, v in zip(KS, t2) if v < t1)
    lentos = ", ".join(f"k = {k}: {v:.2f} s" for k, v in zip(KS, t2) if v >= t1)
    crec = [m[c[:1] + (N_GRANDE,) + c[1:]] / m[c[:1] + (NS[0],) + c[1:]] for c in
            (("serial", None, 1), ("p1", None, 12), ("p2", 64, 12))]
    i12, i24 = HILOS.index(12), HILOS.index(24)
    s1_desde80 = [v for n, v in zip(NS, s1) if n >= 80000]
    n_max2 = NS[s2.index(max(s2))]
    texto = f"""# Gráficas del reporte (versión 3)

Fuente única: `mediciones.csv` (2,320 corridas; SHA-256 `{sha[:16]}…`, sin cambios). Script: `scripts/graficas_v3.py`.
**Convenciones.** Cada punto es la media de las 10 repeticiones de `t_total` (construcción + paso 1 + paso 2, sin leer ni escribir CSV); se omiten barras de error porque la desviación estándar es a lo más {cv_max:.1f} % de la media en las configuraciones graficadas, muy por debajo de las diferencias que muestran las figuras. Todo speedup es una **razón de medias** (media de referencia / media de la versión).
**Configuraciones fijas:** Serial (1 hilo), P1 (12 hilos) y P2 espacial (12 hilos, k = 64). Se usan 12 hilos porque el equipo tiene 12 núcleos físicos, aunque heterogéneos (8 de rendimiento y 4 de eficiencia), y k = 64 porque fue la configuración más rápida de P2 con 12 hilos en {k64_mejor} de los 8 tamaños oficiales.

**Fig. 1 — Tiempo contra N** (`fig1_tiempo_vs_N.png`). *Pie:* Tiempo medio de ejecución contra N; (a) las tres versiones, (b) acercamiento a las versiones paralelas. Media de 10 repeticiones.
*Interpretación:* con N = {ng}, Serial tarda {v5[0]:.3f} s, P1 {v5[1]:.3f} s y P2 {v5[2]:.3f} s. De {ns} a {ng} puntos (10 veces más), Serial tarda {crec[0]:.0f}× más (≈ 10² = 100, crece como N² porque compara todos los pares); P1 y P2 crecen algo menos ({crec[1]:.0f}× y {crec[2]:.0f}×) porque con N pequeño pesan más las partes que crecen más despacio que N² (sincronizar los hilos y, en P2, construir las regiones).

**Fig. 2 — Speedup contra N** (`fig2_speedup_vs_N.png`). *Pie:* Speedup de P1 y P2 respecto de Serial con el mismo N (razón de tiempos medios).
*Interpretación:* P1 pasa de {s1[0]:.1f}× ({ns} puntos) a {s1[-1]:.1f}× ({ng}) y se estabiliza entre {min(s1_desde80):.1f}× y {max(s1_desde80):.1f}× desde 80 000; P2 pasa de {s2[0]:.1f}× a {s2[-1]:.1f}× (máximo {max(s2):.1f}× con {miles(n_max2)}). Ambas superan el mínimo de 1.5× en todos los tamaños.

**Fig. 3 — Aceleración al paralelizar** (`fig3_aceleracion_paralela_vs_hilos.png`). *Pie:* Con N = {ng}, tiempo de cada versión con 1 hilo entre su tiempo con T hilos (P2 con k = 64). Mide solo el efecto de OpenMP.
*Interpretación:* con 12 hilos P1 acelera {a1[i12]:.1f}× y P2 {a2[i12]:.1f}×; con 24 hilos, {a1[i24]:.1f}× y {a2[i24]:.1f}×, sin ganancia real porque no hay más núcleos. P1 con 1 hilo tarda casi lo mismo que Serial ({m[("p1", N_GRANDE, None, 1)]:.3f} s contra {v5[0]:.3f} s), por eso su valor con 12 hilos coincide con el de la Fig. 2 en N = {ng}. P2 se paraleliza peor que P1 (su región más cargada limita el reparto); su ventaja de la Fig. 2 viene de que, ya con 1 hilo, es {alg:.1f}× más rápido que Serial al comparar menos pares: {alg:.1f} × {a2[i12]:.1f} ≈ {alg * a2[i12]:.1f}×.

**Fig. 4 — Efecto de k** (`fig4_p2_tiempo_vs_k.png`). *Pie:* Tiempo medio de P2 para cada k con N = {ng} y 12 hilos; línea punteada: P1 con los mismos 12 hilos.
*Interpretación:* con pocas regiones P2 es más lento que P1 ({lentos}, contra {t1:.2f} s): hay solo 2 o 4 tareas para 12 hilos. Desde k = {cruce} es más rápido, y con k = 64 tarda {t2[-1]:.3f} s: más regiones significan menos pares por comparar y más tareas para repartir.

**Fig. 5 (opcional) — Barras con N = {ng}** (`fig5_barras_200k.png`). Repite el extremo derecho de la Fig. 1; se recomienda sustituirla por esta tabla:

| Versión | Tiempo medio (s) | Speedup |
|---|---|---|
| Serial | {v5[0]:.3f} | 1.0× |
| P1, 12 hilos | {v5[1]:.3f} | {v5[0] / v5[1]:.1f}× |
| P2, 12 hilos, k = 64 | {v5[2]:.3f} | {v5[0] / v5[2]:.1f}× |

**Recomendación:** Fig. 1, 2, 3 y 4 en el reporte, y la tabla en lugar de la Fig. 5.
"""
    (out / "README.md").write_text(texto, encoding="utf-8")


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default="results/official/oficial_20261020")
    args = ap.parse_args()
    run = (REPO / args.run_dir).resolve()
    csv_path = run / "mediciones.csv"

    sha, filas = leer(csv_path)
    tiempos = verificar(filas)
    m = {c: st.fmean(v) for c, v in tiempos.items()}

    # Filtros de las configuraciones fijas y valores de control con N = 200 000.
    usadas = [("serial", n, None, 1) for n in NS] + [("p1", n, None, t) for n in NS for t in (1, 6, 12, 24)
                                                     if t == 12 or n == N_GRANDE]
    usadas += [("p2", n, 64, 12) for n in NS] + [("p2", N_GRANDE, 64, t) for t in HILOS]
    usadas += [("p2", N_GRANDE, k, 12) for k in KS]
    usadas = sorted(set(usadas), key=str)
    for c in usadas:
        assert len(tiempos[c]) == REPS, c
    print(f"filtros: {len(usadas)} configuraciones graficadas, cada una con {REPS} repeticiones")
    for nombre, c in (("Serial", SERIAL), ("P1, 12 hilos", P1), ("P2, 12 hilos, k = 64", P2)):
        clave = (c[0], N_GRANDE, c[1], c[2])
        print(f"  N = {miles(N_GRANDE)}  {nombre:<22} media {m[clave]:.4f} s  (n = {len(tiempos[clave])})")
    cv_max = max(100 * st.stdev(tiempos[c]) / m[c] for c in usadas)
    k64_mejor = sum(1 for n in NS if min(KS, key=lambda k: m[("p2", n, k, 12)]) == 64)

    out = run / "analisis" / "graficas_v3"
    out.mkdir(parents=True, exist_ok=True)
    fig1(m, out)
    s1, s2 = fig2(m, out)
    a1, a2 = fig3(m, out)
    t2, t1 = fig4(m, out)
    v5 = fig5(m, out)
    readme(out, sha, m, cv_max, s1, s2, a1, a2, t2, t1, v5, k64_mejor)

    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != sha:
        sys.exit("ERROR: mediciones.csv cambió durante la ejecución")
    print(f"mediciones.csv sin cambios (SHA-256 {sha})")
    print(f"escrito: {out.relative_to(REPO)} (5 figuras, README.md)")


if __name__ == "__main__":
    main()
