#!/usr/bin/env python3
"""Análisis del benchmark: tablas y gráficas (en español) a partir de las mediciones de scripts/run_experiments.py.

Uso:
    python scripts/analyze_results.py --run-dir results/official/<run_id>
    python scripts/analyze_results.py --run-dir results/runner_test/<run_id>          # pruebas del ejecutor

Solo LEE las mediciones del programa C++ (t_total medido con omp_get_wtime); no ejecuta DBSCAN ni mide nada.
Antes de calcular, comprueba que estén todas las corridas del plan (para el oficial: 2,320 = 232 configuraciones x 10)
y que todas tengan los hashes de la referencia serial; si falta algo, lo dice y no rellena datos.

Escribe en <run-dir>/analisis/:
    mediciones_individuales.csv   todas las corridas (una fila por corrida)
    resumen_por_configuracion.csv media, desviación estándar, mínimo, máximo, mediana y CV por configuración, speedup
    factores_p2.csv               P2: S_total = T_serial / T_P2, A = T_serial / T_P2(1 hilo), Pi = T_P2(1) / T_P2(T)
    p2_vs_p1.csv                  P2 contra P1 con el mismo número de hilos (incluye dónde P2 es más lento)
    mejores_configuraciones.csv   mejor P1 y mejor P2 por tamaño
    anomalias.csv                 repeticiones alejadas de la mediana de su configuración (se señalan, no se borran)
    resumen.md                    tablas principales y explicación de A y Pi
    graficas/*.png                tiempo contra N, speedup contra hilos, speedup contra k, factores A y Pi, mejor P1 vs P2
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Paleta (guía de visualización): categórica para serial / P1 / P2 y una rampa azul de 4 pasos para variables
# ordenadas (N, k o hilos); ambas validadas. Texto en tinta neutra, nunca del color de la serie.
SERIES = {"serial": "#2a78d6", "p1": "#eb6834", "p2": "#1baf7a"}
RAMP4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAMES = {"serial": "Serial", "p1": "P1", "p2": "P2 espacial"}


def num(n):
    """Número con espacio como separador de miles (200 000)."""
    return f"{n:,}".replace(",", " ")


def hilos(t):
    return f"{t} hilo" if t == 1 else f"{t} hilos"


def fail(message):
    sys.exit(f"analyze_results: {message}")


def pick(values, count=4):
    """Hasta `count` valores representativos repartidos a lo largo de la lista (siempre el primero y el último)."""
    values = sorted(values)
    if len(values) <= count:
        return values
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def line(ax, xs, ys, color, label, direct=False, dashed=False):
    ax.plot(xs, ys, color=color, linewidth=2, linestyle="--" if dashed else "-", marker=None if dashed else "o",
            markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    if direct:
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Tablas y gráficas del benchmark (solo lee mediciones del C++).")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifiesto.json").read_text())
    plan = manifest["congelado"]["plan"]
    sizes, threads, ks, reps = plan["tamanos"], plan["hilos"], plan["k"], plan["repeticiones"]
    references = {int(r["n"]): r for r in csv.DictReader((run_dir / "referencias.csv").open())}

    # ---- Carga y verificación: todas las corridas, únicas, válidas y con los hashes de la referencia serial ----
    rows, seen = [], set()
    for r in csv.DictReader((run_dir / "mediciones.csv").open()):
        key = (r["mode"], int(r["n"]), r["k"], int(r["threads"]), int(r["rep"]))
        if key in seen:
            fail(f"corrida duplicada {key}")
        seen.add(key)
        ref = references[int(r["n"])]
        if any(r[h] != ref[h] for h in ("label_hash", "count_hash", "core1", "core2", "noise")):
            fail(f"la corrida {key} no coincide con la referencia serial")
        rows.append({**r, "n": int(r["n"]), "threads": int(r["threads"]), "rep": int(r["rep"]),
                     "k": int(r["k"]) if r["k"] else None, "t_total": float(r["t_total"]),
                     "t_build": float(r["t_build"]), "t_step1": float(r["t_step1"]), "t_step2": float(r["t_step2"]),
                     "pairs_step1": int(r["pairs_step1"]),
                     "max_task_share": float(r["max_task_share"]) if r["max_task_share"] else None,
                     "halo_points": int(r["halo_points"]) if r["halo_points"] else None})
    configs = [("serial", n, None, 1) for n in sizes] + [("p1", n, None, t) for n in sizes for t in threads] + \
              [("p2", n, k, t) for n in sizes for k in ks for t in threads]
    groups = {c: [] for c in configs}
    for r in rows:
        c = (r["mode"], r["n"], r["k"], r["threads"])
        if c not in groups:
            fail(f"configuración fuera del plan: {c}")
        groups[c].append(r)
    incomplete = [(c, len(v)) for c, v in groups.items() if len(v) != reps]
    if incomplete:
        fail(f"{len(incomplete)} de {len(configs)} configuraciones no tienen {reps} repeticiones (p. ej. {incomplete[:3]}); "
             f"hay {len(rows)} de {len(configs) * reps} corridas. No se rellenan datos faltantes.")
    print(f"verificado: {len(rows)} corridas únicas = {len(configs)} configuraciones x {reps} repeticiones; "
          f"todas con los hashes de serial")

    out = run_dir / "analisis"
    (out / "graficas").mkdir(parents=True, exist_ok=True)

    # ---- Estadísticas por configuración ----
    summary = {}
    for c, rs in groups.items():
        t = [r["t_total"] for r in rs]
        mean = st.mean(t)
        summary[c] = {"mode": c[0], "n": c[1], "k": c[2] or "", "threads": c[3], "repeticiones": len(t),
                      "media_s": mean, "desv_est_s": st.stdev(t) if len(t) > 1 else 0.0, "min_s": min(t),
                      "max_s": max(t), "mediana_s": st.median(t),
                      "cv_pct": 100 * (st.stdev(t) if len(t) > 1 else 0.0) / mean,
                      "media_build_s": st.mean(r["t_build"] for r in rs),
                      "media_paso1_s": st.mean(r["t_step1"] for r in rs),
                      "media_paso2_s": st.mean(r["t_step2"] for r in rs),
                      "pares_paso1": rs[0]["pairs_step1"], "halo_points": rs[0]["halo_points"] or "",
                      "max_task_share": rs[0]["max_task_share"] or ""}
    T = lambda mode, n, k, t: summary[(mode, n, k, t)]["media_s"]
    for c, s in summary.items():
        s["speedup"] = T("serial", c[1], None, 1) / s["media_s"]
        s["eficiencia"] = s["speedup"] / c[3] if c[0] == "p1" else ""   # P2: ver Pi / hilos en factores_p2.csv

    def write(name, records, fields):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for rec in records:
                w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items() if k in fields})

    individual_fields = ["mode", "n", "k", "threads", "rep", "seq", "t_total", "t_build", "t_step1", "t_step2",
                         "pairs_step1", "halo_points", "max_task_share", "core1", "core2", "noise", "label_hash",
                         "count_hash", "threads_actual", "schedule", "loadavg_1m", "time_utc"]
    write("mediciones_individuales.csv",
          sorted(rows, key=lambda r: (r["mode"], r["n"], r["k"] or 0, r["threads"], r["rep"])), individual_fields)
    write("resumen_por_configuracion.csv", list(summary.values()), list(next(iter(summary.values())).keys()))

    # ---- P2: S_total = A x Pi ----
    factors = []
    for n in sizes:
        for k in ks:
            t1 = T("p2", n, k, 1)
            A = T("serial", n, None, 1) / t1
            for t in threads:
                s = summary[("p2", n, k, t)]
                Pi = t1 / s["media_s"]
                factors.append({"n": n, "k": k, "threads": t, "S_total": s["speedup"], "A": A, "Pi": Pi,
                                "A_por_Pi": A * Pi, "eficiencia_hilos_Pi_entre_T": Pi / t,
                                "reduccion_pares_N2_entre_pares": n * n / s["pares_paso1"],
                                "max_task_share": s["max_task_share"], "limite_Pi_1_entre_max_share": 1 / s["max_task_share"]})
    write("factores_p2.csv", factors, list(factors[0].keys()))
    F = {(f["n"], f["k"], f["threads"]): f for f in factors}

    comparison = []
    for n in sizes:
        for t in threads:
            for k in ks:
                ratio = T("p1", n, None, t) / T("p2", n, k, t)
                comparison.append({"n": n, "threads": t, "k": k, "t_p1_s": T("p1", n, None, t), "t_p2_s": T("p2", n, k, t),
                                   "p2_mas_rapido_por": ratio, "p2_mas_lento_que_p1": "sí" if ratio < 1 else "no"})
    write("p2_vs_p1.csv", comparison, list(comparison[0].keys()))

    best = []
    for n in sizes:
        p1 = min((summary[("p1", n, None, t)] for t in threads), key=lambda s: s["media_s"])
        p2 = min((summary[("p2", n, k, t)] for k in ks for t in threads), key=lambda s: s["media_s"])
        best.append({"n": n, "t_serial_s": T("serial", n, None, 1), "mejor_p1_hilos": p1["threads"], "mejor_p1_s": p1["media_s"],
                     "mejor_p1_speedup": p1["speedup"], "mejor_p2_k": p2["k"], "mejor_p2_hilos": p2["threads"],
                     "mejor_p2_s": p2["media_s"], "mejor_p2_speedup": p2["speedup"],
                     "p2_entre_p1": p1["media_s"] / p2["media_s"]})
    write("mejores_configuraciones.csv", best, list(best[0].keys()))

    anomalies = []
    for c, rs in groups.items():
        med = summary[c]["mediana_s"]
        for r in rs:
            if med > 0 and (r["t_total"] > 1.25 * med or r["t_total"] < 0.8 * med) and abs(r["t_total"] - med) > 0.002:
                anomalies.append({"mode": c[0], "n": c[1], "k": c[2] or "", "threads": c[3], "rep": r["rep"],
                                  "t_total_s": r["t_total"], "mediana_s": med, "relacion": r["t_total"] / med,
                                  "loadavg_1m": r["loadavg_1m"], "time_utc": r["time_utc"]})
    write("anomalias.csv", anomalies, ["mode", "n", "k", "threads", "rep", "t_total_s", "mediana_s", "relacion",
                                       "loadavg_1m", "time_utc"])

    # ---- Gráficas (en español) ----
    big = max(sizes)
    t_mid = 12 if 12 in threads else max(threads)
    g = out / "graficas"

    fig, ax = plt.subplots(figsize=(7, 4.4))
    line(ax, sizes, [T("serial", n, None, 1) for n in sizes], SERIES["serial"], "Serial", direct=True)
    line(ax, sizes, [T("p1", n, None, t_mid) for n in sizes], SERIES["p1"], f"P1 ({t_mid} hilos)", direct=True)
    best_k = {n: min(ks, key=lambda k: T("p2", n, k, t_mid)) for n in sizes}
    line(ax, sizes, [T("p2", n, best_k[n], t_mid) for n in sizes], SERIES["p2"], f"P2 espacial ({t_mid} hilos, mejor k)", direct=True)
    ax.set_yscale("log")
    style(ax, "Tiempo oficial (t_total) contra número de puntos", "N (puntos)", f"tiempo medio de {reps} repeticiones (s, escala log)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    save(fig, g / "tiempo_vs_N.png")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for color, n in zip(RAMP4, pick(sizes)):
        # Sin etiquetas directas: con N grande las curvas casi coinciden y las etiquetas se enciman (basta la leyenda).
        line(ax, threads, [summary[("p1", n, None, t)]["speedup"] for t in threads], color, f"N = {num(n)}")
    line(ax, threads, threads, INK2, "ideal (speedup = hilos)", dashed=True)
    style(ax, "P1: speedup contra número de hilos", "hilos", "speedup = T_serial / T_P1")
    ax.set_xticks(threads)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    save(fig, g / "speedup_p1_vs_hilos.png")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for color, k in zip(RAMP4, pick(ks)):
        line(ax, threads, [summary[("p2", big, k, t)]["speedup"] for t in threads], color, f"k = {k}", direct=True)
    style(ax, f"P2 espacial (N = {num(big)}): speedup total contra hilos", "hilos", "speedup = T_serial / T_P2")
    ax.set_xticks(threads)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    save(fig, g / "speedup_p2_vs_hilos.png")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for color, t in zip(RAMP4, pick(threads)):
        line(ax, ks, [summary[("p2", big, k, t)]["speedup"] for k in ks], color, hilos(t), direct=True)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    style(ax, f"P2 espacial (N = {num(big)}): speedup total contra k", "k (regiones)", "speedup = T_serial / T_P2")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    save(fig, g / "speedup_p2_vs_k.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    panels = [("S_total = T_serial / T_P2", [F[(big, k, t_mid)]["S_total"] for k in ks]),
              ("A = T_serial / T_P2(1 hilo)", [F[(big, k, t_mid)]["A"] for k in ks]),
              (f"Pi = T_P2(1 hilo) / T_P2({t_mid} hilos)", [F[(big, k, t_mid)]["Pi"] for k in ks])]
    for i, (ax, (title, values)) in enumerate(zip(axes, panels)):
        line(ax, ks, values, SERIES["p2"], title)
        if i == 1:
            line(ax, ks, [F[(big, k, t_mid)]["reduccion_pares_N2_entre_pares"] for k in ks], INK2, "N² / pares evaluados", dashed=True)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        if i == 2:
            line(ax, ks, [F[(big, k, t_mid)]["limite_Pi_1_entre_max_share"] for k in ks], INK2, "límite 1 / max_task_share", dashed=True)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        style(ax, title, "k (regiones)", "factor")
    fig.suptitle(f"P2 espacial, N = {num(big)}, {hilos(t_mid)}: S_total = A × Pi", color=INK, fontsize=12)
    save(fig, g / "factores_A_Pi.png")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    line(ax, sizes, [b["mejor_p1_speedup"] for b in best], SERIES["p1"], "mejor P1", direct=True)
    line(ax, sizes, [b["mejor_p2_speedup"] for b in best], SERIES["p2"], "mejor P2 espacial", direct=True)
    style(ax, "Mejor configuración de P1 y de P2 por tamaño", "N (puntos)", "speedup = T_serial / T_versión")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    save(fig, g / "mejor_p1_vs_p2.png")

    # ---- Resumen en Markdown ----
    md = [f"# Resultados del benchmark — sesión `{manifest['run_id']}`", "",
          f"{len(rows)} corridas únicas y válidas ({len(configs)} configuraciones × {reps} repeticiones), todas con "
          "`label_hash` y `count_hash` idénticos a la referencia serial. Tiempo oficial: `t_total` "
          "(construcción + paso 1 + paso 2, `omp_get_wtime()`), sin lectura ni escritura de CSV. "
          f"Commit `{manifest['congelado']['git_commit'][:12]}`, binario SHA-256 `{manifest['congelado']['sha256_binario'][:16]}…`.", "",
          "## Serial", "", "| N | media (s) | desv. est. (s) | CV % |", "|---|---|---|---|"]
    md += [f"| {num(n)} | {T('serial', n, None, 1):.4f} | {summary[('serial', n, None, 1)]['desv_est_s']:.4f} | "
           f"{summary[('serial', n, None, 1)]['cv_pct']:.2f} |" for n in sizes]
    md += ["", "## P1: speedup (eficiencia = speedup / hilos)", "",
           "| N | " + " | ".join(hilos(t) for t in threads) + " |", "|---|" + "---|" * len(threads)]
    md += [f"| {num(n)} | " + " | ".join(f"{summary[('p1', n, None, t)]['speedup']:.2f}× ({summary[('p1', n, None, t)]['eficiencia']:.2f})"
                                     for t in threads) + " |" for n in sizes]
    md += ["", f"## P2 espacial: speedup total con {hilos(t_mid)}", "",
           "| N | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    md += [f"| {num(n)} | " + " | ".join(f"{summary[('p2', n, k, t_mid)]['speedup']:.2f}×" for k in ks) + " |" for n in sizes]
    md += ["", f"## P2 espacial, N = {num(big)}: S_total = A × Pi ({hilos(t_mid)})", "",
           "| k | N²/pares | A | Pi | límite de Pi (1/max_task_share) | S_total |", "|---|---|---|---|---|---|"]
    md += [f"| {k} | {F[(big, k, t_mid)]['reduccion_pares_N2_entre_pares']:.2f} | {F[(big, k, t_mid)]['A']:.2f} | "
           f"{F[(big, k, t_mid)]['Pi']:.2f} | {F[(big, k, t_mid)]['limite_Pi_1_entre_max_share']:.2f} | "
           f"{F[(big, k, t_mid)]['S_total']:.2f}× |" for k in ks]
    md += ["", "**Cómo leer A y Pi.** A mide la ganancia *algorítmica*: P2 con un solo hilo ya es más rápido que serial "
           "porque cada región solo se compara con su halo y no con todo el plano (A ≈ N²/pares). Pi mide la ganancia "
           "*de OpenMP* al usar varios hilos con el mismo k; está limitada por la región más costosa (1/max_task_share). "
           "S_total = A × Pi, por eso puede superar el número de núcleos: no es paralelismo superlineal, es menos trabajo "
           "más paralelismo. La eficiencia de los hilos de P2 se mide con Pi / hilos, no con S_total / hilos.", "",
           "## Mejor P1 contra mejor P2", "", "| N | mejor P1 | mejor P2 | P2 más rápido que P1 por |", "|---|---|---|---|"]
    md += [f"| {num(b['n'])} | {b['mejor_p1_speedup']:.2f}× ({hilos(b['mejor_p1_hilos'])}) | {b['mejor_p2_speedup']:.2f}× "
           f"(k={b['mejor_p2_k']}, {hilos(b['mejor_p2_hilos'])}) | {b['p2_entre_p1']:.2f}× |" for b in best]
    slower = [c for c in comparison if c["p2_mas_lento_que_p1"] == "sí"]
    md += ["", f"## Configuraciones donde P2 es más lento que P1 con los mismos hilos: {len(slower)} de {len(comparison)}", ""]
    md += [f"- N = {num(c['n'])}, k = {c['k']}, {hilos(c['threads'])}: P1 {c['t_p1_s']:.4f} s, P2 {c['t_p2_s']:.4f} s"
           for c in slower[:40]]
    if len(slower) > 40:
        md.append(f"- … ({len(slower) - 40} más en p2_vs_p1.csv)")
    md += ["", f"## Anomalías de tiempo señaladas (no se borró ninguna): {len(anomalies)}",
           "Repeticiones con t_total > 1.25 × o < 0.8 × la mediana de su configuración (lista completa en anomalias.csv).", "",
           "Nota: la sensibilidad de P2 espacial a la distribución de los puntos se documenta aparte, como prueba "
           "complementaria (results/sensibilidad/); esos datos no se mezclan con este benchmark.", ""]
    (out / "resumen.md").write_text("\n".join(md))
    print(f"escrito: {out} (6 CSV, resumen.md y {len(list(g.glob('*.png')))} gráficas)")


if __name__ == "__main__":
    main()
