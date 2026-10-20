#!/usr/bin/env python3
"""Ejecutor del benchmark oficial: 2,320 corridas de bin/dbscan (serial, P1 y P2) en orden aleatorizado reproducible.

Este script SOLO lanza el programa C++ y guarda lo que reporta en su línea METRICS. No implementa DBSCAN, no hace
trabajo en paralelo, no compila y no mide tiempos por su cuenta: el tiempo oficial es t_total, medido por el propio
programa con omp_get_wtime().

Uso (desde la raíz del repositorio, con el entorno virtual activo):
    python scripts/run_experiments.py --dry-run                        # muestra y verifica el plan; no ejecuta nada
    python scripts/run_experiments.py --run-id oficial_20261020        # inicia o REANUDA el benchmark oficial
    python scripts/run_experiments.py --status --run-id oficial_20261020
    python scripts/run_experiments.py --test --run-id prueba1 --sizes 4000 --reps 2 --threads 1,12 --ks 2,64

Benchmark oficial (enunciado, diapositiva 14): N = 20000 ... 200000 (8 tamaños), hilos 1, 6, 12 y 24, k = 2 ... 64
(6 valores, solo P2), 10 repeticiones: serial 8 x 10 = 80, P1 8 x 4 x 10 = 320, P2 8 x 6 x 4 x 10 = 1,920 -> 2,320.

Carpeta de resultados: results/official/<run_id>/ (las pruebas del ejecutor van a results/runner_test/<run_id>/):
    manifiesto.json       procedencia: commit, SHA-256 de código, binario y datos, compilador, máquina, plan y semilla
    referencias.csv       una corrida serial de referencia por dataset (no cuenta como repetición)
    mediciones.csv        UNA fila por corrida exitosa (solo corridas válidas; se escribe y sincroniza fila por fila)
    fallas.csv            cada intento fallido, con su motivo (nunca aparece en mediciones.csv)
    registro.log          bitácora legible
    progreso.json         avance actual
Reanudar = volver a ejecutar el mismo comando: las corridas ya registradas en mediciones.csv no se repiten.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BINARY = REPO / "bin" / "dbscan"
SOURCE_FILES = ("src/dbscan.cpp", "Makefile", "scripts/run_experiments.py")   # código congelado

OFFICIAL_SIZES = (20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000)
OFFICIAL_THREADS = (1, 6, 12, 24)
OFFICIAL_KS = (2, 4, 8, 16, 32, 64)
OFFICIAL_REPS = 10
EXPECTED_TOTAL = 2320
EPS, MIN_SAMPLES = "0.03", 10          # parámetros de la libreta del profesor
EXPECTED_VERSION = "m5"                # versión del programa congelado (campo METRICS "version")
TIMEOUT_S = 300                        # una corrida normal tarda < 10 s
MAX_ATTEMPTS = 3                       # intentos por corrida ante fallas transitorias (error, timeout, METRICS mal formado)

# Hashes de la versión serial validada (scripts/validate.py all; también iguales a P1 y P2). Si la referencia de un
# dataset no coincide, el benchmark no empieza.
VALIDATED = {
    4000: ("f4f5f8bb42f1b83c", "a56eaa310e975371"), 20000: ("35e64851010283cf", "87ae511821cc2727"),
    40000: ("d9cd37edf20b1016", "9686fdc54323e395"), 80000: ("77915b8ef424d612", "7bc9dca94074be67"),
    120000: ("496ce12c02b572e3", "0b638d5538693451"), 140000: ("59c4e666180874dd", "24962a4a141b54f8"),
    160000: ("56182804d776ed5f", "8ff42be7d5ab967f"), 180000: ("dec3fa3398f5b130", "3a6fdae36710bb05"),
    200000: ("5e96964b762d458b", "64aa7f62aaab4893"),
}

FIELDS = ["run_id", "seq", "rep", "mode", "n", "k", "threads", "t_total", "t_build", "t_step1", "t_step2", "t_read",
          "pairs_step1", "halo_points", "tiles_nonempty", "max_task_share", "core1", "core2", "noise", "label_hash",
          "count_hash", "threads_actual", "schedule", "version", "openmp", "attempt", "loadavg_1m", "time_utc"]
FAIL_FIELDS = ["run_id", "seq", "rep", "mode", "n", "k", "threads", "attempt", "status", "detail", "time_utc"]


# ---------------------------------------------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------------------------------------------

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def command_output(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def dataset(n):
    return REPO / "data" / f"{n}_data.csv"


def key_of(run):
    return f"{run['mode']}|{run['n']}|{run['k']}|{run['threads']}|{run['rep']}"


def clean_env():
    """Entorno sin variables OMP_* / KMP_*: todas las corridas usan la configuración por omisión de OpenMP y el número
    de hilos lo fija el propio programa con --threads (omp_set_num_threads)."""
    return {k: v for k, v in os.environ.items() if not (k.startswith("OMP_") or k.startswith("KMP_"))}


def command_for(run):
    """Argumentos reales de bin/dbscan: serial sin --threads ni --k; P1 con --threads; P2 con --k y --threads."""
    cmd = [str(BINARY), "--input", str(dataset(run["n"])), "--mode", run["mode"], "--eps", EPS,
           "--min-samples", str(MIN_SAMPLES)]
    if run["mode"] in ("p1", "p2"):
        cmd += ["--threads", str(run["threads"])]
    if run["mode"] == "p2":
        cmd += ["--k", str(run["k"])]
    return cmd


# ---------------------------------------------------------------------------------------------------------------
# Plan experimental
# ---------------------------------------------------------------------------------------------------------------

def build_plan(sizes, threads, ks, reps, seed):
    """Configuraciones (serial: 1 hilo y sin k; P1: hilos; P2: k x hilos) y, por repetición, un orden barajado con
    una semilla derivada de `seed`: cada repetición recorre todas las configuraciones antes de empezar la siguiente,
    así una deriva de temperatura o de carga afecta a todas por igual."""
    configs = []
    for n in sizes:
        configs.append({"mode": "serial", "n": n, "k": "", "threads": 1})
        configs += [{"mode": "p1", "n": n, "k": "", "threads": t} for t in threads]
        configs += [{"mode": "p2", "n": n, "k": k, "threads": t} for k in ks for t in threads]
    plan = []
    for rep in range(1, reps + 1):
        order = list(configs)
        random.Random(seed * 1000 + rep).shuffle(order)
        plan += [dict(c, rep=rep) for c in order]
    for seq, run in enumerate(plan, 1):
        run["seq"] = seq
    return configs, plan


def check_plan(plan, sizes, threads, ks, reps, official):
    counts = {m: sum(1 for r in plan if r["mode"] == m) for m in ("serial", "p1", "p2")}
    expected = {"serial": len(sizes) * reps, "p1": len(sizes) * len(threads) * reps,
                "p2": len(sizes) * len(ks) * len(threads) * reps}
    problems = []
    if counts != expected:
        problems.append(f"conteo por modo {counts} != esperado {expected}")
    if len({key_of(r) for r in plan}) != len(plan):
        problems.append("hay corridas duplicadas en el plan")
    if official and len(plan) != EXPECTED_TOTAL:
        problems.append(f"el plan oficial tiene {len(plan)} corridas, no {EXPECTED_TOTAL}")
    for r in plan:
        cmd = command_for(r)
        if r["mode"] == "serial" and ("--k" in cmd or "--threads" in cmd):
            problems.append("serial no debe recibir --k ni --threads")
        if r["mode"] == "p1" and ("--k" in cmd or "--threads" not in cmd):
            problems.append("p1 recibe --threads y no --k")
        if r["mode"] == "p2" and ("--k" not in cmd or "--threads" not in cmd):
            problems.append("p2 recibe --k y --threads")
    return counts, sorted(set(problems))


# ---------------------------------------------------------------------------------------------------------------
# Procedencia, condiciones del sistema y candado
# ---------------------------------------------------------------------------------------------------------------

def provenance(sizes):
    git_status = command_output(["git", "-C", str(REPO), "status", "--porcelain", "--", *SOURCE_FILES])
    return {
        "git_commit": command_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_branch": command_output(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"]),
        "codigo_sin_commit": git_status,   # vacío = código congelado en el commit
        "sha256_codigo": {f: sha256(REPO / f) for f in SOURCE_FILES},
        "sha256_binario": sha256(BINARY),
        "sha256_datos": {str(n): sha256(dataset(n)) for n in sizes},
        "build_info": (REPO / "bin" / "build_info.txt").read_text() if (REPO / "bin" / "build_info.txt").is_file() else "",
        "compilador": command_output(["clang++", "--version"]).splitlines()[:1],
    }


def system_conditions():
    top = command_output(["ps", "-Ao", "pcpu,comm", "-r"]).splitlines()[1:6]
    return {
        "fecha_utc": now_utc(),
        "energia": command_output(["pmset", "-g", "batt"]).splitlines()[:1],
        "carga_1_5_15": list(os.getloadavg()),
        "procesos_con_mas_cpu": [line.strip() for line in top],
        "variables_omp_ignoradas": sorted(k for k in os.environ if k.startswith(("OMP_", "KMP_"))),
    }


def machine_info():
    sysctl = lambda key: command_output(["sysctl", "-n", key])
    return {
        "modelo": sysctl("hw.model"), "cpu": sysctl("machdep.cpu.brand_string"),
        "nucleos_fisicos": sysctl("hw.physicalcpu"), "nucleos_logicos": sysctl("hw.logicalcpu"),
        "nucleos_rendimiento": sysctl("hw.perflevel0.physicalcpu"), "nucleos_eficiencia": sysctl("hw.perflevel1.physicalcpu"),
        "memoria_bytes": sysctl("hw.memsize"), "sistema": command_output(["sw_vers"]).replace("\n", "; "),
        "uname": platform.uname()._asdict(), "python": sys.version.split()[0],
    }


def warnings_for(conditions):
    found = []
    if not any("AC Power" in line for line in conditions["energia"]):
        found.append("el Mac NO está conectado a la corriente")
    for line in conditions["procesos_con_mas_cpu"]:
        parts = line.split(None, 1)
        try:
            if float(parts[0]) >= 50 and "dbscan" not in parts[1]:
                found.append(f"proceso con CPU alta: {line}")
        except (ValueError, IndexError):
            pass
    return found


LOCK = REPO / "results" / ".benchmark.lock"


def acquire_lock(run_id):
    """Un solo benchmark (oficial o de prueba) a la vez en esta máquina."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            info = json.loads(LOCK.read_text())
            os.kill(int(info["pid"]), 0)
            sys.exit(f"ya hay un benchmark en curso (pid {info['pid']}, run_id {info['run_id']}); no se inicia otro")
        except (ProcessLookupError, ValueError, KeyError, json.JSONDecodeError):
            print(f"aviso: se elimina un candado abandonado ({LOCK.read_text().strip()})")
            LOCK.unlink()
        except PermissionError:
            sys.exit("hay un candado de otro usuario; no se inicia otro benchmark")
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({"pid": os.getpid(), "run_id": run_id, "inicio_utc": now_utc()}).encode())
    os.close(fd)


def release_lock():
    try:
        if json.loads(LOCK.read_text()).get("pid") == os.getpid():
            LOCK.unlink()
    except (OSError, json.JSONDecodeError):
        pass


# ---------------------------------------------------------------------------------------------------------------
# Archivos de resultados (escritura fila por fila, con fsync)
# ---------------------------------------------------------------------------------------------------------------

def append_row(path, fields, row):
    new = not path.exists()
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    if new:
        writer.writeheader()
    writer.writerow(row)
    with open(path, "a") as f:
        f.write(buffer.getvalue())
        f.flush()
        os.fsync(f.fileno())


def load_done(path, plan_keys, log, repair=True):
    """Corridas ya registradas en mediciones.csv. Una última línea incompleta (corte de energía durante la escritura)
    se guarda aparte y se descarta (con repair=False solo se ignora: --status nunca modifica archivos); cualquier
    otra fila inválida o duplicada detiene el ejecutor."""
    if not path.exists():
        return {}
    data = path.read_text()
    if data and not data.endswith("\n"):
        cut = data.rfind("\n") + 1
        if repair:
            (path.parent / "lineas_incompletas.txt").open("a").write(data[cut:] + "\n")
            path.write_text(data[:cut])
            log(f"aviso: se descartó una última línea incompleta de {path.name} (guardada en lineas_incompletas.txt)")
        data = data[:cut]
    done = {}
    for row in csv.DictReader(io.StringIO(data)):
        if None in row or any(row.get(f) in (None, "") for f in ("mode", "n", "rep", "t_total", "label_hash")):
            sys.exit(f"fila inválida en {path}: {row}; revise el archivo antes de reanudar")
        key = f"{row['mode']}|{row['n']}|{row['k']}|{row['threads']}|{row['rep']}"
        if key not in plan_keys:
            sys.exit(f"{path} contiene una corrida que no pertenece al plan: {key}")
        if key in done:
            sys.exit(f"{path} contiene una corrida duplicada: {key}")
        done[key] = row
    return done


# ---------------------------------------------------------------------------------------------------------------
# Una corrida
# ---------------------------------------------------------------------------------------------------------------

def run_once(run):
    """Ejecuta bin/dbscan una vez. Devuelve (metrics, None) o (None, motivo)."""
    try:
        proc = subprocess.run(command_for(run), capture_output=True, text=True, timeout=TIMEOUT_S, env=clean_env())
    except subprocess.TimeoutExpired:
        return None, f"timeout de {TIMEOUT_S} s"
    if proc.returncode != 0:
        return None, f"exit {proc.returncode}: {proc.stderr.strip()[:300]}"
    lines = proc.stdout.splitlines()
    if len(lines) != 1 or not lines[0].startswith("METRICS "):
        return None, f"stdout no es una sola línea METRICS ({len(lines)} líneas)"
    try:
        return json.loads(lines[0][len("METRICS "):]), None
    except json.JSONDecodeError as exc:
        return None, f"METRICS no es JSON válido ({exc})"


def check_run(run, metrics, reference):
    """Problemas de una corrida frente a lo pedido y a la referencia serial de su dataset (lista vacía = válida).
    Devuelve (problemas_de_configuración, problemas_de_resultado)."""
    config, result = [], []
    expected = {"mode": run["mode"], "n": run["n"], "min_samples": MIN_SAMPLES, "version": EXPECTED_VERSION,
                "threads_requested": run["threads"], "threads_actual": run["threads"],
                "k": run["k"] if run["mode"] == "p2" else None}
    for key, value in expected.items():
        if metrics.get(key) != value:
            config.append(f"{key} = {metrics.get(key)!r}, se esperaba {value!r}")
    if metrics.get("eps") != float(EPS):
        config.append(f"eps = {metrics.get('eps')!r}")
    if not (isinstance(metrics.get("openmp"), int) and metrics["openmp"] > 0):
        config.append("el binario no reporta OpenMP")
    for key in ("label_hash", "count_hash", "core1", "core2", "noise"):
        if metrics.get(key) != reference[key]:
            result.append(f"{key} = {metrics.get(key)!r}, serial = {reference[key]!r}")
    return config, result


# ---------------------------------------------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ejecutor del benchmark oficial (2,320 corridas de bin/dbscan).")
    parser.add_argument("--run-id", help="nombre de la sesión experimental (carpeta de resultados)")
    parser.add_argument("--seed", type=int, default=20261020, help="semilla del orden experimental")
    parser.add_argument("--dry-run", action="store_true", help="solo mostrar y verificar el plan")
    parser.add_argument("--status", action="store_true", help="mostrar el avance de --run-id")
    parser.add_argument("--test", action="store_true", help="prueba del ejecutor: results/runner_test/, sin exigir commit")
    parser.add_argument("--sizes", help="(solo --test) tamaños separados por comas")
    parser.add_argument("--threads", help="(solo --test) hilos separados por comas")
    parser.add_argument("--ks", help="(solo --test) valores de k separados por comas")
    parser.add_argument("--reps", type=int, help="(solo --test) repeticiones")
    parser.add_argument("--max-runs", type=int, help="(solo --test) detenerse después de este número de corridas nuevas")
    parser.add_argument("--permitir-sin-commit", action="store_true",
                        help="continuar aunque Makefile o este ejecutor no estén en el commit (src/dbscan.cpp sí debe "
                             "estarlo); los archivos sin commit y sus SHA-256 quedan en el manifiesto")
    args = parser.parse_args()

    ints = lambda text: tuple(int(v) for v in text.split(",") if v.strip())
    official = not args.test
    if official and any(v is not None for v in (args.sizes, args.threads, args.ks, args.reps, args.max_runs)):
        sys.exit("--sizes, --threads, --ks, --reps y --max-runs solo se permiten con --test")
    sizes = ints(args.sizes) if args.sizes else OFFICIAL_SIZES
    threads = ints(args.threads) if args.threads else OFFICIAL_THREADS
    ks = ints(args.ks) if args.ks else OFFICIAL_KS
    reps = args.reps or OFFICIAL_REPS
    configs, plan = build_plan(sizes, threads, ks, reps, args.seed)
    counts, problems = check_plan(plan, sizes, threads, ks, reps, official)

    if args.dry_run:
        print(f"plan {'OFICIAL' if official else 'de prueba'}: {len(configs)} configuraciones x {reps} repeticiones = "
              f"{len(plan)} corridas  {counts}  (semilla {args.seed})")
        for mode in ("serial", "p1", "p2"):
            example = next(r for r in plan if r["mode"] == mode)
            print(f"  ejemplo {mode:6s}: {' '.join(command_for(example)[1:])}")
        print("  primeras 5 corridas:", [key_of(r) for r in plan[:5]])
        missing = [str(dataset(n)) for n in sizes if not dataset(n).is_file()]
        print("  datasets:", "faltan " + ", ".join(missing) if missing else "todos presentes")
        print("  verificación del plan:", "OK" if not problems else "; ".join(problems))
        return 0 if not problems and not missing else 1

    if not args.run_id:
        sys.exit("indique --run-id (por ejemplo --run-id oficial_20261020)")
    run_dir = REPO / "results" / ("official" if official else "runner_test") / args.run_id
    measurements, failures = run_dir / "mediciones.csv", run_dir / "fallas.csv"
    plan_keys = {key_of(r) for r in plan}

    if args.status:
        done = load_done(measurements, plan_keys, print, repair=False) if measurements.exists() else {}
        n_fail = max(0, len(failures.read_text().splitlines()) - 1) if failures.exists() else 0
        by_mode = {m: sum(1 for r in done.values() if r["mode"] == m) for m in ("serial", "p1", "p2")}
        running = LOCK.exists() and args.run_id in LOCK.read_text()
        print(f"{run_dir.relative_to(REPO)}: {len(done)}/{len(plan)} corridas exitosas {by_mode}; "
              f"faltan {len(plan) - len(done)}; intentos fallidos registrados: {n_fail}; "
              f"{'EN CURSO' if running else 'sin proceso activo'}")
        return 0

    if problems:
        sys.exit("plan inválido: " + "; ".join(problems))
    if not BINARY.is_file():
        sys.exit("falta bin/dbscan: compile con `make` (el ejecutor nunca compila)")
    for f in ("src/dbscan.cpp", "Makefile"):
        if (REPO / f).stat().st_mtime > BINARY.stat().st_mtime:
            sys.exit(f"{f} es más reciente que bin/dbscan: compile con `make` antes de medir")
    for n in sizes:
        if n not in VALIDATED or not dataset(n).is_file():
            sys.exit(f"dataset {n} sin referencia validada o inexistente")

    # El código debe estar congelado en un commit ANTES de crear cualquier archivo de la sesión oficial.
    prov = provenance(sizes)
    if official and prov["codigo_sin_commit"]:
        if "src/dbscan.cpp" in prov["codigo_sin_commit"] or not args.permitir_sin_commit:
            sys.exit("el código tiene cambios sin commit:\n" + prov["codigo_sin_commit"] +
                     "\nhaga el commit (o use --test para pruebas) antes del benchmark oficial")

    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = (run_dir / "registro.log").open("a")

    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    acquire_lock(args.run_id)
    try:
        manifest_path = run_dir / "manifiesto.json"
        frozen = {"plan": {"tamanos": sizes, "hilos": threads, "k": ks, "repeticiones": reps, "semilla": args.seed,
                           "total": len(plan), "por_modo": counts, "eps": EPS, "min_samples": MIN_SAMPLES},
                  "git_commit": prov["git_commit"], "sha256_codigo": prov["sha256_codigo"],
                  "sha256_binario": prov["sha256_binario"], "sha256_datos": prov["sha256_datos"]}
        conditions = system_conditions()
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            changed = [k for k, v in frozen.items() if json.loads(json.dumps(v)) != manifest.get("congelado", {}).get(k)]
            if changed:
                sys.exit(f"cambió {', '.join(changed)} respecto a la sesión {args.run_id}: no se mezclan tiempos de "
                         "versiones distintas; inicie una nueva sesión con otro --run-id")
            manifest.setdefault("reanudaciones", []).append(conditions)
            log(f"REANUDANDO la sesión {args.run_id}")
        else:
            manifest = {"run_id": args.run_id, "tipo": "oficial" if official else "prueba del ejecutor",
                        "creado_utc": now_utc(), "congelado": frozen, "procedencia": prov, "maquina": machine_info(),
                        "condiciones_inicio": conditions,
                        "metodo": "un proceso por corrida, una corrida a la vez, orden barajado por repetición; "
                                  "tiempo oficial = t_total (omp_get_wtime en C++, sin lectura/escritura de CSV)"}
            log(f"NUEVA sesión {args.run_id} en {run_dir.relative_to(REPO)}")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        for warning in warnings_for(conditions):
            log(f"ADVERTENCIA: {warning}")
        if prov["codigo_sin_commit"]:
            log("ADVERTENCIA: archivos sin commit (se continúa por --permitir-sin-commit; sus SHA-256 quedan en el "
                "manifiesto): " + prov["codigo_sin_commit"].replace("\n", ", "))
        log(f"commit {prov['git_commit'][:12]}, binario sha256 {prov['sha256_binario'][:16]}, plan {len(plan)} corridas "
            f"{counts}, semilla {args.seed}")

        # Referencia serial de cada dataset (no cuenta como repetición) y verificación contra los hashes validados.
        references = {}
        ref_path = run_dir / "referencias.csv"
        if ref_path.exists():
            for row in csv.DictReader(ref_path.open()):
                references[int(row["n"])] = {k: (int(row[k]) if k in ("core1", "core2", "noise") else row[k])
                                             for k in ("label_hash", "count_hash", "core1", "core2", "noise")}
        for n in sizes:
            if n in references:
                continue
            metrics, reason = run_once({"mode": "serial", "n": n, "k": "", "threads": 1, "rep": 0})
            if reason:
                sys.exit(f"falló la referencia serial de N={n}: {reason}")
            if (metrics["label_hash"], metrics["count_hash"]) != VALIDATED[n]:
                sys.exit(f"la referencia serial de N={n} no coincide con los hashes validados: se detiene el benchmark")
            references[n] = {k: metrics[k] for k in ("label_hash", "count_hash", "core1", "core2", "noise")}
            append_row(ref_path, ["n", "label_hash", "count_hash", "core1", "core2", "noise", "t_total", "time_utc"],
                       {"n": n, **references[n], "t_total": metrics["t_total"], "time_utc": now_utc()})
            log(f"referencia serial N={n}: {metrics['label_hash']} / {metrics['count_hash']} (coincide con la validada)")

        done = load_done(measurements, plan_keys, log)
        pending = [r for r in plan if key_of(r) not in done]
        log(f"{len(done)} corridas ya registradas; faltan {len(pending)}")
        started, new_runs = time.time(), 0
        for run in pending:
            if args.max_runs is not None and new_runs >= args.max_runs:
                log(f"se alcanzó --max-runs {args.max_runs}; se detiene (puede reanudarse)")
                break
            # El binario, el código y el dataset deben seguir siendo los mismos durante toda la sesión.
            if sha256(BINARY) != prov["sha256_binario"] or sha256(dataset(run["n"])) != prov["sha256_datos"][str(run["n"])]:
                sys.exit("el binario o un dataset cambió durante la sesión: se detiene (no se mezclan versiones)")
            if any(sha256(REPO / f) != h for f, h in prov["sha256_codigo"].items()):
                sys.exit("el código fuente cambió durante la sesión: se detiene (no se mezclan versiones)")
            for attempt in range(1, MAX_ATTEMPTS + 1):
                load = os.getloadavg()[0]
                metrics, reason = run_once(run)
                if metrics is not None:
                    config_problems, result_problems = check_run(run, metrics, references[run["n"]])
                    if result_problems:
                        append_row(failures, FAIL_FIELDS, {**{k: run[k] for k in ("seq", "rep", "mode", "n", "k", "threads")},
                                   "run_id": args.run_id, "attempt": attempt, "status": "RESULTADO_DISTINTO",
                                   "detail": "; ".join(result_problems), "time_utc": now_utc()})
                        log(f"DISCREPANCIA en {key_of(run)}: {'; '.join(result_problems)}. SE DETIENE EL BENCHMARK.")
                        return 2
                    reason = "; ".join(config_problems) if config_problems else None
                if reason is None:
                    break
                append_row(failures, FAIL_FIELDS, {**{k: run[k] for k in ("seq", "rep", "mode", "n", "k", "threads")},
                           "run_id": args.run_id, "attempt": attempt, "status": "FALLA", "detail": reason,
                           "time_utc": now_utc()})
                log(f"falla en {key_of(run)} (intento {attempt}/{MAX_ATTEMPTS}): {reason}")
            else:
                log(f"{key_of(run)} falló {MAX_ATTEMPTS} veces: se detiene (reanudar tras revisar fallas.csv)")
                return 1
            row = {"run_id": args.run_id, "seq": run["seq"], "rep": run["rep"], "mode": run["mode"], "n": run["n"],
                   "k": run["k"], "threads": run["threads"], "attempt": attempt, "loadavg_1m": round(load, 2),
                   "time_utc": now_utc(), **{k: metrics[k] for k in FIELDS if k in metrics and k not in ("k", "n", "mode")}}
            row = {k: ("" if row.get(k) is None else row.get(k)) for k in FIELDS}
            append_row(measurements, FIELDS, row)
            done[key_of(run)] = row
            new_runs += 1
            elapsed = time.time() - started
            remaining = len(plan) - len(done)
            eta = elapsed / new_runs * remaining / 60
            log(f"[{len(done)}/{len(plan)}] rep {run['rep']:2d} {run['mode']:6s} N={run['n']:6d} k={str(run['k']):>2s} "
                f"T={run['threads']:2d}  t_total={metrics['t_total']:.4f} s  OK  | faltan {remaining} (~{eta:.0f} min)")
            (run_dir / "progreso.json").write_text(json.dumps({"exitosas": len(done), "total": len(plan),
                                                               "faltan": remaining, "ultima": key_of(run),
                                                               "actualizado_utc": now_utc()}) + "\n")
        if len(done) == len(plan):
            log(f"TERMINADO: {len(done)} corridas exitosas y únicas en {measurements.relative_to(REPO)}")
        return 0
    finally:
        release_lock()
        log_file.close()


if __name__ == "__main__":
    sys.exit(main())
