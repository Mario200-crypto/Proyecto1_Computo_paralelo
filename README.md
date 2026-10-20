# Proyecto1_Computo_paralelo — DBSCAN paralelo para detección de ruido (OpenMP)

Proyecto Apertura · Cómputo Paralelo y en la Nube · ITAM, Otoño 2026B · Dr. J. Octavio Gutiérrez García
Equipo: _(nombre 1)_, _(nombre 2)_

> **Estado (2026-10-20):** código final con tres versiones (serial, P1 y P2), validado contra la versión serial y
> contra scikit-learn; benchmark oficial de 2,320 corridas en `results/official/oficial_20261020/`. Decisiones de diseño
> en [`docs/DECISIONS.md`](docs/DECISIONS.md); guía para la presentación en [`docs/guia_defensa.md`](docs/guia_defensa.md).

## Qué es
El detector de ruido de dos pasos visto en clase (DBSCAN "ingenuo"):
- **Paso 1:** un punto con al menos `min_samples` vecinos a distancia ≤ `eps` (contándose a sí mismo) es *core de primer grado*.
- **Paso 2:** un punto que no es core pero está a distancia ≤ `eps` de un core de primer grado es *core de segundo grado*
  (borde); los demás son *ruido*. El Paso 2 solo usa los resultados completos del Paso 1.

Tres versiones en un solo programa, `src/dbscan.cpp`, con el mismo predicado de distancia
(`dx*dx + dy*dy <= eps*eps`, en `double`):

| Modo | Estrategia |
|---|---|
| `--mode serial` | referencia: ciclos simples, sin OpenMP en el cálculo |
| `--mode p1` | "toda la matriz indivisible": `#pragma omp parallel for schedule(static)` sobre los puntos |
| `--mode p2 --k K` | "dividiendo la matriz por k": k regiones rectangulares del plano (k = 2: mitades; k = 4: cuadrantes), cada una con un halo de ancho `eps`; regiones como tareas `schedule(dynamic, 1)` |

La entrada y la salida son CSV compatibles con la libreta del curso `DBSCAN_noise.ipynb`.

## Mapa del repositorio
| Ruta | Contenido |
|---|---|
| `src/dbscan.cpp` | programa C++ (serial, P1, P2) |
| `Makefile` | compilación (macOS: Apple clang + libomp; Linux/MSYS2: `g++ -fopenmp`) |
| `notebooks/` | libreta original del profesor (sin cambios) y copia de trabajo del equipo |
| `data/` | entradas `<N>_data.csv` generadas con la libreta, `SHA256SUMS`, referencias de scikit-learn |
| `tests/cases/` | casos de prueba pequeños con clases esperadas |
| `scripts/validate.py` | validación (casos, errores, estrés, scikit-learn, consistencia entre versiones, libreta) |
| `scripts/run_experiments.py` | ejecutor del benchmark oficial (2,320 corridas) |
| `scripts/analyze_results.py` | tablas y gráficas del benchmark |
| `scripts/pilot.py` | pilotos cortos (mediciones provisionales) |
| `results/official/` | benchmark oficial (mediciones, manifiesto, análisis y gráficas) |
| `results/validacion/`, `results/pilot/`, `results/sensibilidad/` | evidencia de validación, pilotos y prueba complementaria de sensibilidad |
| `experimentos/p2_xp/` | P2 original de Mario (bloques de la matriz de comparaciones), su documentación y evidencia |
| `docs/` | decisiones de diseño, bitácora de IA y guía de defensa |

## Reproducción (macOS con Apple Silicon)
```bash
uv venv .venv --python 3.11 && source .venv/bin/activate && uv pip install -r requirements.txt
make clean && make                       # requiere Homebrew libomp (brew install libomp)
./bin/dbscan --input data/4000_data.csv --mode p2 --k 64 --threads 12 --output results/viz/4000_results.csv
python scripts/validate.py all           # validación completa (~4 min)
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --run-id <sesión>        # benchmark (reanudable con el mismo comando)
python scripts/analyze_results.py --run-dir results/official/<sesión>
```
Los datos se generan con la copia de trabajo de la libreta (celdas W1 + A con cada `n_points`) y se verifican con
`data/SHA256SUMS`. Linux/WSL: `make GNU=1` (`g++ -fopenmp`). Windows: MSYS2 UCRT64 (64 bits).

## Experimentos
N ∈ {20 000, 40 000, 80 000, 120 000, 140 000, 160 000, 180 000, 200 000}; hilos ∈ {1, 6, 12, 24} (12 núcleos
lógicos); k ∈ {2, 4, 8, 16, 32, 64} (solo P2); 10 repeticiones: serial 80 + P1 320 + P2 1,920 = **2,320 corridas**,
un proceso por corrida, orden aleatorizado reproducible (semilla 20261020). Tiempo oficial: `t_total` = construcción +
paso 1 + paso 2 medido con `omp_get_wtime()`, sin lectura ni escritura de CSV. Cada corrida se comprobó contra la
referencia serial (`label_hash` y `count_hash`).

## Resultados (benchmark oficial: 2,320 corridas, 10 repeticiones, 20 de octubre de 2026)
Tablas completas en `results/official/oficial_20261020/analisis/resumen.md` y gráficas en `…/analisis/graficas/`.
Todas las corridas dieron clases y conteos idénticos a serial.
- **Serial:** de 0.087 s (N = 20 000) a 8.68 s (N = 200 000); crece como N².
- **P1:** ≈ 5.8× con 6 hilos (eficiencia ≈ 0.96), 7.9–8.3× con 12 hilos y hasta 8.5× con 24 hilos (12 núcleos:
  8 de rendimiento y 4 de eficiencia, más lentos).
- **P2 espacial:** hasta 48.6× (N = 180 000, k = 64, 12 hilos). Con N = 200 000 y 12 hilos: 3.2× (k=2), 5.3× (4),
  11.8× (8), 13.9× (16), 41.8× (32), 47.8× (64). Se separa en S_total = A × Π: con k = 64, A = 7.8 (8.1× menos pares
  por las regiones) y Π = 6.1 (OpenMP, limitado por la región más costosa).
- **P2 es más lento que P1** con k = 2 y k = 4 y 6 o más hilos (48 de 192 comparaciones): solo hay 2 o 4 tareas.
- El requisito de speedup > 1.5 se cumple con ambas versiones paralelas.
- La sensibilidad de P2 a la distribución de los puntos (de 1.0× con datos concentrados a 122× con datos uniformes)
  se documenta aparte en `results/sensibilidad/`; no forma parte del benchmark oficial.
- Condiciones: Mac con corriente; el proceso del sistema `BTLEServer` ocupó un núcleo durante todas las mediciones.

## Hardware y software
MacBook Pro (Mac16,8), Apple M4 Pro con 12 núcleos (8 de rendimiento + 4 de eficiencia, 12 lógicos), 24 GB,
macOS 26.5.2. Apple clang 21.0.0 + libomp 23.1.3 (OpenMP 5.1), banderas `-std=c++17 -O3 -Wall -Wextra -ffp-contract=off`.
Python 3.11.9 con numpy 2.4.6, scikit-learn 1.9.1, matplotlib 3.11.2 (`requirements.txt`).

## Uso responsable de IA generativa
Se usó Claude Opus 5.5 (en Claude Code) para planeación, implementación, validación y documentación, bajo supervisión
del equipo, que revisó y aprobó cada etapa; cualquier otra herramienta de IA usada se declara en el reporte.
La bitácora detallada está en [`docs/DECISIONS.md`](docs/DECISIONS.md#ai-evidence-log) y la declaración completa en el
reporte. El equipo es responsable de todo el contenido.

## Procedencia y créditos
- **Mario** diseñó e implementó la primera versión paralela P2 (bloques de la matriz de comparaciones, conteos
  parciales y uniones por suma y "O"); se conserva con su documentación y evidencia en `experimentos/p2_xp/`.
- `notebooks/DBSCAN_noise.ipynb`: libreta del Dr. J. Octavio Gutiérrez García, copiada sin cambios de
  <https://github.com/octavio-gutierrez/computoparalelo2026b> (commit `dff4ef2d03db72d19facfc2d535a1fcd2bd6f1f0`,
  blob `992a6b0eeafc89a5a2c35074127af38dc9c1736f`). El repositorio original no declara licencia: este repositorio se
  mantiene **privado** hasta contar con autorización del profesor.
- `notebooks/DBSCAN_noise_equipo.ipynb`: copia de trabajo; su primera celda enumera todos los cambios.
