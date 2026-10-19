# Proyecto1_Computo_paralelo — DBSCAN paralelo para detección de ruido (OpenMP)

Proyecto Apertura · Cómputo Paralelo y en la Nube · ITAM, Otoño 2026B · Dr. J. Octavio Gutiérrez García
Equipo: _(nombre 1)_, _(nombre 2)_

> **Estado (2026-10-18): andamiaje del repositorio.** El código, los datos y los resultados se agregan por hitos; las decisiones de diseño están en [`docs/DECISIONS.md`](docs/DECISIONS.md) y los requisitos en [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

## Qué es
Una versión serial y dos versiones paralelas con OpenMP del detector de ruido de dos pasos visto en clase (DBSCAN "ingenuo"):
- **Paso 1:** un punto con al menos `min_samples` vecinos a distancia ≤ `eps` (contándose a sí mismo) es *core de primer grado*.
- **Paso 2:** un punto que no es core pero está a distancia ≤ `eps` de un core de primer grado es *core de segundo grado* (borde); los demás son *ruido*.
- **P1:** toda la nube de puntos es indivisible; los hilos se reparten los puntos.
- **P2:** el plano se divide en `k` regiones (k = 2 mitades verticales, k = 4 cuadrantes, …); cada región incluye una franja de ancho `eps` con los vecinos de las regiones contiguas, y los resultados se unen por el índice original de cada punto.

La entrada y la salida son archivos CSV compatibles con la libreta del curso `DBSCAN_noise.ipynb`.

## Mapa del repositorio
| Ruta | Contenido |
|---|---|
| `src/dbscan.cpp` | programa C++ (serial, P1, P2) — *hito M1 en adelante* |
| `Makefile` | compilación en macOS (clang + libomp), Linux/WSL y Windows (MSYS2) — *M1* |
| `notebooks/DBSCAN_noise.ipynb` | libreta original del profesor, sin modificar |
| `notebooks/DBSCAN_noise_equipo.ipynb` | copia de trabajo: genera los datos, valida contra scikit-learn y grafica la salida C++ |
| `scripts/` | validación, ejecución de experimentos y gráficas (Python) — *M1, M2, M4* |
| `tests/cases/` | casos de prueba pequeños con etiquetas esperadas — *M1* |
| `data/` | entradas `<N>_data.csv`, `SHA256SUMS`, referencias de scikit-learn |
| `results/` | datos crudos, agregados y gráficas de los experimentos |
| `docs/` | decisiones de diseño, bitácora de uso de IA y reporte |

## Reproducción (macOS con Apple Silicon)
1. **Entorno de Python**
   ```bash
   uv venv .venv --python 3.11 && source .venv/bin/activate
   uv pip install numpy scikit-learn matplotlib ipykernel nbconvert
   ```
2. **Datos** — abrir `notebooks/DBSCAN_noise_equipo.ipynb` y ejecutar W1 + A con `n_points` = 4000, 20000, 40000, 80000, 120000, 140000, 160000, 180000 y 200000; ejecutar B + T1 solo para 4000, 20000 y 40000. Después: `shasum -a 256 data/*_data.csv > data/SHA256SUMS`.
3. **Compilación** — `make` _(disponible desde el hito M1)_. Requiere Homebrew `libomp` (`brew install libomp`).
4. **Ejecución** — `./bin/dbscan --input data/4000_data.csv --mode serial --output results/viz/4000_results.csv`
5. **Visualización** — en la copia de trabajo: W1, `n_points = 4000`, celda C1.
6. **Validación** — `make test` y `make viz-test` _(desde M1)_.

Linux/WSL: `g++ -fopenmp`. Windows: MSYS2 UCRT64 (`g++ … -fopenmp`), solo 64 bits.

## Experimentos
_(Se completa con el hito M2b: ejecución de los 2,320 experimentos y metadatos de la máquina.)_

## Resultados
_(Se completa después de los experimentos oficiales.)_

## Hardware y software
Máquina oficial: MacBook Pro (Mac16,8), Apple M4 Pro con 12 núcleos (8 de rendimiento + 4 de eficiencia, 12 lógicos), 24 GB, macOS 26.5.2. Compilador: Apple clang 21.0.0 + libomp 23.1.3 (Homebrew). _(Versiones de Python, numpy y scikit-learn: se registran al generar los datos.)_

## Uso responsable de IA generativa
Este proyecto usa asistentes de IA (Claude para planeación y auditoría; Codex para implementación) bajo supervisión humana. La bitácora está en [`docs/DECISIONS.md`](docs/DECISIONS.md#ai-evidence-log) y la declaración completa estará en el reporte.

## Procedencia y créditos
- `notebooks/DBSCAN_noise.ipynb`: libreta del Dr. J. Octavio Gutiérrez García, copiada sin cambios de <https://github.com/octavio-gutierrez/computoparalelo2026b> (commit `dff4ef2d03db72d19facfc2d535a1fcd2bd6f1f0`, blob `992a6b0eeafc89a5a2c35074127af38dc9c1736f`). El repositorio original no declara licencia: este repositorio se mantiene **privado** hasta contar con autorización del profesor.
- `notebooks/DBSCAN_noise_equipo.ipynb`: copia de trabajo; su primera celda enumera todos los cambios.
