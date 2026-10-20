# P2 por bloques de la matriz de comparaciones (versión original de Mario)

**Reconocimiento.** Mario diseñó e implementó la primera versión paralela P2 del proyecto: la división de la matriz
de comparaciones en k bloques, los conteos parciales por bloque y las uniones por suma y por "O" lógico. Fue la
primera P2 funcional del equipo y la base contra la que se evaluó la alternativa espacial que finalmente se entregó.
Su código y su evidencia se conservan aquí, separados del programa oficial (`src/dbscan.cpp`) y del benchmark
oficial (`results/official/`).

## Archivos
| Archivo | Contenido |
|---|---|
| `dbscan_paralelo2_0.cpp` | Copia **byte por byte** de `dbscan_paralelo2.cpp` de la rama `origin/mario` (commit `0d680a5`, blob `a5ce8a74aa44`). |
| `p2_bloques_adaptado_extracto.cpp` | Extracto (no compila por sí solo) de la versión integrada al programa del equipo el 19-oct-2026 (versión `m3`): misma estrategia, adaptada a `double`, al predicado común `within_eps` y a las tres clases. |
| `evidencia/p2_bloques_validate_all.txt` | Validación completa con la versión integrada (casos, consistencia contra serial, libreta). |
| `evidencia/p2_bloques_piloto.jsonl` | 147 corridas del piloto (versión `m2a`; ahí `p2` = bloques de Mario). |
| `evidencia/comparacion_bloques_vs_espacial_*.jsonl` | Comparaciones bloques contra espacial (versión `m3`; `p2` = bloques, `p2s`/`p2spatial` = espacial). |

En los archivos de evidencia, el campo `metrics.version` dice qué significaba cada modo: en `m2a` y `m3`, `--mode p2`
era el P2 por bloques; desde la versión final (`m5`), `--mode p2` es el P2 espacial.

## Estrategia original
1. Se imagina la matriz de comparaciones de n × n (fila i = punto i, columna j = punto j); nunca se guarda en memoria.
2. Se divide en k bloques, una cuadrícula `bloques_fila × bloques_col` por **índice** de punto:
   `bloques_col` = la mayor potencia de 2 con `bloques_col² ≤ k` (k = 2 → 2×1, 4 → 2×2, 8 → 4×2, 16 → 4×4,
   32 → 8×4, 64 → 8×8).
3. Cada bloque es una tarea (`#pragma omp parallel for schedule(dynamic, 1)`): compara las filas de su franja contra
   las columnas de su franja.
4. **Paso 1:** como un bloque solo ve parte de las columnas, su conteo es parcial; lo escribe en su propia
   "rebanada" `parcial[bc * n + i]`, así que dos bloques nunca escriben la misma casilla.
5. **Unión por suma** (paralela por puntos): `conteo[i] = Σ parcial[bc][i]`; core de primer grado si
   `conteo[i] ≥ min_samples`.
6. **Paso 2:** cada bloque busca, para sus filas que no son core, algún core de primer grado en su franja de
   columnas y escribe 1/0 en su rebanada.
7. **Unión por "O" lógico:** basta con que un bloque haya encontrado un core cercano para que el punto sea core de
   segundo grado.

Las barreras implícitas al final de cada `parallel for` separan bloques, uniones y pasos.

## Pruebas y mediciones (versión integrada)
- **Corrección:**
  - 1,260 casos de prueba (14 casos × 2 órdenes × 10 valores de k × hilos 1/6/12/24);
  - 108 corridas de consistencia en los 9 datasets (6 k × 6 y 12 hilos);
  - 147 corridas de piloto.
  - Todas dieron `label_hash` y `count_hash` idénticos a serial.
  - Dos versiones alteradas a propósito (unión sin la última rebanada; "Y" en lugar de "O") fueron detectadas.
- **Rendimiento a 200,000 puntos, 12 hilos** (mediciones provisionales; serial ≈ 8.67 s):
  - speedup 1.97× (k=2), 3.89× (k=4), 7.72× (k=8), 6.78× (k=16), 8.48× (k=32), 8.31× (k=64);
  - las uniones costaron < 1 % del tiempo.
- **Límite de paralelismo:** cada fase tiene k tareas; con núcleos iguales el speedup máximo es k / ⌈k/p⌉. Por eso
  k = 2 no pasa de 2× y k = 16 con 12 hilos es más lento que k = 8 (dos rondas de tareas).

## Por qué se seleccionó P2 espacial
1. **Trabajo:** los bloques reparten los mismos N² pares que serial; la versión espacial evita comparar regiones
   lejanas (1.7× a 8.1× menos pares en los datos del curso).
2. **Tiempo:** en la comparación con el mismo binario y 5 repeticiones a 200k, el espacial fue 5.2× (k=32) y 6.0×
   (k=64) más rápido que los bloques, sin traslape entre repeticiones.
3. **Enunciado:** el dibujo del enunciado muestra regiones del plano (mitades y cuadrantes); la versión espacial
   corresponde directamente a esa figura.
4. **Corrección:** ambas versiones reproducen exactamente a serial.

**Lo que los bloques hacen mejor.** No dependen de la distribución de los puntos: siempre reparten el mismo trabajo.
En la prueba de sensibilidad (`results/sensibilidad/`) con datos casi todos concentrados en una región, el P2
espacial no aceleró (1.0×), mientras que los bloques se mantuvieron en ≈ 8×.
