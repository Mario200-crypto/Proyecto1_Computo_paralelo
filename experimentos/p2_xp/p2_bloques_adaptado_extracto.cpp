// EXTRACTO (no compila por sí solo) de src/dbscan.cpp, versión m3 (19-oct-2026): el P2 por bloques de Mario
// adaptado a double, within_eps y tres clases, tal como se validó y midió antes de elegir P2 espacial.
// Usa Points, Result, Timers, count_neighbors y has_core_neighbor del programa principal.

// P2: datos de la cuadrícula de bloques, para METRICS y el resumen (se calculan fuera del tiempo medido).
struct P2Info {
    int bloques_fila = 0, bloques_col = 0;   // cuadrícula bloques_fila x bloques_col
    int blocks_nonempty = 0;                 // bloques (tareas) que tienen al menos un par (i, j) que comparar
    double max_task_share = 0;               // fracción de los N*N pares que le toca al bloque más grande
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 6. P2 por bloques con OpenMP (estrategia de Mario): "dividiendo la matriz por k"   (--mode p2)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Se imagina la matriz de comparaciones de n x n (fila i = punto i, columna j = punto j; nunca se guarda en
// memoria) y se divide en k bloques, una cuadrícula de bloques_fila x bloques_col bloques por ÍNDICE de punto:
//
//     k = 2 -> 2 x 1    k = 4 -> 2 x 2    k = 8 -> 4 x 2    k = 16 -> 4 x 4    k = 32 -> 8 x 4    k = 64 -> 8 x 8
//
// Cada bloque es una tarea: compara las filas de su franja de filas contra las columnas de su franja de
// columnas. Como un bloque solo ve una parte de las columnas, su conteo de vecinos es PARCIAL; cada bloque lo
// escribe en su propia "rebanada" parcial[bc * n + i] (bc = franja de columnas) y luego una fase de UNIÓN suma
// las rebanadas. En el Paso 2 se hace lo mismo con un "O" lógico (basta con que un bloque encuentre un core).
//
// Nota de interpretación: esta división es por índices de la matriz de comparaciones, no una partición espacial
// del plano (el dibujo del enunciado muestra regiones del plano). Cada bloque evalúa exactamente sus pares, así
// que en total P2 evalúa los mismos N*N pares que serial y P1; la diferencia está en cómo se reparte el trabajo.
//
// Adaptaciones respecto al código original de Mario (el esquema de bloques, parciales y uniones no cambió):
//   - double en lugar de float y el predicado común within_eps (dx*dx + dy*dy <= eps*eps) en lugar de sqrt, a
//     través de count_neighbors y has_core_neighbor aplicados a la franja de columnas del bloque;
//   - tres clases (NOISE, CORE1, CORE2) en lugar de 0/1; la salida binaria de la libreta sigue siendo clase != 0;
//   - el tiempo oficial incluye la construcción del arreglo parcial (t_build) y las dos uniones (t_step1, t_step2);
//   - default(none) con la lista explícita de variables compartidas, como en P1.

// Llena res (ya dimensionado a n), registra los tiempos y devuelve en info los datos de la cuadrícula.
void dbscan_p2(const Points& pts, double eps, int min_samples, int k, Result& res, Timers& t, P2Info& info) {
    const int n = (int)pts.x.size();
    const double* x = pts.x.data();
    const double* y = pts.y.data();
    int* count = res.count.data();
    unsigned char* is_core = res.is_core.data();
    unsigned char* label = res.label.data();
    const double eps2 = eps * eps;

    // ---- Construcción (t_build): cuadrícula de bloques y arreglo de resultados parciales ----
    const double t0 = omp_get_wtime();
    // bloques_col = la mayor potencia de 2 tal que bloques_col^2 <= k; bloques_fila = k / bloques_col.
    int bloques_col = 1;
    while ((bloques_col * 2) * (bloques_col * 2) <= k) bloques_col *= 2;
    const int bloques_fila = k / bloques_col;
    // Tamaño (en puntos) de cada franja de filas y de columnas.
    const int tam_fila = (n + bloques_fila - 1) / bloques_fila;
    const int tam_col = (n + bloques_col - 1) / bloques_col;
    // parcial[bc * n + i] = resultado de la fila i dentro de la franja de columnas bc (una rebanada de n enteros
    // por franja de columnas). Los índices se calculan en size_t para que bc * n no se desborde.
    std::vector<int> parcial((size_t)bloques_col * n, 0);
    int* parc = parcial.data();
    const double t1 = omp_get_wtime();

    // ---- Paso 1 por bloques: conteos parciales ----
    // Cada iteración b es UN bloque; schedule(dynamic, 1): cada hilo toma un bloque nuevo al terminar el anterior.
    // Cada bloque (bf, bc) escribe solo las filas de su franja bf dentro de SU rebanada bc: dos bloques nunca
    // escriben la misma casilla (no hay condición de carrera). x, y, n, eps2 y los tamaños solo se leen.
    #pragma omp parallel for default(none) shared(k, n, x, y, parc, eps2, bloques_col, tam_fila, tam_col) \
        schedule(dynamic, 1)
    for (int b = 0; b < k; b++) {
        const int bf = b / bloques_col;   // índice del bloque en las filas
        const int bc = b % bloques_col;   // índice del bloque en las columnas
        const int fila_ini = bf * tam_fila;
        const int fila_fin = std::min(fila_ini + tam_fila, n);
        const int col_ini = bc * tam_col;
        const int col_fin = std::min(col_ini + tam_col, n);
        if (col_ini >= col_fin) continue;   // franja de columnas vacía (solo si n < bloques_col): su rebanada queda en 0
        for (int i = fila_ini; i < fila_fin; i++) {
            // Vecinos de i entre las columnas col_ini..col_fin-1 (el propio i cuenta si está en esta franja).
            parc[(size_t)bc * n + i] = count_neighbors(x[i], y[i], x + col_ini, y + col_ini, col_fin - col_ini, eps2);
        }
    }
    // Barrera implícita: todos los bloques terminaron antes de la unión.
    const double t_union1 = omp_get_wtime();

    // ---- Unión del Paso 1 (paralela, por puntos): suma de las rebanadas -> conteo y core de primer grado ----
    #pragma omp parallel for default(none) shared(n, parc, bloques_col, count, is_core, min_samples)
    for (int i = 0; i < n; i++) {
        int total = 0;
        for (int bc = 0; bc < bloques_col; bc++) total += parc[(size_t)bc * n + i];
        count[i] = total;
        is_core[i] = total >= min_samples;
    }
    // Barrera implícita: is_core está completo antes del Paso 2.
    const double t2 = omp_get_wtime();

    // ---- Paso 2 por bloques: ¿algún core de primer grado a distancia eps en esta franja de columnas? ----
    // Se reutiliza parcial como bandera (1 = se encontró un core). Solo se consulta is_core (Paso 1), que ya no
    // cambia, así que el resultado no depende del orden en que los hilos procesen los bloques.
    #pragma omp parallel for default(none) shared(k, n, x, y, parc, is_core, eps2, bloques_col, tam_fila, tam_col) \
        schedule(dynamic, 1)
    for (int b = 0; b < k; b++) {
        const int bf = b / bloques_col;
        const int bc = b % bloques_col;
        const int fila_ini = bf * tam_fila;
        const int fila_fin = std::min(fila_ini + tam_fila, n);
        const int col_ini = bc * tam_col;
        const int col_fin = std::min(col_ini + tam_col, n);
        if (col_ini >= col_fin) continue;   // franja vacía: su rebanada sigue en 0 desde el Paso 1
        for (int i = fila_ini; i < fila_fin; i++) {
            int alcanzable = 0;
            if (!is_core[i]) {   // solo interesan los puntos que no son core de primer grado
                alcanzable = has_core_neighbor(x[i], y[i], x + col_ini, y + col_ini, is_core + col_ini,
                                               col_fin - col_ini, eps2);
            }
            parc[(size_t)bc * n + i] = alcanzable;   // rebanada propia del bloque
        }
    }
    const double t_union2 = omp_get_wtime();

    // ---- Unión del Paso 2 (paralela, por puntos): "O" lógico entre las rebanadas -> clase final ----
    #pragma omp parallel for default(none) shared(n, parc, bloques_col, is_core, label)
    for (int i = 0; i < n; i++) {
        int alcanzable = 0;
        for (int bc = 0; bc < bloques_col; bc++) {
            if (parc[(size_t)bc * n + i] == 1) alcanzable = 1;
        }
        if (is_core[i]) {
            label[i] = CORE1;
        } else if (alcanzable) {
            label[i] = CORE2;
        } else {
            label[i] = NOISE;
        }
    }
    const double t3 = omp_get_wtime();

    t.build = t1 - t0;
    t.step1 = t2 - t1;   // bloques del Paso 1 + su unión
    t.step2 = t3 - t2;   // bloques del Paso 2 + su unión
    t.merge = (t2 - t_union1) + (t3 - t_union2);

    // Datos de la cuadrícula para METRICS (fuera del tiempo medido).
    info.bloques_fila = bloques_fila;
    info.bloques_col = bloques_col;
    long long max_pairs = 0;
    for (int b = 0; b < k; b++) {
        const int fila_ini = (b / bloques_col) * tam_fila;
        const int col_ini = (b % bloques_col) * tam_col;
        const long long filas = std::max(0, std::min(fila_ini + tam_fila, n) - fila_ini);
        const long long cols = std::max(0, std::min(col_ini + tam_col, n) - col_ini);
        if (filas * cols > 0) info.blocks_nonempty++;
        max_pairs = std::max(max_pairs, filas * cols);
    }
    info.max_task_share = (double)max_pairs / ((double)n * n);
}

