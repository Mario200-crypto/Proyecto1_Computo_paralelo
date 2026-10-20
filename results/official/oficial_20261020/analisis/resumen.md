# Resultados del benchmark — sesión `oficial_20261020`

2320 corridas únicas y válidas (232 configuraciones × 10 repeticiones), todas con `label_hash` y `count_hash` idénticos a la referencia serial. Tiempo oficial: `t_total` (construcción + paso 1 + paso 2, `omp_get_wtime()`), sin lectura ni escritura de CSV. Commit `9f558024ea66`, binario SHA-256 `4a9900bc4fe89943…`.

## Serial

| N | media (s) | desv. est. (s) | CV % |
|---|---|---|---|
| 20 000 | 0.0874 | 0.0002 | 0.20 |
| 40 000 | 0.3488 | 0.0029 | 0.83 |
| 80 000 | 1.3899 | 0.0027 | 0.20 |
| 120 000 | 3.1283 | 0.0075 | 0.24 |
| 140 000 | 4.2517 | 0.0020 | 0.05 |
| 160 000 | 5.5539 | 0.0066 | 0.12 |
| 180 000 | 7.0392 | 0.0225 | 0.32 |
| 200 000 | 8.6758 | 0.0161 | 0.19 |

## P1: speedup (eficiencia = speedup / hilos)

| N | 1 hilo | 6 hilos | 12 hilos | 24 hilos |
|---|---|---|---|---|
| 20 000 | 0.99× (0.99) | 5.64× (0.94) | 5.99× (0.50) | 7.48× (0.31) |
| 40 000 | 1.00× (1.00) | 5.77× (0.96) | 7.20× (0.60) | 8.13× (0.34) |
| 80 000 | 1.00× (1.00) | 5.78× (0.96) | 8.25× (0.69) | 8.21× (0.34) |
| 120 000 | 1.00× (1.00) | 5.82× (0.97) | 7.91× (0.66) | 8.17× (0.34) |
| 140 000 | 1.00× (1.00) | 5.77× (0.96) | 7.94× (0.66) | 8.37× (0.35) |
| 160 000 | 1.00× (1.00) | 5.78× (0.96) | 8.20× (0.68) | 8.42× (0.35) |
| 180 000 | 1.00× (1.00) | 5.79× (0.97) | 7.87× (0.66) | 8.53× (0.36) |
| 200 000 | 1.00× (1.00) | 5.78× (0.96) | 8.29× (0.69) | 8.48× (0.35) |

## P2 espacial: speedup total con 12 hilos

| N | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 |
|---|---|---|---|---|---|---|
| 20 000 | 2.64× | 4.78× | 10.31× | 12.73× | 26.39× | 29.92× |
| 40 000 | 3.11× | 5.51× | 12.63× | 13.90× | 37.64× | 38.28× |
| 80 000 | 3.30× | 5.18× | 12.59× | 13.77× | 43.48× | 42.83× |
| 120 000 | 3.26× | 5.24× | 12.88× | 13.96× | 43.53× | 46.36× |
| 140 000 | 3.17× | 5.00× | 12.47× | 13.56× | 41.62× | 47.37× |
| 160 000 | 3.18× | 5.21× | 12.03× | 13.75× | 42.51× | 47.29× |
| 180 000 | 3.18× | 5.20× | 12.04× | 13.77× | 43.73× | 48.64× |
| 200 000 | 3.18× | 5.29× | 11.80× | 13.90× | 41.84× | 47.77× |

## P2 espacial, N = 200 000: S_total = A × Pi (12 hilos)

| k | N²/pares | A | Pi | límite de Pi (1/max_task_share) | S_total |
|---|---|---|---|---|---|
| 2 | 1.67 | 1.67 | 1.90 | 1.92 | 3.18× |
| 4 | 2.63 | 2.62 | 2.02 | 2.04 | 5.29× |
| 8 | 4.06 | 4.04 | 2.92 | 2.97 | 11.80× |
| 16 | 5.15 | 5.10 | 2.72 | 2.74 | 13.90× |
| 32 | 6.47 | 6.35 | 6.59 | 6.88 | 41.84× |
| 64 | 8.13 | 7.83 | 6.10 | 6.79 | 47.77× |

**Cómo leer A y Pi.** A mide la ganancia *algorítmica*: P2 con un solo hilo ya es más rápido que serial porque cada región solo se compara con su halo y no con todo el plano (A ≈ N²/pares). Pi mide la ganancia *de OpenMP* al usar varios hilos con el mismo k; está limitada por la región más costosa (1/max_task_share). S_total = A × Pi, por eso puede superar el número de núcleos: no es paralelismo superlineal, es menos trabajo más paralelismo. La eficiencia de los hilos de P2 se mide con Pi / hilos, no con S_total / hilos.

## Mejor P1 contra mejor P2

| N | mejor P1 | mejor P2 | P2 más rápido que P1 por |
|---|---|---|---|
| 20 000 | 7.48× (24 hilos) | 29.92× (k=64, 12 hilos) | 4.00× |
| 40 000 | 8.13× (24 hilos) | 39.98× (k=64, 24 hilos) | 4.92× |
| 80 000 | 8.25× (12 hilos) | 44.05× (k=64, 24 hilos) | 5.34× |
| 120 000 | 8.17× (24 hilos) | 46.36× (k=64, 12 hilos) | 5.67× |
| 140 000 | 8.37× (24 hilos) | 47.37× (k=64, 12 hilos) | 5.66× |
| 160 000 | 8.42× (24 hilos) | 47.29× (k=64, 12 hilos) | 5.62× |
| 180 000 | 8.53× (24 hilos) | 48.64× (k=64, 12 hilos) | 5.70× |
| 200 000 | 8.48× (24 hilos) | 47.77× (k=64, 12 hilos) | 5.64× |

## Configuraciones donde P2 es más lento que P1 con los mismos hilos: 48 de 192

- N = 20 000, k = 2, 6 hilos: P1 0.0155 s, P2 0.0321 s
- N = 20 000, k = 4, 6 hilos: P1 0.0155 s, P2 0.0183 s
- N = 20 000, k = 2, 12 hilos: P1 0.0146 s, P2 0.0331 s
- N = 20 000, k = 4, 12 hilos: P1 0.0146 s, P2 0.0183 s
- N = 20 000, k = 2, 24 hilos: P1 0.0117 s, P2 0.0322 s
- N = 20 000, k = 4, 24 hilos: P1 0.0117 s, P2 0.0184 s
- N = 40 000, k = 2, 6 hilos: P1 0.0604 s, P2 0.1120 s
- N = 40 000, k = 4, 6 hilos: P1 0.0604 s, P2 0.0634 s
- N = 40 000, k = 2, 12 hilos: P1 0.0485 s, P2 0.1120 s
- N = 40 000, k = 4, 12 hilos: P1 0.0485 s, P2 0.0633 s
- N = 40 000, k = 2, 24 hilos: P1 0.0429 s, P2 0.1129 s
- N = 40 000, k = 4, 24 hilos: P1 0.0429 s, P2 0.0633 s
- N = 80 000, k = 2, 6 hilos: P1 0.2406 s, P2 0.4201 s
- N = 80 000, k = 4, 6 hilos: P1 0.2406 s, P2 0.2679 s
- N = 80 000, k = 2, 12 hilos: P1 0.1685 s, P2 0.4212 s
- N = 80 000, k = 4, 12 hilos: P1 0.1685 s, P2 0.2683 s
- N = 80 000, k = 2, 24 hilos: P1 0.1694 s, P2 0.4212 s
- N = 80 000, k = 4, 24 hilos: P1 0.1694 s, P2 0.2688 s
- N = 120 000, k = 2, 6 hilos: P1 0.5371 s, P2 0.9617 s
- N = 120 000, k = 4, 6 hilos: P1 0.5371 s, P2 0.5982 s
- N = 120 000, k = 2, 12 hilos: P1 0.3955 s, P2 0.9601 s
- N = 120 000, k = 4, 12 hilos: P1 0.3955 s, P2 0.5975 s
- N = 120 000, k = 2, 24 hilos: P1 0.3828 s, P2 0.9608 s
- N = 120 000, k = 4, 24 hilos: P1 0.3828 s, P2 0.5977 s
- N = 140 000, k = 2, 6 hilos: P1 0.7363 s, P2 1.3428 s
- N = 140 000, k = 4, 6 hilos: P1 0.7363 s, P2 0.8489 s
- N = 140 000, k = 2, 12 hilos: P1 0.5353 s, P2 1.3412 s
- N = 140 000, k = 4, 12 hilos: P1 0.5353 s, P2 0.8504 s
- N = 140 000, k = 2, 24 hilos: P1 0.5082 s, P2 1.3404 s
- N = 140 000, k = 4, 24 hilos: P1 0.5082 s, P2 0.8494 s
- N = 160 000, k = 2, 6 hilos: P1 0.9607 s, P2 1.7496 s
- N = 160 000, k = 4, 6 hilos: P1 0.9607 s, P2 1.0618 s
- N = 160 000, k = 2, 12 hilos: P1 0.6777 s, P2 1.7489 s
- N = 160 000, k = 4, 12 hilos: P1 0.6777 s, P2 1.0651 s
- N = 160 000, k = 2, 24 hilos: P1 0.6594 s, P2 1.7497 s
- N = 160 000, k = 4, 24 hilos: P1 0.6594 s, P2 1.0660 s
- N = 180 000, k = 2, 6 hilos: P1 1.2152 s, P2 2.2127 s
- N = 180 000, k = 4, 6 hilos: P1 1.2152 s, P2 1.3511 s
- N = 180 000, k = 2, 12 hilos: P1 0.8943 s, P2 2.2121 s
- N = 180 000, k = 4, 12 hilos: P1 0.8943 s, P2 1.3529 s
- … (8 más en p2_vs_p1.csv)

## Anomalías de tiempo señaladas (no se borró ninguna): 3
Repeticiones con t_total > 1.25 × o < 0.8 × la mediana de su configuración (lista completa en anomalias.csv).

Nota: la sensibilidad de P2 espacial a la distribución de los puntos se documenta aparte, como prueba complementaria (results/sensibilidad/); esos datos no se mezclan con este benchmark.
