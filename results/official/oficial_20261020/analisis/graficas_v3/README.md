# Gráficas del reporte (versión 3)

Fuente única: `mediciones.csv` (2,320 corridas; SHA-256 `d6b8231c7d052137…`, sin cambios). Script: `scripts/graficas_v3.py`.
**Convenciones.** Cada punto es la media de las 10 repeticiones de `t_total` (construcción + paso 1 + paso 2, sin leer ni escribir CSV); se omiten barras de error porque la desviación estándar es a lo más 7.4 % de la media en las configuraciones graficadas, muy por debajo de las diferencias que muestran las figuras. Todo speedup es una **razón de medias** (media de referencia / media de la versión).
**Configuraciones fijas:** Serial (1 hilo), P1 (12 hilos) y P2 espacial (12 hilos, k = 64). Se usan 12 hilos porque el equipo tiene 12 núcleos físicos, aunque heterogéneos (8 de rendimiento y 4 de eficiencia), y k = 64 porque fue la configuración más rápida de P2 con 12 hilos en 7 de los 8 tamaños oficiales.

**Fig. 1 — Tiempo contra N** (`fig1_tiempo_vs_N.png`). *Pie:* Tiempo medio de ejecución contra N; (a) las tres versiones, (b) acercamiento a las versiones paralelas. Media de 10 repeticiones.
*Interpretación:* con N = 200 000, Serial tarda 8.676 s, P1 1.047 s y P2 0.182 s. De 20 000 a 200 000 puntos (10 veces más), Serial tarda 99× más (≈ 10² = 100, crece como N² porque compara todos los pares); P1 y P2 crecen algo menos (72× y 62×) porque con N pequeño pesan más las partes que crecen más despacio que N² (sincronizar los hilos y, en P2, construir las regiones).

**Fig. 2 — Speedup contra N** (`fig2_speedup_vs_N.png`). *Pie:* Speedup de P1 y P2 respecto de Serial con el mismo N (razón de tiempos medios).
*Interpretación:* P1 pasa de 6.0× (20 000 puntos) a 8.3× (200 000) y se estabiliza entre 7.9× y 8.3× desde 80 000; P2 pasa de 29.9× a 47.8× (máximo 48.6× con 180 000). Ambas superan el mínimo de 1.5× en todos los tamaños.

**Fig. 3 — Aceleración al paralelizar** (`fig3_aceleracion_paralela_vs_hilos.png`). *Pie:* Con N = 200 000, tiempo de cada versión con 1 hilo entre su tiempo con T hilos (P2 con k = 64). Mide solo el efecto de OpenMP.
*Interpretación:* con 12 hilos P1 acelera 8.3× y P2 6.1×; con 24 hilos, 8.5× y 5.9×, sin ganancia real porque no hay más núcleos. P1 con 1 hilo tarda casi lo mismo que Serial (8.670 s contra 8.676 s), por eso su valor con 12 hilos coincide con el de la Fig. 2 en N = 200 000. P2 se paraleliza peor que P1 (su región más cargada limita el reparto); su ventaja de la Fig. 2 viene de que, ya con 1 hilo, es 7.8× más rápido que Serial al comparar menos pares: 7.8 × 6.1 ≈ 47.8×.

**Fig. 4 — Efecto de k** (`fig4_p2_tiempo_vs_k.png`). *Pie:* Tiempo medio de P2 para cada k con N = 200 000 y 12 hilos; línea punteada: P1 con los mismos 12 hilos.
*Interpretación:* con pocas regiones P2 es más lento que P1 (k = 2: 2.73 s, k = 4: 1.64 s, contra 1.05 s): hay solo 2 o 4 tareas para 12 hilos. Desde k = 8 es más rápido, y con k = 64 tarda 0.182 s: más regiones significan menos pares por comparar y más tareas para repartir.

**Fig. 5 (opcional) — Barras con N = 200 000** (`fig5_barras_200k.png`). Repite el extremo derecho de la Fig. 1; se recomienda sustituirla por esta tabla:

| Versión | Tiempo medio (s) | Speedup |
|---|---|---|
| Serial | 8.676 | 1.0× |
| P1, 12 hilos | 1.047 | 8.3× |
| P2, 12 hilos, k = 64 | 0.182 | 47.8× |

**Recomendación:** Fig. 1, 2, 3 y 4 en el reporte, y la tabla en lugar de la Fig. 5.
