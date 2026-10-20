# Prueba complementaria de sensibilidad (NO es parte de las 2,320 corridas oficiales)

Datos sintéticos de `results/sensibilidad/generar_datos.py` (semilla fija), eps = 0.03, min_samples = 10, 12 hilos salvo donde se indica, media de 3 repeticiones en orden barajado; speedup = media serial / media. Mediciones provisionales (Mac con corriente; BTLEServer ocupando un núcleo). Todas las corridas dieron label_hash y count_hash idénticos a serial.

| Distribución | N | Serial | P1 (12) | Mario k=32 / k=64 | **Espacial k=32 / k=64** | Pares evaluados por espacial (N²/pares) k=32 / k=64 | max_task_share k=32 / k=64 | Espacial k=64: A (1 hilo) × Π (12/1) | Espacial vs P1 (k=64) |
|---|---|---|---|---|---|---|---|---|---|
| uniforme | 20k | 0.0833 s | 8.1× | 7.3× / 7.7× | **64.8× / 63.0×** | 21.07× / 36.17× | 0.038 / 0.020 | 11.82 × 5.33 | 7.75× |
| concentrada | 20k | 0.0841 s | 8.3× | 7.2× / 8.1× | **1.0× / 1.0×** | 1.01× / 1.01× | 1.000 / 1.000 | 1.00 × 1.02 | 0.12× |
| desbalanceada | 20k | 0.0861 s | 7.3× | 7.1× / 7.2× | **7.0× / 8.6×** | 3.75× / 4.23× | 0.500 / 0.476 | 3.76 × 2.28 | 1.17× |
| uniforme | 40k | 0.3366 s | 8.0× | 7.3× / 7.6× | **102.8× / 121.9×** | 21.12× / 36.38× | 0.037 / 0.020 | 18.66 × 6.53 | 15.28× |
| concentrada | 40k | 0.3398 s | 7.9× | 7.4× / 7.8× | **1.0× / 1.0×** | 1.01× / 1.01× | 1.000 / 1.000 | 1.01 × 1.00 | 0.13× |
| desbalanceada | 40k | 0.3417 s | 8.4× | 7.6× / 7.9× | **5.8× / 7.0×** | 3.55× / 4.08× | 0.600 / 0.566 | 3.87 × 1.81 | 0.83× |

```
uniforme       20000  clases core1/core2/ruido = 19999/1/0  regiones no vacías k=64: 64  halo/N k=64: 0.765  t_build k=64: 0.91 ms  desv. est. máx: 0.0011 s
concentrada    20000  clases core1/core2/ruido = 19900/0/100  regiones no vacías k=64: 54  halo/N k=64: 0.000  t_build k=64: 0.33 ms  desv. est. máx: 0.0011 s
desbalanceada  20000  clases core1/core2/ruido = 19782/160/58  regiones no vacías k=64: 45  halo/N k=64: 1.024  t_build k=64: 0.69 ms  desv. est. máx: 0.0013 s
uniforme       40000  clases core1/core2/ruido = 40000/0/0  regiones no vacías k=64: 64  halo/N k=64: 0.758  t_build k=64: 1.50 ms  desv. est. máx: 0.0021 s
concentrada    40000  clases core1/core2/ruido = 39800/0/200  regiones no vacías k=64: 59  halo/N k=64: 0.000  t_build k=64: 0.61 ms  desv. est. máx: 0.0028 s
desbalanceada  40000  clases core1/core2/ruido = 39801/134/65  regiones no vacías k=64: 45  halo/N k=64: 0.986  t_build k=64: 1.00 ms  desv. est. máx: 0.0023 s
```
