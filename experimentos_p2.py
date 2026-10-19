# =============================================================================
#  Experimento para evaluar el rendimiento de la VERSIÓN PARALELA 2 (P2)
#  DBSCAN paralelo para la detección de ruido con OpenMP
# =============================================================================
#  Qué hace este script (lo que pide la diapositiva "Experimento para evaluar
#  Rendimiento"):
#    1. Genera los archivos de PUNTOS ALEATORIOS (<n>_data.csv) igual que la
#       libreta DBSCAN_noise.ipynb, para cada número de puntos.
#    2. Compila dbscan_paralelo2.cpp con -fopenmp.
#    3. Corre TODAS las configuraciones:
#         k      = {2, 4, 8, 16, 32, 64}
#         puntos = {20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000}
#         hilos  = {1, cores_virtuales/2, cores_virtuales, cores_virtuales*2}
#       y cada configuración la repite 10 veces.
#    4. Guarda TODAS las corridas en "resultados_p2_crudos.csv" y los
#       PROMEDIOS de las 10 iteraciones en "resultados_p2_promedios.csv"
#       (este es el "archivo con los datos de los experimentos").
#    5. Genera las gráficas de tiempos y de speedup.
#
#  PENDIENTE: comparar contra la versión SERIAL y la P1. Mientras no estén,
#  el speedup se calcula contra la propia P2 con 1 hilo (ver la sección del
#  final). Cuando tengan el tiempo serial, sólo hay que cambiar esa parte.
#
#  Uso (en la terminal, dentro de la carpeta del proyecto):
#     pip install scikit-learn numpy matplotlib
#     python experimentos_p2.py
#
#  Requiere g++ con soporte para OpenMP (el mismo que usan en clase).
# =============================================================================

import os
import re
import csv
import subprocess
import platform

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# ----------------------------- Parámetros ------------------------------------
VALORES_K = [2, 4, 8, 16, 32, 64]
VALORES_PUNTOS = [20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000]
ITERACIONES = 10

# Número de cores virtuales (procesadores lógicos) de la máquina
CORES_VIRTUALES = os.cpu_count()
VALORES_HILOS = sorted(set([1, max(1, CORES_VIRTUALES // 2),
                            CORES_VIRTUALES, CORES_VIRTUALES * 2]))

FUENTE = "dbscan_paralelo2.cpp"
# En Windows el ejecutable termina en .exe
EJECUTABLE = "dbscan_paralelo2.exe" if platform.system() == "Windows" else "./dbscan_paralelo2"

ARCHIVO_CRUDOS = "resultados_p2_crudos.csv"
ARCHIVO_PROMEDIOS = "resultados_p2_promedios.csv"


# ----------------------------- 1. Datos --------------------------------------
def generar_datos(n_points):
    """Genera <n>_data.csv exactamente como en la libreta DBSCAN_noise.ipynb."""
    nombre = str(n_points) + "_data.csv"
    if os.path.exists(nombre):          # si ya existe no lo volvemos a generar
        return
    points, y_true = make_blobs(n_samples=n_points,
                                centers=4,
                                cluster_std=0.06,
                                random_state=11,
                                center_box=(0, 1.0))
    points = np.round(np.abs(points[:, ::-1]), 3)
    np.savetxt(nombre, points, delimiter=",", fmt="%.3f")
    print("Generado", nombre)


# ----------------------------- 2. Compilar -----------------------------------
def compilar():
    salida = "dbscan_paralelo2.exe" if platform.system() == "Windows" else "dbscan_paralelo2"
    comando = ["g++", FUENTE, "-o", salida, "-fopenmp", "-O2"]
    print("Compilando:", " ".join(comando))
    subprocess.run(comando, check=True)


# ----------------------------- 3. Una corrida --------------------------------
def correr(n_puntos, k, hilos):
    """Ejecuta el programa una vez y regresa el tiempo (segundos) que imprime."""
    resultado = subprocess.run([EJECUTABLE, str(n_puntos), str(k), str(hilos)],
                               capture_output=True, text=True, check=True)
    # El programa imprime "... Tiempo: 0.1234 s"
    tiempo = re.search(r"Tiempo:\s*([0-9.eE+-]+)", resultado.stdout)
    return float(tiempo.group(1))


def main():
    print("Cores virtuales detectados:", CORES_VIRTUALES)
    print("Hilos a probar:", VALORES_HILOS)

    for n in VALORES_PUNTOS:
        generar_datos(n)
    compilar()

    total = len(VALORES_PUNTOS) * len(VALORES_K) * len(VALORES_HILOS)
    contador = 0
    promedios = []   # filas: (puntos, k, hilos, tiempo_promedio)

    # Se escriben los datos crudos conforme se van obteniendo, así si se
    # interrumpe el experimento no se pierde lo que ya se corrió.
    with open(ARCHIVO_CRUDOS, "w", newline="") as f_crudos:
        esc = csv.writer(f_crudos)
        esc.writerow(["puntos", "k", "hilos", "iteracion", "tiempo_s"])

        for n in VALORES_PUNTOS:
            for k in VALORES_K:
                for hilos in VALORES_HILOS:
                    contador += 1
                    tiempos = []
                    for it in range(ITERACIONES):
                        t = correr(n, k, hilos)
                        tiempos.append(t)
                        esc.writerow([n, k, hilos, it + 1, t])
                    f_crudos.flush()
                    prom = sum(tiempos) / len(tiempos)   # promedio de 10 iteraciones
                    promedios.append((n, k, hilos, prom))
                    print(f"[{contador}/{total}] puntos={n} k={k} hilos={hilos} "
                          f"promedio={prom:.4f} s")

    # ------------------------- 4. Promedios + speedup -------------------------
    # PENDIENTE: cuando esté la versión serial, el speedup debe ser
    #     speedup = tiempo_serial(n) / tiempo_P2(n, k, hilos)
    # Por ahora usamos como referencia la P2 con 1 hilo (mismo n y k).
    referencia = {(n, k): t for (n, k, h, t) in promedios if h == 1}

    with open(ARCHIVO_PROMEDIOS, "w", newline="") as f:
        esc = csv.writer(f)
        esc.writerow(["puntos", "k", "hilos", "tiempo_promedio_s", "speedup_vs_1hilo"])
        for (n, k, h, t) in promedios:
            esc.writerow([n, k, h, f"{t:.6f}", f"{referencia[(n, k)] / t:.4f}"])
    print("Guardado", ARCHIVO_CRUDOS, "y", ARCHIVO_PROMEDIOS)

    graficar(promedios, referencia)


# ----------------------------- 5. Gráficas -----------------------------------
def graficar(promedios, referencia):
    # (a) Tiempo vs número de puntos: una gráfica por cada k, una línea por hilos
    fig, ejes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for eje, k in zip(ejes.flat, VALORES_K):
        for h in VALORES_HILOS:
            t = [tp for (n, kk, hh, tp) in promedios if kk == k and hh == h]
            eje.plot(VALORES_PUNTOS, t, marker="o", label=f"{h} hilos")
        eje.set_title(f"k = {k}")
        eje.set_xlabel("Número de puntos")
        eje.set_ylabel("Tiempo promedio (s)")
        eje.grid(True)
        eje.legend()
    fig.suptitle("P2: tiempo de ejecución (promedio de 10 iteraciones)")
    fig.tight_layout()
    fig.savefig("grafica_tiempos_p2.png", dpi=150)

    # (b) Speedup vs número de puntos: una gráfica por cada k
    fig, ejes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for eje, k in zip(ejes.flat, VALORES_K):
        for h in VALORES_HILOS:
            s = [referencia[(n, kk)] / tp for (n, kk, hh, tp) in promedios
                 if kk == k and hh == h]
            eje.plot(VALORES_PUNTOS, s, marker="o", label=f"{h} hilos")
        eje.axhline(1.5, color="gray", linestyle="--", label="speedup 1.5")
        eje.set_title(f"k = {k}")
        eje.set_xlabel("Número de puntos")
        eje.set_ylabel("Speedup")
        eje.grid(True)
        eje.legend()
    fig.suptitle("P2: speedup (referencia temporal: P2 con 1 hilo; PENDIENTE: vs serial)")
    fig.tight_layout()
    fig.savefig("grafica_speedup_p2.png", dpi=150)

    # (c) Speedup vs k con el mayor número de puntos (efecto de las divisiones)
    n_max = VALORES_PUNTOS[-1]
    plt.figure(figsize=(7, 5))
    for h in VALORES_HILOS:
        s = [referencia[(n_max, k)] / tp for (n, k, hh, tp) in promedios
             if n == n_max and hh == h]
        plt.plot(VALORES_K, s, marker="o", label=f"{h} hilos")
    plt.xscale("log", base=2)
    plt.xticks(VALORES_K, VALORES_K)
    plt.xlabel("k (número de divisiones)")
    plt.ylabel("Speedup")
    plt.title(f"P2: speedup vs k ({n_max} puntos)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafica_speedup_vs_k_p2.png", dpi=150)

    print("Gráficas guardadas: grafica_tiempos_p2.png, grafica_speedup_p2.png, "
          "grafica_speedup_vs_k_p2.png")


if __name__ == "__main__":
    main()
