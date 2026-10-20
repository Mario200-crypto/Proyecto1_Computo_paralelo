#!/usr/bin/env python3
"""PRUEBA COMPLEMENTARIA de sensibilidad (no forma parte de las 2,320 corridas oficiales).

Genera datos SINTÉTICOS (no son los de la libreta del profesor) para ver cómo cambia el rendimiento de P2 espacial
cuando las regiones quedan muy desbalanceadas. Semilla fija; mismo formato que la libreta ("%.3f,%.3f", sin
encabezado). Escribe results/sensibilidad/datos/<distribución>_<N>.csv.

Uso: python results/sensibilidad/generar_datos.py
"""

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "datos"


def uniforme(rng, n):
    """Puntos repartidos de manera uniforme en [0, 1.3] x [0, 1.3] (la misma caja que los datos del curso)."""
    return rng.random((n, 2)) * 1.3


def concentrada(rng, n):
    """99.5 % de los puntos en un cuadrado de 0.06 x 0.06 y 0.5 % de puntos lejanos que estiran la caja envolvente
    hasta [0, 10] x [0, 10]: con 8 x 8 regiones de 1.25 x 1.25, casi todo cae en UNA sola región."""
    lejanos = max(1, n // 200)
    denso = 0.40 + rng.random((n - lejanos, 2)) * 0.06
    otros = rng.random((lejanos, 2)) * 10.0
    otros[0], otros[1] = (0.0, 0.0), (10.0, 10.0)   # fija las esquinas de la caja envolvente
    puntos = np.vstack([denso, otros])
    return puntos[rng.permutation(n)]


def desbalanceada(rng, n):
    """Cinco grupos gaussianos: uno con el 80 % de los puntos y cuatro con el 5 % cada uno."""
    centros = np.array([[0.35, 0.40], [1.05, 0.25], [0.90, 1.05], [0.20, 1.10], [0.65, 0.75]])
    pesos = np.array([0.80, 0.05, 0.05, 0.05, 0.05])
    sd = np.array([0.08, 0.04, 0.04, 0.04, 0.04])
    grupo = rng.choice(len(centros), size=n, p=pesos)
    return np.abs(centros[grupo] + rng.normal(size=(n, 2)) * sd[grupo, None])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for nombre, generar in (("uniforme", uniforme), ("concentrada", concentrada), ("desbalanceada", desbalanceada)):
        for n in (20000, 40000):
            rng = np.random.default_rng(20261019 + n)
            puntos = np.round(generar(rng, n), 3)
            np.savetxt(OUT / f"{nombre}_{n}.csv", puntos, delimiter=",", fmt="%.3f")
            print(f"{nombre}_{n}.csv: {n} puntos, caja x [{puntos[:, 0].min():.3f}, {puntos[:, 0].max():.3f}] "
                  f"y [{puntos[:, 1].min():.3f}, {puntos[:, 1].max():.3f}]")


if __name__ == "__main__":
    main()
