# Reporte final — DBSCAN para detección de ruido con OpenMP

Reporte académico del Proyecto Apertura (Cómputo Paralelo y en la Nube, ITAM, Otoño 2026).

| Archivo | Contenido |
|---|---|
| `reporte.pdf` | Versión compilada (9 páginas, carta). |
| `reporte.tex` | Fuente LaTeX completa (un solo archivo; la Figura 1 está dibujada con TikZ). |
| `figuras/` | Las cinco figuras que usa el reporte (PDF vectorial). |
| `generar_figuras.py` | Genera las figuras y recalcula todas las cifras del reporte desde los datos oficiales. |
| `cifras_verificadas.txt` | Salida de `generar_figuras.py`: las cifras que cita el reporte. |

## Datos de origen (no se modifican ni se copian aquí)
- `results/official/oficial_20261020/mediciones.csv`: las 2,320 ejecuciones oficiales; es la única fuente de tiempos.
- `results/official/oficial_20261020/referencias.csv`: huellas serial usadas para verificar cada ejecución.
- `results/official/oficial_20261020/manifiesto.json`: commit, huellas SHA-256, máquina, plan y semilla.
- `data/200000_data.csv`: solo para reconstruir el costo de cada región de P2. El script comprueba que la suma
  coincide con `pairs_step1` del CSV.
- Prueba de sensibilidad (Tabla 4): `results/sensibilidad/tabla.md`. No forma parte del benchmark oficial.

Todas las aceleraciones son razones de medias de diez repeticiones, no medias de cocientes.

## Regenerar figuras y cifras
Desde la raíz del repositorio, con el entorno del proyecto (`requirements.txt`: numpy y matplotlib):
```bash
.venv/bin/python entrega/reporte/generar_figuras.py
```
Esto reescribe `figuras/*.pdf` y `cifras_verificadas.txt`, y además imprime las cifras.

## Compilar el PDF
Se compiló con **Tectonic 0.15.0** (motor XeTeX), que descarga solo los paquetes que necesita:
```bash
cd entrega/reporte
tectonic -X compile reporte.tex        # brew install tectonic
```
También debería compilar con `latexmk -xelatex reporte.tex` o con `lualatex` (MacTeX o TeX Live). La rama de
`pdflatex` que incluye el preámbulo no se probó.

Paquetes: babel (español), amsmath, amssymb, graphicx, booktabs, tabularx, xcolor, tikz, caption, enumitem,
listings, titlesec, microtype y hyperref.

## Antes de entregar
- Completar los nombres de los integrantes en la portada: los marcadores rojos `\pendiente{...}`.
- Confirmar o completar las dos notas en rojo de la sección 7, sobre el uso de IA.
- Después de editar, buscar `\pendiente` en `reporte.tex` y recompilar.
