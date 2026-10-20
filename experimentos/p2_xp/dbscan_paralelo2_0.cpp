/*
 * ============================================================================
 *  DBSCAN paralelo para la detección de ruido (outliers) con OpenMP
 *  VERSIÓN PARALELA 2 (P2): dividiendo la matriz en "k" bloques
 * ============================================================================
 *
 *  Idea general (algoritmo ingenuo de la presentación del proyecto):
 *    Paso 1: Para cada punto contamos cuántos puntos están a distancia <= epsilon.
 *            Si ese número es >= min_samples, el punto es CORE (de primer orden);
 *            si no, es RUIDO (outlier).
 *    Paso 2: Para cada punto RUIDO revisamos si es épsilon-alcanzable desde
 *            algún punto CORE de primer orden. Si es así, lo re-etiquetamos como
 *            CORE (de segundo orden).
 *
 *  Estrategia de paralelización P2:
 *    Imaginamos la matriz de distancias de n x n (fila i = punto i,
 *    columna j = punto j). En vez de tratarla como un todo indivisible (P1),
 *    la dividimos en k bloques (sub-matrices) organizados como una
 *    cuadrícula de  bloques_fila x bloques_col  (bloques_fila * bloques_col = k):
 *
 *        k = 2  ->  2 x 1          k = 4  ->  2 x 2
 *        ┌─────┐                   ┌──┬──┐
 *        ├─────┤                   ├──┼──┤
 *        └─────┘                   └──┴──┘
 *
 *    Cada bloque es una tarea independiente que se reparte entre los hilos con
 *    "#pragma omp parallel for schedule(dynamic, 1)".
 *
 *    ¿Cómo unir los resultados para mantener consistencia?
 *      Cada bloque (bf, bc) sólo ve una PARTE de las columnas, así que el conteo
 *      de vecinos que obtiene para la fila i es PARCIAL. Para no tener
 *      condiciones de carrera, cada bloque escribe su resultado parcial en su
 *      propia "rebanada" del arreglo:  parcial[bc][i]
 *      (dos bloques distintos nunca escriben la misma casilla).
 *      Después, en una fase de UNIÓN, sumamos las rebanadas de todos los
 *      bloques de columnas:  conteo[i] = parcial[0][i] + parcial[1][i] + ...
 *      Para el paso 2 se hace lo mismo, pero en vez de sumar se hace un "O"
 *      lógico (basta con que UN bloque encuentre un core cercano).
 *
 *  Compilar:   g++ dbscan_paralelo2.cpp -o dbscan_paralelo2 -fopenmp -O2
 *  Ejecutar:   ./dbscan_paralelo2 <n_puntos> <k> <n_hilos>
 *     ej.:     ./dbscan_paralelo2 20000 4 8
 *  Entrada:    <n_puntos>_data.csv     (generado con DBSCAN_noise.ipynb)
 *  Salida:     <n_puntos>_results.csv  (x, y, etiqueta) -> se visualiza en la libreta
 *              etiqueta = 1 -> punto core (primer o segundo orden)
 *              etiqueta = 0 -> ruido / outlier
 * ============================================================================
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

// Parámetros de DBSCAN (los mismos que usa la libreta DBSCAN_noise.ipynb)
const float EPSILON = 0.03;    // distancia épsilon
const int MIN_SAMPLES = 10;    // núm. mínimo de puntos a distancia épsilon para ser denso

// ----------------------------------------------------------------------------
// Lee el archivo csv con los puntos (cada renglón: x,y)
// ----------------------------------------------------------------------------
void leer_puntos(const string &nombre, vector<float> &x, vector<float> &y) {
    ifstream archivo(nombre);
    if (!archivo.is_open()) {
        cout << "No se pudo abrir el archivo " << nombre << endl;
        exit(1);
    }
    string linea;
    while (getline(archivo, linea)) {
        stringstream ss(linea);
        string valor;
        getline(ss, valor, ',');  x.push_back(stof(valor));  // coordenada x
        getline(ss, valor, ',');  y.push_back(stof(valor));  // coordenada y
    }
    archivo.close();
}

// ----------------------------------------------------------------------------
// Escribe el csv de resultados: x, y, etiqueta
// ----------------------------------------------------------------------------
void escribir_resultados(const string &nombre, const vector<float> &x,
                         const vector<float> &y, const vector<int> &etiqueta) {
    ofstream archivo(nombre);
    for (size_t i = 0; i < x.size(); i++) {
        archivo << x[i] << "," << y[i] << "," << etiqueta[i] << "\n";
    }
    archivo.close();
}

// ----------------------------------------------------------------------------
// Distancia euclidiana entre los puntos i y j
// ----------------------------------------------------------------------------
inline float distancia(const vector<float> &x, const vector<float> &y, int i, int j) {
    float dx = x[i] - x[j];
    float dy = y[i] - y[j];
    return sqrt(dx * dx + dy * dy);
}

int main(int argc, char **argv) {

    // ---------------------- Lectura de parámetros ---------------------------
    if (argc < 4) {
        cout << "Uso: " << argv[0] << " <n_puntos> <k> <n_hilos>" << endl;
        return 1;
    }
    int n_puntos = atoi(argv[1]);   // número de puntos (para el nombre del archivo)
    int k        = atoi(argv[2]);   // número de bloques en que se divide la matriz
    int n_hilos  = atoi(argv[3]);   // número de hilos de OpenMP

    omp_set_num_threads(n_hilos);

    // ---------------------- Lectura de datos --------------------------------
    vector<float> x, y;
    leer_puntos(to_string(n_puntos) + "_data.csv", x, y);
    int n = x.size();

    // ---------------------- Cuadrícula de bloques ---------------------------
    // Buscamos bloques_fila x bloques_col = k, lo más "cuadrado" posible.
    // bloques_col = la mayor potencia de 2 tal que bloques_col^2 <= k
    // (k = 2 -> 2x1, k = 4 -> 2x2, k = 8 -> 4x2, k = 16 -> 4x4, ...)
    int bloques_col = 1;
    while ((bloques_col * 2) * (bloques_col * 2) <= k) bloques_col *= 2;
    int bloques_fila = k / bloques_col;

    // Tamaño (en puntos) de cada franja de filas y de columnas
    int tam_fila = (n + bloques_fila - 1) / bloques_fila;
    int tam_col  = (n + bloques_col  - 1) / bloques_col;

    // Arreglos de resultados
    vector<int> conteo(n, 0);       // núm. total de vecinos de cada punto
    vector<int> etiqueta(n, 0);     // 1 = core, 0 = ruido
    // Resultados parciales: una "rebanada" de tamaño n por cada bloque de columnas.
    // parcial[bc * n + i] = vecinos de i encontrados en la franja de columnas bc
    vector<int> parcial(bloques_col * n, 0);

    double inicio = omp_get_wtime();   // medimos el tiempo con OpenMP

    // =========================================================================
    // PASO 1: conteo de vecinos a distancia épsilon (por bloques)
    // =========================================================================
    // Cada iteración de este for es UN bloque de la matriz. Se usa
    // schedule(dynamic, 1) para que cada hilo tome un bloque nuevo en cuanto
    // termine el anterior.
    #pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < k; b++) {
        int bf = b / bloques_col;   // índice del bloque en las filas
        int bc = b % bloques_col;   // índice del bloque en las columnas

        // Rango de filas y columnas que le tocan a este bloque
        int fila_ini = bf * tam_fila;
        int fila_fin = min(fila_ini + tam_fila, n);
        int col_ini  = bc * tam_col;
        int col_fin  = min(col_ini + tam_col, n);

        for (int i = fila_ini; i < fila_fin; i++) {
            int vecinos = 0;   // variable privada (declarada dentro del for)
            for (int j = col_ini; j < col_fin; j++) {
                // El punto i cuenta como vecino de sí mismo (distancia 0),
                // igual que en la implementación de DBSCAN de la libreta.
                if (distancia(x, y, i, j) <= EPSILON) vecinos++;
            }
            // Cada bloque escribe en SU rebanada -> no hay condición de carrera
            parcial[bc * n + i] = vecinos;
        }
    }
    // Al terminar el "parallel for" hay una barrera implícita: todos los
    // bloques terminaron antes de pasar a la unión.

    // ---------------- UNIÓN de resultados del paso 1 -------------------------
    // Sumamos los conteos parciales de todos los bloques de columnas y
    // determinamos la categoría de cada punto.
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int total = 0;
        for (int bc = 0; bc < bloques_col; bc++) total += parcial[bc * n + i];
        conteo[i] = total;
        // Punto en zona densa -> core de primer orden; si no -> ruido
        etiqueta[i] = (total >= MIN_SAMPLES) ? 1 : 0;
    }

    // =========================================================================
    // PASO 2: ¿el punto ruido es épsilon-alcanzable desde un core (1er orden)?
    // =========================================================================
    // Reutilizamos el arreglo "parcial" como bandera: 1 si en esa franja de
    // columnas se encontró un core de primer orden a distancia <= épsilon.
    // Importante: sólo se consultan los cores de PRIMER orden (etiqueta del
    // paso 1), que no cambia durante este paso, así el resultado es consistente
    // sin importar el orden en que los hilos procesen los bloques.
    #pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < k; b++) {
        int bf = b / bloques_col;
        int bc = b % bloques_col;

        int fila_ini = bf * tam_fila;
        int fila_fin = min(fila_ini + tam_fila, n);
        int col_ini  = bc * tam_col;
        int col_fin  = min(col_ini + tam_col, n);

        for (int i = fila_ini; i < fila_fin; i++) {
            int alcanzable = 0;
            if (etiqueta[i] == 0) {               // sólo nos interesan los puntos ruido
                for (int j = col_ini; j < col_fin; j++) {
                    if (etiqueta[j] == 1 && distancia(x, y, i, j) <= EPSILON) {
                        alcanzable = 1;
                        break;                    // con un core cercano basta
                    }
                }
            }
            parcial[bc * n + i] = alcanzable;     // rebanada propia del bloque
        }
    }

    // ---------------- UNIÓN de resultados del paso 2 -------------------------
    // "O" lógico entre las rebanadas: si algún bloque encontró un core cercano,
    // el punto ruido se re-etiqueta como core de segundo orden.
    // Se guarda en un arreglo nuevo para no mezclar cores de 1er y 2do orden.
    vector<int> etiqueta_final(n, 0);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int alcanzable = 0;
        for (int bc = 0; bc < bloques_col; bc++) {
            if (parcial[bc * n + i] == 1) alcanzable = 1;
        }
        etiqueta_final[i] = (etiqueta[i] == 1 || alcanzable == 1) ? 1 : 0;
    }

    double fin = omp_get_wtime();

    // ---------------------- Resultados --------------------------------------
    int n_ruido = 0;
    for (int i = 0; i < n; i++) if (etiqueta_final[i] == 0) n_ruido++;

    escribir_resultados(to_string(n_puntos) + "_results.csv", x, y, etiqueta_final);

    // Se imprime: puntos, k, hilos, tiempo (para registrar los experimentos)
    cout << "Puntos: " << n << "  k: " << k << " (" << bloques_fila << "x"
         << bloques_col << ")  Hilos: " << n_hilos
         << "  Ruido: " << n_ruido << "  Tiempo: " << (fin - inicio) << " s" << endl;

    return 0;
}
