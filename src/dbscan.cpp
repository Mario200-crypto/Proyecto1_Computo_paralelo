//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// dbscan.cpp — DBSCAN noise detector (the brief's two-step "algoritmo ingenuo")
// Proyecto Apertura 2026B · Cómputo Paralelo y en la Nube · ITAM · Dr. J. Octavio Gutiérrez García
//
// Versiones oficiales (las tres usan within_eps, count_neighbors y has_core_neighbor; resultados idénticos):
//   serial  referencia: el algoritmo ingenuo de dos pasos, sin OpenMP en el cálculo.
//   p1      "considerando toda la matriz indivisible": parallel for sobre los puntos.
//   p2      "dividiendo la matriz por k": k regiones rectangulares del plano con halos de ancho eps (D5).
// La primera P2 del equipo, por bloques de la matriz de comparaciones, la diseñó e implementó Mario; se conserva
// con sus pruebas en experimentos/p2_xp/ y fue la referencia con la que se eligió la versión espacial.
//
// What is computed (docs/DECISIONS.md, D2):
//   Step 1  count[i] = number of points j with distance(i, j) <= eps, i itself and duplicates included.
//           count[i] >= min_samples  =>  i is a first-order core (class 1).
//   Step 2  a point that is not a first-order core but lies within eps of one is a second-order core
//           (class 2, the textbook "border" point); every other point is noise (class 0).
//           Step 2 reads only the Step-1 core flags, so a second-order core never promotes another point.
//   The notebook's binary label is class != 0 (0 = noise, 1 = not noise), as np.where(clusters == -1, 0, 1).
//
// Build:  make            (Makefile: Apple clang + Homebrew libomp on macOS, g++ -fopenmp elsewhere)
// Run:    bin/dbscan --input data/4000_data.csv --mode serial --output results/viz/4000_results.csv
//         bin/dbscan --input data/200000_data.csv --mode p1 --threads 12
//         bin/dbscan --input data/200000_data.csv --mode p2 --k 64 --threads 12
// Help:   bin/dbscan --help
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

const char* const PROGRAM_VERSION = "m5";

// Exit codes (D8).
const int EXIT_OK = 0;
const int EXIT_USAGE = 1;             // bad command line or parameter value
const int EXIT_INPUT = 2;             // input file missing, malformed, empty or outside the supported domain
// (El código 3, "modo no implementado", ya no se usa: serial, p1 y p2 existen.)
const int EXIT_WRITE = 4;             // an output file cannot be written

// Point classes (D2). The notebook's binary label is (class != NOISE).
const unsigned char NOISE = 0;   // not a core, and no first-order core within eps
const unsigned char CORE1 = 1;   // first-order core: count >= min_samples
const unsigned char CORE2 = 2;   // second-order core (border): not CORE1, but within eps of a CORE1 point

// Supported numerical domain (D5). Inside it no intermediate value overflows, eps*eps is a normal double and
// the P2 halo argument holds; the program refuses any input outside it.
const double MAX_ABS_COORDINATE = 1e100;
const double MIN_EPS = 1e-100;
const double MAX_EPS = 1e100;

// Command-line options (D8). Empty strings and zeros mean "not given".
struct Options {
    std::string input;             // --input FILE (required)
    std::string mode;              // --mode serial|p1|p2 (required)
    std::string output;            // --output FILE: "<row>,<0|1>" per input row
    std::string detailed_output;   // --detailed-output FILE: "<row>,<class>,<count>" per input row
    double eps = 0.03;             // --eps (notebook default)
    int min_samples = 10;          // --min-samples (notebook default)
    int threads = 0;               // --threads (p1, p2; serial accepts only 1)
    int k = 0;                     // --k: número de regiones de P2
};

// The input points. x and y are separate contiguous arrays, so the distance loops read memory sequentially.
struct Points {
    std::vector<double> x, y;
    std::vector<std::string> rows;   // trimmed text of each input row; the outputs echo it unchanged
};

// Per-point results, indexed by input row (the merge order of every mode).
// Flags are unsigned char, never std::vector<bool>: vector<bool> packs 8 flags into one byte, so threads that
// write the flags of different points would write the same byte (a data race in the parallel versions).
struct Result {
    std::vector<int> count;               // Step 1: neighbors within eps, the point itself included
    std::vector<unsigned char> is_core;   // Step 1: count >= min_samples; read-only after Step 1
    std::vector<unsigned char> label;     // Step 2: NOISE, CORE1 or CORE2
};

// Wall-clock seconds from omp_get_wtime() (D6). Official time: t_total = build + step1 + step2.
// Reading and writing files are timed separately and are never part of t_total.
struct Timers {
    double read = 0, build = 0, step1 = 0, step2 = 0, write = 0;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 1. Command line
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

void print_usage(FILE* out) {
    std::fprintf(out,
        "uso: dbscan --input ARCHIVO --mode serial|p1|p2 [opciones]\n"
        "\n"
        "  --input ARCHIVO            CSV sin encabezado, un punto 'x,y' por línea (generado con la libreta)\n"
        "  --mode MODO                serial | p1 (toda la matriz) | p2 (k regiones del plano con halos de ancho eps)\n"
        "  --output ARCHIVO           escribe '<x>,<y>,<0|1>' por fila (0 = ruido); lo lee la libreta\n"
        "  --detailed-output ARCHIVO  escribe '<x>,<y>,<clase>,<conteo>' (0 ruido, 1 core 1er grado, 2 core 2o grado)\n"
        "  --eps X                    radio de vecindad, 1e-100 <= X <= 1e100 (por omisión 0.03)\n"
        "  --min-samples M            vecinos para ser core, incluido el propio punto (por omisión 10)\n"
        "  --threads T                hilos de OpenMP para p1/p2 (por omisión, todos los lógicos); serial solo acepta 1\n"
        "  --k K                      número de regiones del plano (solo p2, obligatorio ahí)\n"
        "  --help                     muestra esta ayuda\n"
        "\n"
        "Las coordenadas deben ser finitas con |valor| <= 1e100. Las salidas se escriben solo si se piden,\n"
        "conservan el orden de entrada y no se crean directorios.\n"
        "stdout: una línea 'METRICS {json}'. stderr: un resumen legible.\n"
        "códigos de salida: 0 ok, 1 uso o parámetro, 2 archivo de entrada, 4 escritura de salida\n");
}

// Prints a command-line error and returns false, so callers can write `return usage_error(...)`.
bool usage_error(const std::string& message) {
    std::fprintf(stderr, "dbscan: %s (see dbscan --help)\n", message.c_str());
    return false;
}

// Whole-token number parsing: strtod/strtol must consume every character, so "", "0.5x" and "1e" are rejected.
bool parse_double(const char* text, double* value) {
    char* end = nullptr;
    double v = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(v)) return false;
    *value = v;
    return true;
}

bool parse_int(const char* text, int* value) {
    char* end = nullptr;
    errno = 0;
    long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX) return false;
    *value = (int)v;
    return true;
}

bool contains(const std::vector<std::string>& list, const std::string& item) {
    for (const std::string& s : list) {
        if (s == item) return true;
    }
    return false;
}

// Reads "--option value" pairs into opt and validates them (D8). Returns false after printing an error.
bool parse_args(int argc, char* argv[], Options& opt) {
    const std::vector<std::string> known = {"--input", "--mode", "--output", "--detailed-output",
                                            "--eps", "--min-samples", "--threads", "--k"};
    std::vector<std::string> seen;

    for (int a = 1; a < argc; a += 2) {
        const std::string flag = argv[a];
        if (!contains(known, flag)) return usage_error("unknown option '" + flag + "'");
        if (contains(seen, flag)) return usage_error("option " + flag + " given twice");
        seen.push_back(flag);
        // A value that starts with "--" is the next option, so this one has no value.
        if (a + 1 >= argc || std::strncmp(argv[a + 1], "--", 2) == 0) return usage_error("missing value for " + flag);
        const char* value = argv[a + 1];

        if (flag == "--input") {
            opt.input = value;
        } else if (flag == "--mode") {
            opt.mode = value;
        } else if (flag == "--output") {
            opt.output = value;
        } else if (flag == "--detailed-output") {
            opt.detailed_output = value;
        } else if (flag == "--eps") {
            if (!parse_double(value, &opt.eps))
                return usage_error("--eps: '" + std::string(value) + "' is not a finite number");
        } else if (flag == "--min-samples") {
            if (!parse_int(value, &opt.min_samples))
                return usage_error("--min-samples: '" + std::string(value) + "' is not an integer");
        } else if (flag == "--threads") {
            if (!parse_int(value, &opt.threads))
                return usage_error("--threads: '" + std::string(value) + "' is not an integer");
        } else if (flag == "--k") {
            if (!parse_int(value, &opt.k)) return usage_error("--k: '" + std::string(value) + "' is not an integer");
        }
    }

    if (opt.input.empty()) return usage_error("--input is required");
    if (opt.mode.empty()) return usage_error("--mode is required");
    if (opt.mode != "serial" && opt.mode != "p1" && opt.mode != "p2")
        return usage_error("--mode must be serial, p1 or p2, not '" + opt.mode + "'");
    // Supported numerical domain (D5). scikit-learn itself only requires eps > 0; these bounds keep eps*eps a
    // normal double and every intermediate value finite.
    if (!(opt.eps >= MIN_EPS && opt.eps <= MAX_EPS))
        return usage_error("--eps must satisfy 1e-100 <= eps <= 1e100 (supported domain)");
    if (opt.min_samples < 1) return usage_error("--min-samples must be an integer >= 1");
    if (contains(seen, "--threads") && opt.threads < 1) return usage_error("--threads must be an integer >= 1");
    if (opt.mode == "serial" && contains(seen, "--threads") && opt.threads != 1)
        return usage_error("--mode serial runs on one thread: --threads must be 1 or omitted");
    if (contains(seen, "--k") && opt.mode != "p2") return usage_error("--k is only valid with --mode p2");
    if (opt.mode == "p2" && !contains(seen, "--k")) return usage_error("--mode p2 requires --k");
    if (contains(seen, "--k") && opt.k < 1) return usage_error("--k must be an integer >= 1");
    // Never truncate the input file, and never write both outputs into one file.
    if (opt.output == opt.input || opt.detailed_output == opt.input)
        return usage_error("an output file cannot be the input file");
    if (!opt.output.empty() && opt.output == opt.detailed_output)
        return usage_error("--output and --detailed-output must be different files");
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 2. CSV input and output (D7: the notebook's format)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Removes spaces and tabs at both ends, and the '\r' that Windows line endings leave at the end.
std::string trim(const std::string& s) {
    size_t begin = 0, end = s.size();
    while (begin < end && (s[begin] == ' ' || s[begin] == '\t')) begin++;
    while (end > begin && (s[end - 1] == ' ' || s[end - 1] == '\t' || s[end - 1] == '\r')) end--;
    return s.substr(begin, end - begin);
}

// One coordinate: surrounding spaces are allowed; the rest must be a complete finite number inside the
// supported domain |value| <= 1e100 (D5). On error stores the reason and returns false.
bool parse_coordinate(const std::string& token, double* value, std::string* reason) {
    const std::string text = trim(token);
    if (text.empty()) {
        *reason = "empty value";
        return false;
    }
    char* end = nullptr;
    double v = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0') {
        *reason = "'" + text + "' is not a number";
        return false;
    }
    if (!std::isfinite(v)) {
        *reason = "'" + text + "' is not a finite number";
        return false;
    }
    if (std::fabs(v) > MAX_ABS_COORDINATE) {
        *reason = "'" + text + "' is outside the supported domain |value| <= 1e100";
        return false;
    }
    *value = v;
    return true;
}

// Reads the notebook's CSV: one "x,y" row per line, no header; empty lines are skipped.
// Keeps each row's trimmed text so the outputs echo it byte for byte (output columns 1-2 = input row).
// Returns EXIT_OK, or prints "path:line: reason" and returns EXIT_INPUT.
int read_points(const std::string& path, Points& pts) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "%s: cannot open input file: %s\n", path.c_str(), std::strerror(errno));
        return EXIT_INPUT;
    }
    std::string line;
    long line_number = 0;
    while (std::getline(in, line)) {
        line_number++;
        const std::string row = trim(line);
        if (row.empty()) continue;

        size_t commas = 0;
        for (char c : row) {
            if (c == ',') commas++;
        }
        if (commas != 1) {
            std::fprintf(stderr, "%s:%ld: expected 2 comma-separated values, found %zu\n", path.c_str(), line_number,
                         commas + 1);
            return EXIT_INPUT;
        }
        const size_t comma = row.find(',');
        double x = 0, y = 0;
        std::string reason;
        if (!parse_coordinate(row.substr(0, comma), &x, &reason)) {
            std::fprintf(stderr, "%s:%ld: column 1: %s\n", path.c_str(), line_number, reason.c_str());
            return EXIT_INPUT;
        }
        if (!parse_coordinate(row.substr(comma + 1), &y, &reason)) {
            std::fprintf(stderr, "%s:%ld: column 2: %s\n", path.c_str(), line_number, reason.c_str());
            return EXIT_INPUT;
        }
        // Points are indexed with int (supported domain: N <= 2^31 - 1).
        if (pts.x.size() == (size_t)INT_MAX) {
            std::fprintf(stderr, "%s:%ld: more than 2^31 - 1 points\n", path.c_str(), line_number);
            return EXIT_INPUT;
        }
        pts.x.push_back(x);
        pts.y.push_back(y);
        pts.rows.push_back(row);
    }
    if (in.bad()) {
        std::fprintf(stderr, "%s: read error\n", path.c_str());
        return EXIT_INPUT;
    }
    if (pts.x.empty()) {
        std::fprintf(stderr, "%s: no data rows\n", path.c_str());
        return EXIT_INPUT;
    }
    return EXIT_OK;
}

// Opens an output file before the computation starts, so an unwritable path fails before the long part.
// Binary mode: every line ends in '\n' on every platform. Directories are not created.
FILE* open_output(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) std::fprintf(stderr, "dbscan: cannot write %s: %s\n", path.c_str(), std::strerror(errno));
    return f;
}

// Closes an output file; reports any write error (for example a full disk).
bool close_output(FILE* f, const std::string& path) {
    bool ok = !std::ferror(f);
    if (std::fclose(f) != 0) ok = false;
    if (!ok) std::fprintf(stderr, "dbscan: error while writing %s\n", path.c_str());
    return ok;
}

// Notebook format, read by its second cell with np.loadtxt: "<row text>,<0|1>" in input order, no header.
// 1 = not noise, the notebook's np.where(clusters == -1, 0, 1).
void write_binary(FILE* f, const Points& pts, const Result& res) {
    for (size_t i = 0; i < pts.rows.size(); i++) {
        std::fprintf(f, "%s,%d\n", pts.rows[i].c_str(), res.label[i] != NOISE ? 1 : 0);
    }
}

// Detailed format: "<row text>,<class>,<count>" (class 0 noise, 1 first-order core, 2 second-order core).
void write_detailed(FILE* f, const Points& pts, const Result& res) {
    for (size_t i = 0; i < pts.rows.size(); i++) {
        std::fprintf(f, "%s,%d,%d\n", pts.rows[i].c_str(), (int)res.label[i], res.count[i]);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 3. Distance predicate and kernels — shared by every mode
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

// The only place where "j is a neighbor of i" is decided (D2): distance <= eps  <=>  dx*dx + dy*dy <= eps*eps.
// Squared distances avoid sqrt and give the same decision for exact numbers. In IEEE double each product and
// the sum are rounded separately; -ffp-contract=off forbids fusing them into one FMA, so every compiler and
// every mode makes exactly the same decision. eps2 = eps*eps, computed once by the caller.
inline bool within_eps(double dx, double dy, double eps2) {
    return dx * dx + dy * dy <= eps2;
}

// Step-1 kernel: how many of the n candidates (xs[j], ys[j]) lie within eps of the query point (xq, yq).
// The query point itself is one of the candidates (distance 0), so it counts itself; duplicates count separately.
// No early exit: all n candidates are tested, so every query does the same amount of work.
int count_neighbors(double xq, double yq, const double* xs, const double* ys, int n, double eps2) {
    int count = 0;
    for (int j = 0; j < n; j++) {
        if (within_eps(xs[j] - xq, ys[j] - yq, eps2)) count++;
    }
    return count;
}

// Step-2 kernel: is any first-order core among the candidates within eps of (xq, yq)?
// It reads only the frozen Step-1 flags core[] (never Step-2 labels), so there is no propagation.
// It may stop at the first hit: the answer is an "exists", independent of the scan order.
bool has_core_neighbor(double xq, double yq, const double* xs, const double* ys, const unsigned char* core, int n,
                       double eps2) {
    for (int j = 0; j < n; j++) {
        if (core[j] && within_eps(xs[j] - xq, ys[j] - yq, eps2)) return true;
    }
    return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 4. Serial version (D3: brute force, every pair evaluated in Step 1)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Fills res (already sized to n) and records the Step-1 and Step-2 times.
void dbscan_serial(const Points& pts, double eps, int min_samples, Result& res, Timers& t) {
    const int n = (int)pts.x.size();
    const double* x = pts.x.data();
    const double* y = pts.y.data();
    const double eps2 = eps * eps;

    t.build = 0.0;   // nothing to build: every point is compared with the whole point set

    // Step 1: neighbor counts and first-order core flags.
    const double t0 = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        res.count[i] = count_neighbors(x[i], y[i], x, y, n, eps2);
        res.is_core[i] = res.count[i] >= min_samples;
    }
    const double t1 = omp_get_wtime();

    // Step 2: final classes. is_core is complete and no longer written, so each label depends only on Step 1.
    for (int i = 0; i < n; i++) {
        if (res.is_core[i]) {
            res.label[i] = CORE1;
        } else if (has_core_neighbor(x[i], y[i], x, y, res.is_core.data(), n, eps2)) {
            res.label[i] = CORE2;
        } else {
            res.label[i] = NOISE;
        }
    }
    const double t2 = omp_get_wtime();

    t.step1 = t1 - t0;
    t.step2 = t2 - t1;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 5. P1 con OpenMP: "considerando toda la matriz indivisible" (D4)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Planificación (schedule) de los dos ciclos de P1: static (decisión D4). static reparte de antemano a cada hilo un
// bloque contiguo de unas N/p iteraciones. Se comparó contra dynamic,64 en el Mac: dynamic fue en promedio ~5 % más
// rápido con 12 hilos, pero la diferencia no superó el umbral fijado antes de medir (2 desviaciones estándar) y,
// en ese caso, la regla de D4 elige static por ser la opción más simple. El nombre queda en METRICS ("schedule").
const char* const P1_SCHEDULE_NAME = "static";

// Fija el número de hilos y crea el equipo de hilos ANTES de empezar a medir (D6): la primera región paralela
// del programa es la que crea los hilos y ese costo no es parte del algoritmo. Devuelve cuántos hilos tiene el
// equipo en realidad (debe ser igual a `threads`).
int start_threads(int threads) {
    omp_set_dynamic(0);              // el runtime no puede decidir usar menos hilos de los pedidos
    omp_set_num_threads(threads);
    int actual = 0;
    #pragma omp parallel default(none) shared(actual)
    {
        #pragma omp single
        actual = omp_get_num_threads();   // un solo hilo escribe; la barrera implícita al final de single basta
    }
    return actual;
}

// P1: los mismos dos pasos y las mismas funciones que dbscan_serial; lo único nuevo son los dos #pragma.
// La "matriz" (todos los puntos x, y) no se divide: cada hilo clasifica distintos puntos i, pero todos consultan
// el arreglo completo para contar vecinos. Llena res (ya dimensionado a n) y registra los tiempos de cada paso.
void dbscan_p1(const Points& pts, double eps, int min_samples, Result& res, Timers& t) {
    const int n = (int)pts.x.size();
    const double* x = pts.x.data();
    const double* y = pts.y.data();
    int* count = res.count.data();
    unsigned char* is_core = res.is_core.data();
    unsigned char* label = res.label.data();
    const double eps2 = eps * eps;

    t.build = 0.0;   // igual que serial: no hay estructuras que construir

    // Paso 1: conteo de vecinos y cores de primer grado.
    // shared: n, x, y, eps2 y min_samples solo se leen; en count e is_core cada iteración escribe únicamente su
    //         posición i, y cada i pertenece a un solo hilo, así que dos hilos nunca escriben el mismo elemento.
    // private: el índice i (OpenMP lo hace privado) y las variables locales de count_neighbors (cada llamada
    //          tiene las suyas).
    const double t0 = omp_get_wtime();
    #pragma omp parallel for default(none) shared(n, x, y, count, is_core, eps2, min_samples) schedule(static)
    for (int i = 0; i < n; i++) {
        count[i] = count_neighbors(x[i], y[i], x, y, n, eps2);
        is_core[i] = count[i] >= min_samples;
    }
    // Al final del "parallel for" hay una barrera implícita: ningún hilo sigue hasta que todos terminaron el
    // Paso 1. Por eso is_core está completo antes del Paso 2 y no hace falta un "#pragma omp barrier" extra.
    const double t1 = omp_get_wtime();

    // Paso 2: clase final. is_core solo se lee (nunca se modifica aquí); cada iteración escribe solo label[i].
    #pragma omp parallel for default(none) shared(n, x, y, is_core, label, eps2) schedule(static)
    for (int i = 0; i < n; i++) {
        if (is_core[i]) {
            label[i] = CORE1;
        } else if (has_core_neighbor(x[i], y[i], x, y, is_core, n, eps2)) {
            label[i] = CORE2;
        } else {
            label[i] = NOISE;
        }
    }
    const double t2 = omp_get_wtime();   // también después de una barrera implícita: todas las clases ya están

    t.step1 = t1 - t0;
    t.step2 = t2 - t1;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 6. P2 con OpenMP: "dividiendo la matriz por k" en regiones rectangulares del PLANO con halos de ancho eps (D5)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Idea: en lugar de dividir la matriz de comparaciones por índices (la primera P2 del equipo, de Mario; ver
// experimentos/p2_xp/), se divide el plano en k = filas x columnas
// rectángulos iguales, como en el dibujo del enunciado (k = 2: dos mitades verticales; k = 4: cuadrantes).
// Cada punto pertenece a UNA sola región (su "dueña"). Un vecino de un punto está, a lo más, a distancia eps en x
// y en y; por eso basta con que cada región compare sus puntos contra los suyos MÁS los puntos de una franja
// ("halo") de ancho eps alrededor de su rectángulo. Las regiones lejanas nunca se comparan entre sí: así se
// evalúan menos pares que en serial, y ese ahorro se reporta aparte del efecto de los hilos (pairs_step1).
//
// Todo lo siguiente está dentro del tiempo oficial:
//   construcción (t_build): caja que envuelve a los puntos, región dueña de cada punto, regiones que alcanza cada
//            punto con su halo y, por región, arreglos locales (x, y, índice original): primero sus puntos y
//            después los del halo;
//   paso 1 (t_step1): por región, conteo de vecinos de SUS puntos contra su arreglo local -> count e is_core;
//   (barrera implícita)
//   paso 2 (t_step2): por región, copia de is_core de los puntos locales (así recibe las banderas de los puntos del
//            halo, que calculó otra región) y clase de sus puntos.
// Cada punto lo clasifica solo su región dueña y se escribe en su índice original: el orden de salida no cambia.

// Arreglos locales de una región: primero sus n_owned puntos y después los del halo.
struct Region {
    int n_owned = 0;
    std::vector<double> lx, ly;         // coordenadas (copias contiguas para el ciclo de distancias)
    std::vector<int> lid;               // índice original de cada punto local
    std::vector<unsigned char> lcore;   // Paso 2: copia de is_core de cada punto local
};

// Datos de las regiones para METRICS y el resumen (se calculan fuera del tiempo medido).
struct P2Info {
    int rows = 0, cols = 0;
    long long pairs = 0;         // pares evaluados en el Paso 1: suma de n_owned * L sobre las regiones
    long long halo_points = 0;   // copias de puntos en halos: suma de (L - n_owned)
    int regions_nonempty = 0;    // regiones con al menos un punto propio
    double max_task_share = 0;   // fracción de los pares que tiene la región más costosa (límite del paralelismo)
};

// Filas de la cuadrícula (D5): el mayor divisor de k que sea <= raíz de k; columnas = k / filas.
// k = 2 -> 1x2, 4 -> 2x2, 8 -> 2x4, 16 -> 4x4, 32 -> 4x8, 64 -> 8x8.
int region_rows(int k) {
    int rows = 1;
    for (int r = 1; r * r <= k; r++) {
        if (k % r == 0) rows = r;
    }
    return rows;
}

// Franja (0 .. n-1) a la que pertenece la coordenada v cuando las franjas empiezan en lo y miden w.
// Es la MISMA función para decidir la región dueña y el alcance del halo, y nunca disminuye cuando v crece
// (es monótona): eso garantiza que ningún vecino quede fuera del halo (demostración en D5). Los casos especiales
// evitan conversiones indefinidas a int: ancho 0 (todos los puntos en una línea), v antes del inicio y v en el
// borde máximo o después (se asigna a la última franja).
int strip_index(double v, double lo, double w, int n) {
    if (!(w > 0)) return 0;
    const double q = (v - lo) / w;
    if (!(q >= 0)) return 0;
    if (q >= n) return n - 1;
    return (int)q;
}

void dbscan_p2(const Points& pts, double eps, int min_samples, int k, Result& res, Timers& t, P2Info& info) {
    const int n = (int)pts.x.size();
    const double* x = pts.x.data();
    const double* y = pts.y.data();
    int* count = res.count.data();
    unsigned char* is_core = res.is_core.data();
    unsigned char* label = res.label.data();
    const double eps2 = eps * eps;

    // ---- Construcción (t_build) ----
    const double t0 = omp_get_wtime();
    const int rows = region_rows(k);
    const int cols = k / rows;
    double xmin = x[0], xmax = x[0], ymin = y[0], ymax = y[0];   // caja que envuelve a todos los puntos
    for (int i = 1; i < n; i++) {
        xmin = std::min(xmin, x[i]);
        xmax = std::max(xmax, x[i]);
        ymin = std::min(ymin, y[i]);
        ymax = std::max(ymax, y[i]);
    }
    const double W = (xmax - xmin) / cols;   // ancho de cada región
    const double H = (ymax - ymin) / rows;   // alto de cada región
    // Alcance del halo: eps más un margen numérico s = 1e-9 (M + eps) (D5), con M = la coordenada más grande en
    // valor absoluto. El margen cubre los errores de redondeo de x[j] - x[i], de dx*dx y de x[j] +- m; es miles de
    // veces mayor que esos errores y, con los datos del curso (M ~ 1.3, eps = 0.03), vale ~1.3e-9.
    const double M = std::max(std::max(std::fabs(xmin), std::fabs(xmax)), std::max(std::fabs(ymin), std::fabs(ymax)));
    const double m = eps + 1e-9 * (M + eps);

    // Por punto: región dueña y el rango de columnas y filas de regiones que alcanza con su halo.
    std::vector<int> owner(n), c_lo(n), c_hi(n), r_lo(n), r_hi(n);
    int* own = owner.data();
    int* clo = c_lo.data();
    int* chi = c_hi.data();
    int* rlo = r_lo.data();
    int* rhi = r_hi.data();
    #pragma omp parallel for default(none) shared(n, x, y, xmin, ymin, W, H, rows, cols, m, own, clo, chi, rlo, rhi)
    for (int i = 0; i < n; i++) {
        own[i] = strip_index(y[i], ymin, H, rows) * cols + strip_index(x[i], xmin, W, cols);
        clo[i] = strip_index(x[i] - m, xmin, W, cols);
        chi[i] = strip_index(x[i] + m, xmin, W, cols);
        rlo[i] = strip_index(y[i] - m, ymin, H, rows);
        rhi[i] = strip_index(y[i] + m, ymin, H, rows);
    }

    // Por región: sus puntos (en orden de índice original) y después los puntos de OTRAS regiones cuyo alcance
    // incluye a esta región (el halo). El arreglo de regiones se dimensiona antes de la región paralela y cada
    // iteración r solo modifica la región r (sus propios vectores): no hay condición de carrera.
    std::vector<Region> regions(k);
    Region* reg = regions.data();
    #pragma omp parallel for default(none) shared(k, n, cols, reg, own, clo, chi, rlo, rhi, x, y) schedule(dynamic, 1)
    for (int r = 0; r < k; r++) {
        Region& R = reg[r];
        for (int j = 0; j < n; j++) {
            if (own[j] == r) {
                R.lx.push_back(x[j]);
                R.ly.push_back(y[j]);
                R.lid.push_back(j);
            }
        }
        R.n_owned = (int)R.lid.size();
        if (R.n_owned == 0) continue;   // región vacía: no tiene puntos que clasificar
        const int fila = r / cols;
        const int col = r % cols;
        for (int j = 0; j < n; j++) {
            if (own[j] != r && clo[j] <= col && col <= chi[j] && rlo[j] <= fila && fila <= rhi[j]) {
                R.lx.push_back(x[j]);
                R.ly.push_back(y[j]);
                R.lid.push_back(j);
            }
        }
        R.lcore.assign(R.lid.size(), 0);
    }

    // Orden de las tareas: primero las regiones más costosas (n_owned * L), para que la última en terminar sea
    // pequeña (balance de carga con schedule(dynamic, 1)). Ordenamiento por inserción: k <= 64 elementos.
    std::vector<int> order(k);
    for (int r = 0; r < k; r++) {
        const long long cost = (long long)reg[r].n_owned * (long long)reg[r].lid.size();
        int pos = r;
        while (pos > 0 && (long long)reg[order[pos - 1]].n_owned * (long long)reg[order[pos - 1]].lid.size() < cost) {
            order[pos] = order[pos - 1];
            pos--;
        }
        order[pos] = r;
    }
    const int* ord = order.data();
    const double t1 = omp_get_wtime();

    // ---- Paso 1 por regiones: conteo completo de vecinos de los puntos propios contra el arreglo local ----
    // Cada punto tiene una sola región dueña, así que count[i] e is_core[i] los escribe una sola iteración.
    #pragma omp parallel for default(none) shared(k, ord, reg, count, is_core, min_samples, eps2) schedule(dynamic, 1)
    for (int o = 0; o < k; o++) {
        const Region& R = reg[ord[o]];
        const int L = (int)R.lid.size();
        for (int a = 0; a < R.n_owned; a++) {
            const int i = R.lid[a];
            count[i] = count_neighbors(R.lx[a], R.ly[a], R.lx.data(), R.ly.data(), L, eps2);
            is_core[i] = count[i] >= min_samples;
        }
    }
    // Barrera implícita: todas las regiones terminaron el Paso 1; is_core está completo (también para el halo).
    const double t2 = omp_get_wtime();

    // ---- Paso 2 por regiones: importar las banderas del halo y clasificar los puntos propios ----
    #pragma omp parallel for default(none) shared(k, ord, reg, is_core, label, eps2) schedule(dynamic, 1)
    for (int o = 0; o < k; o++) {
        Region& R = reg[ord[o]];
        const int L = (int)R.lid.size();
        for (int b = 0; b < L; b++) R.lcore[b] = is_core[R.lid[b]];   // solo lee is_core (ya no cambia)
        for (int a = 0; a < R.n_owned; a++) {
            const int i = R.lid[a];
            if (is_core[i]) {
                label[i] = CORE1;
            } else if (has_core_neighbor(R.lx[a], R.ly[a], R.lx.data(), R.ly.data(), R.lcore.data(), L, eps2)) {
                label[i] = CORE2;
            } else {
                label[i] = NOISE;
            }
        }
    }
    const double t3 = omp_get_wtime();

    t.build = t1 - t0;
    t.step1 = t2 - t1;
    t.step2 = t3 - t2;

    // Datos para METRICS (fuera del tiempo medido).
    info.rows = rows;
    info.cols = cols;
    long long max_pairs = 0;
    for (int r = 0; r < k; r++) {
        const long long pairs = (long long)reg[r].n_owned * (long long)reg[r].lid.size();
        info.pairs += pairs;
        if (reg[r].n_owned > 0) {
            info.halo_points += (long long)reg[r].lid.size() - reg[r].n_owned;
            info.regions_nonempty++;
        }
        max_pairs = std::max(max_pairs, pairs);
    }
    info.max_task_share = (double)max_pairs / (double)info.pairs;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 7. Métricas: hashes de resultados y la línea METRICS (D8)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

// FNV-1a, 64 bits: a fingerprint of the results, so modes, thread counts and k can be compared from the
// METRICS line alone (identical results give identical hashes).
const uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
const uint64_t FNV_PRIME = 1099511628211ULL;

uint64_t fnv1a_add(uint64_t hash, unsigned char byte) {
    return (hash ^ byte) * FNV_PRIME;
}

// One byte per point (its class), in input order.
uint64_t label_hash(const std::vector<unsigned char>& label) {
    uint64_t h = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < label.size(); i++) h = fnv1a_add(h, label[i]);
    return h;
}

// Four bytes per point (its count as a 32-bit little-endian integer), so the hash is the same on every platform.
uint64_t count_hash(const std::vector<int>& count) {
    uint64_t h = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < count.size(); i++) {
        const uint32_t c = (uint32_t)count[i];
        for (int b = 0; b < 4; b++) h = fnv1a_add(h, (unsigned char)(c >> (8 * b)));
    }
    return h;
}

// Small JSON writers. Every value is produced here, so no escaping is needed (no quotes or backslashes).
const std::string JSON_NULL = "null";

std::string json_text(const std::string& s) {
    return "\"" + s + "\"";
}

std::string json_int(long long v) {
    return std::to_string(v);
}

std::string json_seconds(double s) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.9f", s);
    return buf;
}

std::string json_hash(uint64_t h) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "\"%016llx\"", (unsigned long long)h);
    return buf;
}

// Shortest decimal text that reads back as exactly the same double (0.03 -> "0.03").
std::string json_double(double v) {
    char buf[64];
    for (int digits = 1; digits <= 17; digits++) {
        std::snprintf(buf, sizeof buf, "%.*g", digits, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    return buf;
}

// Appends "key":value to a JSON object under construction.
void json_add(std::string& json, const char* key, const std::string& value) {
    json += (json.size() > 1) ? ",\"" : "\"";
    json += key;
    json += "\":";
    json += value;
}

// La única línea de stdout, para las máquinas. Los campos que no aplican al modo valen null.
// totals[c] = número de puntos de la clase c. Se llama después de terminar el cálculo (fuera de t_total).
// p2 = datos de las regiones de P2 (nullptr en serial y P1).
void print_metrics(const Options& opt, int n, const Result& res, const int totals[3], const Timers& t,
                   bool wrote_files, int threads_requested, int threads_actual, const P2Info* p2) {
    std::string schedule = JSON_NULL;
    if (opt.mode == "p1") schedule = json_text(P1_SCHEDULE_NAME);
    if (opt.mode == "p2") schedule = json_text("dynamic,1");   // regiones de P2, una tarea por región
    std::string json = "{";
    json_add(json, "version", json_text(PROGRAM_VERSION));
    json_add(json, "openmp", json_int(_OPENMP));
    json_add(json, "mode", json_text(opt.mode));
    json_add(json, "n", json_int(n));
    json_add(json, "eps", json_double(opt.eps));
    json_add(json, "min_samples", json_int(opt.min_samples));
    json_add(json, "threads_requested", json_int(threads_requested));   // serial: 1 (no hay región paralela)
    json_add(json, "threads_actual", json_int(threads_actual));
    json_add(json, "k", p2 ? json_int(opt.k) : JSON_NULL);
    json_add(json, "schedule", schedule);
    json_add(json, "t_read", json_seconds(t.read));
    json_add(json, "t_build", json_seconds(t.build));
    json_add(json, "t_step1", json_seconds(t.step1));
    json_add(json, "t_step2", json_seconds(t.step2));
    json_add(json, "t_total", json_seconds(t.build + t.step1 + t.step2));
    json_add(json, "t_write", wrote_files ? json_seconds(t.write) : JSON_NULL);
    // serial y P1 evalúan todos los pares (i, j); P2 solo los de cada región contra su arreglo local.
    json_add(json, "pairs_step1", json_int(p2 ? p2->pairs : (long long)n * n));
    json_add(json, "halo_points", p2 ? json_int(p2->halo_points) : JSON_NULL);           // copias en halos
    json_add(json, "tiles_nonempty", p2 ? json_int(p2->regions_nonempty) : JSON_NULL);   // regiones con puntos
    json_add(json, "max_task_share", p2 ? json_double(p2->max_task_share) : JSON_NULL);  // región más costosa
    json_add(json, "core1", json_int(totals[CORE1]));
    json_add(json, "core2", json_int(totals[CORE2]));
    json_add(json, "noise", json_int(totals[NOISE]));
    json_add(json, "label_hash", json_hash(label_hash(res.label)));
    json_add(json, "count_hash", json_hash(count_hash(res.count)));
    json += "}";
    std::printf("METRICS %s\n", json.c_str());
}

// Resumen para personas, en stderr; también después del cálculo.
void print_summary(const Options& opt, int n, const int totals[3], const Timers& t, int threads_actual,
                   const P2Info* p2) {
    char grid[64] = "";
    if (p2) std::snprintf(grid, sizeof grid, ", k=%d (%dx%d regiones)", opt.k, p2->rows, p2->cols);
    std::fprintf(stderr,
                 "dbscan %s (%d hilos%s): n=%d eps=%g min_samples=%d -> core1=%d core2=%d ruido=%d | "
                 "lectura %.3f s, construcción %.3f s, paso1 %.3f s, paso2 %.3f s, total %.3f s\n",
                 opt.mode.c_str(), threads_actual, grid, n, opt.eps, opt.min_samples, totals[CORE1], totals[CORE2],
                 totals[NOISE], t.read, t.build, t.step1, t.step2, t.build + t.step1 + t.step2);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 8. main
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    for (int a = 1; a < argc; a++) {
        if (std::strcmp(argv[a], "--help") == 0) {
            print_usage(stdout);
            return EXIT_OK;
        }
    }

    Options opt;
    if (!parse_args(argc, argv, opt)) return EXIT_USAGE;

    Timers t;
    Points pts;
    double start = omp_get_wtime();
    const int status = read_points(opt.input, pts);
    t.read = omp_get_wtime() - start;
    if (status != EXIT_OK) return status;
    const int n = (int)pts.x.size();

    // Open the requested outputs now, so an unwritable path fails before the computation.
    const bool writing = !opt.output.empty() || !opt.detailed_output.empty();
    FILE* binary_file = nullptr;
    FILE* detailed_file = nullptr;
    if (!opt.output.empty()) {
        binary_file = open_output(opt.output);
        if (!binary_file) return EXIT_WRITE;
    }
    if (!opt.detailed_output.empty()) {
        detailed_file = open_output(opt.detailed_output);
        if (!detailed_file) {
            if (binary_file) std::fclose(binary_file);
            return EXIT_WRITE;
        }
    }

    // Hilos: serial usa uno. P1 y P2 usan --threads o, si no se indica, el máximo de OpenMP (los núcleos
    // lógicos). start_threads crea el equipo de hilos aquí, antes de medir.
    int threads_requested = 1;
    int threads_actual = 1;
    if (opt.mode == "p1" || opt.mode == "p2") {
        threads_requested = (opt.threads > 0) ? opt.threads : omp_get_max_threads();
        threads_actual = start_threads(threads_requested);
    }

    // Result arrays are allocated before any clock starts.
    Result res;
    res.count.assign(n, 0);
    res.is_core.assign(n, 0);
    res.label.assign(n, NOISE);

    // Cálculo (aquí están los únicos tiempos oficiales: t_build, t_step1, t_step2). Sin impresiones ni archivos.
    P2Info p2_info;
    if (opt.mode == "serial") {
        dbscan_serial(pts, opt.eps, opt.min_samples, res, t);
    } else if (opt.mode == "p1") {
        dbscan_p1(pts, opt.eps, opt.min_samples, res, t);
    } else {
        dbscan_p2(pts, opt.eps, opt.min_samples, opt.k, res, t, p2_info);
    }
    const P2Info* p2 = (opt.mode == "p2") ? &p2_info : nullptr;

    // Outputs: timed as t_write, never part of t_total.
    bool ok = true;
    start = omp_get_wtime();
    if (binary_file) {
        write_binary(binary_file, pts, res);
        ok = close_output(binary_file, opt.output) && ok;
    }
    if (detailed_file) {
        write_detailed(detailed_file, pts, res);
        ok = close_output(detailed_file, opt.detailed_output) && ok;
    }
    t.write = omp_get_wtime() - start;
    if (!ok) return EXIT_WRITE;

    int totals[3] = {0, 0, 0};
    for (int i = 0; i < n; i++) totals[res.label[i]]++;
    print_metrics(opt, n, res, totals, t, writing, threads_requested, threads_actual, p2);
    print_summary(opt, n, totals, t, threads_actual, p2);
    return EXIT_OK;
}
