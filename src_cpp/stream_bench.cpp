/**
 * STREAM Triad Benchmark for OpenMP
 * Measures peak sustained memory bandwidth in GB/s.
 */
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

int main(int argc, char* argv[]) {
    int num_threads = (argc >= 2) ? std::atoi(argv[1]) : 1;
    omp_set_num_threads(num_threads);

    const size_t N = 20000000; // 20M elements (~160 MB per array, 480 MB total)
    std::vector<double> a(N, 0.0);
    std::vector<double> b(N, 1.0);
    std::vector<double> c(N, 2.0);
    double scalar = 3.0;

    // Warm-up & First-touch parallel initialization
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 3.0;
    }

    const int REPEATS = 10;
    double min_time = 1e9;

    for (int rep = 0; rep < REPEATS; ++rep) {
        double t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            a[i] = b[i] + scalar * c[i];
        }
        double t1 = omp_get_wtime();
        double dt = t1 - t0;
        if (dt < min_time) min_time = dt;
    }

    // STREAM Triad moves 3 * 8 bytes = 24 bytes per element
    double bytes_moved = 3.0 * sizeof(double) * N;
    double gb_per_sec = (bytes_moved / min_time) / 1e9;

    std::cout << "Threads: " << num_threads 
              << " | Min Time: " << min_time * 1e3 << " ms"
              << " | Bandwidth: " << gb_per_sec << " GB/s\n";

    return 0;
}

