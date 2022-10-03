#include "neurad.h"

using nrd::NRMatrix;
using nrd::NRMatrixHeap;
using nrd::NRSize;

void matMulBench(NRMatrixHeap *ph) {
    vector<NRSize> matDims = {2, 3, 4, 5, 6, 8, 10, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000, 12000, 14000, 20000};
    vector<int> matReps = {5000, 1000, 1000, 1000, 1000, 1000, 500, 500, 100, 50, 50, 50, 50, 20, 10, 10, 5, 5, 5, 5};
#ifdef USE_SINGLE_PRECISION_FLOAT
    cout << "Benchmark, single precision" << endl
         << endl;
#else
    cout << "Benchmark, double precision" << endl
         << endl;
#endif
    cout << "                                            Min              Mean             Max" << endl;
    for (int j = 0; j < matDims.size(); j++) {
        auto dim = matDims[j];
        auto reps = matReps[j];
        string name = "m" + std::to_string(dim);
        long min_time, max_time;
        NRMatrix m = NRMatrix(ph, dim, dim, name);
        unsigned long mean = 0;
        int act = 0;
        for (int i = 0; i < reps; i++) {
            int perc = (i * 100) / reps;
            cout << perc << "%"
                 << "   "
                 << "\r" << std::flush;
            m.pm->randNormal(0, 1);
            // Start BENCHMARK
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            NRMatrix n = m * m;
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            // End BENCHMARK
            auto td = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
            if ((j == 0 && i > 400) || (j > 0 && i > reps / 10 + 3)) {  // skip warmup phases
                if (act == 0 || td < min_time) min_time = td;
                if (act == 0 || td > max_time) max_time = td;
                mean += td;
                act += 1;
            }
            n.pm->name = "n";
            ph->erase("(" + name + "*" + name + ")");
            ph->erase("n");
        }
        mean /= act;
        cout << "\r";
        bool tooSlow = false;
        string unit;
        int unicode_frickel = 0;
        if (min_time < 100000) {
            unit = "ns";
        } else if (min_time < 100000000) {
            unit = "µs";
            unicode_frickel = 1;
            min_time /= 1000;
            mean /= 1000;
            max_time /= 1000;
        } else if (min_time < 100000000000) {
            unit = "ms";
            min_time /= 1000000;
            mean /= 1000000;
            max_time /= 1000000;
            if (min_time > 5000) tooSlow = true;
        } else {
            tooSlow = true;
            unit = "s";
            min_time /= 1000000000;
            mean /= 1000000000;
            max_time /= 1000000000;
        }
        // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        std::cout << "Matrix shape = " << std::setw(16) << "[" + std::to_string(dim) + "," + std::to_string(dim) + "] ";
        cout << std::setw(16 + unicode_frickel) << std::to_string(min_time) + " " + unit;
        cout << std::setw(16) << std::to_string(mean) + " " << unit;
        cout << std::setw(16 + unicode_frickel) << std::to_string(max_time) + " " + unit << endl;
        ph->erase(name);
        ph->erase(m.pm->name);
        if (tooSlow) break;
    }
}

int main(int, char **) {
#ifndef NDEBUG
    cout << "---------------------------------------" << endl;
    cout << "ERROR: Debug build, benchmarks invalid!" << endl;
    cout << "---------------------------------------" << endl
         << endl;
#endif
    NRMatrixHeap h;
    matMulBench(&h);
}