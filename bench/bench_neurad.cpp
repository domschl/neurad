#include "neurad.h"

using nrd::NRMatrix;
using nrd::NRMatrixHeap;
using nrd::NRSize;

void matMulBench(NRMatrixHeap *ph) {
    vector<NRSize> matDims = {2, 3, 4, 5, 6, 8, 10, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000, 12000, 14000, 20000};
    for (NRSize dim : matDims) {
        string name = "d" + std::to_string(dim);
        NRMatrix m = NRMatrix(ph, dim, dim, name);
        m.pm->randNormal(0, 1);
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        NRMatrix n = m * m;
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        bool tooSlow = false;
        string unit;
        auto td = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        int unicode_frickel = 0;
        if (td < 100000) {
            unit = "ns";
        } else if (td < 100000000) {
            unit = "µs";
            unicode_frickel = 1;
            td /= 1000;
        } else if (td < 100000000000) {
            unit = "ms";
            td /= 1000000;
            if (td > 5000) tooSlow = true;
        } else {
            tooSlow = true;
            unit = "s";
            td /= 1000000000;
        }
        // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        std::cout << "Matrix shape = " << std::setw(16) << "[" + std::to_string(dim) + "," + std::to_string(dim) + "] " << std::setw(16 + unicode_frickel) << std::to_string(td) + " " + unit << std::endl;
        ph->erase(name);
        ph->erase(m.pm->name);
        if (tooSlow) break;
    }
}

int main(int, char **) {
    NRMatrixHeap h;
    matMulBench(&h);
}