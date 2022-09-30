#include <iostream>
#include <iomanip>
#include <random>
#include <math.h>

using std::cout;
using std::endl;

// XXX Apple only:
#include <Accelerate/Accelerate.h>

typedef float NTFloat;
// extern "C" void sgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N, const int *K, NTFloat *ALPHA, NTFloat *A, const int *LDA, NTFloat *B, const int *LDB, NTFloat *BETA, NTFloat *C, const int *LDC);

class NRTensor {
  public:
    NTFloat *data = nullptr;
    int x;
    int y;
    enum MatrixInitType { None,
                          Zero,
                          Unit,
                          Random };
    NRTensor(int y, int x, MatrixInitType t)
        : y(y), x(x) {
        data = (NTFloat *)malloc(x * y * sizeof(NTFloat));
        switch (t) {
        case None:
            break;
        case Zero:
            for (int iy = 0; iy < y; iy++) {
                int ry = iy * x;
                for (int ix = 0; ix < x; ix++) {
                    data[ry + ix] = 0.0;
                }
            }
            break;
        case Unit:
            for (int iy = 0; iy < y; iy++) {
                int ry = iy * x;
                for (int ix = 0; ix < x; ix++) {
                    if (iy != ix)
                        data[ry + ix] = 0.0;
                    else
                        data[ry + ix] = 1.0;
                }
            }
            break;
        case Random:
            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::normal_distribution<> d{0, 1};
            for (int iy = 0; iy < y; iy++) {
                int ry = iy * x;
                for (int ix = 0; ix < x; ix++) {
                    data[ry + ix] = d(gen);
                }
            }
            break;
        }
    }
    ~NRTensor() {
        if (data != nullptr) {
            free(data);
        }
    }
    NTFloat get(int iy, int ix) {
        return data[iy * x + ix];
    }
    void set(int iy, int ix, NTFloat v) {
        data[iy + x + ix] = v;
    }
    NRTensor *operator+(NRTensor &r) {
        if (this->x != r.x || this->y != r.y) return new NRTensor(0, 0, None);
        NRTensor *s = new NRTensor(this->y, this->x, None);
        for (int iy = 0; iy < this->y; iy++) {
            int ry = iy * r.x;
            for (int ix = 0; ix < this->x; ix++) {
                s->data[ry + ix] = this->data[ry + ix] + r.data[ry + ix];
            }
        }
        return s;
    }
    void print(int precision = 2, bool brackets = true) const {
        cout << std::fixed << std::setprecision(precision);
        for (int iy = 0; iy < y; iy++) {
            int ry = iy * x;
            if (y == 1)
                cout << "[";
            else {
                if (iy == 0)
                    cout << "⎛";
                else if (iy > 0 && iy < y - 1)
                    cout << "⎜";
                else
                    cout << "⎝";
            }
            for (int ix = 0; ix < x; ix++) {
                cout << data[ry + ix];
                if (ix < x - 1) cout << " ";
            }
            if (y == 1)
                cout << "]";
            else {
                if (iy == 0)
                    cout << "⎞";
                else if (iy > 0 && iy < y - 1)
                    cout << "⎟";
                else
                    cout << "⎠";
            }
            cout << endl;
        }
    }
};

std::ostream &operator<<(std::ostream &os, const NRTensor &mat) {
    mat.print();
    return os;
}

void testmat() {
    NTFloat A[4] = {1, 2, 3, 4};
    NTFloat B[4] = {1, 0, 1, 0};
    char TRANS = 'N';
    int M = 2;
    int N = 2;
    int K = 2;
    NTFloat ALPHA = 1.0;
    int LDA = 2;
    int LDB = 2;
    NTFloat BETA = 0.0;
    NTFloat C[4];
    int LDC = 2;

    sgemm_(&TRANS, &TRANS, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);

    cout << C[0] << endl;
    cout << C[1] << endl;
    cout << C[2] << endl;
    cout << C[3] << endl;
    // getchar();
    return;
}
int main(int, char **) {
    NRTensor t1 = NRTensor(1, 1, NRTensor::MatrixInitType::Unit);
    NRTensor t2 = NRTensor(1, 1, NRTensor::MatrixInitType::Random);
    NRTensor *t3 = t1 + t2;
    cout << t1 << t2 << *t3;
    // testmat();
}
