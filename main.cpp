#include <iostream>
#include <math.h>

using std::cout;
using std::endl;

//#include <BLAS.h>
//#include <cblas>
typedef float NTFloat;
extern "C" void sgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N, const int *K, NTFloat *ALPHA, NTFloat *A, const int *LDA, NTFloat *B, const int *LDB, NTFloat *BETA, NTFloat *C, const int *LDC);

class NRTensor {
  public:
    NTFloat data;
    NRTensor() {
    }
    ~NRTensor() {
    }
};

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
    NRTensor t1 = NRTensor();
    testmat();
}
