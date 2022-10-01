#include <iostream>
#include <iomanip>
#include <random>
#include <math.h>
#include <cmath>
#include <vector>
#include <set>
#include <functional>

using std::cout;
using std::endl;
using std::vector;

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#error "Put your BLAS here"
#endif

#ifdef USE_SINGLE_PRECISION_FLOAT
typedef float NRFloat;
typedef int NRSize;
NRFloat NaN = std::nanf("0");
#else
typedef double NRFloat;
typedef int NRSize;
NRFloat NaN = std::nan("0");
#endif

class NRMatrix;

class NRMatrix {
  public:
    vector<NRFloat> mx;
    NRSize x;
    NRSize y;
    NRSize l;
    std::string name = "";
    std::string op = "";
    std::set<NRMatrix *> children;
    std::function<void()> backprop;

  private:
    NRSize i, ix, iy, ry, rx;
    // A bit of FORTRAN-charm for BLAS:
    int M, K, N, LDA, LDB, LDC;
    NRFloat ALPHA = 1.0, BETA = 0.0;
    CBLAS_TRANSPOSE TRANSA = CblasNoTrans, TRANSB = CblasNoTrans;

  public:
    NRMatrix() {
        x = 0;
        y = 0;
        l = 0;
    }
    NRMatrix(NRSize y, NRSize x, std::string name = "", std::string op = "")
        : y(y), x(x), name(name), op(op) {
        l = x * y;
        mx = vector<NRFloat>(l);
    }
    NRMatrix(NRSize y, NRSize x, vector<NRFloat> v, std::string name = "", std::string op = "")
        : y(y), x(x), name(name), op(op) {
        l = x * y;
        if (v.size() == l)
            mx = v;
        else {
            mx = vector<NRFloat>(l);
            l = std::min(v.size(), mx.size());
            for (int i = 0; i < l; i++)
                mx[i] = v[i];
        }
    }
    ~NRMatrix() {
    }
    /* We get all this for free, since using std::vector:
    //! Copy
    NRMatrix(const NRMatrix &source_matrix)
        : x(source_matrix.x), y(source_matrix.y), l(source_matrix.l), mx(source_matrix.mx) {
    }
    // Assignment copy
    NRMatrix &operator=(const NRMatrix &source_matrix) {
        if (this != &source_matrix) {
            x = source_matrix.x;
            y = source_matrix.y;
            l = source_matrix.l;
            mx = source_matrix.mx;
        }
        return *this;
    }

    // Move constructor.
    NRMatrix(NRMatrix &&source_matrix) noexcept
        : mx({}), l(0), x(0), y(0) {
        x = source_matrix.x;
        y = source_matrix.y;
        l = source_matrix.l;
        mx = source_matrix.mx;

        source_matrix.x = 0;
        source_matrix.y = 0;
        source_matrix.l = 0;
        source_matrix.mx = {};
    }

    // Move assignment operator.
    NRMatrix &operator=(NRMatrix &&source_matrix) noexcept {
        if (this != &source_matrix) {
            // Free the existing resource.
            // delete[] mx;
            x = source_matrix.x;
            y = source_matrix.y;
            l = source_matrix.l;
            mx = source_matrix.mx;

            source_matrix.x = 0;
            source_matrix.y = 0;
            source_matrix.l = 0;
            source_matrix.mx = {};
        }
        return *this;
    }
    */
    void zero() {
        for (NRSize i = 0; i < l; i++)
            mx[i] = 0.0;
    }
    void unit() {
        for (iy = 0; iy < y; iy++) {
            ry = iy * x;
            for (ix = 0; ix < x; ix++) {
                if (iy != ix)
                    mx[ry + ix] = 0.0;
                else
                    mx[ry + ix] = 1.0;
            }
        }
    }
    void t() {
        for (iy = 0; iy < y; iy++) {
            ry = iy * x;
            for (ix = 0; ix < x; ix++) {
                if (iy != ix) {
                    rx = ix * y;
                    NRFloat s = mx[ry + ix];
                    mx[ry + ix] = mx[rx + iy];
                    mx[rx + iy] = s;
                }
            }
        }
        NRFloat sw = x;
        x = y;
        y = sw;
    }
    void randNormal(NRFloat mean, NRFloat var) {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> dn{mean, var};
        for (i = 0; i < l; i++)
            mx[i] = dn(gen);
    }
    void randInt(int a, int b) {  // Inclusive [a,b]
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_int_distribution<> di{a, b};
        for (i = 0; i < l; i++)
            mx[i] = di(gen);
    }
    NRFloat get(NRSize yi, NRSize xi) {
        return mx[yi * x + xi];
    }
    void set(NRSize yi, NRSize xi, NRFloat v) {
        mx[yi * x + xi] = v;
    }
    NRMatrix operator+(NRMatrix &r) {
        NRMatrix s;
        if (this->x != r.x || this->y != r.y) {
            s = NRMatrix(0, 0);
        } else {
            s = NRMatrix(this->y, this->x);
            for (iy = 0; iy < this->y; iy++) {
                ry = iy * r.x;
                for (ix = 0; ix < this->x; ix++) {
                    s.mx[ry + ix] = this->mx[ry + ix] + r.mx[ry + ix];
                }
            }
        }
        return std::move(s);
    }
    //! this is the single part of any neural network implementation that has influence on performance:
    //! matrix multiplication, everything else matters much, much less:
    NRMatrix operator*(NRMatrix &r) {
        if (this->x != r.y) {
            return std::move(NRMatrix(0, 0));
        } else {
            M = this->y;
            K = this->x;
            N = r.x;
            LDA = K;
            LDB = N;
            LDC = N;
            NRMatrix C(M, N);
            C.zero();
#if defined(USE_SINGLE_PRECISION_FLOAT)
            cblas_sgemm(CblasRowMajor, TRANSA, TRANSB, M, N, K, ALPHA, (float *)&(this->mx[0]), LDA,
                        (float *)&(r.mx[0]), LDB, BETA, (float *)&(C.mx[0]), LDC);
#else
            cblas_dgemm(CblasRowMajor, TRANSA, TRANSB, M, N, K, ALPHA, (double *)&(this->mx[0]), LDA,
                        (double *)&(r.mx[0]), LDB, BETA, (double *)&(C.mx[0]), LDC);
#endif
            return std::move(C);
        }
    }

    NRFloat max() const {
        if (l == 0) return NaN;
        NRFloat m = mx[0];
        for (NRSize i = 1; i < l; i++)
            if (mx[i] > m) m = mx[i];
        return m;
    }
    NRFloat min() const {
        if (l == 0) return NaN;
        NRFloat m = mx[0];
        for (NRSize i = 1; i < l; i++)
            if (mx[i] < m) m = mx[i];
        return m;
    }
    bool isInt() const {
        if (l == 0) return false;
        for (NRSize i = 0; i < l; i++) {
            if ((NRFloat)(int)mx[i] != mx[i]) return false;
        }
        return true;
    }
    bool isPos() const {
        if (l == 0) return false;
        for (NRSize i = 0; i < l; i++) {
            if (mx[i] < 0.0) return false;
        }
        return true;
    }
    // print() is used by cout << overload, hence it and all methods is uses
    // need to be const.
    void print(int precision = 2, bool brackets = true) const {
        NRSize xi, yi, yr;
        NRFloat ma, mn;
        int lp;
        bool isint;
        bool issci;
        bool use_pref = false;
        std::string pref, spref;
        NRSize prefline;
        if (name != "") {
            pref = name + " = ";
            spref = "";
            for (auto c : pref)
                spref += " ";
            use_pref = true;
            prefline = y / 2;
        }
        cout << std::fixed << std::setprecision(precision);
        mn = min();
        ma = max();
        if (-mn > ma) {
            ma = -mn;
            lp = 0;
        } else
            lp = 0;
        if (isInt()) {
            isint = true;
            lp -= 3;
        } else
            isint = false;
        if (isPos()) lp -= 1;
        if (ma > 10000 || ma < 0.1) {
            cout << std::scientific;
            issci = true;
            lp += 9;
        } else {
            issci = false;
            lp += (int)log10(ma) + 5;
        }
        for (yi = 0; yi < y; yi++) {
            if (use_pref) {
                if (yi == prefline)
                    cout << pref;
                else
                    cout << spref;
            }
            yr = yi * x;
            if (y == 1)
                cout << "[";
            else {
                if (yi == 0)
                    cout << "⎛";
                else if (yi > 0 && yi < y - 1)
                    cout << "⎜";
                else
                    cout << "⎝";
            }
            for (xi = 0; xi < x; xi++) {
                cout << std::setw(lp);
                if (isint && !issci)
                    cout << (int)(mx[yr + xi]);
                else
                    cout << mx[yr + xi];
                if (xi < x - 1) cout << " ";
            }
            if (y == 1)
                cout << "]";
            else {
                if (yi == 0)
                    cout << "⎞";
                else if (yi > 0 && yi < y - 1)
                    cout << "⎟";
                else
                    cout << "⎠";
            }
            cout << endl;
        }
    }
};

std::ostream &operator<<(std::ostream &os, const NRMatrix &mat) {
    mat.print();
    return os;
}

int main(int, char **) {
    NRMatrix t1 = NRMatrix(3, 2, (vector<NRFloat>){1, 2, 3, 4, 5, 6}, "t1");
    NRMatrix t2 = NRMatrix(2, 3, (vector<NRFloat>){1, 0, 1, 0, 1, 0}, "t2");
    // t1.randInt(0, 1000000000);
    // t2.randInt(-100, 1000000000);
    //  t1.randNormal(0, 1e20);
    //  t2.randNormal(0, 1e20);
    cout << t1 << t2;
    NRMatrix t3 = t1 * t2;
    t3.name = "t1*t2 = t3";
    cout << t3;
    // Output:
    //       ⎛1 2⎞
    //  t1 = ⎜3 4⎟
    //       ⎝5 6⎠
    //       ⎛1 0 1⎞
    //  t2 = ⎝0 1 0⎠
    //               ⎛1 2 1⎞
    //  t1*t2 = t3 = ⎜3 4 3⎟
    //               ⎝5 6 5⎠
}
