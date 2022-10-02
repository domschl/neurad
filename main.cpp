#include <iostream>
#include <iomanip>
#include <random>
#include <math.h>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <functional>

using std::cout;
using std::endl;
using std::map;
using std::string;
using std::vector;

#ifdef __APPLE__
// Apple's include directories are a somewhat unusual.
// If this or dependencies like cblas.h aren't found by language servers,
// try to fix CMakeList.txt, include_directories() directiv.
#include <Accelerate/Accelerate.h>
#else
//  "Put your BLAS here"
#include <cblas.h>
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

struct NRMatrixCore {
    vector<NRFloat> mx;
    vector<NRFloat> grad;
    NRSize x;
    NRSize y;
    NRSize l;
    std::string name = "";
    vector<NRMatrixCore *> pDeps;
    bool bValid = false;

  private:
    NRSize i, ix, iy, rx, ry;

  public:
    NRMatrixCore() {
        mx = {};
        grad = {};
        x = 0;
        y = 0;
        l = 0;
        name = "null";
        bValid = false;
    }
    NRMatrixCore(NRSize y, NRSize x, string name)
        : y(y), x(x), name(name) {
        l = x * y;
        mx = vector<NRFloat>(l);
        grad = mx;
        bValid = true;
        zeroGrad();
    }
    NRMatrixCore(NRSize y, NRSize x, string name, vector<NRFloat> v)
        : y(y), x(x), name(name) {
        l = x * y;
        if (v.size() == l)
            mx = v;
        else {
            cout << "ERROR: incompatible length of initializing vector for matrix " << name << ", partial init!" << endl;
            mx = vector<NRFloat>(l);
            l = std::min(v.size(), mx.size());
            for (int i = 0; i < l; i++) {
                mx[i] = v[i];
            }
        }
        grad = mx;
        bValid = true;
        zeroGrad();
    }
    void zero() {
        for (NRSize i = 0; i < l; i++) {
            mx[i] = 0.0;
            grad[i] = 0.0;
        }
    }
    void zeroGrad() {
        for (NRSize i = 0; i < l; i++) {
            grad[i] = 0.0;
        }
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
                    s = grad[ry + ix];
                    grad[ry + ix] = grad[rx + iy];
                    grad[rx + iy] = s;
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
        grad[i] = 0;
    }
    void randInt(int a, int b) {  // Inclusive [a,b]
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_int_distribution<> di{a, b};
        for (i = 0; i < l; i++)
            mx[i] = di(gen);
        grad[i] = 0;
    }
    NRFloat get(NRSize yi, NRSize xi) {
        return mx[yi * x + xi];
    }
    void set(NRSize yi, NRSize xi, NRFloat v) {
        mx[yi * x + xi] = v;
    }
};

class NRMatrixHeap {
  public:
    map<string, NRMatrixCore> h;
    NRMatrixHeap() {
        h["null"] = NRMatrixCore();
    }
    ~NRMatrixHeap() {
    }
    bool exists(string name) {
        auto srch = h.find(name);
        if (srch == h.end()) return false;
        return true;
    }
    bool isCompatible(string name, NRSize y, NRSize x) {
        NRMatrixCore *pmc = getP(name);
        if (pmc == nullptr) return false;
        if (pmc->y == y && pmc->x == x) return true;
        return false;
    }
    NRMatrixCore *getP(string name) {
        auto srch = h.find(name);
        if (srch == h.end()) return (NRMatrixCore *)nullptr;
        return &(h[name]);
    }
    NRMatrixCore *add(NRSize y, NRSize x, string name) {
        if (exists(name)) {
            cout << "FATAL: tried to add existing matrix " << name << " ignored." << endl;
            return nullptr;
        }
        h[name] = NRMatrixCore(y, x, name);
        return getP(name);
    }
    NRMatrixCore *add(NRSize y, NRSize x, string name, vector<NRFloat> v) {
        if (exists(name)) {
            cout << "FATAL: tried to add existing matrix " << name << " ignored." << endl;
            return nullptr;
        }
        h[name] = NRMatrixCore(y, x, name, v);
        return getP(name);
    }
    bool erase(string name) {
        auto srch = h.find(name);
        if (srch == h.end()) return false;
        h.erase(srch);
        return true;
    }
};

class NRMatrix {
  public:
    NRMatrixHeap *ph;
    NRMatrixCore *pm;
    int defaultPrecision = 3;
    std::vector<NRMatrixCore *> children;
    std::function<void()> backprop;

    NRMatrix(NRMatrixHeap *ph)
        : ph(ph) {
        pm = ph->getP("null");
    }
    NRMatrix(NRMatrixHeap *ph, NRMatrixCore *pmc)
        : ph(ph), pm(pmc) {
    }
    NRMatrix(NRMatrixHeap *ph, NRSize y, NRSize x, std::string name)
        : ph(ph) {
        if (ph->exists(name)) {
            if (ph->isCompatible(name, y, x)) {
                pm = ph->getP(name);
                cout << "INFO: recycling existing matrix " << name << endl;
                return;
            } else {
                cout << "FATAL: tried to recreate existing matrix " << name << " with different size, ignored!" << endl;
                return;
            }
        } else {
            pm = ph->add(y, x, name);
        }
    }
    NRMatrix(NRMatrixHeap *ph, NRSize y, NRSize x, string name, vector<NRFloat> v)
        : ph(ph) {
        if (ph->exists(name)) {
            if (ph->isCompatible(name, y, x)) {
                pm = ph->getP(name);
                cout << "INFO: recycling existing matrix " << name << endl;
                return;
            } else {
                cout << "FATAL: tried to recreate existing matrix " << name << " with different size, ignored!" << endl;
                return;
            }
        } else {
            pm = ph->add(y, x, name, v);
        }
    }
    ~NRMatrix() {
    }

  private:
    NRSize ix, iy, rx, ry, i;  // don't always re-allocate

  public:
    NRMatrixCore *matAdd(NRMatrix *pma, NRMatrix *pmb) {
        NRMatrixCore *pa = pma->pm;
        NRMatrixCore *pb = pmb->pm;
        NRMatrixCore *pc;
        if (pa->x != pb->x || pa->y != pb->y) {
            std::cerr << "Invalid matrix add: " << pa->name << '+' << pb->name << " " << pa->x << "," << pa->y << "!=" << pb->x << "," << pb->y << " -> abort!" << endl;
            return ph->getP("null");
        }
        string name = "(" + pa->name + "+" + pb->name + ")";
        // XXX: commutative case!
        if (ph->exists(name)) {
            cout << "INFO: Using cached version of " << name << endl;
            return ph->getP(name);
        }
        pc = ph->add(pa->y, pa->x, name);
        for (iy = 0; iy < pa->y; iy++) {
            ry = iy * pa->x;
            for (ix = 0; ix < pa->x; ix++) {
                pc->mx[ry + ix] = pa->mx[ry + ix] + pb->mx[ry + ix];
            }
        }
        return ph->getP(name);  // XXX redundant call
        // s.children.push_back();
        // s.children.push_back();
    }

    NRMatrix operator+(NRMatrix &r) {
        NRMatrixCore *pmc = matAdd(this, &r);
        NRMatrix s = NRMatrix(ph, pmc);
        // s.children.push_back(NRMatrix(*this));
        // s.children.push_back(NRMatrix(r));
        return std::move(s);
    }
    NRMatrix operator+(NRMatrix &&r) {
        NRMatrixCore *pmc = matAdd(this, &r);
        NRMatrix s = NRMatrix(ph, pmc);
        // s.children.push_back(NRMatrix(*this));
        // s.children.push_back(NRMatrix(r));
        return std::move(s);
    }

    NRMatrixCore *matMul(NRMatrix *pma, NRMatrix *pmb) {
        NRMatrixCore *pa = pma->pm;
        NRMatrixCore *pb = pmb->pm;
        NRMatrixCore *pc;
        if (pa->x != pb->y) {
            std::cerr << "Invalid matrix mult: " << pa->name << '+' << pb->name << " " << pa->x << "," << pa->y << "<->" << pb->x << "," << pb->y << pa->x << "!=" << pb->y << " -> abort!" << endl;
            return ph->getP("null");
        }
        string name = "(" + pa->name + "*" + pb->name + ")";
        // XXX: commutative case!
        if (ph->exists(name)) {
            cout << "INFO: Using cached version of " << name << endl;
            return ph->getP(name);
        }
        pc = ph->add(pa->y, pb->x, name);
#if defined(USE_SINGLE_PRECISION_FLOAT)
        // A bit of FORTRAN-charm for BLAS:
        // int M, K, N, LDA, LDB, LDC;
        // NRFloat ALPHA = 1.0, BETA = 0.0;
        // CBLAS_TRANSPOSE TRANSA = CblasNoTrans, TRANSB = CblasNoTrans;
        // M = this->y; K = this->x; N = r.x; LDA = K; LDB = N; LDC = N;
        // cblas_sgemm(CblasRowMajor, TRANSA, TRANSB, M, N, K, ALPHA, (float *)&(this->mx[0]), LDA,
        //            (float *)&(r.mx[0]), LDB, BETA, (float *)&(C.mx[0]), LDC);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pa->y, pb->x, pa->x, 1.0, (NRFloat *)&(pa->mx[0]), pa->x,
                    (NRFloat *)&(pb->mx[0]), pb->x, 0.0, (NRFloat *)&(pc->mx[0]), pb->x);
#else
        // cblas_dgemm(CblasRowMajor, TRANSA, TRANSB, M, N, K, ALPHA, (double *)&(this->mx[0]), LDA,
        //             (double *)&(r.mx[0]), LDB, BETA, (double *)&(C.mx[0]), LDC);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pa->y, pb->x, pa->x, 1.0, (NRFloat *)&(pa->mx[0]), pa->x,
                    (NRFloat *)&(pb->mx[0]), pb->x, 0.0, (NRFloat *)&(pc->mx[0]), pb->x);
#endif
        // C.children.push_back(NRMatrix(*this));
        // C.children.push_back(NRMatrix(r));
        return ph->getP(name);  // XXX redundant call
    }
    NRMatrix operator*(NRMatrix &r) {
        NRMatrixCore *pmc = matMul(this, &r);
        NRMatrix s = NRMatrix(ph, pmc);
        // s.children.push_back(NRMatrix(*this));
        // s.children.push_back(NRMatrix(r));
        return std::move(s);
    }
    NRMatrix operator*(NRMatrix &&r) {
        NRMatrixCore *pmc = matMul(this, &r);
        NRMatrix s = NRMatrix(ph, pmc);
        // s.children.push_back(NRMatrix(*this));
        // s.children.push_back(NRMatrix(r));
        return std::move(s);
    }

    NRFloat max() const {
        if (pm->l == 0) return NaN;
        NRFloat m = pm->mx[0];
        for (NRSize i = 1; i < pm->l; i++)
            if (pm->mx[i] > m) m = pm->mx[i];
        return m;
    }
    NRFloat min() const {
        if (pm->l == 0) return NaN;
        NRFloat m = pm->mx[0];
        for (NRSize i = 1; i < pm->l; i++)
            if (pm->mx[i] < m) m = pm->mx[i];
        return m;
    }
    bool isInt() const {
        if (pm->l == 0) return false;
        for (NRSize i = 0; i < pm->l; i++) {
            if ((NRFloat)(int)pm->mx[i] != pm->mx[i]) return false;
        }
        return true;
    }
    bool isPos() const {
        if (pm->l == 0) return false;
        for (NRSize i = 0; i < pm->l; i++) {
            if (pm->mx[i] < 0.0) return false;
        }
        return true;
    }
    void setDefaultPrecision(int precision = 3) {
        defaultPrecision = precision;
    }
    int getDefaultPrecision() const {
        return defaultPrecision;
    }

    // print() is used by cout << overload, hence it and all methods is uses
    // need to be const.
    void print(int precision = -1, bool brackets = true) const {
        NRSize xi, yi, yr;
        NRFloat ma, mn;
        int lp;
        bool isint;
        bool issci;
        bool use_pref = false;
        if (precision == -1) precision = getDefaultPrecision();
        std::string pref, spref;
        NRSize prefline;
        if (pm->name != "") {
            pref = pm->name + " = ";
            spref = "";
            for (auto c : pref)
                spref += " ";
            use_pref = true;
            prefline = pm->y / 2;
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
        for (yi = 0; yi < pm->y; yi++) {
            if (use_pref) {
                if (yi == prefline)
                    cout << pref;
                else
                    cout << spref;
            }
            yr = yi * pm->x;
            if (pm->y == 1)
                cout << "[";
            else {
                if (yi == 0)
                    cout << "⎛";
                else if (yi > 0 && yi < pm->y - 1)
                    cout << "⎜";
                else
                    cout << "⎝";
            }
            for (xi = 0; xi < pm->x; xi++) {
                cout << std::setw(lp);
                if (isint && !issci)
                    cout << (int)(pm->mx[yr + xi]);
                else
                    cout << pm->mx[yr + xi];
                if (xi < pm->x - 1) cout << " ";
            }
            if (pm->y == 1)
                cout << "]";
            else {
                if (yi == 0)
                    cout << "⎞";
                else if (yi > 0 && yi < pm->y - 1)
                    cout << "⎟";
                else
                    cout << "⎠";
            }
            cout << endl;
        }
    }
    void family(int gen = 0) {
        if (gen == 0) cout << "Family:" << endl;
        // cout << "gen " << gen + 1 << endl;
        for (int i = 0; i < gen; i++)
            cout << "  ";
        cout << pm->name;
        if (pm->l == 0) {
            cout << " = INVALID MATRIX, abort!" << endl;
            return;
        }
        if (children.size() == 0) {
            cout << "]" << endl;
            return;
        }
        cout << endl;
        // for (NRMatrix child : children) {
        //     child.family(gen + 1);
        // }
    }
};

std::ostream &operator<<(std::ostream &os, const NRMatrix &mat) {
    mat.print();
    return os;
}

int main(int, char **) {
    NRMatrixHeap h;
    NRMatrix t1 = NRMatrix(&h, 3, 2, "t1", (vector<NRFloat>){1, 2, 3, 4, 5, 6});
    NRMatrix t2 = NRMatrix(&h, 3, 2, "t2", (vector<NRFloat>){1, 4, 1, 3, 1, 2});

    cout << t1 << t2;
    NRMatrix t3 = (t1 + t2) + (t1 + t2);
    cout << t3;
    NRMatrix t4 = t1 + t2 + t3;
    cout << t4;
    NRMatrix t5 = t3 + t4;
    cout << t5;

    NRMatrix t10 = NRMatrix(&h, 3, 2, "t10", (vector<NRFloat>){1, 2, 3, 4, 5, 6});
    NRMatrix t11 = NRMatrix(&h, 2, 3, "t11", (vector<NRFloat>){1, 4, 1, 3, 1, 2});
    cout << t10 << t11;
    NRMatrix t12 = (t10 * t11) * (t10 * t11);
    cout << t12;
}

// Test-output:
//      ⎛1 2⎞
// t1 = ⎜3 4⎟
//      ⎝5 6⎠
//      ⎛1 4⎞
// t2 = ⎜1 3⎟
//      ⎝1 2⎠
// INFO: Using cached version of (t1+t2)
//                     ⎛ 4 12⎞
// ((t1+t2)+(t1+t2)) = ⎜ 8 14⎟
//                     ⎝12 16⎠
// INFO: Using cached version of (t1+t2)
//                               ⎛ 6 18⎞
// ((t1+t2)+((t1+t2)+(t1+t2))) = ⎜12 21⎟
//                               ⎝18 24⎠
//                                                   ⎛10 30⎞
// (((t1+t2)+(t1+t2))+((t1+t2)+((t1+t2)+(t1+t2)))) = ⎜20 35⎟
//                                                   ⎝30 40⎠
//       ⎛1 2⎞
// t10 = ⎜3 4⎟
//       ⎝5 6⎠
//       ⎛1 4 1⎞
// t11 = ⎝3 1 2⎠
// INFO: Using cached version of (t10*t11)
//                         ⎛254 268 186⎞
// ((t10*t11)*(t10*t11)) = ⎜598 632 438⎟
//                         ⎝942 996 690⎠
