#pragma once

#include <iostream>
#include <iomanip>
#include <random>
#include <math.h>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <functional>
#include <chrono>

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

namespace nrd {
#ifdef USE_SINGLE_PRECISION_FLOAT
typedef float NRFloat;
typedef int NRSize;
NRFloat NaN = std::nanf("0");
#else
typedef double NRFloat;
typedef int NRSize;
NRFloat NaN = std::nan("0");
#endif

struct NRTensorAtom {
    vector<NRFloat> t;
    vector<NRSize> d;
    NRTensorAtom() {
        t = {};
        d = {};
    }
    NRTensorAtom(vector<NRSize> d)
        : d(d) {
        t = vector<NRFloat>(size());
    }
    NRSize size() {
        NRSize l = 1;
        for (int i = 0; i < d.size(); i++)
            l *= d[i];
        return l;
    }
};

struct NRMatrixCore {
    NRTensorAtom m;
    NRTensorAtom grad;
    /*
    vector<NRFloat> mx;
    vector<NRFloat> grad;
    NRSize x;
    NRSize y;
    NRSize l;
    */
    std::string name = "";
    string op;
    vector<NRMatrixCore *> children;
    bool bValid = false;

  private:
    NRSize i, ix, iy, rx, ry;

  public:
    NRMatrixCore() {
        m = NRTensorAtom();
        grad = NRTensorAtom();
        /*
        x = 0;
        y = 0;
        l = 0;
        */
        name = "null";
        op = "";
        bValid = false;
    }
    NRMatrixCore(NRSize y, NRSize x, string name, string op = "", vector<NRMatrixCore *> children = {})
        : name(name), op(op), children(children) {
        // l = x * y;
        // mx = vector<NRFloat>(l);
        // grad = mx;
        m = NRTensorAtom({y, x});
        grad = NRTensorAtom({y, x});
        bValid = true;
        zeroGrad();
    }
    NRMatrixCore(NRSize y, NRSize x, string name, vector<NRFloat> v, string op = "", vector<NRMatrixCore *> children = {})
        : name(name), op(op), children(children) {
        // l = x * y;
        m = NRTensorAtom({y, x});
        if (m.size() == v.size())
            m.t = v;
        else {
            cout << "ERROR: incompatible length of initializing vector for matrix " << name << ", NO init!" << endl;
        }
        grad = NRTensorAtom({y, x});
        bValid = true;
        zeroGrad();
    }
    void zero() {
        // XXX children!
        for (NRSize i = 0; i < m.t.size(); i++) {
            m.t[i] = 0.0;
            grad.t[i] = 0.0;
        }
    }
    void ones() {
        // XXX children!
        for (NRSize i = 0; i < m.t.size(); i++) {
            m.t[i] = 1.0;
            grad.t[i] = 0.0;
        }
    }
    void zeroGrad() {
        for (NRSize i = 0; i < m.t.size(); i++) {
            grad.t[i] = 0.0;
        }
    }
    void onesGrad() {
        for (NRSize i = 0; i < m.t.size(); i++) {
            grad.t[i] = 1.0;
        }
    }
    void unit() {
        for (iy = 0; iy < m.d[0]; iy++) {
            ry = iy * m.d[1];
            for (ix = 0; ix < m.d[1]; ix++) {
                if (iy != ix)
                    m.t[ry + ix] = 0.0;  // XXX: call overload
                else
                    m.t[ry + ix] = 1.0;
                grad.t[i] = 0.0;
            }
        }
    }
    void t() {
        for (iy = 0; iy < m.d[0]; iy++) {
            ry = iy * m.d[1];
            for (ix = 0; ix < m.d[1]; ix++) {
                if (iy != ix) {
                    rx = ix * m.d[0];
                    std::swap(m.t[ry + ix], m.t[rx + iy]);
                    std::swap(grad.t[ry + ix], grad.t[rx + iy]);
                }
            }
        }
        // std::swap(x, y);
    }
    void randNormal(NRFloat mean, NRFloat var) {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> dn{mean, var};
        for (i = 0; i < m.t.size(); i++) {
            m.t[i] = dn(gen);
            grad.t[i] = 0;
        }
    }
    void randInt(int a, int b) {  // Inclusive [a,b]
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_int_distribution<> di{a, b};
        for (i = 0; i < m.t.size(); i++) {
            m.t[i] = di(gen);
            grad.t[i] = 0;
        }
    }
    NRFloat get(NRSize yi, NRSize xi) {
        return m.t[yi * m.d[1] + xi];  // XXX: call overload
    }
    void set(NRSize yi, NRSize xi, NRFloat v) {
        m.t[yi * m.d[1] + xi] = v;
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
        if (pmc->m.d[0] == y && pmc->m.d[1] == x) return true;
        return false;
    }
    NRMatrixCore *getP(string name) {
        auto srch = h.find(name);
        if (srch == h.end()) return (NRMatrixCore *)nullptr;
        return &(h[name]);
    }
    NRMatrixCore *add(NRSize y, NRSize x, string name, string op = "", vector<NRMatrixCore *> children = {}) {
        if (exists(name)) {
            cout << "FATAL: tried to add existing matrix " << name << " ignored." << endl;
            return nullptr;
        }
        h[name] = NRMatrixCore(y, x, name, op, children);
        return getP(name);
    }
    NRMatrixCore *add(NRSize y, NRSize x, string name, vector<NRFloat> v, string op = "", vector<NRMatrixCore *> children = {}) {
        if (exists(name)) {
            cout << "FATAL: tried to add existing matrix " << name << " ignored." << endl;
            return nullptr;
        }
        h[name] = NRMatrixCore(y, x, name, v, op, children);
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
    void zero() {
        pm->zero();
    }

    void matAddDiff(NRMatrixCore *pm) {
        if (pm->children.size() != 2) {
            cout << "FATAL: matAddDiff requires two children!";
            return;
        }
        NRMatrixCore *pma, *pmb;
        pma = pm->children[0];
        pmb = pm->children[1];
        pma->grad = pma->grad + pm->grad;
    }
    NRMatrixCore *matAdd(NRMatrixCore *pa, NRMatrixCore *pb) {
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
        pc = ph->add(pa->y, pa->x, name, "+", (vector<NRMatrixCore *>){pa, pb});
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
        NRMatrixCore *pmc = matAdd(this->pm, &(r->pm));
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

    void matMulDiff(NRMatrixCore *pm) {
    }

    NRMatrixCore *matMul(NRMatrix *pma, NRMatrix *pmb) {
        NRMatrixCore *pa = pma->pm;
        NRMatrixCore *pb = pmb->pm;
        NRMatrixCore *pc;
        if (pa->x != pb->y) {
            std::cerr << "Invalid matrix mult: " << pa->name << '+' << pb->name << " " << pa->x << "," << pa->y << "<->" << pb->x << "," << pb->y << ": " << pa->x << "!=" << pb->y << " -> abort!" << endl;
            return ph->getP("null");
        }
        string name = "(" + pa->name + "*" + pb->name + ")";
        // XXX: commutative case!
        if (ph->exists(name)) {
            cout << "INFO: Using cached version of " << name << endl;
            return ph->getP(name);
        }
        pc = ph->add(pa->y, pb->x, name, "*", (vector<NRMatrixCore *>){pa, pb});
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
    NRMatrixCore *_tr(NRMatrixCore *pa) {
        string name = "t(" + pa->name + ")";
        if (ph->exists(name)) {
            cout << "INFO: Using cached version of " << name << endl;
            return ph->getP(name);
        }
        NRMatrixCore *pc = ph->add(pa->x, pa->y, name, "T", (vector<NRMatrixCore *>){pa});
        pc->mx = pa->mx;
        pc->t();
        return pc;
    }
    NRMatrix t() {
        NRMatrixCore *pc = _tr(pm);
        NRMatrix s = NRMatrix(ph, pc);
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
        std::string dimline;
        NRSize prefline;
        dimline = "[" + std::to_string(pm->y) + "," + std::to_string(pm->x) + "]";
        if (pm->name != "") {
            pref = pm->name + " = ";
        } else {
            pref = "";
            dimline += " = ";
        }
        if (pref.length() > 25) {
            pref = pref.substr(0, 11) + " .. " + pref.substr(pref.length() - 12);
        }
        int pl = std::max(pref.length(), dimline.length());
        spref = "";
        for (int i = 0; i < pl; i++) {
            if (pref.length() < pl) pref += " ";
            if (dimline.length() < pl) dimline += " ";
            if (spref.length() < pl) spref += " ";
        }
        use_pref = true;
        prefline = pm->y / 2;
        if (prefline > 0) --prefline;
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
            lp += 10;
        } else {
            issci = false;
            lp += (int)log10(ma) + precision + 3;
        }
        for (yi = 0; yi < pm->y; yi++) {
            if (yi > 6 && yi < pm->y - 6 && ((yi != prefline && yi != prefline + 1) || !use_pref)) {
                if (use_pref) {
                    if (yi == prefline - 1) cout << spref << "   ..." << endl;
                    if (yi == prefline + 2) cout << spref << "   ..." << endl;
                } else {
                    if (yi == pm->y / 2) cout << "..." << endl;
                }
            } else {
                if (use_pref) {
                    if (yi == prefline)
                        cout << pref;
                    else if (yi == prefline + 1)
                        cout << dimline;
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
                    if (xi > 3 && xi < pm->x - 4) {
                        if (xi == 5) cout << "..  ";
                    } else {
                        if (isint && !issci)
                            cout << (int)(pm->mx[yr + xi]);
                        else
                            cout << pm->mx[yr + xi];
                        if (xi < pm->x - 1) cout << " ";
                    }
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
    }
    void family() {
        MFamily(pm, 0);
    }

    void MFamily(NRMatrixCore *p, int gen) {
        if (gen == 0) cout << "Family:" << endl;
        for (int i = 0; i < gen; i++)
            cout << "  ";
        cout << p->name;
        if (p->op != "") cout << " " << p->op << " ";
        if (p->l == 0) {
            cout << " = INVALID MATRIX, abort!" << endl;
            return;
        }
        if (p->children.size() == 0) {
            cout << "]" << endl;
            return;
        }
        cout << endl;
        for (NRMatrixCore *child : p->children) {
            MFamily(child, gen + 1);
        }
    }
};

std::ostream &operator<<(std::ostream &os, const NRMatrix &mat) {
    mat.print();
    return os;
}
}  // namespace nrd
