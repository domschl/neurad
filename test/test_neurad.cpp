#include "neurad.h"

using nrd::NRFloat;
using nrd::NRMatrix;
using nrd::NRMatrixHeap;

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

    NRMatrix t10 = NRMatrix(&h, 10, 2, "t10", (vector<NRFloat>){7, 5, 8, 3, 3, 9, 10, 1, 2, 3, 4, 5, 6, 1, 2, 3, 3, 4, 2, 3});
    NRMatrix t11 = NRMatrix(&h, 2, 10, "t11", (vector<NRFloat>){1, 4, 1, 3, 1, 2, 3, 7, 8, 1, 5, 5, 8, 9, 1, 3, 2, 5, 4, 1});
    cout << t10 << t11;
    NRMatrix t12 = (t10 * t11) * (t10 * t11);
    cout << t12;

    t12.family();

    cout << t1;
    cout << t1.t();
    NRMatrix tt = t1.t();
    cout << tt;
    NRMatrix tt3 = t1 * tt + tt.t() * t1.t();
    cout << tt3;

    tt3.family();
    /*
        NRSize dim = 12000;
        NRMatrix m = NRMatrix(&h, dim, dim, "m");
        NRMatrix n = NRMatrix(&h, dim, dim, "n");
        m.pm->randNormal(0, 1);
        n.pm->randNormal(0, 1);
        auto m1 = m + n;
        auto m2 = n * m + m;
        auto o = m1 * m2;
        cout << m << n << o;
    */
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
