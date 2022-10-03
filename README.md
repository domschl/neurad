# neurad
Target: Neural nets with auto-diff with BLAS as only dep.
WIP! unfinished, just started, ignore!

## Build

Requirements: `cmake`, `ninja`, for non-Apple: `openblas`.

- Linux: `pacman -S cmake ninja openblas` (or `apt...`) and some c compiler.
- Mac: `brew install cmake ninja`, command-line tools required.

Clone repo, create `build` directory:

```bash
mkdir build
cd build
cmake -G Ninja ..
ninja
```

Run (in build folder):
```bash
./bench/bench_neurad
```
or
```bash
./test/test_neurad
```

## Matrix multiplication benchmarks (does not generalise! internal testing!)

Benchmark (single run, imprecise!) of matrix multiplication of given dimension
of two matrices with random normal intialization.

### Single precision

| dim / computer | Raspberry Pi 4 | i5-7500 CPU @ 3.40GHz | Mac mini M1 | Mac Studio M1 Pro |
| -------------- | -------------- | --------------------- | ----------- | ----------------- |
|           BLAS |       OpenBLAS |              OpenBLAS |  Accelerate |        Accelerate |
|         [2,2]  |    8129 ns     |     1783 ns           |   2250 ns   |    2709 ns        |
|         [3,3]  |    7259 ns     |     1458 ns           |   2208 ns   |    2666 ns        |
|         [4,4]  |    7351 ns     |     1305 ns           |   2083 ns   |    2417 ns        |
|         [5,5]  |   16555 ns     |     1435 ns           |   2250 ns   |    2708 ns        |
|         [6,6]  |    7648 ns     |     3156 ns           |   2334 ns   |    2792 ns        |
|         [8,8]  |   17426 ns     |     1353 ns           |   2292 ns   |    2709 ns        |
|       [10,10]  |    9889 ns     |     1583 ns           |   2542 ns   |    3375 ns        |
|       [32,32]  |     122 µs     |     7150 ns           |  30333 ns   |   35542 ns        |
|       [64,64]  |     408 µs     |    27197 ns           |  18375 ns   |   67541 ns        |
|     [128,128]  |    2172 µs     |    42026 µs           |    103 µs   |   58750 ns        |
|     [256,256]  |    6040 µs     |     8234 µs           |    412 µs   |     219 µs        |
|     [512,512]  |   29476 µs     |    10760 µs           |   1214 µs   |     881 µs        |
|   [1024,1024]  |     188 ms     |    25486 µs           |   3702 µs   |    2086 µs        |
|   [2048,2048]  |    1404 ms     |      137 ms           |  22357 µs   |   14166 µs        |
|   [4096,4096]  |   11123 ms     |      670 ms           |    159 ms   |   87925 µs        |
|   [8192,8192]  |                |     4242 ms           |   1230 ms   |     658 ms        |
| [10000,10000]  |                |     7535 ms           |   2129 ms   |    1058 ms        |
| [12000,12000]  |                |                       |   3490 ms   |    1717 ms        |
| [14000,14000]  |                |                       |   5947 ms   |    2843 ms        |
| [20000,20000]  |                |                       |             |    7872 ms        |

Strange things happen in ARM & M1-land at 128x128, sudden high performance boost.

### Double precision

| dim / computer | Raspberry Pi 4 | i5-7500 CPU @ 3.40GHz | Mac mini M1 | Mac Studio M1 Pro |
| -------------- | -------------- | --------------------- | ----------- | ----------------- |
|           BLAS |       OpenBLAS |              OpenBLAS |  Accelerate |        Accelerate |
|         [2,2]  |    8129 ns     |     1659 ns           |   2750 ns   |    2708 ns        |
|         [3,3]  |    7259 ns     |     1328 ns           |   2875 ns   |    2792 ns        |
|         [4,4]  |    7351 ns     |     1209 ns           |   2542 ns   |    2208 ns        |
|         [5,5]  |   16555 ns     |     1340 ns           |   3500 ns   |    2916 ns        |
|         [6,6]  |    7648 ns     |     3370 ns           |   3042 ns   |    2750 ns        |
|         [8,8]  |   17426 ns     |     1392 ns           |   3125 ns   |    3000 ns        |
|       [10,10]  |    9889 ns     |     1645 ns           |   3459 ns   |    3042 ns        |
|       [32,32]  |     122 µs     |    14753 ns           |    355 µs   |   27750 ns        |
|       [64,64]  |     408 µs     |    54334 ns           |  33167 ns   |   28333 ns        |
|     [128,128]  |    2172 µs     |    24113 µs           |    159 µs   |     113 µs        |
|     [256,256]  |    6040 µs     |    17583 µs           |    767 µs   |     463 µs        |
|     [512,512]  |   29476 µs     |    24362 µs           |   3544 µs   |    1580 µs        |
|   [1024,1024]  |     188 ms     |    17310 µs           |  13818 µs   |    7215 µs        |
|   [2048,2048]  |    1404 ms     |      111 ms           |  73564 µs   |   39971 µs        |
|   [4096,4096]  |   11123 ms     |     1376 ms           |    623 ms   |     316 ms        |
|   [8192,8192]  |                |     9835 ms           |   5054 ms   |    2452 ms        |
| [10000,10000]  |                |                       |             |    3667 ms        |
| [12000,12000]  |                |                       |             |    6220 ms        |
| [14000,14000]  |                |                       |             |                   |
