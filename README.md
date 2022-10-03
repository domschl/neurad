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

## Matrix multiplication benchmarks

Benchmark of matrix multiplication of given dimension
of two matrices with random normal intialization. Measured after warm-up, minimum time given.

### Single precision

| dim / computer | Raspberry Pi 4 | i5-7500 CPU @ 3.40GHz | Mac mini M1 | Mac Studio M1 Max |
| -------------- | -------------- | --------------------- | ----------- | ----------------- |
|           BLAS |       OpenBLAS |              OpenBLAS |  Accelerate | INVALID 1-shot old|
|         [2,2]  |    3240 ns     |      470 ns           |    291 ns   |                   |
|         [3,3]  |    3518 ns     |      486 ns           |    333 ns   |                   |
|         [4,4]  |    3352 ns     |      477 ns           |    291 ns   |                   |
|         [5,5]  |    3722 ns     |      546 ns           |    291 ns   |                   |
|         [6,6]  |    3574 ns     |      555 ns           |    333 ns   |                   |
|         [8,8]  |    3296 ns     |      562 ns           |    333 ns   |                   |
|       [10,10]  |    4796 ns     |      654 ns           |    583 ns   |                   |
|       [32,32]  |   24796 ns     |     2199 ns           |    791 ns   |                   |
|       [64,64]  |   93499 ns     |     8356 ns           |   1833 ns   |                   |
|     [128,128]  |     268 µs     |    25748 ns           |   7125 ns   |                   |
|     [256,256]  |    2294 µs     |      277 µs           |  37500 ns   |                   |
|     [512,512]  |   15743 µs     |     1488 µs           |    249 µs   |                   |
|   [1024,1024]  |     112 ms     |     7818 µs           |   2357 µs   |                   |
|   [2048,2048]  |     812 ms     |    48093 µs           |  21963 µs   |                   |
|   [4096,4096]  |    6172 ms     |      348 ms           |    164 ms   |                   |
|   [8192,8192]  |                |     2656 ms           |   1296 ms   |     658 ms        |
| [10000,10000]  |                |     4716 ms           |   2262 ms   |    1058 ms        |
| [12000,12000]  |                |     8062 ms           |   3730 ms   |    1717 ms        |
| [14000,14000]  |                |                       |   6244 ms   |    2843 ms        |
| [20000,20000]  |                |                       |             |    7872 ms        |

### Double precision

| dim / computer | Raspberry Pi 4 | i5-7500 CPU @ 3.40GHz | Mac mini M1 | Mac Studio M1 Max |
| -------------- | -------------- | --------------------- | ----------- | ----------------- |
|           BLAS |       OpenBLAS |              OpenBLAS |  Accelerate | INVALID 1-shot old|
|         [2,2]  |    3111 ns     |      441 ns           |    291 ns   |                   |
|         [3,3]  |    3259 ns     |      478 ns           |    333 ns   |                   |
|         [4,4]  |    3185 ns     |      436 ns           |    291 ns   |                   |
|         [5,5]  |    3481 ns     |      506 ns           |    333 ns   |                   |
|         [6,6]  |    3648 ns     |      480 ns           |    334 ns   |                   |
|         [8,8]  |    3667 ns     |      513 ns           |    375 ns   |                   |
|       [10,10]  |    4352 ns     |      673 ns           |    500 ns   |                   |
|       [32,32]  |   34074 ns     |     3426 ns           |   1167 ns   |                   |
|       [64,64]  |     161 µs     |    16517 ns           |   3625 ns   |                   |
|     [128,128]  |     494 µs     |    45452 ns           |  19791 ns   |                   |
|     [256,256]  |    4459 µs     |      409 µs           |    116 µs   |                   |
|     [512,512]  |   29686 µs     |     3350 µs           |   1036 µs   |                   |
|   [1024,1024]  |     192 ms     |    15562 µs           |   9264 µs   |                   |
|   [2048,2048]  |    1481 ms     |      108 ms           |  74874 µs   |   39971 µs        |
|   [4096,4096]  |   11568 ms     |      821 ms           |    599 ms   |     316 ms        |
|   [8192,8192]  |                |     6433 ms           |   4803 ms   |    2452 ms        |
| [10000,10000]  |                |                       |   8110 ms   |    3667 ms        |
| [12000,12000]  |                |                       |             |    6220 ms        |
| [14000,14000]  |                |                       |             |                   |
