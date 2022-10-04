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

| dim / computer | Raspberry Pi 4 | i5-7500 CPU @ 3.40GHz | Mac mini M1 | Mac Studio M1 Pro |
| -------------- | -------------- | --------------------- | ----------- | ----------------- |
|           BLAS |       OpenBLAS |              OpenBLAS |  Accelerate |        Accelerate |
|         [2,2]  |    3240 ns     |      470 ns           |    291 ns   |     333 ns        |
|         [3,3]  |    3518 ns     |      486 ns           |    333 ns   |     375 ns        |
|         [4,4]  |    3352 ns     |      477 ns           |    291 ns   |     250 ns        |
|         [5,5]  |    3722 ns     |      546 ns           |    291 ns   |     292 ns        |
|         [6,6]  |    3574 ns     |      555 ns           |    333 ns   |     375 ns        |
|         [8,8]  |    3296 ns     |      562 ns           |    333 ns   |     333 ns        |
|       [10,10]  |    4796 ns     |      654 ns           |    583 ns   |     583 ns        |
|       [32,32]  |   24796 ns     |     2199 ns           |    791 ns   |     833 ns        |
|       [64,64]  |   93499 ns     |     8356 ns           |   1833 ns   |    1833 ns        |
|     [128,128]  |     268 µs     |    25748 ns           |   7125 ns   |    7167 ns        |
|     [256,256]  |    2294 µs     |      277 µs           |  37500 ns   |   38292 ns        |
|     [512,512]  |   15743 µs     |     1488 µs           |    249 µs   |     270 µs        |
|   [1024,1024]  |     112 ms     |     7818 µs           |   2357 µs   |    1601 µs        |
|   [2048,2048]  |     812 ms     |    48093 µs           |  21963 µs   |   10194 µs        |
|   [4096,4096]  |    6172 ms     |      348 ms           |    164 ms   |   85232 µs        |
|   [8192,8192]  |                |     2656 ms           |   1296 ms   |     641 ms        |
| [10000,10000]  |                |     4716 ms           |   2262 ms   |    1029 ms        |
| [12000,12000]  |                |     8062 ms           |   3730 ms   |    1741 ms        |
| [14000,14000]  |                |                       |   6244 ms   |    2803 ms        |
| [20000,20000]  |                |                       |             |    7624 ms        |

### Double precision

| dim / computer | Raspberry Pi 4 | i5-7500 CPU @ 3.40GHz | Mac mini M1 | Mac Studio M1 Pro |
| -------------- | -------------- | --------------------- | ----------- | ----------------- |
|           BLAS |       OpenBLAS |              OpenBLAS |  Accelerate |        Accelerate |
|         [2,2]  |    3111 ns     |      441 ns           |    291 ns   |     333 ns        |
|         [3,3]  |    3259 ns     |      478 ns           |    333 ns   |     333 ns        |
|         [4,4]  |    3185 ns     |      436 ns           |    291 ns   |     333 ns        |
|         [5,5]  |    3481 ns     |      506 ns           |    333 ns   |     375 ns        |
|         [6,6]  |    3648 ns     |      480 ns           |    334 ns   |     375 ns        |
|         [8,8]  |    3667 ns     |      513 ns           |    375 ns   |     375 ns        |
|       [10,10]  |    4352 ns     |      673 ns           |    500 ns   |     458 ns        |
|       [32,32]  |   34074 ns     |     3426 ns           |   1167 ns   |    1208 ns        |
|       [64,64]  |     161 µs     |    16517 ns           |   3625 ns   |    3667 ns        |
|     [128,128]  |     494 µs     |    45452 ns           |  19791 ns   |   20042 ns        |
|     [256,256]  |    4459 µs     |      409 µs           |    116 µs   |     117 µs        |
|     [512,512]  |   29686 µs     |     3350 µs           |   1036 µs   |     794 µs        |
|   [1024,1024]  |     192 ms     |    15562 µs           |   9264 µs   |    5628 µs        |
|   [2048,2048]  |    1481 ms     |      108 ms           |  74874 µs   |   35481 µs        |
|   [4096,4096]  |   11568 ms     |      821 ms           |    599 ms   |     299 ms        |
|   [8192,8192]  |                |     6433 ms           |   4803 ms   |    2296 ms        |
| [10000,10000]  |                |                       |   8110 ms   |    3709 ms        |
| [12000,12000]  |                |                       |             |    6319 ms        |
| [14000,14000]  |                |                       |             |                   |

Benchmark, double precision

                                            Min              Mean             Max
Matrix shape =           [2,2]           333 ns            532 ns        12375 ns
Matrix shape =           [3,3]           333 ns            436 ns        13125 ns
Matrix shape =           [4,4]           333 ns            385 ns        10875 ns
Matrix shape =           [5,5]           375 ns            436 ns         6792 ns
Matrix shape =           [6,6]           375 ns            474 ns         6792 ns
Matrix shape =           [8,8]           375 ns            462 ns        13167 ns
Matrix shape =         [10,10]           458 ns            516 ns          666 ns
Matrix shape =         [32,32]          1208 ns           1346 ns        10833 ns
Matrix shape =         [64,64]          3667 ns           4125 ns        12500 ns
Matrix shape =       [128,128]         20042 ns          20123 ns        20209 ns
Matrix shape =       [256,256]           117 µs            124 µs          236 µs
Matrix shape =       [512,512]           794 µs            846 µs         1092 µs
Matrix shape =     [1024,1024]          5628 µs           5873 µs         6387 µs
Matrix shape =     [2048,2048]         35481 µs          35737 µs        36196 µs
Matrix shape =     [4096,4096]           299 ms            305 ms          314 ms
Matrix shape =     [8192,8192]          2296 ms           2328 ms         2359 ms
Matrix shape =   [10000,10000]          3709 ms           3709 ms         3709 ms
Matrix shape =   [12000,12000]          6319 ms           6319 ms         6319 ms