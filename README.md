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
|           BLAS |       OpenBLAS |              OpenBLAS |  Accelerate | INVALID 1-shot old|
|         [2,2]  |    3240 ns     |      470 ns           |    291 ns   |    2709 ns        |
|         [3,3]  |    3518 ns     |      486 ns           |    333 ns   |    2666 ns        |
|         [4,4]  |    3352 ns     |      477 ns           |    291 ns   |    2417 ns        |
|         [5,5]  |    3722 ns     |      546 ns           |    291 ns   |    2708 ns        |
|         [6,6]  |    3574 ns     |      555 ns           |    333 ns   |    2792 ns        |
|         [8,8]  |    3296 ns     |      562 ns           |    333 ns   |    2709 ns        |
|       [10,10]  |    4796 ns     |      654 ns           |    583 ns   |    3375 ns        |
|       [32,32]  |   24796 ns     |     2199 ns           |    791 ns   |   35542 ns        |
|       [64,64]  |   93499 ns     |     8356 ns           |   1833 ns   |   67541 ns        |
|     [128,128]  |     268 µs     |    25748 ns           |   7125 ns   |   58750 ns        |
|     [256,256]  |    2294 µs     |      277 µs           |  37500 ns   |     219 µs        |
|     [512,512]  |   15743 µs     |     1488 µs           |    249 µs   |     881 µs        |
|   [1024,1024]  |     112 ms     |     7818 µs           |   2357 µs   |    2086 µs        |
|   [2048,2048]  |     812 ms     |    48093 µs           |  21963 µs   |   14166 µs        |
|   [4096,4096]  |    6172 ms     |      348 ms           |    164 ms   |   87925 µs        |
|   [8192,8192]  |                |     2656 ms           |   1296 ms   |     658 ms        |
| [10000,10000]  |                |     4716 ms           |   2262 ms   |    1058 ms        |
| [12000,12000]  |                |     8062 ms           |   3730 ms   |    1717 ms        |
| [14000,14000]  |                |                       |   6244 ms   |    2843 ms        |
| [20000,20000]  |                |                       |             |    7872 ms        |

### Double precision

| dim / computer | Raspberry Pi 4 | i5-7500 CPU @ 3.40GHz | Mac mini M1 | Mac Studio M1 Pro |
| -------------- | -------------- | --------------------- | ----------- | ----------------- |
|           BLAS |       OpenBLAS |              OpenBLAS |  Accelerate | INVALID 1-shot old|
|         [2,2]  |    3111 ns     |      441 ns           |    291 ns   |    2708 ns        |
|         [3,3]  |    3259 ns     |      478 ns           |    333 ns   |    2792 ns        |
|         [4,4]  |    3185 ns     |      436 ns           |    291 ns   |    2208 ns        |
|         [5,5]  |    3481 ns     |      506 ns           |    333 ns   |    2916 ns        |
|         [6,6]  |    3648 ns     |      480 ns           |    334 ns   |    2750 ns        |
|         [8,8]  |    3667 ns     |      513 ns           |    375 ns   |    3000 ns        |
|       [10,10]  |    4352 ns     |      673 ns           |    500 ns   |    3042 ns        |
|       [32,32]  |   34074 ns     |     3426 ns           |   1167 ns   |   27750 ns        |
|       [64,64]  |     161 µs     |    16517 ns           |   3625 ns   |   28333 ns        |
|     [128,128]  |     494 µs     |    45452 ns           |  19791 ns   |     113 µs        |
|     [256,256]  |    4459 µs     |      409 µs           |    116 µs   |     463 µs        |
|     [512,512]  |   29686 µs     |     3350 µs           |   1036 µs   |    1580 µs        |
|   [1024,1024]  |     192 ms     |    15562 µs           |   9264 µs   |    7215 µs        |
|   [2048,2048]  |    1481 ms     |      108 ms           |  74874 µs   |   39971 µs        |
|   [4096,4096]  |   11568 ms     |      821 ms           |    599 ms   |     316 ms        |
|   [8192,8192]  |                |     6433 ms           |   4803 ms   |    2452 ms        |
| [10000,10000]  |                |                       |   8110 ms   |    3667 ms        |
| [12000,12000]  |                |                       |             |    6220 ms        |
| [14000,14000]  |                |                       |             |                   |


                                            Min              Mean             Max
Matrix shape =           [2,2]           291 ns            416 ns        15792 ns
Matrix shape =           [3,3]           333 ns            421 ns         5625 ns
Matrix shape =           [4,4]           291 ns            393 ns         6042 ns
Matrix shape =           [5,5]           333 ns            462 ns         5125 ns
Matrix shape =           [6,6]           334 ns            485 ns         6209 ns
Matrix shape =           [8,8]           375 ns            503 ns         6083 ns
Matrix shape =         [10,10]           500 ns            599 ns         5791 ns
Matrix shape =         [32,32]          1167 ns           1422 ns         5458 ns
Matrix shape =         [64,64]          3625 ns           3829 ns         4250 ns
Matrix shape =       [128,128]         19791 ns          19859 ns        20000 ns
Matrix shape =       [256,256]           116 µs            117 µs          127 µs
Matrix shape =       [512,512]          1036 µs           1223 µs         1440 µs
Matrix shape =     [1024,1024]          9264 µs          10377 µs        11553 µs
Matrix shape =     [2048,2048]         74874 µs          75085 µs        75451 µs
Matrix shape =     [4096,4096]           599 ms            607 ms          633 ms
Matrix shape =     [8192,8192]          4803 ms           4812 ms         4829 ms
Matrix shape =   [10000,10000]          8110 ms           8110 ms         8110 ms