# Benchmarks

OpenBlas, Accelerate, Pytorch and Tensorflow with various M1 accelerators.

## Matrix multiplication benchmarks BLAS

Benchmark of matrix multiplication of given dimension
of two matrices with random normal intialization. Measured after warm-up, minimum time given.

### Single precision

| dim / computer | Raspberry Pi 4 | i5-7500 CPU @ 3.40GHz | Mac mini M1 | Mac Studio M1 Max |
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

| dim / computer | Raspberry Pi 4 | i5-7500 CPU @ 3.40GHz | Mac mini M1 | Mac Studio M1 Max |
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

## M1 Accelerate BLAS vs Torch MPS on M1 vs Tensorflow GPU on M1

Note: the benchmarks for Tensorflow and Pytorch forced calculation by accessing one
element of the result of the matrix multiplication. 

This introduces considerable python-overhead for columns marked with `1x` (one single
matrix multiplication). So M1 GPUs are no more 'unified' than other external graphics
cards.

The columns `nx` measure time for one multiplication within several fused matrix 
multiplications.

Performance between Torch MPS and Tensorflow GPU on M1 seems pretty much identical.

If fusing is not possible, or for matrix dims < 1000, Accelerate is the better choice.


| dim / computer | M1 Accel | TF Metal 1x | TF Metal nx |Torch MPS 1x | Torch MPS nx |
| -------------- | -------- | ----------- | ----------- | ----------- | ------------ |
| [2,2]          | 291 ns   |  560.760 µs |  48.073 µs  |  2.422 ms  |  38.831 µs   |
| [3,3]          | 333 ns   |  544.071 µs |  49.844 µs  |  3.179 ms  |  37.671 µs   |
| [4,4]          | 291 ns   |  550.032 µs |  50.068 µs  |  3.177 ms  |  37.754 µs   |
| [5,5]          | 291 ns   |  575.066 µs |  50.373 µs  |  3.070 ms  |  37.773 µs   |
| [6,6]          | 333 ns   |  591.040 µs |  49.852 µs  |  3.040 ms  |  37.181 µs   |
| [8,8]          | 333 ns   |  557.661 µs |  49.926 µs  |  3.155 ms  |  36.961 µs   |
| [10,10]        | 583 ns   |  613.928 µs |  55.102 µs  |  3.233 ms  |  41.356 µs   |
| [32,32]        | 791 ns   |  535.965 µs |  53.460 µs  |  3.126 ms  |  41.022 µs  |
| [64,64]        | 1833 ns  |  543.118 µs |  70.541 µs  |  3.222 ms  |  65.658 µs  |
| [128,128]      | 7125 ns  |  607.967 µs |  97.060 µs  |  3.244 ms  |  96.602 µs  |
| [256,256]      | 37500 ns |  766.277 µs |  159.478 µs |  3.164 ms  |  145.297 µs |
| [512,512]      | 249 µs   |  1.358 ms   |  440.722 µs |  3.628 ms  |  429.482 µs |
| [1024,1024]    | 2357 µs  |  5.490 ms   |  1.246 ms   |  7.052 ms  |  1.275 ms   |
| [2048,2048]    | 21963 µs |  15.224 ms  |  10.392 ms  |  13.426 ms |  14.085 ms  |
| [4096,4096]    | 164 ms   |  110.070 ms |  97.281 ms  |  88.938 ms |  79.480 ms  |
| [8192,8192]    | 1296 ms  |  741.490 ms |  771.039 ms |  625.143 m |  648.684 ms |
| [10000,10000]  | 2262 ms  |  1.742 s    |  1.572 s    |  1.403 s   |  1.584 s    |
| [12000,12000]  | 3730 ms  |  2.783 s    |  2.607 s    |  2.558 s   |  2.349 s    |
| [14000,14000]  | 6244 ms  |  4.334 s    |  4.253 s    |  3.958 s   |  4.141 s    |
| [20000,20000]  |          |  11.705 s   |  11.703 s   |  11.744 s  |  13.375 s   |
 
- OS: Darwin Ventura 13 beta 10, Python: 3.10.6 (Conda) Tensorflow:  2.10.0, GPU: METAL
- OS: Darwin Ventura 13 beta 10, Python: 3.10.6 (Conda) Pytorch: 1.13.0.dev20221004, GPU: MPS Metal accelerator
