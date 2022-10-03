# neurad
Target: Neural nets with auto-diff with BLAS as only dep.
WIP! unfinished, just started, ignore!

## Matrix benchmarks (does not generalise! internal testing!)

### Single precision

| dim / computer | Raspberry Pi 4 | i5-7500 CPU @ 3.40GHz | Mac mini M1 |
| -------------- | -------------- | --------------------- | ----------- |
|         [2,2]  |                |     1783 ns           |             |
|         [3,3]  |                |     1458 ns           |             |
|         [4,4]  |                |     1305 ns           |             |
|         [5,5]  |                |     1435 ns           |             |
|         [6,6]  |                |     3156 ns           |             |
|         [8,8]  |                |     1353 ns           |             |
|       [10,10]  |                |     1583 ns           |             |
|       [32,32]  |                |     7150 ns           |             |
|       [64,64]  |                |    27197 ns           |             |
|     [128,128]  |                |    42026 µs           |             |
|     [256,256]  |                |     8234 µs           |             |
|     [512,512]  |                |    10760 µs           |             |
|   [1024,1024]  |                |    25486 µs           |             |
|   [2048,2048]  |                |      137 ms           |             |
|   [4096,4096]  |                |      670 ms           |             |
|   [8192,8192]  |                |     4242 ms           |             |
| [10000,10000]  |                |     7535 ms           |             |


### Double precision

| dim / computer | Raspberry Pi 4 | i5-7500 CPU @ 3.40GHz | Mac mini M1 |
| -------------- | -------------- | --------------------- | ----------- |
|         [2,2]  |                |     1659 ns           |             |
|         [3,3]  |                |     1328 ns           |             |
|         [4,4]  |                |     1209 ns           |             |
|         [5,5]  |                |     1340 ns           |             |
|         [6,6]  |                |     3370 ns           |             |
|         [8,8]  |                |     1392 ns           |             |
|       [10,10]  |                |     1645 ns           |             |
|       [32,32]  |                |    14753 ns           |             |
|       [64,64]  |                |    54334 ns           |             |
|     [128,128]  |                |    24113 µs           |             |
|     [256,256]  |                |    17583 µs           |             |
|     [512,512]  |                |    24362 µs           |             |
|   [1024,1024]  |                |    17310 µs           |             |
|   [2048,2048]  |                |      111 ms           |             |
|   [4096,4096]  |                |     1376 ms           |             |
|   [8192,8192]  |                |     9835 ms           |             |
| [10000,10000]  |                |                       |             |


Matrix shape =           [2,2]          1659 ns
Matrix shape =           [3,3]          1328 ns
Matrix shape =           [4,4]          1209 ns
Matrix shape =           [5,5]          1340 ns
Matrix shape =           [6,6]          3370 ns
Matrix shape =           [8,8]          1392 ns
Matrix shape =         [10,10]          1645 ns
Matrix shape =         [32,32]         14753 ns
Matrix shape =         [64,64]         54334 ns
Matrix shape =       [128,128]         24113 µs
Matrix shape =       [256,256]         17583 µs
Matrix shape =       [512,512]         24362 µs
Matrix shape =     [1024,1024]         17310 µs
Matrix shape =     [2048,2048]           111 ms
Matrix shape =     [4096,4096]          1376 ms
Matrix shape =     [8192,8192]          9835 ms
