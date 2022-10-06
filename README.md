# neurad
Target: Neural nets with auto-diff with BLAS as only dep.
WIP! unfinished, just started, ignore!

## Benchmarks

See [Benchmarks](bench/README.md) for matrix multiplication comparisation between
Intel OpenBlas, M1 Accelerate and M1 with Tensorflow (GPU) or Pytorch (MPS)

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

