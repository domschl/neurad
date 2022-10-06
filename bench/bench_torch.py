import torch
import time
from ml_indie_tools.env_tools import MLEnv

menv = MLEnv(platform="pt")
mdev = torch.device("mps")

matDims = [
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    10000,
    12000,
    14000,
    20000,
]
matReps = [
    5000,
    1000,
    1000,
    1000,
    1000,
    1000,
    500,
    500,
    100,
    50,
    50,
    50,
    50,
    20,
    10,
    10,
    5,
    5,
    5,
    5,
]

def humanTime(tsec):
    if tsec>1:
        return f"{tsec:.3f} s"
    if tsec>0.001:
        return f"{tsec*1000:.3f} ms"
    return f"{tsec*1000000:.3f} µs"

results=[]
for i, dim in enumerate(matDims):
    min = -1
    for rep in range(1):
        m = torch.randn(dim, dim, device=mdev)
        p = m
        t0 = time.time()
        for _ in range(matReps[i]):
            p = p @ m
        print(p[0,0])  # Force caculations, introduces python overhead!
        t1 = time.time()
        d=(t1-t0)/matReps[i]
        if rep == 0 or d < min:
            min = d

    smin=humanTime(min)
    results+=[(dim,smin)]
    print(f"Dim: {dim}x{dim}: {smin}")
print("------------------------------------------------")
print(f"{menv.describe()}")
for dim, smin in results:
    print(f"Dim: {dim}x{dim}: {smin}")
    