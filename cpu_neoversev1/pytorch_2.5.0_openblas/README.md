# Package List

The container created on May 4, 2025, for **Graviton3E** contains the following packages.

```
Package            Version
------------------ ------------------
astunparse         1.6.3
attrs              25.1.0
certifi            2025.1.31
charset-normalizer 3.4.1
expecttest         0.3.0
filelock           3.17.0
fsspec             2025.2.0
hypothesis         6.127.3
idna               3.10
Jinja2             3.1.5
lintrunner         0.12.7
MarkupSafe         3.0.2
mpmath             1.3.0
networkx           3.4.2
ninja              1.11.1.3
numpy              2.2.3
optree             0.14.0
packaging          24.2
pillow             11.1.0
pip                24.0
pkgconfig          1.5.5
psutil             7.0.0
PyYAML             6.0.2
requests           2.32.3
SCons              4.8.1
setuptools         68.1.2
six                1.17.0
sortedcontainers   2.4.0
sympy              1.13.1
torch              2.5.0a0+git32f585d
torchaudio         2.6.0a0+c670ad8
torchvision        0.22.0a0+dcd1e42
types-dataclasses  0.6.6
typing_extensions  4.12.2
urllib3            2.3.0
wheel              0.42.0
```

# **Performance test**

Measure the execution time of matrix multiplication for a 1000x1000 matrix.
Theoretical flop count is **200G flops**.

```bash
import torch
import time

def benchmark_matmul():
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    start = time.time()
    for _ in range(100):
        z = torch.matmul(x, y)
    end = time.time()
    print(f"Execution time: {end - start:.5f} seconds")

benchmark_matmul()
```

**The execution time at **Graviton3E(hpc7g.16xlarge) using 64vCPU**

| # of threads | Execution time[sec] |
| ---- | ---- |
|  1 | 2.55920 |
|  2 | 1.29891 |
|  4 | 0.67570 |
|  8 | 0.37352 |
| 16 | 0.22710 |
| 32 | 0.17172 |
| 64 | 0.21237 |

This benchmark test is executed by the following script.

```bash
#!/bin/bash

for j in 1 2 4 8 16 32 52;do
END=`expr $j - 1`
export OMP_NUM_THREADS=$j
export GOMP_CPU_AFFINITY="0-$END"
for i in `seq 1 5`;do
  singularity run pytorch.sif python3 test.py
done
done
```
