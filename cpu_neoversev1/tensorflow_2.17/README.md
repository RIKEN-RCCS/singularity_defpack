# **Package List**

The container created on May 4, 2025, for Graviton3E contains the following packages.

```
Package            Version
------------------ ------------------
absl-py                 2.1.0
astunparse              1.6.3
certifi                 2025.1.31
charset-normalizer      3.4.1
flatbuffers             25.2.10
gast                    0.6.0
google-pasta            0.2.0
grpcio                  1.70.0
h5py                    3.13.0
idna                    3.10
keras                   3.9.0
libclang                18.1.1
Markdown                3.7
markdown-it-py          3.0.0
MarkupSafe              3.0.2
mdurl                   0.1.2
ml-dtypes               0.4.1
mpi4py                  4.0.3
namex                   0.0.8
numpy                   1.26.4
opt_einsum              3.4.0
optree                  0.14.1
packaging               24.2
pip                     24.0
pkgconfig               1.5.5
protobuf                4.25.6
Pygments                2.17.2
PyYAML                  6.0.1
requests                2.32.3
rich                    13.9.4
setuptools              68.1.2
six                     1.17.0
tensorboard             2.17.1
tensorboard-data-server 0.7.2
tensorflow-cpu          2.17.1
termcolor               2.5.0
typing_extensions       4.12.2
urllib3                 2.3.0
Werkzeug                3.1.3
wheel                   0.42.0
wrapt                   1.17.2
```

# **Performance test**

Measure the execution time of matrix multiplication for a 1000x1000 matrix.
Theoretical flop count is **200G flops**.

```bash
import tensorflow as tf
import time

class MatMulBenchmark(tf.test.Benchmark):
    def benchmark_matmul(self):
        x = tf.random.normal([1000, 1000])
        y = tf.random.normal([1000, 1000])
        start = time.time()
        for _ in range(100):
            z = tf.matmul(x, y)
        end = time.time()
        print(f"Execution time: {end - start:.5f} seconds")

bench = MatMulBenchmark()
bench.benchmark_matmul()
```

**The execution time at Graviton3E(hpc7g.16xlarge) using 64vCPU is 0.08588sec.**
