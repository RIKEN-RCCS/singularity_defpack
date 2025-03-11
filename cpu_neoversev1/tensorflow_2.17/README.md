# **Building TensorFlow from source**

## **definition file**

Install Ubuntu standard packages and scons.

```bash
  apt-get update && apt-get install -y --no-install-recommends build-essential
  apt-get install -y python3-dev python3-pip python3-pkgconfig python3-venv libhdf5-dev
  apt-get install -y llvm-17 clang-17 libomp-17-dev libomp-17-doc
  apt-get install -y openmpi-bin openmpi-common openmpi-doc libopenmpi-dev libopenmpi3t64
  apt-get install -y wget git patchelf unzip cmake
  apt-get install -y libgoogle-perftools4t64

  python3 -m pip install --break-system-packages mpi4py
```

Create the `/opt/dist` directory to store the wheels that will be built later.

```bash
  mkdir /opt/dist
```

Setup bazel v6.5.0.

```bash
  # Setup Bazel v6.5.0
  wget https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-arm64
  mv bazelisk-linux-arm64 /usr/local/bin/bazel
  chmod +x /usr/local/bin/bazel
  export USE_BAZEL_VERSION=6.5.0
  /usr/local/bin/bazel version
```

Clone the TensorFlow source from Git and checkout release **2.17.1**.

> **note1:** To address an error occurring with Clang 17, I have added a patch to compute_library.patch that includes <string> in IPrinter.h. The official patch will be applied from version v2.19.0-rc0.
> **note2:** The ACL version is specified as v23.5.0 in ./tensorflow/workspace2.bzl and ./third_party/xla/tsl_workspace2.bzl. Since Stateless GEMM is applied from v25.02.1, the performance improvement with oneDNN may be limited depending on the problem size.

Create a wheel according to the above declarations, move the wheel file to `/opt/dist`, and install it.
To reduce the container file size, delete the cloned directories.

```bash
  # Build TensorFlow from the tip of the tree
  cd /opt
  git clone https://github.com/tensorflow/tensorflow.git
  cd tensorflow
  git checkout v2.17.1

  export TF_PYTHON_VERSION=3.12

  yes "" | python3 configure.py

  FILE=$(find /tmp/build-temp-* | grep 'tmp/compute_library.patch' )
  cp $FILE /opt/tensorflow/third_party/compute_library
  bazel build --config=opt --copt=-march=native --config=mkl_aarch64_threadpool --linkopt=-fuse-ld=bfd //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow_cpu
  FILE=`ls -1 bazel-bin/tensorflow/tools/pip_package/wheel_house`
  mv bazel-bin/tensorflow/tools/pip_package/wheel_house/$FILE /opt/dist
  python3 -m pip install --break-system-packages /opt/dist/${FILE}
  cd /opt && rm -rf /opt/tensorflow
```

Freeze the environment for the reuse of the PyTorch environment.

```bash
  python3 -m pip freeze > /opt/requirements.txt
```

## **Package List**

The container created on May 11, 2025, for Graviton3E contains the following packages.

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
grpcio                  1.71.0
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

## **MATMUL**
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

**The execution time at Graviton3E(hpc7g.16xlarge) using 64vCPU**

> note: tcmalloc refers to pre-loadiing libtcmalloc.so.4 befor python3.

| Mode | Execution time[sec] | GFlop/s |
| ---- | ----: | ----: |
| oneDNN=OFF | 0.07938 | 2,519.5 |
| oneDNN=ON, FP32 | 0.07678 | 2,604.8 |
| oneDNN=ON, BF16 | 0.06026 | 3,319.0 |
| oneDNN=ON, BF16, tcmalloc | 0.05867 | 3,408.9 |


## **INFERENCE**

Measure the execution time of inference performance.
This script is quoted from the [PyTorch tutorial](https://pytorch.org/tutorials/recipes/inference_tuning_on_aws_graviton.html).

```python
import tensorflow as tf
import time

print(f"Using CPU device")

class MyNeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.linear_relu_stack = tf.keras.Sequential([
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dense(11008, activation="relu"),
            tf.keras.layers.Dense(10)
        ])

    def call(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

model = MyNeuralNetwork()

X = tf.random.uniform((256, 64, 64))

for _ in range(50):
    _ = model(X, training=False)

start = time.time()
for _ in range(100):
    _ = model(X, training=False)
end = time.time()
print(f"Execution time: {end - start:.5f} seconds")
```

**The execution time at Graviton3E(hpc7g.16xlarge) using 64vCPU**

> note: tcmalloc refers to pre-loadiing libtcmalloc.so.4 befor python3.

| Mode | Execution time[sec] |
| ---- | ----: |
| oneDNN=OFF | 1.74958 |
| oneDNN=ON, FP32 | 2.84469 |
| oneDNN=ON, FP16 | 2.04007 |
| oneDNN=ON, BF16, tcmalloc | 1.3516 |
