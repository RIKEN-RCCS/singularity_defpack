# **Building PyTorch from source**

## **definition file**

Install Ubuntu standard packages and scons.
Set the installed gcc-14 and g++-14 as the default compilers.

```bash
  apt-get update && apt-get install -y --no-install-recommends build-essential
  apt-get install -y python3-dev python3-pip python3-pkgconfig python3-venv libhdf5-dev
  apt-get install -y gcc-14 g++-14
  apt-get install -y openmpi-bin openmpi-common openmpi-doc libopenmpi-dev libopenmpi3t64
  apt-get install -y wget git patchelf unzip cmake
  apt-get install -y libgoogle-perftools4t64

  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100

  gcc --version
  g++ --version

  python3 -m pip install --break-system-packages scons
```

Clone the Arm Compute Library (ACL) source code from Git and checkout release **25.02.1**.
Release **25.02.1** includes stateless support for GEMM kernels, which improves performance.
Install ACL with neon, openmp, multi_isa, and fixed_format_kernels enabled.

```bash
  # Build Arm Compute Library (ACL)
  ACL_VER=25.02.1
  cd /opt
  git clone https://github.com/ARM-software/ComputeLibrary.git
  cd ComputeLibrary
  git checkout v${ACL_VER}
  scons Werror=0 -j32 benchmark_tests=1 embed_kernels=0 debug=0 neon=1 opencl=0 os=linux openmp=1 cppthreads=0 arch=armv8.2-a multi_isa=1 build=native fixed_format_kernels=1
```

Clone the OpenBLAS source code from Git and checkout release **0.3.27**.
OpenBLAS improves performance for GEMM kernels where ACL is not utilized.
Install OpenBLAS with openmp enabled.

```bash
  # Build OpenBLAS
  OB_VER=0.3.27
  cd /opt
  git clone https://github.com/OpenMathLib/OpenBLAS.git
  cd OpenBLAS
  git checkout v${OB_VER}
  mkdir build
  cd build
  cmake -DUSE_OPENMP=ON -DCORE=ARMV8SVE -DTARGET=ARMV8_2 ..
  make -j32
  make install
  rm -rf /opt/OpenBLAS
```

Define the root directory of ACL and the use of MKLDNN, MKLDNN_ACL, MPI, and OpenMP for PyTorch installation.
To prevent job failures due to insufficient memory during parallel compilation, limit the maximum number of parallel processes to 4.

```bash
  export ACL_ROOT_DIR=/opt/ComputeLibrary
  export USE_MKL=ON USE_MKLDNN=ON USE_MKLDNN_ACL=ON USE_CUDA=0 USE_MPI=1 USE_OPENMP=1
  export MAX_JOBS=4
```

Create the `/opt/dist` directory to store the wheels that will be built later.

```bash
  mkdir /opt/dist
```

Clone the PyTorch source from Git and checkout release **2.5.0**.
Also, checkout release **3.7.1** of oneDNN.
Create a wheel according to the above declarations, move the wheel file to `/opt/dist`, and install it.
To reduce the container file size, delete the cloned directories.

```bash
  # Build PyTorch from the tip of the tree
  PYTORCH_VER=2.5.0
  ONEDNN_VER=3.7.1
  cd /opt
  git clone --recursive http://github.com/pytorch/pytorch
  cd pytorch
  git checkout v${PYTORCH_VER}
  git submodule sync
  git submodule update --init --recursive
  cd third_party/ideep/mkl-dnn
  git checkout v${ONEDNN_VER}
  cd ../../..
  python3 -m pip install --break-system-packages -r requirements.txt
  python3 setup.py bdist_wheel
  FILE=`cd ./dist; ls -1 *.whl | head -1`
  mv ./dist/${FILE} /opt/dist
  python3 -m pip install --break-system-packages /opt/dist/${FILE}
  cd /opt; rm -rf /opt/pytorch
```

Clone the source of TorchVision and TorchAudio from Git.
Create a wheel, move the wheel file to `/opt/dist`, and install it.
To reduce the container file size, delete the cloned directories.
Since the maximum number of parallel processes of TorchAudio cannot be controlled with MAX_JOB, specify CMAKE_BUILD_PARALLEL_LEVEL instead and limit the maximum number of parallel processes to 2.

```bash
  # Build TorchVision
  cd /opt
  git clone https://github.com/pytorch/vision
  cd vision
  python3 setup.py bdist_wheel
  FILE=`cd ./dist; ls -1 *.whl | head -1`
  mv ./dist/${FILE} /opt/dist
  python3 -m pip install --break-system-packages /opt/dist/${FILE}
  cd /opt; rm -rf /opt/vision

  # Build TorchAudio
  cd /opt
  export CMAKE_BUILD_PARALLEL_LEVEL=2
  git clone https://github.com/pytorch/audio
  cd audio
  python3 setup.py bdist_wheel
  FILE=`cd ./dist; ls -1 *.whl | head -1`
  mv ./dist/${FILE} /opt/dist
  python3 -m pip install --break-system-packages /opt/dist/${FILE}
  cd /opt; rm -rf /opt/audio
```

Freeze the environment for the reuse of the PyTorch environment.

```bash
  python3 -m pip freeze > /opt/requirements.txt
```

## Package List

The container created on May 9, 2025, for Graviton3E contains the following packages.

```
Package            Version
------------------ ------------------
astunparse         1.6.3
attrs              25.2.0
certifi            2025.1.31
charset-normalizer 3.4.1
expecttest         0.3.0
filelock           3.17.0
fsspec             2025.3.0
hypothesis         6.129.0
idna               3.10
Jinja2             3.1.6
lintrunner         0.12.7
MarkupSafe         3.0.2
mpmath             1.3.0
networkx           3.4.2
ninja              1.11.1.3
numpy              2.2.3
optree             0.14.1
packaging          24.2
pillow             11.1.0
pip                24.0
pkgconfig          1.5.5
psutil             7.0.0
PyYAML             6.0.2
requests           2.32.3
SCons              4.9.0
setuptools         68.1.2
six                1.17.0
sortedcontainers   2.4.0
sympy              1.13.1
torch              2.5.0a0+git32f585d
torchaudio         2.6.0a0+c670ad8
torchvision        0.22.0a0+124dfa4
types-dataclasses  0.6.6
typing_extensions  4.12.2
urllib3            2.3.0
wheel              0.42.0
``

# **Performance test**

## **MATMUL**

Measure the execution time of matrix multiplication for a 1000x1000 matrix.
Theoretical flop count is **200G flops**.

```python
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

**The execution time at Fugaku(A64FX) using 48core**

> note: tcmalloc refers to pre-loadiing libtcmalloc.so.4 befor python3.

| Mode | Execution time[sec] | GFlop/s |
| ---- | ----: | ----: |
| FP32 | 0.3097 | 645.8 |
| FP32+tcmalloc | 0.3022 | 661.9 |


## **INFERENCE**

Measure the execution time of inference performance.
This script is quoted from the [PyTorch tutorial](https://pytorch.org/tutorials/recipes/inference_tuning_on_aws_graviton.html).

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

device = ("cpu")
print(f"Using {device} device")

class MyNeuralNetwork(nn.Module):
  def __init__(self):
      super().__init__()
      self.flatten = nn.Flatten()
      self.linear_relu_stack = nn.Sequential(
          nn.Linear(4096, 4096),
          nn.ReLU(),
          nn.Linear(4096, 11008),
          nn.ReLU(),
          nn.Linear(11008, 10),
      )

  def forward(self, x):
      x = self.flatten(x)
      logits = self.linear_relu_stack(x)
      return logits

model = MyNeuralNetwork().to(device)

X = torch.rand(256, 64, 64, device=device)

with torch.set_grad_enabled(False):
    for _ in range(50):
        model(X) #Warmup
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with record_function("mymodel_inference"):
            for _ in range(100):
                model(X)

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```

**The execution time at Fugaku(A64FX) using 48core**

| Mode | Execution time[sec] |
| ---- | ----: |
| FP32 | 14.58 |
| FP32+tcmalloc | 10.39 |

Following is the profiler output with the default PyTorch configuration:

```
Using cpu device
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
           aten::addmm        80.22%       11.698s        88.17%       12.858s      42.859ms           300
       aten::clamp_min         8.33%        1.215s         8.33%        1.215s       6.073ms           200
           aten::copy_         7.93%        1.157s         7.93%        1.157s       3.856ms           300
     mymodel_inference         3.32%     484.873ms       100.00%       14.583s       14.583s             1
          aten::linear         0.06%       8.957ms        88.28%       12.875s      42.916ms           300
         aten::flatten         0.03%       4.520ms         0.04%       6.154ms      61.536us           100
               aten::t         0.02%       3.632ms         0.05%       7.919ms      26.398us           300
            aten::relu         0.02%       2.916ms         8.35%        1.217s       6.087ms           200
      aten::as_strided         0.02%       2.675ms         0.02%       2.675ms       4.458us           600
       aten::transpose         0.02%       2.409ms         0.03%       4.288ms      14.293us           300
            aten::view         0.01%       1.634ms         0.01%       1.634ms      16.336us           100
          aten::expand         0.01%       1.579ms         0.02%       2.375ms       7.917us           300
    aten::resolve_conj         0.00%     473.552us         0.00%     473.552us       0.789us           600
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 14.583s
```

Following is the profiler output with the FP32+tcmalloc:

```
$ LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtcmalloc.so.4 python3 infer.py
```

```
Using cpu device
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
           aten::addmm        77.25%        8.030s        87.97%        9.144s      30.479ms           300
       aten::clamp_min        11.29%        1.174s        11.29%        1.174s       5.870ms           200
           aten::copy_        10.70%        1.112s        10.70%        1.112s       3.706ms           300
     mymodel_inference         0.56%      58.432ms       100.00%       10.394s       10.394s             1
          aten::linear         0.04%       4.501ms        88.09%        9.156s      30.521ms           300
               aten::t         0.04%       3.702ms         0.08%       8.046ms      26.821us           300
            aten::relu         0.03%       2.865ms        11.32%        1.177s       5.884ms           200
      aten::as_strided         0.02%       2.561ms         0.02%       2.561ms       4.268us           600
       aten::transpose         0.02%       2.318ms         0.04%       4.345ms      14.482us           300
            aten::view         0.02%       1.587ms         0.02%       1.587ms      15.874us           100
          aten::expand         0.01%       1.409ms         0.02%       1.943ms       6.478us           300
         aten::flatten         0.01%     911.614us         0.02%       2.499ms      24.990us           100
    aten::resolve_conj         0.00%     467.842us         0.00%     467.842us       0.780us           600
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 10.394s
```
