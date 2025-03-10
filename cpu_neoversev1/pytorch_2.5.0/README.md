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

```bash
  export ACL_ROOT_DIR=/opt/ComputeLibrary
  export USE_MKL=ON USE_MKLDNN=ON USE_MKLDNN_ACL=ON USE_CUDA=0 USE_MPI=1 USE_OPENMP=1
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
attrs              25.1.0
certifi            2025.1.31
charset-normalizer 3.4.1
expecttest         0.3.0
filelock           3.17.0
fsspec             2025.3.0
hypothesis         6.127.9
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
```

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

**The execution time at Graviton3E(hpc7g.16xlarge) using 64vCPU**

| # of threads | Execution time[sec] | GFlop/s |
| ---- | ----: | ----: |
|  1 | 2.54832 |    78.5 |
|  2 | 1.29556 |   154.4 |
|  4 | 0.67472 |   296.4 |
|  8 | 0.37038 |   540.0 |
| 16 | 0.22970 |   870.7 |
| 32 | 0.18053 | 1,107.8 |
| 64 | 0.20754 |   963.7 |


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

**The execution time at Graviton3E(hpc7g.16xlarge) using 64vCPU**

| Mode | Execution time[sec] |
| ---- | ----: |
| FP32 | 2.127 |
| BF16 | 1.236 |
| BF16+tcmalloc | 0.672 |

Following is the profiler output with the default PyTorch configuration:

```
Using cpu device
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
           aten::addmm        92.68%        1.971s        94.16%        2.002s       6.675ms           300
       aten::clamp_min         2.97%      63.117ms         2.97%      63.117ms     315.585us           200
     mymodel_inference         2.34%      49.772ms       100.00%        2.127s        2.127s             1
           aten::copy_         1.43%      30.491ms         1.43%      30.491ms     101.635us           300
          aten::linear         0.14%       3.006ms        94.52%        2.010s       6.700ms           300
               aten::t         0.10%       2.040ms         0.21%       4.493ms      14.975us           300
            aten::relu         0.07%       1.505ms         3.04%      64.622ms     323.110us           200
      aten::as_strided         0.07%       1.408ms         0.07%       1.408ms       2.347us           600
         aten::flatten         0.06%       1.378ms         0.10%       2.228ms      22.283us           100
       aten::transpose         0.06%       1.305ms         0.12%       2.453ms       8.175us           300
            aten::view         0.04%     850.213us         0.04%     850.213us       8.502us           100
          aten::expand         0.03%     659.868us         0.04%     919.915us       3.066us           300
    aten::resolve_conj         0.01%     227.359us         0.01%     227.359us       0.379us           600
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.127s
```

Following is the profiler output with the bfload16:

```
$ export DNNL_DEFAULT_FPMATH_MODE=BF16
```

```
Using cpu device
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
           aten::addmm        87.83%        1.085s        90.34%        1.116s       3.721ms           300
       aten::clamp_min         4.99%      61.665ms         4.99%      61.665ms     308.323us           200
     mymodel_inference         3.90%      48.198ms       100.00%        1.236s        1.236s             1
           aten::copy_         2.41%      29.823ms         2.41%      29.823ms      99.411us           300
          aten::linear         0.17%       2.153ms        90.87%        1.123s       3.743ms           300
               aten::t         0.16%       2.000ms         0.36%       4.451ms      14.836us           300
            aten::relu         0.12%       1.469ms         5.11%      63.133ms     315.667us           200
      aten::as_strided         0.11%       1.407ms         0.11%       1.407ms       2.345us           600
       aten::transpose         0.11%       1.319ms         0.20%       2.450ms       8.168us           300
            aten::view         0.08%     952.788us         0.08%     952.788us       9.528us           100
          aten::expand         0.05%     668.986us         0.08%     945.287us       3.151us           300
         aten::flatten         0.04%     487.127us         0.12%       1.440ms      14.399us           100
    aten::resolve_conj         0.02%     231.992us         0.02%     231.992us       0.387us           600
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.236s
```

Following is the profiler output with the bfload16 with tcmalloc:

```
$ export DNNL_DEFAULT_FPMATH_MODE=BF16
$ LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtcmalloc.so.4 python3 infer.py
```

```
Using cpu device
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
           aten::addmm        89.03%     598.708ms        91.75%     617.012ms       2.057ms           300
       aten::clamp_min         3.83%      25.748ms         3.83%      25.748ms     128.742us           200
     mymodel_inference         3.14%      21.118ms       100.00%     672.507ms     672.507ms             1
           aten::copy_         2.57%      17.290ms         2.57%      17.290ms      57.633us           300
          aten::linear         0.29%       1.959ms        92.66%     623.138ms       2.077ms           300
               aten::t         0.27%       1.837ms         0.62%       4.166ms      13.887us           300
       aten::transpose         0.19%       1.266ms         0.35%       2.329ms       7.763us           300
      aten::as_strided         0.18%       1.238ms         0.18%       1.238ms       2.064us           600
            aten::relu         0.18%       1.196ms         4.01%      26.944ms     134.721us           200
            aten::view         0.13%     866.526us         0.13%     866.526us       8.665us           100
          aten::expand         0.09%     604.474us         0.12%     779.890us       2.600us           300
         aten::flatten         0.07%     440.311us         0.19%       1.307ms      13.068us           100
    aten::resolve_conj         0.03%     234.495us         0.03%     234.495us       0.391us           600
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 672.507ms
```
