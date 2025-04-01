# **Singularity Container Definition File and Performance**

## Definition Files

|  | virtual_fugaku | Compiler | AI | Misc. |
| ---- | ---- | ---- | ---- | ---- |
|  A64FX |  |  | [pytorch](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_a64fx/pytorch_2.5.0) : v2.5.0 <br> [Pytorch Install Guide](https://github.com/fujitsu/pytorch/wiki) | |
|  Neoverse V1 | spack-ver1-1 | [gcc](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/gcc_14.1.0) : v14.1.0 <br> [llvm](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/llvm_19.1.4) : v19.1.4 <br> [acl](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/acfl_24.10.1) : v24.10.1 |  [pytorch](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/pytorch_2.5.0) : v2.5.0 <br> [tensorflow](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/tensorflow_2.17) : v2.17.1 <br> [llama.cpp](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/llama.cpp) : b4953+OpenWebUI <br> [Megatron-DeepSpeed](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/Megatron-DeepSpeed)| |
|  Sapphire Rapids | [application](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/application) | [gcc](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/gcc_14.1.0) : v14.1.0 <br> [llvm](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/llvm_19.1.4) : v19.1.4 <br> [oneapi](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/oneapi_2025.0.1) | [pytorch](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/pytorch_2.5.0) : v2.5.0 <br> [tensorflow](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/tensorflow_2.17) : v2.17.1 <br> [llama.cpp](https://github.com/RIKEN-RCCS/singularity_defpack/tree/main/cpu_sapphirerapids/llama.cpp) : b4953+OpenWebUI | [btop](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/btop) <br> [glow](https://github.com/RIKEN-RCCS/singularity_defpack/tree/main/cpu_sapphirerapids/glow) <br> [gromacs](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/gromacs_2024.04) : v2024.04|
|  Zen4 |  | [aocc](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_zen4/aocc) |  | |
|  Nvidia GPU |  |  | [pytorch](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/gpu_nvidia/pytorch) : v2.2.0 <br> [Megatron-DeepSpeed](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/gpu_nvidia/Megatron-DeepSpeed) <br> [llama.cpp](https://github.com/RIKEN-RCCS/singularity_defpack/tree/main/gpu_nvidia/llama.cpp) : b4953+OpenWebUI<br> [llm-jp-eval](https://github.com/RIKEN-RCCS/singularity_defpack/tree/main/gpu_nvidia/llm-jp-eval) | [opensplat+colmap](https://github.com/RIKEN-RCCS/singularity_defpack/tree/main/gpu_nvidia/opensplat)|
|  AMD GPU    |  |  | [pytorch](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/gpu_amd/pytorch) : v2.1.2 <br> [Megatron-DeepSpeed](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/gpu_amd/Megatron-DeepSpeed)| |

### ARM A64FX (Fugaku)

 - PyTorch v2.5.0 : Tensors and Dynamic neural networks in Python with oneDNN v3.7.1, ACL v25.02.1, OpenBLAS v0.3.27

### ARM Neoverse V1 (Probably, V2 should be fine as well)

 - GCC v14.1.0 : The GNU Compiler Collection includes C, C++ and Fortran with *openblas, fftw, armpl*
 - LLVM v19.1.4 : Collection of modular and reusable compiler and toolchain technologies with *openblas, fftw, armpl*
 - Arm Compiler for Linux v24.10.1 : Suite of tools containing Arm C/C++ Compiler, Arm Fortran Compiler, Arm Performance Libraries
 - llama.cpp b4953 : LLM inference in C/C++
 - PyTorch v2.5.0 : Tensors and Dynamic neural networks in Python with *oneDNN v3.7.1, ACL v25.02.1, OpenBLAS v0.3.27*
 - TensorFlow v2.17.1 : An Open Source Machine Learning Framework with *oneDNN v3.2.1, ACL v23.05.1*
 - Megatron DeepSpeed : Ongoing research training transformer language models at scale, including: BERT & GPT-2

### Intel Sapphire Rapids

 - GCC v14.1.0 : The GNU Compiler Collection includes C, C++ and Fortran with *openblas, fftw*
 - LLVM v19.1.4 : Collection of modular and reusable compiler and toolchain technologies with *openblas, fftw*
 - oneAPI : Intel software development tool kit
 - llama.cpp b4953 : LLM inference in C/C++
 - PyTorch v2.5.0 : Tensors and Dynamic neural networks in Python
 - TensorFlow v2.17.1 : An Open Source Machine Learning Framework
 - HPC Applications : VirtualFugaku v1.1
 - btop : A monitor of resources
 - glow : Render markdown on the CLI, with pizzazz!
 - Gromacs 2024.04 : Molecular simulation toolkit

### AMD Zen4 (EPYC 9004 series)

 - AMD Optimizin C/C++ and Fortran compilers : AMD Optimizing C/C++ Compiler with *amd-aocl, aocl-compression, aocl-da, aocl-utils, aocl-libmem, aocl-sparse*

### NVIDIA GPU

 - PyTorch v2.2.0 : Tensors and Dynamic neural networks in Python
 - Megatron DeepSpeed : Ongoing research training transformer language models at scale, including: BERT & GPT-2
 - llama.cpp b4953 : LLM inference in C/C++
 - llm-jp-eval : Automatically evaluating large-scale Japanese language models across multiple datasets.
 - OpenSplat+colmap : Production-grade 3D gaussian splatting

### AMD GPU

 - PyTorch v2.1.2 : Tensors and Dynamic neural networks in Python
 - Megatron DeepSpeed : Ongoing research training transformer language models at scale, including: BERT & GPT-2

## How to Create Container

Assign the definition file to the environment variable NAME and create a container with the fakeroot option enabled.

```bash
#!/bin/sh

NAME=gcc

singularity -v build --force --fakeroot $NAME.sif $NAME.def > $NAME.log 2>&1
```

----

# **Extra: Installing SingularityCE v4.2.1 on Graviton 3E**

## Install Development packages.

```bash
sudo dnf groupinstall "Development Tools"
sudo dnf install -y \
   autoconf \
   automake \
   crun \
   cryptsetup \
   fuse \
   fuse3 \
   fuse3-devel \
   git \
   glib2-devel \
   libseccomp-devel \
   libtool \
   squashfs-tools \
   wget \
   zlib-devel
```

## Install Go v.1.22.6

```bash
$ export VERSION=1.22.6 OS=linux ARCH=arm64 && \
    wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && \
    sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz && \
    rm go$VERSION.$OS-$ARCH.tar.gz

$ echo 'export GOPATH=${HOME}/go' >> ~/.bashrc && \
    echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc && \
    source ~/.bashrc
```

## Install Singularity v.4.2.1

```bash
$ export VERSION=4.2.1 &&
    wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-ce-${VERSION}.tar.gz && \
    tar -xzf singularity-ce-${VERSION}.tar.gz && \
    cd singularity-ce-${VERSION}

$ ./mconfig && \
    make -C ./builddir && \
    sudo make -C ./builddir install
```

## Set timezone to Asia/Tokyo

```bash
$ sudo timedatectl set-timezone Asia/Tokyo
```

## Install extra tools

```
$ sudo dnf install -y gcc-gfortran
$ sudo dnf install -y perf
$ sudo dnf install -y numactl*
```

## Fakeroot setting

```
$ sudo singularity config fakeroot --add username
$ sudo singularity config fakeroot --enable username
```
