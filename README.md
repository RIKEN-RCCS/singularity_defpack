# **Singularity Container Definition File Package**

## Introduction

Definition files for creating a singularity container.

## Definition Files

|  | virtual_fugaku | Compiler | Misc. |
| ---- | ---- | ---- | ---- |
|  A64FX |  |  | [Pytorch Install Guide](https://github.com/fujitsu/pytorch/wiki) |
|  Neoverse V1 | [spack-ver1-1.def](https://github.com/RIKEN-RCCS/spack/blob/virtual_fugaku/spack-ver1-1.def) | [gcc.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/gcc_14.1.0) : v14.1.0 <br> [llvm.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/llvm_19.1.4) : v19.1.4 <br> [acl.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/acfl_24.10.1) |  [pytorch.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/pytorch_2.5.0) : v2.5.0 <br> [tensorflow.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/tensorflow_2.17) : v2.17 <br> [llama.cpp.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/llama.cpp) <br> [Megatron-DeepSpeed.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_neoversev1/Megatron-DeepSpeed)|
|  Sapphire Rapids | [application.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/application) | [gcc.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/gcc_14.1.0) : v14.1.0 <br> [llvm.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/llvm_19.1.4) : v19.1.4 <br> [oneapi.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/oneapi_2025.0.1) | [pytorch.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/pytorch_2.5.0) : v2.5.0 <br> [tensorflow.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_sapphirerapids/tensorflow_2.17) : v2.17 |
|  Zen4 |  | [aocc.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/cpu_zen4/aocc) |  |
|  Nvidia GPU |  |  | [pytorch.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/gpu_nvidia/pytorch) : v2.2.0 <br> [Megatron-DeepSpeed.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/gpu_nvidia/Megatron-DeepSpeed)|
|  AMD GPU    |  |  | [pytorch.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/gpu_amd/pytorch) : v2.1.2 <br> [Megatron-DeepSpeed.def](https://github.com/RIKEN-RCCS/singularity_defpack/blob/main/gpu_amd/Megatron-DeepSpeed)|

### ARM Neoverse V1 (Probably, V2 should be fine as well)

 - GCC v14.1.0 : gcc.def (includes openblas, fftw, armpl)
 - LLVM v19.1.4 : llvm.def (includes openblas, fftw, armpl)
 - Arm Compiler for Linux : acfl.def (includes armpl)
 - llama.cpp : llama.cpp.def (requires GCC v14.1.0 container image)
 - PyTorch v2.5.0 : pytorch.def (with oneDNN v3.7.1, ACL v25.02.1, OpenBLAS v0.3.27)
 - TensorFlow v2.17 : tensorflow.def (with oneDNN v3.2.1, ACL v23.05.1)
 - Megatron DeepSpeed : Megatron-DeepSpeed.def (requires PyTorch v2.5.0 container image)

### Intel Sapphire Rapids

 - GCC v14.1.0 : gcc.def (includes openblas, fftw)
 - LLVM v19.1.4 : llvm.def (includes openblas, fftw)
 - oneAPI : oneapi.def (includes mkl)
 - PyTorch v2.5.0 : pytorch.def
 - TensorFlow v2.17 : tensorflow.def
 - HPC Applications : application.def (based on [VirtualFugaku v1.1](https://github.com/RIKEN-RCCS/spack/blob/virtual_fugaku/spack-ver1-1.def))

### AMD Zen4 (EPYC 9004 series)

 - AMD Optimizin C/C++ and Fortran compilers : aocc.def (includes amd-aocl, aocl-compression, aocl-da, aocl-utils, aocl-libmem, aocl-sparse)

### NVIDIA GPU

 - PyTorch v2.2.0 : pytorch.def
 - Megatron DeepSpeed : Megatron-DeepSpeed.def (requires PyTorch container image)

### AMD GPU

 - PyTorch v2.1.2 : pytorch.def
 - Megatron DeepSpeed : Megatron-DeepSpeed.def (requires PyTorch container image)

---

## How to Create Container

Assign the definition file to the environment variable NAME and create a container with the fakeroot option enabled.

```bash
#!/bin/sh

NAME=gcc

singularity -v build --force --fakeroot $NAME.sif $NAME.def > $NAME.log 2>&1
```

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

## Set timezon to Asia/Tokyo

```bash
$ sudo timedatectl set-timezone Asia/Tokyo
```

## Install extra tools

```
$ sudo dnf install -y gcc-gfortran
$ sudo dnf install -y perf
$ sudo dnf install -y numactl*
```
