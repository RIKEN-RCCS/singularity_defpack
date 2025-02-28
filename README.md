
# Singularity Container Definition File Package

## Introduction

Definition files for creating a singularity container.

---

## Definition Files

### ARM Neoverse V1 (Probably, V2 should be fine as well)

 - GCC v14.1.0 : gcc.def (includes openblas, fftw, armpl)
 - LLVM v19.1.4 : llvm.def (includes openblas, fftw, armpl)
 - Arm Compiler for Linux : acfl.def (includes armpl)
 - llama.cpp : llama.cpp.def (requires GCC v14.1.0 container image)
 - PyTorch v2.5.0 : pytorch.def (with oneDNN v3.5.3, ACL v25.02)
 - TensorFlow v2.17 : tensorflow.def (without ACL)
 - Megatron DeepSpeed : Megatron-DeepSpeed.def (requires PyTorch v2.5.0 container image)

### Intel Sapphire Rapids

 - GCC v14.1.0 : gcc.def (includes openblas, fftw)
 - LLVM v19.1.4 : llvm.def (includes openblas, fftw)
 - oneAPI : oneapi.def (includes mkl)
 - PyTorch v2.5.0 : pytorch.def (with oneDNN v3.5.3) and (OpenBLAS v0.3.29)
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
