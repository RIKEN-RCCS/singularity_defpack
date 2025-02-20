
# Singularity Container Definition File Package

## Introduction

Definition files for creating a singularity container.

---

## Definition Files

### Neoverse V1 (Probably, V2 should be fine as well)

 - GCC v14.1.0 : gcc.def (includes openblas, fftw, armpl)
 - LLVM v19.1.4 : llvm.def (includes openblas, fftw, armpl)
 - Arm Compiler for Linux : acfl.def (includes armpl)
 - llama.cpp : llama.cpp.def (requires GCC v14.1.0 localimage)
 - PyTorch v2.5.0 : pytorch.def (requires GCC v14.1.0 localimage)

### Sapphire Rapids

 - GCC v14.1.0 : gcc.def (includes openblas, fftw)
 - LLVM v19.1.4 : llvm.def (includes openblas, fftw)
 - oneAPI : oneapi.def (includes mkl)

---

## How to Create Container

Assign the definition file to the environment variable NAME and create a container with the fakeroot option enabled.

```bash
#!/bin/sh

NAME=gcc

singularity -v build --force --fakeroot $NAME.sif $NAME.def > $NAME.log 2>&1
```
