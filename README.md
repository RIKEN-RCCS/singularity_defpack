
# Singularity Container Definition File Package

## Introduction

Definition files for creating a singularity container.

---

## Definition Files

### Neoverse V1,V2

 - GCC v14.1.0 : gcc.def (includes openblas v0.3.28, fftw v3.3.10, armpl v24.10)
 - LLVM v19.1.4 : llvm.def (includes openblas v0.3.28, fftw v3.3.10, armpl v24.10)
 - Arm Compiler for Linux v24.10.1 : acfl.def (includes armpl v24.10)

### Sapphire Rapids

 - GCC v14.1.0 : gcc.def (includes openblas v0.3.27, fftw v3.3.10)
 - LLVM v19.1.4 : llvm.def (includes openblas v0.3.28, fftw v3.3.10)
 - oneAPI v2025.0.1 : oneapi.def (includes mkl v2024.2.2)

---

## How to Create Container

Assign the definition file to the environment variable NAME and create a container with the fakeroot option enabled.

```bash
#!/bin/sh

NAME=gcc

singularity -v build --force --fakeroot $NAME.sif $NAME.def > $NAME.log 2>&1
```
