
# Singularity Container Definition File Package

## Introduction

Definition files for creating a singularity container.

---

## Definition Files

### Neoverse V1,V2

 - GCC v14.1.0 : gcc.def
 - LLVM v19.1.4 : llvm.def
 - Arm Compiler for Linux v24.10.1 : acfl.def

### Sapphire Rapids

 - GCC v14.1.0 : gcc.def
 - LLVM v19.1.4 : llvm.def
 - oneAPI v2025.0.1 : oneapi.def

---

## How to Create Container

Assign the definition file to the environment variable NAME and create a container with the fakeroot option enabled.

```bash
#!/bin/sh

NAME=gcc

singularity -v build --force --fakeroot $NAME.sif $NAME.def > $NAME.log 2>&1
```
