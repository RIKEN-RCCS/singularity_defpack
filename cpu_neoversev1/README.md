# **About Language System Containers**

The base OS is set to **Rocky Linux 9.4** for its ease of installing standard OS packages. Rocky Linux is an open-source Linux distribution that is compatible with Red Hat Enterprise Linux (RHEL). The environment is primarily built using **Spack**, with various packages installed into the virtual environment **virtual_fugaku**. For Spack, the [RIKEN-RCCS repository](https://github.com/RIKEN-RCCS/spack.git) is used.  

## **GCC version 14.1.0**

Since some applications require `gcc@8.5.0`, `gcc@14.1.0` is built using the following steps:  

1. Install `clang` (OS default)  
2. Use `clang` to build `gcc@8.5.0`  
3. Use `gcc@8.5.0` to build `gcc@14.1.0`  

The following mathematical libraries are installed, including both commonly used BLAS and FFT libraries as well as vendor-specific **ARM Performance Library for GCC (armpl)**:  

- OpenBLAS  
- FFTW  
- ARMPL for GCC  

For profiling purposes, **Google Performance Tools (gperftools)** and the visualization tool **`pprof`** (which requires `go`) are also installed:  

- gperftools  
- perf_helper  
- go  

### **Compilation Example**

In the example below, a compilation script `.compile.sh` is created using a **here document**, then passed to the container for execution.  Since library path have already been set in the container, additional path specifications are unnecessary.  

> **Note:** For usage details of `perf_helper`, refer to the [RIKEN-RCCS repository](https://github.com/RIKEN-RCCS/perf_helper)

```sh
#!/bin/sh

SIFFILE=./gcc.sif

cat << EOF > .compile.sh
rm -f a.out* *.a *.o *.mod

FC=gfortran
OMP="-fopenmp -fPIC"

\$FC -c \$OMP main.f90 -o main_f.o -J/usr/local/lib
\$FC -c \$OMP test.f90 -o test.o
\$FC \$OMP main_f.o test.o -lperf_helper -lprofiler -o a.out_f
EOF

singularity run ${SIFFILE} sh ./.compile.sh
```

## **ARM Compiler For Linux (acfl)**

`acfl` is installed using **Spack**.  

Since `armpl` is already included in `acfl`, additional mathematical libraries like OpenBLAS are **not** installed.  

For profiling, `gperftools` and `go` (for `pprof`) are also installed:  

- gperftools  
- perf_helper  
- go  

(*Compilation steps are the same as for GCC and are omitted here.*)  

---

## **LLVM version 19.1.4**

LLVM is downloaded and installed from the official LLVM repository.  Before installing LLVM, `ninja` and `cmake` are installed using the OS default `clang`.  Since `spack compiler find` cannot detect `flang`, it must be manually added to `compilers.yaml`.  

Installed mathematical libraries:  

- OpenBLAS  
- FFTW  
- ARMPL for GCC  

Profiling tools:  

- gperftools  
- perf_helper  
- go  

(*Compilation steps are the same as for GCC and are omitted here.*)  

---

# **About AI System Containers**

The base OS is set to **Ubuntu 24.04**, as it is widely used for AI-related tasks.
This environment is built using the **OS default compilers** for package installation.
Since AI environments typically require **Python modules**, instead of using Spack to manage packages in a virtual environment, **wheels** are created to enhance reusability.  

## **PyTorch version 2.5.0**

The ARM Compute Library (`acl`) is installed using the OS default compiler **GCC-14**.
`acl` is specifically optimized for **machine learning and computer vision** tasks, and works as a **backend for oneDNN** which is used by PyTorch.
During the `acl` build, **OpenMP is enabled**, and the **target architecture is set to `armv8.2-a-sve`**, enabling **SVE instructions**.  

- **acl version 25.02**  

During the PyTorch build, the following settings are applied:  

- Enable **oneDNN** as the mathematical library  
- Set `acl` as the **oneDNN backend**  
- Enable **MPI and OpenMP support**  

PyTorch is built along with **TorchVision** and **TorchAudio**.  

Each package is compiled into a **wheel** file and moved to `/opt/dist` before installation.
To reduce container size, cloned directories (`git clone` artifacts) are **removed** after building.  

- **PyTorch version 2.5.0**  
- **TorchVision**  
- **TorchAudio**  

### **Creating a PyTorch Environment**

Required packages for PyTorch execution are listed in `/opt/requirements.txt`.
To set up a PyTorch environment inside a virtual Python environment, execute the following commands inside the PyTorch container or within a new container referencing the PyTorch container as a **local image**:

```sh
python3 -m venv venv
. /opt/venv/bin/activate
python3 -m pip install -r /opt/requirements.txt
```

## **TensorFlow 2.17**

## TensorFlow Version 2.17

Since oneDNN and its backend ACL are enabled by default in TensorFlow, special installation steps are unnecessary.
Installation follows the official [Building from Source](https://www.tensorflow.org/install/source?hl=ja) guide.

> **Note**: On aarch64, a linking error workaround requires adding `--linkopt=-fuse-ld=bfd` to `bazel build`.

### Installed Dependencies
- **Bazel version**: 6.5.0
- **TensorFlow version**: 2.17

### Creating a TensorFlow Environment
Required packages for TensorFlow execution are listed in `/opt/requirements.txt`.
A TensorFlow environment can be quickly set up using the same procedure as PyTorch.
