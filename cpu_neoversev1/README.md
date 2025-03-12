# **Language System Containers**

The base OS is set to **Rocky Linux 9.4** for its ease of installing standard OS packages. Rocky Linux is an open-source Linux distribution that is compatible with Red Hat Enterprise Linux (RHEL). The environment is primarily built using **Spack**, with various packages installed into the virtual environment **virtual_fugaku**. For Spack, the [RIKEN-RCCS repository](https://github.com/RIKEN-RCCS/spack.git) is used.  

## **[GCC version 14.1.0](https://github.com/RIKEN-RCCS/singularity_defpack/tree/main/cpu_neoversev1/gcc_14.1.0)**

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

## **[ARM Compiler For Linux (acfl)](https://github.com/RIKEN-RCCS/singularity_defpack/tree/main/cpu_neoversev1/acfl_24.10.1)**

`acfl` is installed using **Spack**.  

Since `armpl` is already included in `acfl`, additional mathematical libraries like OpenBLAS are **not** installed.  

For profiling, `gperftools` and `go` (for `pprof`) are also installed:  

- gperftools  
- perf_helper  
- go  

## **[LLVM version 19.1.4](https://github.com/RIKEN-RCCS/singularity_defpack/tree/main/cpu_neoversev1/llvm_19.1.4)**

LLVM is downloaded and installed from the official LLVM repository.  Before installing LLVM, `ninja` and `cmake` are installed using the OS default `clang`.  Since `spack compiler find` cannot detect `flang`, it must be manually added to `compilers.yaml`.  

Installed mathematical libraries:  

- OpenBLAS  
- FFTW  
- ARMPL for GCC  

Profiling tools:  

- gperftools  
- perf_helper  
- go  

---

# **AI System Containers**

The base OS is set to **Ubuntu 24.04**, as it is widely used for AI-related tasks.
This environment is built using the **OS default compilers** for package installation.
Since AI environments typically require **Python modules**, instead of Spack to manage packages in a virtual environment, **wheels** are created to enhance reusability.  

## **[PyTorch version 2.5.0](https://github.com/RIKEN-RCCS/singularity_defpack/tree/main/cpu_neoversev1/pytorch_2.5.0)**

The ARM Compute Library (`acl`) is installed using the OS default compiler **GCC-14**.
`acl` is specifically optimized for **machine learning and computer vision** tasks, and works as a **backend for oneDNN** which is used by PyTorch.
During the `acl` build, **OpenMP is enabled**, and the **target architecture is set to `armv8.2-a`**.  

- **acl version 25.02**  

During the PyTorch build, the following settings are applied:  

- Enable **oneDNN** as the mathematical library  
- Set `acl` as the **oneDNN backend**  
- Enable **MPI and OpenMP support**  

**TorchVision** and **TorchAudio** are also installed.

## **[TensorFlow 2.17.1](https://github.com/RIKEN-RCCS/singularity_defpack/tree/main/cpu_neoversev1/tensorflow_2.17)**

Installation follows the official [Building from Source](https://www.tensorflow.org/install/source?hl=ja) guide.

> **Note**: On aarch64, a linking error workaround requires adding `--linkopt=-fuse-ld=bfd` to `bazel build`.

