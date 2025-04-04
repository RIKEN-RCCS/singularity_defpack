# **Setup gcc using SPACK**

## **definition file**

Install Rocky Linux 9.4 standard packages.

```bash
  dnf -y group install "Development Tools"
  dnf -y install wget which perf kernel-tools numactl-devel python3-devel
  dnf -y install llvm* clang*
  dnf -y install libxcrypt-compat
  dnf -y install cmake
  dnf clean all
```

Clone Spack from RIKEN RCCS GitHub, define spack mirror, activate the Spack environment, and detect installed compilers in Spack.

```bash
  cd /opt
  git clone https://github.com/RIKEN-RCCS/spack.git
  cd spack

cat << EOF > /opt/spack/etc/spack/mirrors.yaml
mirrors:
  build_cache_https:
    url: https://spack-mirror.r-ccs.riken.jp
    signed: false
EOF

  . /opt/spack/share/spack/setup-env.sh

  spack compiler find
```

Install gcc@8.5.0 using Clang and detect the compiler in Spack.

```bash
  # Install gcc@8.5.0
  spack install -j 32 gcc@8.5.0%clang +binutils
  spack load gcc@8.5.0
  spack compiler find
```

Install gcc@14.1.0 using gcc@8.5.0 and detect the compiler in Spack.

```bash
  # Install gcc@14.1.0
  spack install -j 32 gcc@14.1.0%gcc@8.5.0 +binutils
  spack load gcc@14.1.0
  spack compiler find
```

Output the list of detected compilers.

```bash
  spack config blame compilers
```

Create a Spack virtual environment `virtual_fugaku`.

```bash
  spack env create virtual_fugaku
```

Install the `openmpi`, fabric library `libfabric`, profiling tool `gperftools` and mathematical libraries `openblas`, `fftw`, `armpl for gcc`.

```bash
  spack -e virtual_fugaku install -j 32 --add libfabric fabrics=sockets,tcp,udp,shm,efa,verbs,ucx,mlx
  spack -e virtual_fugaku install -j 32 --add openmpi fabrics=auto
  spack -e virtual_fugaku install -j 32 --add gperftools%gcc@14.1.0
  spack -e virtual_fugaku install -j 32 --add openblas%gcc@14.1.0 threads=openmp
  spack -e virtual_fugaku install -j 32 --add fftw%gcc@14.1.0 +openmp
  spack -e virtual_fugaku install -j 32 --add armpl-gcc%gcc@14.1.0 threads=openmp
```

Clone the `go` from GitHub and install it into `/usr/local`.
`pprof` included in Go is required for visualizing profile data.

```bash
  # Go for pprof
  VERSION=1.23.5
  OS=linux ARCH=arm64
  wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz
  tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz
  rm go$VERSION.$OS-$ARCH.tar.gz
```

Clone the profiling tool `perf_helper` from GitHub, build it, and install libraries and other dependencies into `/usr/local`.

```bash
  # Perf Helper Library
  cd /opt
  git clone https://github.com/RIKEN-RCCS/perf_helper.git
  cd perf_helper
  make COMPILER=gcc
  cp perf_helper.h /usr/local/include
  cp libperf_helper.so /usr/local/lib
  cp perf_helper_mod.mod /usr/local/lib
```

To automatically activate the `virtual_fugaku` environment when a Singularity container is started, configure environment variable.

```bash
  spack env activate --sh virtual_fugaku >> $SINGULARITY_ENVIRONMENT
  spack gc -y
  spack clean --all
```

Set the library and other search paths in the Singularity container using the `%runscript` section.

```bash
%runscript
  ARMPL_DIR=`spack location -i armpl-gcc`/armpl_24.10_gcc
  OB_DIR=`spack location -i openblas`
  FFTW_DIR=`spack location -i fftw`
  GPERF_DIR=`spack location -i gperftools`
  VIEW_DIR=/opt/spack/var/spack/environments/virtual_fugaku/.spack-env/view

  export PATH=/usr/local/go/pkg/tool/linux_arm64:${PATH}:/usr/local/bin:${ARMPL_DIR}/bin:${GPERF_DIR}/bin:${FFTW_DIR}/bin
  export LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/lib:${VIEW_DIR}/lib:${VIEW_DIR}/lib64:${ARMPL_DIR}/lib:${GPERF_DIR}/lib:${OB_DIR}/lib:${FFTW_DIR}/lib
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib:${VIEW_DIR}/lib:${VIEW_DIR}/lib64:${ARMPL_DIR}/lib:${GPERF_DIR}/lib:${OB_DIR}/lib:${FFTW_DIR}/lib
  export C_INCLUDE_PATH=${C_INCLUDE_PATH}:/usr/local/include:${ARMPL_DIR}/include:${GPERF_DIR}/include/gperftools:${OB_DIR}/include:${FFTW_DIR}/include
  export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:/usr/local/include:${ARMPL_DIR}/include:${GPERF_DIR}/include/gperftools:${OB_DIR}/include:${FFTW_DIR}/include
  export INCLUDE_PATH=${INCLUDE_PATH}:/usr/local/include:${ARMPL_DIR}/include:${GPERF_DIR}/include/gperftools:${OB_DIR}/include:${FFTW_DIR}/include
```

----

# **Compilation Example**

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
