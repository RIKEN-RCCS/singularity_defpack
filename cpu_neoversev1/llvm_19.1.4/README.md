# **Setup llvm using SPACK**

## **definition file**

Install Rocky Linux 9.4 standard packages.

```bash
  dnf -y group install "Development Tools"
  dnf -y install wget which perf kernel-tools numactl-devel python3-devel
  dnf -y install llvm* clang*
  dnf -y install gcc-gfortran
  dnf -y install libxcrypt-compat
  dnf -y install cmake
  dnf clean all
```

Clone Spack from RIKEN RCCS GitHub, activate the Spack environment, and detect installed compilers in Spack.

```bash
  cd /opt
  git clone https://github.com/RIKEN-RCCS/spack.git

cat << EOF > /opt/spack/etc/spack/mirrors.yaml
mirrors:
  build_cache_https:
    url: https://spack-mirror.r-ccs.riken.jp
    signed: false
EOF

  . /opt/spack/share/spack/setup-env.sh

  spack compiler find
```

Create a Spack virtual environment `virtual_fugaku`.

```bash
  spack env create virtual_fugaku
```

Install ninja required for LLVM installation.

```bash
  spack -e virtual_fugaku install -j 32 --add ninja%clang
  spack load ninja
```

Clone LLVM from GitHub, checkout v19.1.4, and install into `/usr/local/llvm-19.1.4`.

```bash
  cd /opt
  git clone https://github.com/llvm/llvm-project.git
  cd llvm-project
  git checkout llvmorg-19.1.4
  mkdir build
  cd build
  cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local/llvm-19.1.4 \
    -DLLVM_ENABLE_PROJECTS="bolt;clang;clang-tools-extra;compiler-rt;cross-project-tests;libclc;lld;mlir;openmp;polly;pstl;flang" \
    ../llvm
  ninja -j24
  ninja install
  cd /opt
  rm -rf /opt/llvm-project
```

The LLVM Fortran compiler is installed as `flang-new`, so create a symbolic link `flang`.
Since Spack does not recognize `flang` as the Fortran compiler by default, edit the `compilers.yaml` file to add `flang`.

```bash
  cd /usr/local/llvm-19.1.4/bin
  ln -s flang-new flang

  LLVM=/usr/local/llvm-19.1.4
  export PATH=${PATH}:${LLVM}/bin
  spack compiler find
  FILE=/root/.spack/linux/compilers.yaml
sed -i '/spec: clang@=19.1.4/,/flags:/{
    /f77:/s/null/\/usr\/local\/llvm-19.1.4\/bin\/flang/
    /fc:/s/null/\/usr\/local\/llvm-19.1.4\/bin\/flang/
}' $FILE
```

Output the list of detected compilers.

```bash
  spack config blame compilers
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
  LLVM=/usr/local/llvm-19.1.4
  export PATH=/usr/local/go/pkg/tool/linux_arm64:${LLVM}/bin:/usr/local/bin:${PATH}
  export LIBRARY_PATH=${LLVM}/lib:${LLVM}/lib/aarch64-unknown-linux-gnu:${LLVM}/lib/clang/19/lib/aarch64-unknown-linux-gnu:/usr/local/lib:${LIBRARY_PATH}
  export LD_LIBRARY_PATH=${LLVM}/lib:${LLVM}/lib/aarch64-unknown-linux-gnu:${LLVM}/lib/clang/19/lib/aarch64-unknown-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}
  export C_INCLUDE_PATH=${LLVM}/include:${LLVM}/lib/clang/19/include:/usr/local/include:${C_INCLUDE_PATH}
  export CPLUS_INCLUDE_PATH=${LLVM}/include:${LLVM}/lib/clang/19/include:/usr/local/include:${CPLUS_INCLUDE_PATH}
  export INCLUDE_PATH=${LLVM}/include:${LLVM}/lib/clang/19/include:/usr/local/include:${INCLUDE_PATH}

  # Perf Helper Library
  cd /opt
  git clone https://github.com/RIKEN-RCCS/perf_helper.git
  cd perf_helper
  make COMPILER=llvm
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
  LLVM=/usr/local/llvm-19.1.4

  export PATH=/usr/local/go/pkg/tool/linux_arm64:${LLVM}/bin:/usr/local/bin:${PATH}:${ARMPL_DIR}/bin:${GPERF_DIR}/bin:${FFTW_DIR}/bin
  export LIBRARY_PATH=${LLVM}/lib:${LLVM}/lib/aarch64-unknown-linux-gnu:${LLVM}/lib/clang/19/lib/aarch64-unknown-linux-gnu:/usr/local/lib:${LIBRARY_PATH}:${VIEW_DIR}/lib:${VIEW_DIR}/lib64:${ARMPL_DIR}/lib:${GPERF_DIR}/lib:${OB_DIR}/lib:${FFTW_DIR}/lib
  export LD_LIBRARY_PATH=${LLVM}/lib:${LLVM}/lib/aarch64-unknown-linux-gnu:${LLVM}/lib/clang/19/lib/aarch64-unknown-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}:${VIEW_DIR}/lib:${VIEW_DIR}/lib64:${ARMPL_DIR}/lib:${GPERF_DIR}/lib:${OB_DIR}/lib:${FFTW_DIR}/lib
  export C_INCLUDE_PATH=${LLVM}/include:${LLVM}/lib/clang/19/include:/usr/local/include:${C_INCLUDE_PATH}:${ARMPL_DIR}/include:${GPERF_DIR}/include/gperftools:${OB_DIR}/include:${FFTW_DIR}/include
  export CPLUS_INCLUDE_PATH=${LLVM}/include:${LLVM}/lib/clang/19/include:/usr/local/include:${CPLUS_INCLUDE_PATH}:${ARMPL_DIR}/include:${GPERF_DIR}/include/gperftools:${OB_DIR}/include:${FFTW_DIR}/include
  export INCLUDE_PATH=${LLVM}/include:${LLVM}/lib/clang/19/include:/usr/local/include:${INCLUDE_PATH}:${ARMPL_DIR}/include:${GPERF_DIR}/include/gperftools:${OB_DIR}/include:${FFTW_DIR}/include
  exec "$@"
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

FC=flang
OMP="-fopenmp -fPIC"

\$FC -c \$OMP main.f90 -o main_f.o -J/usr/local/lib
\$FC -c \$OMP test.f90 -o test.o
\$FC \$OMP main_f.o test.o -lperf_helper -o a.out_f
EOF

singularity run ${SIFFILE} sh ./.compile.sh
```
