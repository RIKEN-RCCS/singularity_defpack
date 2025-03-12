# **Setup llvm using SPACK**

## **definition file**

Install Rocky Linux 9.4 standard packages.

```bash
  dnf -y group install "Development Tools"
  dnf -y install wget which perf kernel-tools numactl-devel python3-devel
  dnf -y install llvm* clang*
  dnf -y install gcc-gfortran
  dnf clean all
```

Clone Spack from RIKEN RCCS GitHub, activate the Spack environment, and detect installed compilers in Spack.

```bash
  cd /opt
  git clone https://github.com/RIKEN-RCCS/spack.git

  . /opt/spack/share/spack/setup-env.sh

  spack compiler find
```

Create a Spack virtual environment `virtual_fugaku`.

```bash
  spack env create virtual_fugaku
```

Install cmake and ninja required for LLVM installation.

```bash
  spack -e virtual_fugaku install -j 32 --add ninja%clang cmake%clang
  spack load ninja cmake
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

Install the profiling tool `gperftools` and mathematical libraries `openblas`, `fftw`, `armpl for gcc`.

```bash
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

Define various environment variables to build `perf_helper` with LLVM.
Clone the profiling tool `perf_helper` from GitHub, build it, and install libraries and other dependencies into `/usr/local`.

```bash
  export PATH=/usr/local/llvm-19.1.4/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/llvm-19.1.4/lib:$LD_LIBRARY_PATH
  export LIBRARY_PATH=/usr/local/llvm-19.1.4/lib:$LIBRARY_PATH
  export C_INCLUDE_PATH=/usr/local/llvm-19.1.4/include:$C_INCLUDE_PATH
  export CC=clang
  export FC=flang
  export OMP="-fopenmp -fPIC -Wall"

  # Perf Helper Library
  cd /opt
  git clone https://github.com/RIKEN-RCCS/perf_helper.git
  cd perf_helper
  ${CC} -c ${OMP} perf_helper.c -o perf_helper.o
  ${CC} -c ${OMP} perf_helper_wrapper.c -o perf_helper_wrapper.o
  ${FC} -c ${OMP} perf_helper_mod.f90 -o perf_helper_mod.o
  ar r libperf_helper.a perf_helper.o perf_helper_wrapper.o perf_helper_mod.o
  cp perf_helper.h /usr/local/include
  cp libperf_helper.a /usr/local/lib
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

  export PATH=/usr/local/go/pkg/tool/linux_arm64:${PATH}:${LLVM}/bin:/usr/local/bin:${ARMPL_DIR}/bin:${GPERF_DIR}/bin:${FFTW_DIR}/bin
  export LIBRARY_PATH=${LIBRARY_PATH}:${LLVM}/lib:${LLVM}/lib/aarch64-unknown-linux-gnu:${LLVM}/lib/clang/19/lib/aarch64-unknown-linux-gnu:/usr/local/lib:${VIEW_DIR}/lib:${VIEW_DIR}/lib64:${ARMPL_DIR}/lib:${GPERF_DIR}/lib:${OB_DIR}/lib:${FFTW_DIR}/lib
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LLVM}/lib:${LLVM}/lib/aarch64-unknown-linux-gnu:${LLVM}/lib/clang/19/lib/aarch64-unknown-linux-gnu:/usr/local/lib:${VIEW_DIR}/lib:${VIEW_DIR}/lib64:${ARMPL_DIR}/lib:${GPERF_DIR}/lib:${OB_DIR}/lib:${FFTW_DIR}/lib
  export C_INCLUDE_PATH=${C_INCLUDE_PATH}:${LLVM}/include:${LLVM}/lib/clang/19/include:/usr/local/include:${ARMPL_DIR}/include:${GPERF_DIR}/include/gperftools:${OB_DIR}/include:${FFTW_DIR}/include
  export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${LLVM}/include:${LLVM}/lib/clang/19/include:/usr/local/include:${ARMPL_DIR}/include:${GPERF_DIR}/include/gperftools:${OB_DIR}/include:${FFTW_DIR}/include
  export INCLUDE_PATH=${INCLUDE_PATH}:${LLVM}/include:${LLVM}/lib/clang/19/include:/usr/local/include:${ARMPL_DIR}/include:${GPERF_DIR}/include/gperftools:${OB_DIR}/include:${FFTW_DIR}/include
```
