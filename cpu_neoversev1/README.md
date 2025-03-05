# 言語系のコンテナについて

言語系のコンテナは、OS標準パッケージの導入の容易さから、Base OSをRocky Linux 9.4とする。
Rocky LinuxとはRed Hat Enterprise Linux(RHEL)と互換性を持つオープンソースのLinuxディストリビューションである。
環境は`spack`による構築を基本とし、仮想環境`virtual_fugaku`へ各種パッケージをインストールする。
`spack`はRIKEN-RCCSの[リポジトリ](https://github.com/RIKEN-RCCS/spack.git)を利用する。

## GCC version14.1.0

- 一部アプリが`gcc@8.5.0`を必要とするため、下記ステップで`gcc@14.1.0`を構築する
  1. clang (OS標準)
  2. clang -> gcc@8.5.0
  3. gcc@8.5.0 -> gcc@14.1.0

- 数学ライブラリを構築する
  - openblas
  - fftw
  - ARM Performance Library(armpl)

- プロファイル情報採取のためのライブラリを構築する、可視化ツール`pprof`のため`go`も合わせて構築する
  - gperftools
  - perf_helper
  - go

- コンパイル方法の例

  下記の例では、コンパイルスクリプト`.compile.sh`をヒアドキュメントで作成し、コンテナに渡してコンパイルを実行している。上記手順で整備したライブラリはコンテナ内部でパスを設定しているため、追加のパス指定なしにリンク可能である。

  注：`perf_helper`の使い方はRIKEN-RCCSの[リポジトリ](https://github.com/RIKEN-RCCS/perf_helper)を参照のこと。
  
  ```bash
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

## ARM Compiler For Linux (acfl)

- acflはspackで構築する

- acflに数学ライブラリ`armpl`が含まれるため、`openblas`等の追加の数学ライブラリは構築しない

- プロファイル情報採取のためのライブラリを構築する、可視化ツール`pprof`のため`go`も合わせて構築する
  - gperftools
  - perf_helper
  - go

- コンパイル方法はgccと同様のため割愛する

## LLVM version19.1.4

- LLVMリポジトリからソースをダウンロード、構築する。LLVMの構築に先立ち、`ninja`と`cmake`をOS標準の`clang`で構築する。なお、`spack compiler find`は`flang`を発見できないため、手動で`compilers.yaml`に`flang`を追加する。

- 数学ライブラリを構築する
  - openblas
  - fftw
  - ARM Performance Library(armpl)

- プロファイル情報採取のためのライブラリを構築する、可視化ツール`pprof`のため`go`も合わせて構築する
  - gperftools
  - perf_helper
  - go

- コンパイル方法はgccと同様のため割愛する

# AI系のコンテナについて

## PyTorch 19.1.4


## TensorFlow 2.17.1


