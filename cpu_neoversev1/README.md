# 言語系のコンテナについて

言語系のコンテナは、OS標準パッケージの導入の容易さから、Base OSをRocky Linux 9.4とする。
Rocky LinuxとはRed Hat Enterprise Linux(RHEL)と互換性を持つオープンソースのLinuxディストリビューションである。
環境はspackによる構築を基本とし、仮想環境virtual_fugakuへ各種パッケージをインストールする。
spackはRIKEN-RCCSのリポジトリを利用する(https://github.com/RIKEN-RCCS/spack.git)

## gcc_14.1.0

- 一部アプリがgcc@8.5.0を必要とするため、下記ステップでgcc@14.1.0を構築する
 1. clang (OS標準)
 2. clang -> gcc@8.5.0
 3. gcc@8.5.0 -> gcc@14.1.0

- 数学ライブラリを構築する
 * openblas
 * fftw
 * ARM Performance Library(armpl)

- プロファイル情報採取のためのライブラリを構築する、可視化ツールpprofのためgoも合わせて構築する
 * gperftools
 * perf_helper
 * go

- コンパイル方法の例
下記の例では、コンパイルスクリプト.compile.shをヒアドキュメントで作成し、コンテナに渡してコンパイルを実行している。
上記手順で整備したライブラリはコンテナ内部でパスを設定しているため、追加のパス指定なしにリンク可能である。
注：perf_helperの使い方は(Gitリポジトリ)[https://github.com/RIKEN-RCCS/perf_helper]を参照のこと。

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

## ARM Compiler For Linux(acfl)

- acflはspackで構築する

- acflに数学ライブラリarmplが含まれるため、openblas等の追加の数学ライブラリは構築しない

- プロファイル情報採取のためのライブラリを構築する、可視化ツールpprofのためgoも合わせて構築する
 * gperftools
 * perf_helper
 * go

- コンパイル方法はgccと同様のため割愛する

## llvm 19.1.4

- LLVMリポジトリからソースをダウンロード、構築する
LLVMの構築に先立ち、ninjaとcmakeをOS標準のclangで構築する
なお、spack compiler findはflangを発見できないため、手動でcompilers.yamlにflangを追加する

- 数学ライブラリを構築する
 * openblas
 * fftw
 * ARM Performance Library(armpl)

- プロファイル情報採取のためのライブラリを構築する、可視化ツールpprofのためgoも合わせて構築する
 * gperftools
 * perf_helper
 * go

- コンパイル方法はgccと同様のため割愛する

# AI系のコンテナについて

## PyTorch 19.1.4


## TensorFlow 2.17.1


