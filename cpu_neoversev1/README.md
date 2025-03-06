# 言語系のコンテナについて

OS標準パッケージの導入の容易さから、Base OSをRocky Linux 9.4とする。
Rocky LinuxとはRed Hat Enterprise Linux(RHEL)と互換性を持つオープンソースのLinuxディストリビューションである。
環境はspackによる構築を基本とし、仮想環境 **virtual_fugaku** へ各種パッケージをインストールする。
spackはRIKEN-RCCSの[リポジトリ](https://github.com/RIKEN-RCCS/spack.git)を利用する。

## GCC version14.1.0

一部アプリが`gcc@8.5.0`を必要とするため、下記ステップで`gcc@14.1.0`を構築する
1. clang (OS標準)
2. clang -> gcc@8.5.0
3. gcc@8.5.0 -> gcc@14.1.0

数学ライブラリは一般的に利用されるBLASとFFT、ベンダー固有のARM Performance Library for gcc(armpl)をインストールする
- openblas
- fftw
- armpl for gcc

プロファイル情報採取のためのGoogle Performance Tools(gperftools)、可視化ツール`pprof`のためのgoも合わせてインストールする
- gperftools
- perf_helper
- go

コンパイル方法の例

下記の例では、コンパイルスクリプト`.compile.sh`をヒアドキュメントで作成し、コンテナに渡してコンパイルを実行している。上記手順で整備したライブラリはコンテナ内部でパスを設定しているため、追加のパス指定なしにリンク可能である。

  注：perf_helperの使い方はRIKEN-RCCSの[リポジトリ](https://github.com/RIKEN-RCCS/perf_helper)を参照のこと。
  
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

acflはspackでインストールする

acflに数学ライブラリarmplが含まれるため、openblas等の追加の数学ライブラリはインストールしない

プロファイル情報採取のためのGoogle Performance Tools(gperftools)、可視化ツール`pprof`のためのgoも合わせてインストールする
- gperftools
- perf_helper
- go

コンパイル方法はgccと同様のため割愛する

## LLVM version19.1.4

LLVMリポジトリからソースをダウンロード、インストールする。LLVMのインストールに先立ち、`ninja`と`cmake`をOS標準の`clang`でインストールする。なお、`spack compiler find`は`flang`を発見できないため、手動で`compilers.yaml`に`flang`を追加する。

数学ライブラリは一般的に利用されるBLASとFFT、ベンダー固有のARM Performance Library for gcc(armpl)をインストールする
- openblas
- fftw
- armpl for gcc

プロファイル情報採取のためのGoogle Performance Tools(gperftools)、可視化ツール`pprof`のためのgoも合わせてインストールする
- gperftools
- perf_helper
- go

コンパイル方法はgccと同様のため割愛する

---

# AI系のコンテナについて

Base OSをAI関連で広く使われるUbuntu 24.04とする。
環境はOS標準のコンパイラを使い各種パッケージを構築する。AI系ではPythonのモジュールを必要に応じて`pip`コマンドで整備することが一般的であるため、spackで仮想環境にインストールする方法ではなく、wheelを作成することで再利用性を高めている。

## PyTorch version2.5.0

数学ライブラリはARMが公開するARM Compute Library(acl)をOS標準コンパイラ gcc-14でインストールする。
ARMが公開する数学ライブラリとしてArm Performance Libraryが存在するが、aclは機械学習やコンピュータビジョン向けに最適化されたライブラリである。
また、aclはPyTorchが使用する数学ライブラリoneDNNのバックエンドとして機能する。
aclのビルドではOpenMPを有効化し、対象アーキをarmv8.2-a-sveとしてSVE命令を有効化する。
- acl version25.02

PyTorchのビルドでは数学ライブラリoneDNN、oneDNNのバックエンドとしてaclを有効化、MPIおよびOpenMPのサポートを有効化する。
PyTorchと共に利用されるTorchVisionとTorchAudioを合わせて構築する。
各パッケージの構築ではwheelを作成し、wheelは`/opt/dist`へ移動したのちにインストールする。
コンテナサイズの削減のため`git clone`したディレクトリは削除する。
- PyTorch version2.5.0
- TorchVision
- TorchAudio

PyTorchの実行に必要なパッケージは`/opt/requirements.txt`に一覧としてまとめている。
例えばPythonの仮想環境でPyTorch環境を構築したい場合は、pytorchコンテナ内あるいはlocalimageとしてpytorchコンテナを引用した新規コンテナ内で下記コマンドを実行することで即座にPyTorch環境を作成可能である。
```bash
python3 -m venv venv
. /opt/venv/bin/activate
python3 -m pip install -r /opt/requirements.txt
```

## TensorFlow 2.17

TensorFlowでは数学ライブラリoneDNNおよびバックエンドaclがデフォルトで有効化されているため特別な手順なしにインストールが可能である。
公式Webページの[ソースからビルド](https://www.tensorflow.org/install/source?hl=ja)する手順に従いインストールする。
なお、aarch64ではリンク時のエラーに対応するため、リンクオプション`--linkopt=-fuse-ld=bfd`を`bazel build`に追加する。
- bazel version6.5.0
- TensorFlow version2.17

TensorFlowの実行に必要なパッケージは`/opt/requirements.txt`に一覧としてまとめている。
PyTorchと同様な手順で即座にTensorFlow環境を作成可能である。
