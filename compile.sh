#!/bin/bash

set -e

OS=ubuntu
TF_DIR=~/tensorflow

mkdir -p egbbso64-nn-${OS}-cpu
mkdir -p egbbso64-nn-${OS}-gpu

cd src
sed -i 's/^USE_TF.*/USE_TF = 0/g' Makefile
sed -i 's/^USE_TRT.*/USE_TRT = 1/g' Makefile
make clean; make
cp egbbso64.so ../egbbso64-nn-${OS}-gpu/
sed -i 's/^USE_TF.*/USE_TF = 2/g' Makefile
sed -i 's/^USE_TRT.*/USE_TRT = 0/g' Makefile
cd ..

DLL=`ldd egbbso64-nn-${OS}-gpu/egbbso64.so | awk '{ print $3 }' | grep libnv`
if ! [ -z "$DLL" ]; then
    cp $DLL egbbso64-nn-${OS}-gpu
fi

set +e
DLL=`ldd egbbso64-nn-${OS}-gpu/egbbso64.so | awk '{ print $3 }' | grep libcu`
set -e
if ! [ -z "$DLL" ]; then
    cp $DLL egbbso64-nn-${OS}-gpu
fi

mkdir -p $TF_DIR/tensorflow/cc/egbbdll/
cp src/*.cpp src/*.h src/BUILD $TF_DIR/tensorflow/cc/egbbdll/
cd $TF_DIR
bazel build --config=opt --config=monolithic //tensorflow/cc/egbbdll:egbbdll64.dll
cd -
cp $TF_DIR/bazel-bin/tensorflow/cc/egbbdll/egbbdll64.dll egbbso64-nn-${OS}-cpu/egbbso64.so
chmod 755 egbbso64-nn-${OS}-cpu/egbbso64.so

zip -r egbbso64-nn-${OS}-cpu.zip egbbso64-nn-${OS}-cpu
zip -r egbbso64-nn-${OS}-gpu.zip egbbso64-nn-${OS}-gpu
