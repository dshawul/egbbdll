
# What is it

egbbdll is a library for probing scorpio bitbases and optionally neural networks via

  * [libnnprobe](https://github.com/dshawul/nn-probe) for any NN.
  * [libnnueprobe](https://github.com/dshawul/nnue-probe) specifically for NNUE.
  * [libnncpuprobe](https://github.com/dshawul/nncpu-probe) specifically for tiny nets on CPU.

# How to build

To compile without libnnprobe

    make clean; make COMP=gcc 

Cross-compiling for windows from linux is possible by setting `COMP=win`

To compile with libnnprobe set NN_PROBE_PATH

    make clean; make COMP=gcc NN_PROBE_PATH=~/nn-probe

To compile with libnnueprobe set NNUEPROBE_PATH

    make clean; make COMP=gcc NNUE_PROBE_PATH=~/nnue-probe

To compile with libnncpuprobe set NNCPUPROBE_PATH

    make clean; make COMP=gcc NNCPU_PROBE_PATH=~/nncpu-probe
