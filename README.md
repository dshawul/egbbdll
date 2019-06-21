
# What is it

egbbdll is a library for probing scorpio bitbases and
optionally to probe neural networks via [libnnprobe](https://github.com/dshawul/nn-probe).

# How to build

To compile without libnnprobe

    make clean; make COMP=gcc 

Cross-compiling for windows and android from linux is possible by setting COMP=[win/arm]

To compile with libnnprobe set NNPROBE_PATH

    make clean; make COMP=gcc NNPROBE_PATH=~/nn-probe
