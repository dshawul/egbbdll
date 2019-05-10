
# How to build

The egbbdll serves dual purpose, namely, for probing bitbases and neural networks.
If you just want the former, the build processess is very easy.

First clone the repository

	git clone https://github.com/dshawul/egbbdll.git

## Building without neural network support

Then set the following variables to 0 in the Makefile
    
    USE_TF = 0
    USE_TRT = 0

Then
	
	make

That is it.

## Building with neural network support

To build egbbdll with neural network support is a lot more complicated. Neural network inference is done via 
tensorflow and/or TensorRT backends. This means you are limited to NVIDIA GPUs atleast until OpenCL support is 
complete in these libraries. 

### Downloading binaries (for Windows)

The easiest way on a Windows machines is to download the binaries I provide for both 
CPU and GPU (with CUDA + cuDNN libraries). Here are the links for convenience

[egbbdll64-nn-windows-cpu.zip](https://github.com/dshawul/Scorpio/releases/download/2.9.0/egbbdll64-nn-windows-cpu.zip)

[egbbdll64-nn-windows-gpu.zip](https://github.com/dshawul/Scorpio/releases/download/2.9.0/egbbdll64-nn-windows-gpu.zip)

Extract it in a directory where bitbases are located so it can serve its dual purpose. 
For the GPU version of egbbdll, we need to set the Path environment variable ( in the case of linux 
the LD_LIBRARY_PATH variable) so that the system can find cudnn.dll, nvinfer.dll and other dependencies.

### Building from sources (for linux)

For those who feel adventurous, here is how you build from sources.
Egbbdll needs tensorflow for neural network inference on the CPU, however, on the
GPU there is another option that is less complicated to build AND also significantly faster.
That option is using TensorRT and UFF format network files.
We need to install the followind dependencies for the TensorRT backend:

    * cuDNN
    * CUDA
    * TensorRT

These can be download [NVIDA developer page](https://developer.nvidia.com/)
Make sure you download compatible versions (e.g. cuDNN 7.3 + CUDA 10.0 + TensorRT 5.0)

####   Building egbbdll with TensorRT backend

Clone the egbbdll repository

    git clone https://github.com/dshawul/egbbdll.git

Then go into the Makefile and set the paths to the dependencies

    #########################################
    #  USE_TF      0 = Don't use tensorflow
    #              1 = tensorlow_cc
    #              2 = manually built tensorflow
    #  USE_TRT     0 = Don't use TensorRT
    #              1 = Use TensorRT
    #  USE_SHARED  0 = static linking if possible
    #              1 = dynamic linking to TF/TRT
    ########################################
    USE_TF = 0
    USE_TRT = 1
    USE_SHARED = 1

    ########################################
    # Set directories to dependenies
    ########################################

    ifeq ($(USE_TF),1)
        TF_DIR=/usr/local
    else ifeq ($(USE_TF),2)
        TF_DIR=/home/daniel/tensorflow
    endif

    ifneq ($(USE_TRT),0)
        TRT_DIR = /home/daniel/TensorRT-5.0.0.10
        CUDA_DIR = /usr/local/cuda
    endif

Then build egbbso64.so with
    
    make

That is it.

#### Building egbbdll with tensorflow backend

To build egbbdll with libtensorflow_cc.so dependency, do one of the following.

##### Option 1 - using tensorflow_cc on linux only

The first option is to use tensorflow_cc (USE_TF = 1). Build the C++ tensorflow library following the steps 
given [here](https://github.com/FloopCZ/tensorflow_cc).

It may be easier to use docker using the command below instead of building the shared library with bazel yourself.

    docker run --runtime=nvidia -it floopcz/tensorflow_cc:ubuntu-shared-cuda /bin/bash

Once the build and install (or docker load) is complete, you should see the tensorflow and protobuf libraries

    $ ls /usr/local/lib/tensorflow_cc/
    libprotobuf.a  libtensorflow_cc.so

##### Option 2 - manually building tensorflow

The second option is to build tensorflow manually without using scripts (USE_TF = 2). This option also works on windows.
Go into the tensorflow director and execute

    bazel build --config=opt --config=monolithic //tensorflow:libtensorflow_cc.so

This will take hours to build so be patient. Once it completes, you will find the library in bazel-bin/tensorflow directory.

#### Build egbbdll directly using bazel -- i.e. without using Makefile

To compile egbbdll directly using bazel, put the egbbdll source directory in tensorflow/tenorflow/cc/ and then compile with

    bazel build --config=opt --config=monolithic //tensorflow/cc/egbbdll:egbbdll64.dll

You will then find the egbbdll64.dll in bazel-bin/tensorflow/cc/egbbdll directory. This option currently doesn't work
with the TensorRT backend.
