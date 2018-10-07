#########################################
# USE tensorflow NN lib
#  USE_TF      0 = Don't use tensorflow
#              1 = tensorlow_cc
#              2 = manually built tensorflow
#  USE_TRT     0 = Don't use TensorRT
#              1 = Use TensorRT
#  USE_SHARED  0 = static linking if possible
#              1 = dynamic linking to TF/TRT
########################################
USE_TF = 2
USE_TRT = 0
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

############################
# Choose compiler and flags
############################
CC = g++
CFLAGS = -O3 -Wno-unused-result -std=c++11
LDFLAGS =

#CC = x86_64-w64-mingw32-g++
#CFLAGS = -O3 -Wno-unused-result -std=c++11
#LDFLAGS = -static 

DEFINES =

#DEFINES += -DBIGENDIAN

ifneq ($(USE_TF),0)
    DEFINES += -DTENSORFLOW
endif

ifneq ($(USE_TRT),0)
    DEFINES += -DTRT
endif

############################
# Target so and files
############################
EXE = egbbso64.so
RM = rm -rf
OBJ = egbbdll.o moves.o index.o decompress.o codec.o cache.o eval_nn.o

TF_LIB=
TF_INC=
#######################
#  TensorFlow
#######################

ifneq ($(USE_TF),0)

ifeq ($(USE_TF),1)
    TF_DIR_INC=$(TF_DIR)/include/tensorflow
    TF_DIR_LIB=$(TF_DIR)/lib/tensorflow_cc
else
    TF_DIR_INC=$(TF_DIR)
    TF_DIR_LIB=$(TF_DIR)/bazel-bin/tensorflow
endif

TF_INC =-I$(TF_DIR_INC)
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/downloads
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/downloads/eigen
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/downloads/gemmlowp
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/downloads/nsync/public
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/gen/protobuf-host/include

ifeq ($(USE_SHARED),1)
    TF_INC+=-I$(TF_DIR_INC)/bazel-genfiles
    TF_LIB = -Wl,-rpath=$(TF_DIR_LIB) 
    TF_LIB += $(TF_DIR_LIB)/libtensorflow_cc.so 
else
    TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/gen/host_obj
    TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/gen/proto
    TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/gen/proto_text

    ifeq ($(USE_TF),1)
        TF_LIB = $(TF_DIR_LIB)/nsync.a
        TF_LIB += -Wl,--whole-archive ${TF_DIR_LIB}/libtensorflow-core.a -Wl,--no-whole-archive
    else
        TF_LIB = $(TF_DIR_INC)/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11/nsync.a
        TF_LIB += -Wl,--whole-archive ${TF_DIR_INC}/tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a -Wl,--no-whole-archive
    endif
endif

ifeq ($(USE_TF),1)
    TF_LIB += $(TF_DIR_LIB)/libprotobuf.a
else ifeq ($(USE_TF),2)
    TF_LIB += $(TF_DIR_INC)/tensorflow/contrib/makefile/gen/protobuf-host/lib/libprotobuf.a
endif

LDFLAGS += $(TF_LIB)

endif

######################
#  TensorRT
#######################

ifneq ($(USE_TRT),0)

TF_INC += -I$(TRT_DIR)
TF_INC += -I$(CUDA_DIR)
TF_LIB += -Wl,-rpath=$(TRT_DIR)/lib
TF_LIB += $(TRT_DIR)/lib/libnvinfer.so
TF_LIB += $(TRT_DIR)/lib/libnvparsers.so

LDFLAGS += $(TF_LIB)

endif

######################
# Rules
######################

$(EXE): $(OBJ)
	$(CC) -o $@ $(OBJ) $(LDFLAGS) -shared -lm -lpthread

%.o: %.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -fPIC -o $@ $<

eval_nn.o: eval_nn.cpp
	$(CC) $(CFLAGS) $(DEFINES) ${TF_INC} -c -fPIC -o $@ $<

clean:
	$(RM) $(OBJ)
