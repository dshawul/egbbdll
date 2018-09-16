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

############################
# USE tensorflow NN lib
############################
USE_TF = 2

ifneq ($(USE_TF),0)
	DEFINES += -DTENSORFLOW
endif
############################
# Target so and files
############################
EXE = egbbso64.so
RM = rm -rf
OBJ = egbbdll.o moves.o index.o decompress.o codec.o cache.o
ifneq ($(USE_TF),0)
	OBJ += eval_nn.o
endif

#######################
#  TensorFlow
#######################

ifeq ($(USE_TF),1)
TF_DIR=/usr/local
TF_DIR_INC=$(TF_DIR)/include/tensorflow
TF_DIR_LIB=$(TF_DIR)/lib/tensorflow_cc
else ifeq ($(USE_TF),2) 
TF_DIR_INC=/home/dabdi/stests/vals/tensorflow
TF_DIR_LIB=$(TF_DIR_INC)/bazel-bin/tensorflow
endif

TF_INC =-I$(TF_DIR_INC)
TF_INC+=-I$(TF_DIR_INC)/bazel-genfiles
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/downloads
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/downloads/eigen
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/downloads/gemmlowp
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/downloads/nsync/public
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/gen/protobuf-host/include

TF_LIB = -Wl,-rpath=$(TF_DIR_LIB) 
TF_LIB += $(TF_DIR_LIB)/libtensorflow_cc.so 

ifeq ($(USE_TF),1)
TF_LIB += $(TF_DIR_LIB)/libprotobuf.a
else ifeq ($(USE_TF),2)
TF_LIB += $(TF_DIR_INC)/tensorflow/contrib/makefile/gen/protobuf-host/lib/libprotobuf.a
endif

ifneq ($(USE_TF),0)
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
