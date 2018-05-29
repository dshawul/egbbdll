############################
# Choose compiler and flags
############################
CC = g++
CFLAGS = -O3 -Wno-unused-result

#CC = x86_64-w64-mingw32-g++
#CFLAGS = -O3 -Wno-unused-result -static

DEFINES =
LDFLAGS = 

#DEFINES += -DBIGENDIAN

############################
# USE tensorflow NN lib
############################
USE_TF = 1

ifeq ($(USE_TF),1)
	DEFINES += -DTENSORFLOW
endif
############################
# Target so and files
############################
EXE = egbbso64.so
RM = rm -rf
OBJ = egbbdll.o moves.o index.o decompress.o codec.o cache.o
ifeq ($(USE_TF),1)
	OBJ += eval_nn.o
endif

#######################
#  TensorFlow
#######################
TF_DIR=/usr/local
TF_DIR_INC=$(TF_DIR)/include/tensorflow
TF_DIR_LIB=$(TF_DIR)/lib/tensorflow_cc

TF_INC =-I$(TF_DIR_INC)
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/gen/host_obj
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/downloads
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/downloads/eigen
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/downloads/gemmlowp
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/downloads/nsync/public
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/gen/proto
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/gen/proto_text
TF_INC+=-I$(TF_DIR_INC)/tensorflow/contrib/makefile/gen/protobuf-host/include

#static linking
#TF_LIB = -Wl,--whole-archive ${TF_DIR_LIB}/libtensorflow-core.a -Wl,--no-whole-archive
#TF_LIB += $(TF_DIR_LIB)/libprotobuf.a $(TF_DIR_LIB)/nsync.a

#dynamic linking
TF_LIB = -Wl,-rpath=$(TF_DIR_LIB) 
TF_LIB += $(TF_DIR_LIB)/libprotobuf.a
TF_LIB += -Wl,-Bdynamic $(TF_DIR_LIB)/libtensorflow_cc.so 

ifeq ($(USE_TF),1)
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
	$(CC) $(CFLAGS) $(DEFINES) ${TF_INC} -std=c++11 -c -fPIC -o $@ $<

clean:
	$(RM) $(OBJ)
