#CC = g++
#CFLAGS = -O3 -Wno-unused-result

CC = x86_64-w64-mingw32-g++
CFLAGS = -O3 -Wno-unused-result -static

DEFINES =
LDFLAGS = 

#DEFINES += -DBIGENDIAN

EXE = egbbso64.so
RM = rm -rf
OBJ = egbbdll.o moves.o index.o decompress.o codec.o cache.o

$(EXE): $(OBJ)
	$(CC) $(CFLAGS) $(DEFINES) $(LDFLAGS) $(OBJ) -shared -o $(EXE) -lm -lpthread

%.o: %.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -fPIC $<
clean:
	$(RM) $(OBJ)
