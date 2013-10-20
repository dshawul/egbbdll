CC = g++
CFLAGS = -O3
DEFINES =
LDFLAGS = 

EXE = egbbso.so
RM = rm -rf
OBJ = egbbdll.o moves.o index.o decompress.o codec.o cache.o

$(EXE): $(OBJ)
	$(CC) $(CFLAGS) $(DEFINES) $(LDFLAGS) $(OBJ) -shared -o $(EXE) -lm -lpthread

%.o: %.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -fPIC $<
clean:
	$(RM) $(OBJ)
