CC = g++
CFLAGS = -O3
DEFINES =
LDFLAGS = 

RM = rm -rf
OBJ = egbbdll.o moves.o index.o decompress.o codec.o cache.o

egbbso.so: $(OBJ)
	$(CC) $(CFLAGS) $(DEFINES) $(LDFLAGS) $(OBJ) -shared -o egbbso.so -lm -lpthread

%.o: %.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -fPIC $<
clean:
	$(RM) $(OBJ)
