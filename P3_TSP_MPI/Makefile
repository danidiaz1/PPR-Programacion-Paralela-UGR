CC = mpicxx
CFLAGS = -O3 -m64
SRC = src
LIB = lib
OBJ = obj
BIN = bin

$(OBJ)/bbpar: $(OBJ)/bbpar.o $(OBJ)/libbb.o
	$(CC) $(CFLAGS) $(OBJ)/bbpar.o $(OBJ)/libbb.o -o $(BIN)/bbpar

$(OBJ)/bbpar.o: $(SRC)/bbpar.cc
	$(CC) $(CFLAGS) -c $(SRC)/bbpar.cc -o $(OBJ)/bbpar.o

$(OBJ)/libbb.o: $(LIB)/libbb.cc $(LIB)/libbb.h
	$(CC) $(CFLAGS) -c  $(LIB)/libbb.cc -o $(OBJ)/libbb.o

clean:
	rm -f $(OBJ)/*.o $(BIN)/bbpar
