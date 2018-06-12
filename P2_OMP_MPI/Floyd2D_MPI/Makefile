CC = g++
NAME = floyd
CFLAGS = -m64 -O3 -fopenmp
MPICC = mpicxx
MPIFLAGS = -O3 -m64
MODULES = floyd.o Graph.o

all: floyd
Graph.o: 	Graph.cc Graph.h
			$(CC) $(CFLAGS) -c Graph.cc

floyd.o:	floyd.cc
			$(MPICC) $(MPIFLAGS) -c floyd.cc

floyd: $(MODULES)
			$(MPICC) $(MPIFLAGS) $(MODULES) -o $(NAME)

clean:
		rm -f $(MODULES)  $(NAME)
