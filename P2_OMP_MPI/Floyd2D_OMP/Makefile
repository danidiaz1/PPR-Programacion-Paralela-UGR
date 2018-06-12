CC = g++
NAME = floyd
CFLAGS = -m64 -O3 -fopenmp

MODULES = floyd.o Graph.o

all: floyd
Graph.o: 	Graph.cc Graph.h
			$(CC) $(CFLAGS) -c Graph.cc

floyd.o:	floyd.cc
			$(CC) $(CFLAGS) -c floyd.cc

floyd: 		$(MODULES)
			$(CC) $(CFLAGS) $(MODULES) -o $(NAME) $(EXTRALIBS)

clean:
		rm -f $(MODULES)  $(NAME)
