CC=clang++
CMPC=gcc
CMPCPP=g++
CFLAGS= -std=c++11 -O3
FLAGS= -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
DIRECTORY=.

all: dataConstruction generateWalks distance metapath2vec

dataConstruction: $(DIRECTORY)/dataConstruction.cpp $(DIRECTORY)/header.h $(DIRECTORY)/ioFile.h $(DIRECTORY)/parser.hpp $(DIRECTORY)/strPackage.h
	$(CC) $(CFLAGS) $(DIRECTORY)/dataConstruction.cpp -o dataConstruction

generateWalks: $(DIRECTORY)/generateWalks.cpp $(DIRECTORY)/header.h $(DIRECTORY)/ioFile.h $(DIRECTORY)/parser.hpp $(DIRECTORY)/strPackage.h
	$(CC) $(CFLAGS) $(DIRECTORY)/generateWalks.cpp -o generateWalks

distance: $(DIRECTORY)/distance.c
	$(CMPC) $(DIRECTORY)/distance.c -o distance $(FLAGS)

metapath2vec : $(DIRECTORY)/metapath2vec.cpp
	$(CMPCPP) $(DIRECTORY)/metapath2vec.cpp -o metapath2vec $(FLAGS)

clean:
	rm dataConstruction generateWalks distance metapath2vec
