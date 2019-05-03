all:scan run

CC = g++

base:$(objs)
	@$(CC) -fopenmp parbase.c vector.c -o base

run:base
	@./base

clean:
	@/bin/rm -rf *.o
