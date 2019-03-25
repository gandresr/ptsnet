all:scan run

CC=gcc
objs = genbase.o vector.o

base:$(objs)
	@$(CC) $(objs) -o base

run:base
	@./base

clean:
	@/bin/rm -rf *.o
