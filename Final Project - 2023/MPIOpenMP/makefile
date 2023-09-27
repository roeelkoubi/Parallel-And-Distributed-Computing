CC = mpicc
CFLAGS = -Wall -Wextra -g  -fopenmp
TARGET = MPI_OpenMP 
SRC = MPI_OpenMP.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $<

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)

run4:
	mpiexec -n 4 ./MPI_OpenMP

run8:
	mpiexec -n 8 ./MPI_OpenMP

runOn2:
	mpiexec -np 4 -machinefile  run2computers  -map-by  node  ./MPI_OpenMP


	





