CC = mpicc
CFLAGS = -Wall -Wextra -g  -fopenmp
TARGET = taskConstruct  
SRC = taskConstruct.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $<

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)

runTaskConstruct4:
	mpiexec -n 4 ./taskConstruct 

runTaskConstruct8:
	mpiexec -n 8 ./taskConstruct 
	





