CC = mpicc
CFLAGS = -Wall -Wextra -g  -fopenmp
TARGET = Sequential
SRC = Sequential.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $<

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)

seq: 
	./Sequential

	





