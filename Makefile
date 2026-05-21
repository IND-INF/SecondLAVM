.PHONY: clean
CFLAGS = -Wall -g
OBJ = main.o Experiments\first_experiment.o Experiments\second_experiment.o Experiments\third_experiment.o\
	accuracy.o perceptron.o matrix.o

main: $(OBJ)
	gcc $(OBJ) -o main

%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

clean:
	del $(OBJ) main