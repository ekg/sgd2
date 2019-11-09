all: bin/sgd2_test

bin/sgd2_test: src/sgd2.hpp src/main.cpp
	mkdir -p bin
	g++ -o bin/sgd2_test src/main.cpp
