main: main.cpp
	g++ -fopenmp -Wall main.cpp -o main -lOpenCL -lboost_program_options