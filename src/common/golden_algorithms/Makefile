FLAGS = -std=c++11 -O3 
BUILD_DIR = bin

.PHONY: all pagerank
all: \
	pagerank

pagerank: pagerank/pagerank_cpu_main.cpp pagerank/pagerank_cpu.cpp
	mkdir -p $(BUILD_DIR)
	g++ $(FLAGS) pagerank/pagerank_cpu_main.cpp pagerank/pagerank_cpu.cpp -o $(BUILD_DIR)/pagerank
