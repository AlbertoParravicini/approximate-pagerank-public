//
// Created by Francesco Sgherzi on 8/10/19.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include "../csc_matrix/csc_matrix.h"
#include "../utils/options.hpp"
#include "../utils/utils.hpp"
#include "../../fpga/src/csc_fpga/csc_fpga.hpp"
#include <chrono>

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

	/////////////////////////////
	// Input parameters /////////
	/////////////////////////////

    Options options = Options(argc, argv);
    int debug = !options.debug;//options.debug;
    bool use_csc = options.use_csc;
    std::string graph_path = options.graph_path;
    int max_iter = options.max_iter;
    fixed_float alpha = options.alpha;
    fixed_float max_err = options.max_err;
    bool use_sample_graph = options.use_sample_graph;
    uint num_tests = options.num_tests;
    bool undirect_graph = options.undirect_graph;
    std::string xclbin_path = options.xclbin_path;

	std::string golden_res = "../../data/graphs/other/test/results.txt";

	/////////////////////////////
	// Load data ////////////////
	/////////////////////////////

	// Use a sample graph if required;
	if (use_sample_graph) {
		if (debug) {
			std::cout << "Using sample graph located at: " << DEFAULT_MTX_FILE << std::endl;
		}
		use_csc = false;
		graph_path = DEFAULT_MTX_FILE;
	}

	// Load dataset and create auxiliary vectors;
	auto start_1 = clock_type::now();
	csc_t input_m;
	if (!use_csc) {
		input_m = load_graph_mtx(graph_path, debug, undirect_graph);
	} else {
		input_m = load_graph_csc(graph_path, debug);
	}
	auto end_1 = clock_type::now();
	auto start_2 = clock_type::now();
	csc_fixed_fpga_t f_input_m = convert_to_fixed_point_fpga(input_m);
	auto end_2 = clock_type::now();

	auto loading_time = chrono::duration_cast<chrono::milliseconds>(end_1 - start_1).count();
	auto to_fixed_time = chrono::duration_cast<chrono::milliseconds>(end_2 - start_2).count();

	if (debug) {
		std::cout << "\n----------------\n- Graph Summary -\n----------------" << std::endl;
		std::cout << "- |V| = " << f_input_m.col_ptr.size() - 1 << std::endl;
		std::cout << "- |E| = " << f_input_m.col_val.size() << std::endl;
		std::cout << "- Undirected: " << (undirect_graph ? "True" : "False") << std::endl;
		std::cout << "----------------" << std::endl;
		std::cout << "- Loading time: " << loading_time / 1000 << " sec" << std::endl;
		std::cout << "- Float-to-fixed time: " << to_fixed_time / 1000 << " sec" << std::endl;
		std::cout << "- Tot time: " << (loading_time + to_fixed_time) / 1000 << " sec" << std::endl;
		if (!use_csc) {
			long filesize_mb = get_file_size(graph_path) / 1000000;
			double throughput = loading_time > 0 ? 1000 * filesize_mb / loading_time : 0;
			std::cout << "- File size: " << filesize_mb << " MB "<< std::endl;
			std::cout << "- Throughput: " << throughput << " MB/sec" << std::endl;
		}
		std::cout << "----------------" << std::endl;
	}

	print_graph(f_input_m.col_ptr, f_input_m.col_idx);
	print_array_indexed(f_input_m.col_val);
}

