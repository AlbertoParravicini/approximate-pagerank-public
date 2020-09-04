//
// Created by Francesco Sgherzi on 8/10/19.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <ap_int.h>
#include <chrono>

#include "options.hpp"
#include "utils.hpp"
#include "evaluation_utils.hpp"
#include "../ip/multi_spmv.hpp"
#include "gold_algorithms.hpp"

#include "../ip/csc_matrix.hpp"
#include "../ip/coo_matrix.hpp"
#include "../ip/csc_fpga.hpp"
#include "../ip/coo_fpga.hpp"
#include "aligned_allocator.h"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

#define allocator aligned_allocator<input_block>

/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

	/////////////////////////////
	// Input parameters /////////
	/////////////////////////////

    Options options = Options(argc, argv);
    int debug = true;
    bool use_csc = options.use_csc;
	std::string graph_path = options.graph_path;
//	std::string graph_path = "../../data/scf_2019_10_02_13_26_56.mtx";
	unsigned int max_iter = options.max_iter;
    fixed_float alpha = options.alpha;
    fixed_float max_err = options.max_err;
    bool use_sample_graph = options.use_sample_graph;
    uint num_tests = options.num_tests;
    bool undirect_graph = options.undirect_graph;
    std::string xclbin_path = options.xclbin_path;

	std::string golden_res = "../../data/graphs/other/test/results.txt";

	/////////////////////////////
	// Setup ////////////////////
	/////////////////////////////

	csv_results_t results;
	results.fixed_float_width = FIXED_WIDTH;
	results.fixed_float_scale = FIXED_INTEGER_PART;

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

	/////////////////////////////
	// Setup kernel /////////////
	/////////////////////////////

	// Generate a random sparse matrix to test spmv;
//	index_type N = 1000; // Desired number of vertices, multiple of BUFFER_SIZE;
//	std::vector<index_type> ptr(N + 1, 0.0);
//	std::vector<index_type> idx;
//	index_type E = create_random_graph<index_type>(ptr, idx, 2);
//	// Create random values;
//	std::vector<fixed_float> val(E, 0.0);
//	random_array<fixed_float>(val.data(), val.size(), 1);
//	csc_fixed_fpga_t csc = {val, ptr, idx};
//	coo_fixed_fpga_t coo(csc, true);

	csc_t input_m = load_graph_mtx(graph_path, debug, undirect_graph);
	csc_fixed_fpga_t csc = convert_to_fixed_point_fpga(input_m);
	coo_fixed_fpga_t coo(csc, true);
	index_type N = coo.N;
	index_type E = coo.E;
	index_type E_fixed = coo.E_fixed;
	std::vector<index_type> ptr = csc.col_ptr;
	std::vector<index_type> idx = csc.col_idx;
	std::vector<fixed_float> val = csc.col_val;

	std::vector<fixed_float, aligned_allocator<fixed_float>> vec(N_PPR_VERTICES * N, 1.0 / N);
	std::vector<fixed_float, aligned_allocator<fixed_float>> res_f(N_PPR_VERTICES * N, 0.0);
	std::vector<fixed_float, aligned_allocator<fixed_float>> res_gold_f(N_PPR_VERTICES * N, 0.0);
	std::vector<float> res(N_PPR_VERTICES * N, 0.0);
	std::vector<float> res_gold(N_PPR_VERTICES * N, 0.0);

	std::cout << "N: " << ptr.size() - 1 << "; E: " << E << "; E_fixed: " << E_fixed << std::endl;

	// Golden algorithm;
	multi_spmv_gold(ptr.data(), idx.data(), val.data(), N, (index_type) N_PPR_VERTICES, res_gold_f.data(), vec.data());

	for (int i = 0; i < N * N_PPR_VERTICES; i++) {
		res_gold[i] = res_gold_f[i];
	}

	print_matrix_indexed(res_gold.data(), N_PPR_VERTICES, N, N_PPR_VERTICES, N);

	index_type num_blocks_N = (N + BUFFER_SIZE - 1) / BUFFER_SIZE;
	index_type N_padded = num_blocks_N * BUFFER_SIZE;
	index_type num_blocks_E = (E_fixed + BUFFER_SIZE - 1) / BUFFER_SIZE;
	index_type E_padded = num_blocks_E * BUFFER_SIZE;

	std::vector<input_block, allocator> start_in(num_blocks_E);
	std::vector<input_block, allocator> end_in(num_blocks_E);
	std::vector<input_block, allocator> val_in(num_blocks_E);
	std::vector<input_block, allocator> vec_in(N_PPR_VERTICES * num_blocks_N);
	std::vector<input_block, allocator> res_out(N_PPR_VERTICES * num_blocks_N);

	write_packed_array(coo.start.data(), start_in.data(), E_fixed, num_blocks_E);
	write_packed_array(coo.end.data(), end_in.data(), E_fixed, num_blocks_E);
	write_packed_array(coo.val.data(), val_in.data(), E_fixed, num_blocks_E);

	write_packed_matrix(vec.data(), vec_in.data(), N_PPR_VERTICES, N, num_blocks_N);

	/////////////////////////////
	// Execute the kernel ///////
	/////////////////////////////

	num_tests = 3;

	std::vector<double> trans_times(num_tests, 0);
	std::vector<double> exec_times(num_tests, 0);
	std::vector<double> errors(num_tests, 0);
	std::vector<double> ndcgs(num_tests, 0);
	for (uint i = 0; i < num_tests; i++) {
		if (debug) {
			std::cout << "Iteration " << i << ")" << std::endl;
		}

		multi_spmv_main(start_in.data(), end_in.data(), val_in.data(), N, E_padded, res_out.data(), vec_in.data());

		// Read back the result;
		read_packed_matrix(res_f.data(), N_PPR_VERTICES, N, res_out.data(), num_blocks_N);

		for (int i = 0; i < N_PPR_VERTICES * N; i++) {
			res[i] = res_f[i];
		}

		print_matrix_indexed(res.data(), N_PPR_VERTICES, N, N_PPR_VERTICES, N);

		int error = check_array_equality(res.data(), res_gold.data(), N_PPR_VERTICES * N);
		std::cout << "num errors: " << error << std::endl;
		errors[i] = error;

		// Analyze results;
		if (!debug) {
			if (i == 0) {
				std::cout << "fixed_float_width,fixed_float_scale,v,e,"
						<< "execution_time,transfer_time,error_pct,normalized_dcg" << std::endl;
			}
			std::cout << results.fixed_float_width << "," << results.fixed_float_scale << "," << results.n_vertices << ","
					<< results.n_edges << "," << results.execution_time << "," << results.transfer_time << std::endl;
		}
	}

	if (debug) {
		int old_precision = std::cout.precision();
		std::cout.precision(2);
		std::cout << "\n----------------\n- Results ------\n----------------" << std::endl;
		std::cout << "Mean transfer time:  " << mean(trans_times) << "±" << st_dev(trans_times) << " ms;" << std::endl;
		std::cout << "Mean execution time: " << mean(exec_times) << "±" << st_dev(exec_times) << " ms;" << std::endl;
		std::cout << "Mean % error: " << mean(errors) << "±" << st_dev(errors) << " %;" << std::endl;
		std::cout << "Mean NDCG: " << mean(ndcgs) << "±" << st_dev(ndcgs) << ";" << std::endl;
		std::cout << "----------------" << std::endl;
		std::cout.precision(old_precision);
	}
}

