//
// Created by Francesco Sgherzi on 8/10/19.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>  /* For O_RDWR, O_WRONLY */
#include <unistd.h> /* For open(), read(), ... */
#include <sys/time.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <ap_int.h>
#include <chrono>

#include "../ip/csc_matrix.h"
#include "options.hpp"
#include "utils.hpp"
#include "evaluation_utils.hpp"
#include "../ip/pagerank.hpp"
#include "gold_algorithms.hpp"

#include "../ip/csc_fpga.hpp"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

	/////////////////////////////
	// Input parameters /////////
	/////////////////////////////

	Options options = Options(argc, argv);
	int debug = true;	//options.debug;
	bool use_csc = options.use_csc;
	// FIXME: use relative path
	std::string graph_path =
			/*options.graph_path*/"/home/users/francesco.sgherzi/Projects/approximate-pagerank/data/graphs/mtx/scf_2019_10_02_13_26_56.mtx";
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
			std::cout << "Using sample graph located at: " << DEFAULT_MTX_FILE
					<< std::endl;
		}
		use_csc = false;
		graph_path = DEFAULT_MTX_FILE;
	}

	// Load dataset and create auxiliary vectors;
	auto start_1 = clock_type::now();
	csc_t input_m;
	input_m = load_graph_mtx(graph_path, debug, undirect_graph);

	auto end_1 = clock_type::now();
	auto start_2 = clock_type::now();
	csc_fixed_fpga_t f_input_m = convert_to_fixed_point_fpga(input_m);
	auto end_2 = clock_type::now();

	auto loading_time = chrono::duration_cast < chrono::milliseconds
			> (end_1 - start_1).count();
	auto to_fixed_time = chrono::duration_cast < chrono::milliseconds
			> (end_2 - start_2).count();

	results.n_edges = f_input_m.col_val.size();
	results.n_vertices = f_input_m.col_ptr.size() - 1;

	unsigned int N = results.n_vertices;
	unsigned int E = results.n_edges;

	if (debug) {
		std::cout << "Copying csc vectors..." << std::endl;
	}
	auto val = f_input_m.col_val;
	auto ptr = f_input_m.col_ptr;
	auto idx = f_input_m.col_idx;

	if (debug) {
		std::cout << "Creating auxiliary vectors..." << std::endl;
	}

	auto vec = std::vector<fixed_float>(N, 1.0 / N);
	auto res = std::vector<fixed_float>(N, 0.0);
	auto dangling_bitmap = std::vector<index_type>(N, 1);

	if (debug) {
		std::cout << "Setting up dangling bitmap..." << std::endl;
	}

	for (int i = 0; i < idx.size(); ++i) {
		dangling_bitmap[idx[i]] = 0;
	}

	if (debug) {
		std::cout << "\n----------------\n- Graph Summary -\n----------------"
				<< std::endl;
		std::cout << "- |V| = " << results.n_vertices << std::endl;
		std::cout << "- |E| = " << results.n_edges << std::endl;
		std::cout << "- Undirected: " << (undirect_graph ? "True" : "False")
				<< std::endl;
		std::cout << "- Alpha: " << alpha << std::endl;
		std::cout << "- Max. error: " << max_err << std::endl;
		std::cout << "- Max. iterations: " << max_iter << std::endl;
		std::cout << "----------------" << std::endl;
		std::cout << "- Loading time: " << loading_time / 1000 << " sec"
				<< std::endl;
		std::cout << "- Float-to-fixed time: " << to_fixed_time / 1000 << " sec"
				<< std::endl;
		std::cout << "- Tot. time: " << (loading_time + to_fixed_time) / 1000
				<< " sec" << std::endl;
		if (!use_csc) {
			long filesize_mb = get_file_size(graph_path) / 1000000;
			double throughput =
					loading_time > 0 ? 1000 * filesize_mb / loading_time : 0;
			std::cout << "- File size: " << filesize_mb << " MB" << std::endl;
			std::cout << "- Throughput: " << throughput << " MB/sec"
					<< std::endl;
		}
		std::cout << "----------------" << std::endl;
	}

	/////////////////////////////
	// Setup golden pagerank ////
	/////////////////////////////
	auto tmp_pr_golden = std::vector<float>(N, 0.0);
	auto initial_pr_golden = std::vector<float>(N, 1.0 / N);
	auto result_golden = std::vector<float>(N, 0.0);

	unsigned int iterations_to_convergence = 0;

	// TODO: prepare args for pagerank

	pagerank_golden(input_m.col_ptr.data(), input_m.col_idx.data(),
			input_m.col_val.data(), &N, &E, result_golden.data(),
			initial_pr_golden.data(), dangling_bitmap.data(),
			tmp_pr_golden.data(), &options.max_err, &options.alpha, &max_iter,
			&iterations_to_convergence);

	// Make sure that each vector has size multiple of BUFFER_SIZE, i.e. 512 bits.
	// Remove the starting 0 from the ptr vector, to align it with other vectors;
	int E_fixed = ((E + BUFFER_SIZE - 1) / BUFFER_SIZE) * BUFFER_SIZE;
	for (int i = 0; i < E_fixed - E; i++) {
		val.push_back(0);
		idx.push_back(0);
	}

	std::vector<dangling_type> dangling_bitmap_fpga;
	for (unsigned int &el : dangling_bitmap) {
		dangling_bitmap_fpga.push_back(el);
	}

	index_type num_blocks_V = N / BUFFER_SIZE;
	input_block *ptr_in = (input_block *) malloc(
			sizeof(input_block) * num_blocks_V);
	input_block *vec_in = (input_block *) malloc(
			sizeof(input_block) * (num_blocks_V + 1));
	input_block *res_in = (input_block *) malloc(
			sizeof(input_block) * (num_blocks_V + 1));

	// Dangling bitmap is... a bitmap
	// Every item is a single bit, which creates some problems
	// in the packing and the unpacking routines
	input_block *dangling_bmp_in = (input_block *) malloc(
			N > AP_UINT_BITWIDTH ?
					sizeof(input_block) * (N / AP_UINT_BITWIDTH) :
					sizeof(input_block));
	input_block *tmp_pr_in = (input_block *) malloc(
			sizeof(input_block) * (num_blocks_V + 1));

	for (uint i = 0; i < num_blocks_V; i++) {
		input_block ptr_val = 0;
		input_block vec_val = 0;
		input_block res_val = 0;
		input_block tmp_pr_val = 0;
		for (int j = 0; j < BUFFER_SIZE; ++j) {
			index_type curr_1 = ptr[BUFFER_SIZE * i + j + 1];
			fixed_float curr_2 = vec[BUFFER_SIZE * i + j];
			fixed_float curr_3 = res[BUFFER_SIZE * i + j];

			unsigned int lower_range = FIXED_WIDTH * j;
			unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
			unsigned int ptr_val_int = *((unsigned int *) &curr_1);
			unsigned int vec_val_int = *((unsigned int *) &curr_2);
			unsigned int res_val_int = *((unsigned int *) &curr_3);

			ptr_val.range(upper_range, lower_range) = ptr_val_int;
			vec_val.range(upper_range, lower_range) = vec_val_int;
			res_val.range(upper_range, lower_range) = res_val_int;
			tmp_pr_val.range(upper_range, lower_range) = (unsigned int) 0;
		}
		ptr_in[i] = ptr_val;
		vec_in[i] = vec_val;
		res_in[i] = res_val;
		tmp_pr_in[i] = tmp_pr_val;
	}

	// Add the remaining elements in the last chunk
	input_block vec_val_ecc = 0;
	input_block res_val_ecc = 0;
	input_block tmp_pr_val_ecc = 0;

	// Add the real remaining elements
	for (int j = 0; j < BUFFER_SIZE; ++j) {

		fixed_float curr_2 = 0.0;
		fixed_float curr_3 = 0.0;
		index_type curr_4 = 0;

		if (j < N % BUFFER_SIZE) {
			curr_2 = vec[BUFFER_SIZE * num_blocks_V + j];
			curr_3 = res[BUFFER_SIZE * num_blocks_V + j];
		}

		unsigned int lower_range = FIXED_WIDTH * j;
		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
		unsigned int vec_val_int = *((unsigned int *) &curr_2);
		unsigned int res_val_int = *((unsigned int *) &curr_3);
		unsigned int d_bmp = *((unsigned int *) &curr_4);

		vec_val_ecc.range(upper_range, lower_range) = vec_val_int;
		res_val_ecc.range(upper_range, lower_range) = res_val_int;
		tmp_pr_val_ecc.range(upper_range, lower_range) = 0;
	}

	vec_in[num_blocks_V] = vec_val_ecc;
	res_in[num_blocks_V] = res_val_ecc;
	tmp_pr_in[num_blocks_V] = tmp_pr_val_ecc;

	// Dangling bitmap packing routine
	for (int i = 0; i < N / AP_UINT_BITWIDTH; ++i) {

		const unsigned int chunk = i;
		input_block cur = 0;
		for (int j = 1; j < AP_UINT_BITWIDTH; ++j) {
			unsigned int lower = j - 1;
			unsigned int higher = j;
			cur.range(higher, lower) = dangling_bitmap_fpga[AP_UINT_BITWIDTH * i
					+ j];
		}

		dangling_bmp_in[i] = cur;

	}

	input_block cur_dangling = 0;

	// Dangling bitmap packing routine [ecceding partition]
	for (int j = 1; j < AP_UINT_BITWIDTH; ++j) {

		unsigned int lower = j - 1;
		unsigned int higher = j;

		if (j < N % AP_UINT_BITWIDTH) {
			cur_dangling.range(higher, lower) = dangling_bitmap_fpga[N
					- N % AP_UINT_BITWIDTH + j];
		} else {
			cur_dangling.range(higher, lower) = 0;
		}
	}

	index_type num_blocks_E = E_fixed / BUFFER_SIZE;
	input_block *idx_in = (input_block *) malloc(
			sizeof(input_block) * num_blocks_E);
	input_block *val_in = (input_block *) malloc(
			sizeof(input_block) * num_blocks_E);
	for (uint i = 0; i < num_blocks_E; i++) {
		input_block idx_val = 0;
		input_block val_val = 0;

		for (int j = 0; j < BUFFER_SIZE; ++j) {
			index_type curr_1 = idx[BUFFER_SIZE * i + j];
			fixed_float curr_2 = val[BUFFER_SIZE * i + j];

			unsigned int lower_range = FIXED_WIDTH * j;
			unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
			unsigned int idx_val_int = *((unsigned int *) &curr_1);
			unsigned int val_val_int = *((unsigned int *) &curr_2);

			idx_val.range(upper_range, lower_range) = idx_val_int;
			val_val.range(upper_range, lower_range) = val_val_int;
		}
		idx_in[i] = idx_val;
		val_in[i] = val_val;
	}

	/////////////////////////////
	// Execute the kernel ///////
	/////////////////////////////

	std::vector<double> trans_times(num_tests, 0);
	std::vector<double> exec_times(num_tests, 0);
	std::vector<double> errors(num_tests, 0);
	std::vector<double> ndcgs(num_tests, 0);
	for (uint i = 0; i < num_tests; i++) {
		if (debug) {
			std::cout << "Iteration " << i << ")" << std::endl;
		}

		fixed_float max_err = options.max_err;
		fixed_float alpha = options.alpha;
		index_type max_iter = /*options.max_iter*/iterations_to_convergence * 2;

		// TODO: remove the bound iterations_to_convergence
		// for now it is fixed to be twice the iterations that the golden one took to converge
		pagerank_main(ptr_in, idx_in, val_in, &N, &E, res_in, vec_in,
				dangling_bmp_in, tmp_pr_in, &max_err, &alpha, &max_iter);

		// Read back the result;
		for (uint i = 0; i < num_blocks_V; i++) {
			input_block temp_block = res_in[i];
			for (int j = 0; j < BUFFER_SIZE; ++j) {

				unsigned int lower_range = FIXED_WIDTH * j;
				unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
				fixed_float reset_value = 1.0 / N;
				unsigned int temp_res = temp_block.range(upper_range,
						lower_range);
				res[BUFFER_SIZE * i + j] = *((fixed_float *) &temp_res);

			}
		}

	}

	input_block temp_block = res_in[num_blocks_V];
	for (int j = 0; j < N % BUFFER_SIZE; ++j) {

		unsigned int lower_range = FIXED_WIDTH * j;
		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
		unsigned int temp_res = temp_block.range(upper_range, lower_range);
		res[BUFFER_SIZE * num_blocks_V + j] = *((fixed_float *) &temp_res);

	}

	auto idxs_fpga = sort_pr(N, res.data());
	auto idxs_gold = sort_pr(N, result_golden.data());
	auto idx_errors = 0;
	std::cout << "Sorted Positions" << std::endl;
	for (int i = 0; i < N; ++i) {
//		std::cout << "fpga ->" << idxs_fpga[i] << " golden " << idxs_gold[i]
//				<< std::endl;
		if (idxs_fpga[i] != idxs_gold[i]) {
			idx_errors++;
//			std::cout << "error in idx -> " << i << " \n\tfpga: "
//					<< idxs_fpga[i] << " golden " << idxs_gold[i] << std::endl;
		}

	}

	std::cout << "Indexing errors -> " << idx_errors << std::endl;

	fixed_float total_err = 0.0;
	for (int i = 0; i < N; ++i) {
//			std::cout << "Index " << i << " -> \n\tDifference: "
//					<< std::abs((result_golden[i] - res[i].to_float()))
//					<< " \n\t\tWith values := golden -> " << result_golden[i]
//					<< " FPGA -> " << res[i] << std::endl;

		total_err += (fixed_float) std::abs(
				(result_golden[i] - res[i].to_float()));

	}

	std::cout << "average error -> " << (total_err / N) << "." << std::endl;

	if (debug) {
		int old_precision = std::cout.precision();
		std::cout.precision(2);
		std::cout << "\n----------------\n- Results ------\n----------------"
				<< std::endl;
		std::cout << "Mean transfer time:  " << mean(trans_times) << "±"
				<< st_dev(trans_times) << " ms;" << std::endl;
		std::cout << "Mean execution time: " << mean(exec_times) << "±"
				<< st_dev(exec_times) << " ms;" << std::endl;
		std::cout << "Mean % error: " << mean(errors) << "±" << st_dev(errors)
				<< " %;" << std::endl;
		std::cout << "Mean NDCG: " << mean(ndcgs) << "±" << st_dev(ndcgs) << ";"
				<< std::endl;
		std::cout << "----------------" << std::endl;
		std::cout.precision(old_precision);
	}
}
