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
#include "gold_algorithms.hpp"

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 16
#endif
#include "axpb.hpp"
#include "csc_fpga.hpp"

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
	std::string graph_path = options.graph_path;
	unsigned int max_iter = options.max_iter;
	fixed_float alpha = options.alpha;
	fixed_float max_err = options.max_err;
	bool use_sample_graph = options.use_sample_graph;
	uint num_tests = options.num_tests;
	bool undirect_graph = options.undirect_graph;
	std::string xclbin_path = options.xclbin_path;

	index_type N = 10240;

	// Random vector of size N
	std::vector<fixed_float> v1 = std::vector<fixed_float>(N, 0.0);
	random_array<fixed_float>(v1.data(), v1.size(), 1);
	std::vector<fixed_float> v2 = std::vector<fixed_float>(N, 0.0);
	random_array<fixed_float>(v2.data(), v2.size(), 1);

	/////////////////////////////////
	// Test for euclidean distance //
	/////////////////////////////////

	// Partition the vectors into `streams` of 512 bits
	index_type num_blocks_V = N / BUFFER_SIZE;
	input_block *stream_a_axpb = (input_block *) malloc(
			sizeof(input_block) * num_blocks_V);


	for (uint i = 0; i < num_blocks_V; i++) {
		input_block stream_a_partition = 0;

		for (int j = 0; j < BUFFER_SIZE; ++j) {

			fixed_float curr_1 = v1[BUFFER_SIZE * i + j];

			unsigned int lower_range = 32 * j;
			unsigned int upper_range = 32 * (j + 1) - 1;
			unsigned int partition_a = *((unsigned int *) &curr_1);

			stream_a_partition.range(upper_range, lower_range) = partition_a;
		}

		stream_a_axpb[i] = stream_a_partition;

	}

	std::vector<fixed_float> result_golden(N, 0.0);

	fixed_float a = 0.1;
	fixed_float b = 0.2;
	axpb_gold(N, result_golden.data(), &a, v1.data(), &b);

	/////////////////////////////
	// Execute the kernel ///////
	/////////////////////////////
	axpb_main(&N, &a, stream_a_axpb, &b);

	std::vector<fixed_float> result_fixed(N);

	for (int i = 0; i < num_blocks_V; ++i) {

		input_block curr = stream_a_axpb[i];

		for (int j = 0; j < BUFFER_SIZE; ++j) {

			unsigned int lower = 32 * j;
			unsigned int upper = 32 * (j + 1) - 1;

			// Separate extraction and assignment because -Waddress-of-temporary
			unsigned int tmp_block = curr.range(upper, lower);
			result_fixed[i * BUFFER_SIZE + j] = *((fixed_float *) &tmp_block);

		}

	}

	fixed_float sum_axpb_errors = 0.0;

	for (int i = 0; i < N; ++i) {
		sum_axpb_errors += std::abs(result_fixed[i].to_float() - result_golden[i].to_float());
	}

	std::cout << "AXPB -> error: " << sum_axpb_errors << std::endl;


	/////////////////////////
	/// EUCLIDEAN DISTANCE //
	/////////////////////////
	std::cout << "Beginning EUCLIDEAN_DISTANCE" << std::endl;
	v1.clear();
	v2.clear();

	random_array(v1.data(), N, 1);
	random_array(v2.data(), N, 1);



	input_block *stream_a_euc = (input_block *) malloc(
				sizeof(input_block) * num_blocks_V);
	input_block *stream_b_euc = (input_block *) malloc(
				sizeof(input_block) * num_blocks_V);


	for (uint i = 0; i < num_blocks_V; i++) {
		input_block stream_a_partition = 0;
		input_block stream_b_partition = 0;

		for (int j = 0; j < BUFFER_SIZE; ++j) {

			fixed_float curr_1 = v1[BUFFER_SIZE * i + j];
			fixed_float curr_2 = v2[BUFFER_SIZE * i + j];
			unsigned int lower_range = 32 * j;
			unsigned int upper_range = 32 * (j + 1) - 1;
			unsigned int partition_a = *((unsigned int *) &curr_1);
			unsigned int partition_b = *((unsigned int *) &curr_2);

			stream_a_partition.range(upper_range, lower_range) = partition_a;
			stream_b_partition.range(upper_range, lower_range) = partition_b;
		}

		stream_a_euc[i] = stream_a_partition;
		stream_b_euc[i] = stream_b_partition;

	}

	std::vector<fixed_float> result_golden(N, 0.0);

	fixed_float result_euc_golden = 0.0;
	euclidean_distance_gold(N, &result_euc_golden, v1.data(), v2.data());

	fixed_float result_fpga = 0.0;
	euclidean_distance_main(&N, &result_fpga, stream_a_euc, stream_b_euc);

	std::cout << "Euclidean Distance -> diff: " << result_euc_golden - result_fpga << std::endl;

}
