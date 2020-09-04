#pragma once

#include <vector>
#include "opencl_utils.hpp"
#include "../../common/utils/utils.hpp"
#include "../../common/utils/options.hpp"
#include "fpga_utils.hpp"
#include "aligned_allocator.h"
#include <chrono>

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

#define allocator aligned_allocator<input_block>

/////////////////////////////
/////////////////////////////

struct PageRank {
	PageRank(index_type N, index_type E,
			index_type max_iter = MAX_ITER, fixed_float alpha = ALPHA, fixed_float max_error = MAX_ERROR, std::vector<index_type, aligned_allocator<index_type>> personalization_vertices = {}) :
			N(N), E(E), max_iter(max_iter), alpha(alpha), max_error(max_error), personalization_vertices(personalization_vertices) {

		// Initialize support arrays;
		result = std::vector<fixed_float>(N, 1.0 / N);
		pr = std::vector<fixed_float>(N, 1.0 / N);
		dangling_bitmap = std::vector<index_type>(N, 1);
		this->personalization_vertices = std::vector<index_type, aligned_allocator<index_type>>(N_PPR_VERTICES, 0);

		num_blocks_N = (N + BUFFER_SIZE - 1) / BUFFER_SIZE;
        N_padded = num_blocks_N * BUFFER_SIZE;
        num_blocks_E = (E + BUFFER_SIZE - 1) / BUFFER_SIZE;
        E_padded = num_blocks_E * BUFFER_SIZE;
        num_blocks_bitmap = (N + AP_UINT_BITWIDTH - 1) / AP_UINT_BITWIDTH;

        for(int i = 0; i < N_PPR_VERTICES; i++){
        	if (i < personalization_vertices.size()) {
            	this->personalization_vertices[i] = personalization_vertices[i];
        	}
		}
	}

	virtual void setup_inputs(ConfigOpenCL &config, bool debug = false) = 0;
	virtual void initialize_dangling_bitmap() = 0;
	virtual double transfer_input_data(ConfigOpenCL &config, bool measure_time = true, bool debug = false) = 0;
	virtual double execute(ConfigOpenCL &config, bool measure_time = true, bool debug = false) = 0;
	virtual double reset(ConfigOpenCL &config, bool measure_time = true, bool debug = false) = 0;
    virtual void preprocess_inputs() = 0;

	// Instance parameters;
	index_type N;
	index_type E;
	std::vector<index_type, aligned_allocator<index_type>> personalization_vertices;

	index_type max_iter;
	float alpha;
	float max_error;

	// Support arrays;
	std::vector<fixed_float> result;
	std::vector<fixed_float> pr;
	std::vector<index_type> dangling_bitmap;

	// OpenCL stuff;

	// Device vectors;
	cl::Buffer d_pr;
	cl::Buffer d_result;
	cl::Buffer d_dangling_bitmap;
	cl::Buffer d_personalization_vertices;

	// Kernel inputs (partitioned);
    std::vector<input_block, allocator> dangling_bitmap_in;
    std::vector<input_block, allocator> pr_in;

	// Kernel output (partitioned);
	std::vector<input_block, allocator> result_out;

	// Instance dimensions
	index_type N_padded;
	index_type E_padded;
	index_type num_blocks_E;
    index_type num_blocks_N;
    index_type num_blocks_bitmap;

};
