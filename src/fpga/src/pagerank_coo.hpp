#pragma once

#include "pagerank.hpp"
#include "coo_fpga/coo_fpga.hpp"

struct PageRankCOO : PageRank {
	PageRankCOO(index_type N, index_type E, coo_fixed_fpga_t *input_graph,
				index_type max_iter = MAX_ITER, fixed_float alpha = ALPHA, fixed_float max_error = MAX_ERROR, std::vector<index_type, aligned_allocator<index_type>> personalization_vertices = {}) :
				input_graph(input_graph), PageRank(N, E, max_iter, alpha, max_error, personalization_vertices) {
		initialize_dangling_bitmap();

		result = std::vector<fixed_float>(N * N_PPR_VERTICES, 0);

		// Update the number of blocks to reflect the use of additional self-loops;
		E_fixed = input_graph->E_fixed;
        num_blocks_E = (E_fixed + BUFFER_SIZE - 1) / BUFFER_SIZE;
        E_padded = num_blocks_E * BUFFER_SIZE;
	}

	// Number of edges counting extra self-loops;
	index_type E_fixed = E;
	// Graph used in the computation;
	coo_fixed_fpga_t *input_graph;

	// Device vectors;
	cl::Buffer d_coo_start;
	cl::Buffer d_coo_end;
	cl::Buffer d_coo_val;

	// Kernel inputs (partitioned);
	std::vector<input_block, allocator> start_in;
    std::vector<input_block, allocator> end_in;
    std::vector<input_block, allocator> val_in;

	void setup_inputs(ConfigOpenCL &config, bool debug = false) override;
	void initialize_dangling_bitmap() override;
	double transfer_input_data(ConfigOpenCL &config, bool measure_time = true, bool debug = false) override;
	double execute(ConfigOpenCL &config, bool measure_time = true, bool debug = false) override;
	double reset(ConfigOpenCL &config, bool measure_time = true, bool debug = false) override;
    void preprocess_inputs() override;
};
