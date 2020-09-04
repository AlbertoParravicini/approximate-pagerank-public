#pragma once

#include "pagerank.hpp"
#include "csc_fpga/csc_fpga.hpp"

struct PageRankCSC : PageRank {
	PageRankCSC(index_type N, index_type E, csc_fixed_fpga_t *input_graph,
				index_type max_iter = MAX_ITER, fixed_float alpha = ALPHA, fixed_float max_error = MAX_ERROR, std::vector<index_type, aligned_allocator<index_type>> personalization_vertices = {}) :
				input_graph(input_graph), PageRank(N, E, max_iter, alpha, max_error, personalization_vertices) {
		initialize_dangling_bitmap();
		pr_tmp = std::vector<fixed_float>(N, 1.0 / N);
	}

	// Graph used in the computation;
	csc_fixed_fpga_t *input_graph;
	std::vector<fixed_float> pr_tmp;

	// Device vectors;
	cl::Buffer d_csc_col_val;
	cl::Buffer d_csc_col_ptr;
	cl::Buffer d_csc_col_idx;
	cl::Buffer d_pr_tmp;

	// Kernel inputs (partitioned);
	std::vector<input_block, allocator> ptr_in;
    std::vector<input_block, allocator> idx_in;
    std::vector<input_block, allocator> val_in;
	std::vector<input_block, allocator> pr_tmp_in;

	void setup_inputs(ConfigOpenCL &config, bool debug = false) override;
	void initialize_dangling_bitmap() override;
	double transfer_input_data(ConfigOpenCL &config, bool measure_time = true, bool debug = false) override;
	double execute(ConfigOpenCL &config, bool measure_time = true, bool debug = false) override;
	double reset(ConfigOpenCL &config, bool measure_time = true, bool debug = false) override;
    void preprocess_inputs() override;
};
