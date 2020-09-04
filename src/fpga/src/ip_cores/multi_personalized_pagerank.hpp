#pragma once

#include "../fpga_utils.hpp"
//#include "pagerank_ops/spmv/spmv_coo_multi.hpp"
#include "pagerank_ops/spmv/spmv_coo_multi_stream.hpp"

extern "C" {

void compute_scaling_factor(index_type_fpga num_blocks_n,
		fixed_float dangling_scale, fixed_float shift_factor,
		input_block *dangling_bitmap, fixed_float pr[N_PPR_VERTICES][MAX_VERTICES],
		fixed_float *scaling_factor);

void personalized_pagerank_vector_ops_local_buffer_only(
		index_type_fpga num_blocks_n, fixed_float alpha,
		fixed_float scaling_factor[N_PPR_VERTICES], fixed_float pr_in[N_PPR_VERTICES][MAX_VERTICES],
		fixed_float pr_out[N_PPR_VERTICES][MAX_VERTICES], fixed_float shift_factor,
		index_type_fpga personalization_vertices[N_PPR_VERTICES]);

void multi_ppr_main(input_block *start, input_block *end,
		input_block *val, index_type_fpga N, index_type_fpga E,
		input_block *result, input_block *dangling_bitmap,
		float max_err, float alpha, index_type_fpga max_iter, index_type *personalization_vertices);
}

