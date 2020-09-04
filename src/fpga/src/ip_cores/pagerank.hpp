#pragma once

#include "../fpga_utils.hpp"
#include "pagerank_ops/spmv/spmv.hpp"
#include "pagerank_ops/spmv/spmv_coo.hpp"
#include "pagerank_ops/axpb/axpb.hpp"

extern "C" {
inline void pagerank_vector_ops(index_type_fpga N, fixed_float dangling_scale,
		fixed_float shift_factor, fixed_float alpha, fixed_float max_err,
		input_block *dangling_bitmap, fixed_float pr[MAX_VERTICES],
		input_block *pr_tmp);

inline void pagerank_vector_ops_gmem(index_type_fpga N,
		fixed_float dangling_scale, fixed_float shift_factor, fixed_float alpha,
		fixed_float max_err, input_block *dangling_bitmap, input_block *pr,
		input_block *pr_tmp);

inline void compute_scaling_factor(index_type_fpga num_blocks_n,
		fixed_float dangling_scale, fixed_float shift_factor,
		input_block *dangling_bitmap, fixed_float pr[MAX_VERTICES],
		fixed_float *scaling_factor);

inline void personalized_pagerank_vector_ops_local_buffer_only(
		index_type_fpga num_blocks_n, fixed_float alpha,
		fixed_float scaling_factor, fixed_float pr[MAX_VERTICES],
		fixed_float pr_write_back[MAX_VERTICES], fixed_float shift_factor,
		index_type_fpga preferred_index);

void pagerank_main(input_block *ptr, input_block *idx, input_block *val,
		index_type_fpga N, index_type_fpga E, input_block *result,
		input_block *pr, input_block *dangling_bitmap, input_block *pr_tmp,
		float max_err, float alpha, index_type_fpga max_iter);

void personalized_pagerank_coo_main(input_block *start, input_block *end,
		input_block *val, index_type_fpga N, index_type_fpga E,
		input_block *result, input_block *pr, input_block *dangling_bitmap,
		input_block *pr_tmp, float max_err,
		float alpha, index_type_fpga max_iter, index_type_fpga personalization_vertex);
}

