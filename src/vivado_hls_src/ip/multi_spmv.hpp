#pragma once

#include "csc_fpga.hpp"
#include "fpga_utils.hpp"
#include "spmv_utils.hpp"

/////////////////////////////
/////////////////////////////

extern "C" {

// This looks better than 2 (pipeline length 4 instead of 64);
inline void reduction_tree_1(
		fixed_float pointwise_res[BUFFER_SIZE],
		index_type x_local[BUFFER_SIZE],
		fixed_float aggregated_res[2 * BUFFER_SIZE],
		index_type start_in_buffer,
		index_type start_x);

inline void reduction_tree_2(
		fixed_float pointwise_res[BUFFER_SIZE],
		index_type x_local[BUFFER_SIZE],
		fixed_float aggregated_res[2 * BUFFER_SIZE],
		index_type start_in_buffer,
		index_type start_x);

/////////////////////////////
/////////////////////////////

inline void inner_spmv_product(
		index_type x[BUFFER_SIZE],
		index_type y[BUFFER_SIZE],
		fixed_float val[BUFFER_SIZE],
		fixed_float vec[MAX_N],
		fixed_float aggregated_res_local[2 * BUFFER_SIZE],
		index_type start_x,
		index_type start_in_buffer);

/////////////////////////////
/////////////////////////////

inline void spmv_coo_multi(input_block *x, input_block *y, input_block *val,
		index_type E, fixed_float res[N_PPR_VERTICES][MAX_N], fixed_float vec[N_PPR_VERTICES][MAX_N]);

void multi_spmv_main(input_block *x, input_block *y, input_block *val, index_type N,
		index_type E, input_block *res, input_block *vec);
}
