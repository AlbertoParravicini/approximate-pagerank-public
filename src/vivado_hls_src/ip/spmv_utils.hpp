#pragma once

#include "csc_fpga.hpp"
#include "fpga_utils.hpp"

extern "C" {

inline void read_block_vec(input_block block, fixed_float buffer_out[BUFFER_SIZE], fixed_float vec[MAX_VERTICES]) {
#pragma HLS ARRAY_PARTITION variable=vec cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=buffer_out complete dim=1

#pragma HLS INLINE

	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int lower_range = FIXED_WIDTH * j;
		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
		unsigned int block_curr = block.range(upper_range, lower_range);
		index_type_fpga temp_ind = *((index_type_fpga *)&block_curr);
	    buffer_out[j] = vec[temp_ind];
	}
}

inline void read_block_vec_gmem(input_block block, fixed_float buffer_out[BUFFER_SIZE], input_block *vec) {
#pragma HLS ARRAY_PARTITION variable=buffer_out complete dim=1
#pragma HLS INLINE

	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int lower_range = FIXED_WIDTH * j;
		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
		unsigned int block_curr = block.range(upper_range, lower_range);
		index_type_fpga temp_ind = *((index_type_fpga *)&block_curr);

		// Compute the block of "vec" where the required index is;
		index_type_fpga block_ind = temp_ind / BUFFER_SIZE;
		// Compute the position in the block where the required index is;
		index_type_fpga position_in_block_ind = temp_ind % BUFFER_SIZE;

		// Load the required "vec" value from the input block;
		input_block vec_curr_block = vec[block_ind];
		lower_range = FIXED_WIDTH * position_in_block_ind;
		upper_range = FIXED_WIDTH * (position_in_block_ind + 1) - 1;
		unsigned int required_vec_val = vec_curr_block.range(upper_range, lower_range);
		buffer_out[j] = *((fixed_float *)&required_vec_val);
	}
}

inline void reset_buffer(fixed_float buf[BUFFER_SIZE]){
#pragma HLS INLINE
	for(int i = 0; i < BUFFER_SIZE; ++i){
#pragma HLS unroll
		buf[i] = 0.0;
	}
}

inline void reset_large_buffer(fixed_float buf[2 * BUFFER_SIZE]){
#pragma HLS inline
	for(int i = 0; i < 2 * BUFFER_SIZE; ++i){
#pragma HLS unroll
		buf[i] = 0.0;
	}
}

}
