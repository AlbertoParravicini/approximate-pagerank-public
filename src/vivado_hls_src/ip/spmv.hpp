#pragma once

#include "csc_fpga.hpp"
#include "fpga_utils.hpp"

/////////////////////////////
/////////////////////////////

inline void read_block_vec(input_block block, fixed_float buffer_out[BUFFER_SIZE], fixed_float vec[MAX_VERTICES]) {
#pragma HLS array_partition variable=vec cyclic factor=16
#pragma HLS array_partition variable=buffer_out complete dim=1

#pragma HLS inline

	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
		unsigned int lower_range = FIXED_WIDTH * j;
		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
		unsigned int block_curr = block.range(upper_range, lower_range);
		index_type_fpga temp_ind = *((index_type_fpga *)&block_curr);
	    buffer_out[j] = vec[temp_ind];
	}
}

inline void reset_buffer(fixed_float buf[BUFFER_SIZE]){
#pragma HLS inline
	for(int i = 0; i < BUFFER_SIZE; ++i){
#pragma HLS unroll
		buf[i] = 0.0;
	}
}

/////////////////////////////
/////////////////////////////

/*

inline void spmv(input_block *ptr, input_block *idx, input_block *val,
			index_type_fpga N, index_type_fpga E, input_block *result, fixed_float vec[MAX_VERTICES]) {
#pragma HLS ARRAY_PARTITION variable=vec cyclic factor=16

	// Allocate a local buffer that contains values of "ptr", "idx", "val";
	index_type_fpga ptr_local[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=ptr_local complete dim=1
	index_type_fpga idx_local[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=idx_local complete dim=1
	fixed_float val_local[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=val_local complete dim=1
	fixed_float vec_local[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=vec_local complete dim=1
	// Buffer where partial results are stored;
	fixed_float res_local[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=res_local complete dim=1
	// Buffer where the spmv value of each vertex is stored;
	fixed_float curr_sum[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=res_local complete dim=1

	// Initialize curr_sum;
	reset_buffer(curr_sum);

	// Note that the first value of "ptr" is not 0;
	index_type_fpga begin = 0;
	// Keep track of which "edge blocks" we have to read;
	index_type_fpga curr_idx_block = 0;
	// Position on the current buffer;
	index_type_fpga curr_buff_position = 0;

	// Read the first block in the first iteration;
	read_block_vec(idx[0], vec_local, vec);
	read_block_float(val[0], val_local);
	curr_idx_block++;

	// Outer loop executed |N| / BUFFER_SIZE times;
	PTR_OUTER: for (index_type_fpga i = 0; i < N / BUFFER_SIZE; ++i) {
#pragma HLS loop_tripcount min=64 max=64 avg=64
#pragma HLS pipeline II=1

		// Read a block of "ptr" values;
		read_block_index(ptr[i], ptr_local);
		// Go through the current "ptr" values;
		PTR_INNER: for (index_type_fpga ii = 0; ii < BUFFER_SIZE; ii++) {
#pragma HLS unroll
			index_type_fpga end = ptr_local[ii];
			reset_buffer(curr_sum);

			// 3 parts:
			// - 1. Read values on the current chunk;
			// - 2. Load new chunks and fully process them;
			// - 3. Process values on a final chunk if required;
			index_type_fpga values_to_read = end > begin ? end - begin : 0; // The ptr vector is 0-padded, so we enforce reading 0 values in the padded part;
			// Values to read on the current chunk;
			index_type_fpga val_on_curr_block = curr_buff_position + values_to_read < BUFFER_SIZE ? values_to_read : BUFFER_SIZE - curr_buff_position;
			index_type_fpga values_on_next_blocks = values_to_read - val_on_curr_block;
			index_type_fpga full_blocks_to_process = values_on_next_blocks / BUFFER_SIZE;
			index_type_fpga values_on_last_block = values_on_next_blocks % BUFFER_SIZE;

			// Read values on the current chunk.
			// Perform always BUFFER_SIZE iterations, so we can unroll,
			// but multiply by a flag to ignore values outside of the requred boundary;
			VAL_INNER_1: for (index_type_fpga jj = 0; jj < BUFFER_SIZE; ++jj) {
#pragma HLS unroll
				curr_sum[jj] += val_local[jj] * vec_local[jj] * (jj - curr_buff_position < val_on_curr_block);
			}
			curr_buff_position += val_on_curr_block;

			// Read full blocks;
			// Crazy idea: sort incoming vertices by in-degree, so that more important vertices come first
			// in each vertex neighborhood list. Then process a fixed number of blocks for each vertex (e.g. 3)
			// and unroll everything!
			VAL_OUTER: for (index_type_fpga j = 0; j < full_blocks_to_process; ++j, ++curr_idx_block) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=10 max=10 avg=10 // Assume an average in-degree of 10 * BUFFER_SIZE;
				// Read new chunks;
				read_block_vec(idx[curr_idx_block], vec_local, vec);
				read_block_float(val[curr_idx_block], val_local);
				// Process the chunks;
				VAL_INNER_2: for (index_type_fpga jj = 0; jj < BUFFER_SIZE; ++jj) {
#pragma HLS unroll
					curr_sum[jj] += val_local[jj] * vec_local[jj];
				}
				curr_buff_position = BUFFER_SIZE;
			}

			// Process values on the last chunk;
			if (values_on_last_block > 0) {
				read_block_vec(idx[curr_idx_block], vec_local, vec);
				read_block_float(val[curr_idx_block], val_local);
				curr_idx_block++;
				curr_buff_position = values_on_last_block;
			}

			// Perform always BUFFER_SIZE iterations, so we can unroll,
			// but multiply by a flag to ignore values outside of the required boundary;
			VAL_INNER_3: for (index_type_fpga jj = 0; jj < BUFFER_SIZE; ++jj) {
#pragma HLS unroll
				curr_sum[jj] += val_local[jj] * vec_local[jj] * (jj < values_on_last_block);
			}

			// Write the new result;
			res_local[ii] = reduction(curr_sum);
			begin = end;
		}
		// Read a block of "ptr" values;
		input_block temp_block;
		write_block_float(&temp_block, res_local);
		result[i] = temp_block;
	}
}

void spmv_main(input_block *ptr, input_block *idx, input_block *val,
		index_type_fpga N, index_type_fpga E, input_block *result, input_block *vec);


*/
