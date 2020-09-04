#pragma once

#include "csc_fpga/csc_fpga.hpp"
#ifndef uint
#define uint unsigned int
#endif

/////////////////////////////
/////////////////////////////

// Read 16 values from a  bits block, and write them in a buffer of the specified type;
inline void read_block_index(input_block block, index_type_fpga buffer_out[BUFFER_SIZE]) {
#pragma HLS INLINE
	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int lower_range = FIXED_WIDTH * j;
		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
		unsigned int block_curr = block.range(upper_range, lower_range);
		buffer_out[j] = block_curr;
	}
}

inline void write_block_index(input_block *block, index_type_fpga buffer_in[BUFFER_SIZE]) {
#pragma HLS INLINE
	WRITE_BLOCK: for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS unroll
		unsigned int lower = FIXED_WIDTH * i;
		unsigned int upper = FIXED_WIDTH * (i + 1) - 1;
		block->range(upper, lower) = buffer_in[i];
	}
}

inline void read_block_dangling(input_block block, dangling_type buffer_out[AP_UINT_BITWIDTH]) {
#pragma HLS INLINE
	READ_BLOCK: for (int j = 0; j < AP_UINT_BITWIDTH; ++j) {
#pragma HLS unroll
		buffer_out[j] = block.bit(j);
	}
}

inline void read_block_float(input_block block, fixed_float buffer_out[BUFFER_SIZE]) {
#pragma HLS INLINE
	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int lower_range = FIXED_WIDTH * j;
		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
		unsigned int block_curr = block.range(upper_range, lower_range);
		buffer_out[j] = *((fixed_float *) &block_curr);
	}
}

// Write 16 values to a 512 bits block, taking them from a buffer of the specified type;
inline void write_block_float(input_block* block, fixed_float buffer_in[BUFFER_SIZE]) {
#pragma HLS INLINE
	WRITE_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		fixed_float curr_val = buffer_in[j];
		unsigned int lower_range = FIXED_WIDTH * j;
		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
		block->range(upper_range, lower_range) = *((unsigned int *) &curr_val);
	}
}

inline void memcpy_buf_to_buf(input_block *dest, input_block *src) {
#pragma HLS INLINE
	MEMCPY: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int begin = FIXED_WIDTH * j;
		unsigned int end = FIXED_WIDTH * (j + 1) - 1;
		unsigned int value = src->range(end, begin);
		dest->range(end, begin) = value;
	}
}

template<typename T>
inline void write_packed_array(
		T* array_in,
		input_block* array_packed_out,
		index_type array_size,
		index_type array_packed_size,
		uint buffer_size = BUFFER_SIZE,
		uint bitwidth = FIXED_WIDTH) {
	for (uint i = 0; i < array_packed_size; i++) {
		input_block new_block = 0;
		for (uint j = 0; j < buffer_size; j++) {
			T curr_val = buffer_size * i + j < array_size ? array_in[buffer_size * i + j] : (T) 0;
			unsigned int lower_range = bitwidth * j;
			unsigned int upper_range = bitwidth * (j + 1) - 1;
			unsigned int curr_val_in = *((unsigned int *) &curr_val);
			new_block.range(upper_range, lower_range) = curr_val_in;
		}
		array_packed_out[i] = new_block;
	}
}

// Write a matrix to packets, padding to 0 each row.
// For example, the matrix composed of column vectors [[1,2,3], [4,5,6], [7,8,9]]
// will become [[1,2,3,0], [4,5,6,0], [7,8,9,0]];
template<typename T>
inline void write_packed_matrix(
		T* array_in,
		input_block* array_packed_out,
		index_type num_columns,
		index_type num_rows,
		index_type array_packed_size,
		uint buffer_size = BUFFER_SIZE,
		uint bitwidth = FIXED_WIDTH) {
	for (uint c = 0; c < num_columns; c++) {
		int curr_start = c * num_rows; // Start of the current column in the input array;
		for (uint i = 0; i < array_packed_size; i++) {
			input_block new_block = 0;
			for (uint j = 0; j < buffer_size; j++) {
				T curr_val = ((buffer_size * i + j) < num_rows) ? array_in[(buffer_size * i + j) + curr_start] : (T) 0;
				unsigned int lower_range = bitwidth * j;
				unsigned int upper_range = bitwidth * (j + 1) - 1;
				unsigned int curr_val_in = *((unsigned int *) &curr_val);
				new_block.range(upper_range, lower_range) = curr_val_in;
			}
			array_packed_out[i + c * array_packed_size] = new_block;
		}
	}
}

template<typename T>
inline void read_packed_array(
		T* array_out,
		input_block* array_packed_in,
		index_type array_size,
		index_type array_packed_size,
		uint buffer_size = BUFFER_SIZE,
		uint bitwidth = FIXED_WIDTH) {
	for (uint i = 0; i < array_packed_size; i++) {
		input_block curr_block = array_packed_in[i];
		for (uint j = 0; j < buffer_size; j++) {
			if (buffer_size * i + j < array_size) {
				unsigned int lower_range = bitwidth * j;
				unsigned int upper_range = bitwidth * (j + 1) - 1;
				unsigned int val_curr_block = curr_block.range(upper_range, lower_range);
				array_out[buffer_size * i + j] = *((T*) &val_curr_block);
			}
		}
	}
}

template<typename T>
inline void read_packed_matrix(
		T* array_out,
		index_type num_columns, // 5
		index_type num_rows, // N
		input_block* array_packed_in,
		index_type array_packed_size, // num_blocks_n
		uint buffer_size = BUFFER_SIZE,
		uint bitwidth = FIXED_WIDTH) {
	for (uint c = 0; c < num_columns; c++) {
		int curr_start = c * num_rows; // Start of the current column in the input array;
		for (uint i = 0; i < array_packed_size; i++) {
			input_block curr_block = array_packed_in[i + c * array_packed_size];
			for (uint j = 0; j < buffer_size; j++) {
				if (buffer_size * i + j < num_rows) {
					unsigned int lower_range = bitwidth * j;
					unsigned int upper_range = bitwidth * (j + 1) - 1;
					unsigned int val_curr_block = curr_block.range(upper_range, lower_range);
					array_out[buffer_size * i + j + curr_start] = *((T*) &val_curr_block);
				}
			}
		}
	}
}

/////////////////////////////
/////////////////////////////

inline fixed_float reduction_16(fixed_float input[16]) {
#pragma HLS INLINE
#pragma HLS array_partition variable=input complete
	fixed_float acc = 0.0;
	for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS UNROLL
		acc += input[i];
	}
	return acc;
}

template<typename T>
inline T reduction(T input[BUFFER_SIZE]) {
#pragma HLS INLINE
#pragma HLS array_partition variable=input complete
	T acc = 0.0;
	for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS UNROLL
		acc += input[i];
	}
	return acc;
}
