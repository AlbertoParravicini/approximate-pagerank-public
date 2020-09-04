#pragma once
#include "../../../csc_fpga/csc_fpga.hpp"
#include "../../../fpga_utils.hpp"
#ifndef BUFFER_SIZE
#define BUFFER_SIZE 16
#endif

extern "C" {
void euclidean_distance_main(index_type_fpga *dim, fixed_float *result,
		input_block *v1, input_block *v2);
void euclidean_distance_512(fixed_float *result, fixed_float v1[BUFFER_SIZE],
		fixed_float v2[BUFFER_SIZE]);
void euclidean_distance_512_ecceding(fixed_float *result,
		fixed_float v1[BUFFER_SIZE], fixed_float v2[BUFFER_SIZE],
		index_type_fpga real_elements);

void l1_norm_512(fixed_float *result, fixed_float v1[BUFFER_SIZE], fixed_float v2[BUFFER_SIZE]);

void l1_norm_512_ecceding(fixed_float *result, fixed_float v1[BUFFER_SIZE], fixed_float v2[BUFFER_SIZE], index_type_fpga real_elements);


}
