#pragma once

#include "csc_fpga.hpp"
#include "fpga_utils.hpp"

#define ITERATIONS 10240 / BUFFER_SIZE

extern "C" {

void fixed_dot_512(index_type_fpga dim, fixed_float *result, input_block *a,
		input_block *b);

void dot_product_main(input_block *a, input_block *b, index_type_fpga *size,
		fixed_float *result);

void dot_product_512(fixed_float *result, fixed_float a[BUFFER_SIZE],
		dangling_type b[BUFFER_SIZE]);

void dot_product_512_ecceding(fixed_float *result, fixed_float a[BUFFER_SIZE],
		index_type_fpga b[BUFFER_SIZE], index_type_fpga real_elements);
}
