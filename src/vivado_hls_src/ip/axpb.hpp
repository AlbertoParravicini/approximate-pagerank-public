#pragma once

#include "csc_fpga.hpp"
#include "spmv.hpp"

extern "C" {

void axpb_512(fixed_float a, fixed_float x[BUFFER_SIZE], fixed_float b);

void axpb_main(index_type_fpga *dim, fixed_float *a, input_block *block_in,
		fixed_float *b);

void axpb_512_ecceding(fixed_float a, fixed_float x[BUFFER_SIZE], fixed_float b,
		index_type_fpga real_elements);

}

