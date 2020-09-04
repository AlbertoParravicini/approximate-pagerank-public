#pragma once

#include "../../../csc_fpga/csc_fpga.hpp"
#include "../../../fpga_utils.hpp"

extern "C" {

inline void axpb_512(fixed_float *a, fixed_float x[BUFFER_SIZE], fixed_float *b) {
#pragma HLS inline
#pragma HLS array_partition variable=x complete

	fixed_float local_a = *a;
	fixed_float local_b = *b;
	COMPUTE_AXPB: for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS UNROLL
		x[i] = local_a * x[i] + local_b;
	}
}

}

