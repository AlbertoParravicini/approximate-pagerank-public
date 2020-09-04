#pragma once

#include "../../../csc_fpga/csc_fpga.hpp"
#include "../../../fpga_utils.hpp"

#define ITERATIONS 10240 / BUFFER_SIZE

extern "C" {

inline void dot_product_512(fixed_float *result, fixed_float a[BUFFER_SIZE], dangling_type b[BUFFER_SIZE]){
#pragma HLS inline
#pragma HLS array_partition variable=a complete
	fixed_float tmp[BUFFER_SIZE];
#pragma HLS array_partition variable=tmp complete
	DOTP_MULTIPLY: for(int i = 0; i < BUFFER_SIZE; ++i){
#pragma HLS UNROLL
		tmp[i] = a[i] * b[i];
	}

	*result += reduction_16(tmp);
}

}
