#include "dot_product.hpp"

/////////////////////////////
/////////////////////////////


void fixed_dot_512(index_type_fpga dim, fixed_float *result, input_block *a,
		input_block *b) {
	*result = 0;
	fixed_float buffer_res[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer_res complete dim=1

	// Initialize the first buffer;
	for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
		buffer_res[j] = 0;
	}

	DOT_LOOP_S: for (int i = 0; i < (dim / BUFFER_SIZE); ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=640 max=640 avg=640

		input_block a_val = a[i];
		input_block b_val = b[i];

		PROCESS_BUFFER: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
			unsigned int lower_range = 32 * j;
			unsigned int upper_range = 32 * (j + 1) - 1;
			unsigned int a_val_curr = a_val.range(upper_range, lower_range);
			fixed_float a_val_float = *((fixed_float *) &a_val_curr);
			unsigned int b_val_curr = b_val.range(upper_range, lower_range);
			fixed_float b_val_float = *((fixed_float *) &b_val_curr);

			buffer_res[j] += a_val_float * b_val_float;
		}
	}

	// Add the temporary results;
	*result = reduction_16(buffer_res);
}

void dot_product_main(input_block *a, input_block *b, index_type_fpga *size,
		fixed_float *result) {
	fixed_dot_512(*size, result, a, b);
}

void dot_product_512(fixed_float *result, fixed_float a[BUFFER_SIZE], dangling_type b[BUFFER_SIZE]){
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

void dot_product_512_ecceding(fixed_float *result, fixed_float a[BUFFER_SIZE], index_type_fpga b[BUFFER_SIZE], index_type_fpga real_elements){
#pragma HLS inline
	fixed_float tmp[BUFFER_SIZE];
#pragma HLS array_partition variable=tmp complete
	DOTP_MULTIPLY_CONDITIONALLY: for(int i = 0; i < BUFFER_SIZE; ++i){
		tmp[i] = a[i] * b[i] * (i < real_elements);
	}

	*result += reduction_16(tmp);

}


