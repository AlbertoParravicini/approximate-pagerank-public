#include "distance.hpp"
#include "cmath"

void euclidean_distance_main(index_type_fpga *dim, fixed_float *result,
		input_block *v1, input_block *v2) {

//#pragma HLS INTERFACE m_axi port=dim offset=slave bundle=gmem0 depth=1
//#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem0 depth=1
//#pragma HLS INTERFACE m_axi port=v1 offset=slave bundle=gmem1 depth=1000
//#pragma HLS INTERFACE m_axi port=v2 offset=slave bundle=gmem2 depth=1000
//
//#pragma HLS INTERFACE s_axilite port=dim bundle=control
//#pragma HLS INTERFACE s_axilite port=result bundle=control
//#pragma HLS INTERFACE s_axilite port=v1 bundle=control
//#pragma HLS INTERFACE s_axilite port=v2 bundle=control
//#pragma HLS INTERFACE s_axilite port=return bundle=control

	index_type_fpga local_dim = *dim;
	fixed_float local_result_buffer[BUFFER_SIZE];
#pragma HLS array_partition variable=local_result_buffer
	fixed_float local_v1_even[BUFFER_SIZE];
#pragma HLS array_partition variable=local_v1_even
	fixed_float local_v2_even[BUFFER_SIZE];
#pragma HLS array_partition variable=local_v2_even
	fixed_float local_v1_odd[BUFFER_SIZE];
#pragma HLS array_partition variable=local_v1_odd
	fixed_float local_v2_odd[BUFFER_SIZE];
#pragma HLS array_partition variable=local_v2_odd

	for (int i = 0; i < local_dim / BUFFER_SIZE; ++i) {
#pragma HLS pipeline
#pragma HLS loop tripcount min=640 max=640 avg=640

		input_block block_in_v1 = v1[i];
		input_block block_in_v2 = v2[i];
		if (i % 2) {

			fixed_float local_result_even = 0.0;
			UNPACK_EVEN: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll

				int begin = 32 * j;
				int end = 32 * (j + 1) - 1;

				unsigned int val1 = block_in_v1.range(end, begin);
				unsigned int val2 = block_in_v2.range(end, begin);

				fixed_float tmp1 = *((fixed_float *) &val1);
				fixed_float tmp2 = *((fixed_float *) &val2);

				local_v1_even[j] = tmp1;
				local_v2_even[j] = tmp2;

			}

			euclidean_distance_512(&local_result_even, local_v1_even, local_v2_even);
			local_result_buffer[i % BUFFER_SIZE] += local_result_even;

		} else {
			fixed_float local_result_odd = 0.0;
			UNPACK_ODD: for(int j = 0; j < BUFFER_SIZE; ++j){
#pragma HLS unroll
				int begin = 32 * j;
				int end   = 32 * (j + 1) - 1;

				unsigned int val1 = block_in_v1.range(end, begin);
				unsigned int val2 = block_in_v2.range(end, begin);

				fixed_float tmp1 = *((fixed_float *) &val1);
				fixed_float tmp2 = *((fixed_float *) &val2);

				local_v1_odd[j] = tmp1;
				local_v2_odd[j] = tmp2;
			}

			euclidean_distance_512(&local_result_odd, local_v1_odd, local_v2_odd);
			local_result_buffer[i % BUFFER_SIZE] += local_result_odd;
		}

		for(int i = 0; i < BUFFER_SIZE; ++i){
#pragma HLS unroll
			*result += local_result_buffer[i];
		}


	}

	*result = sqrtf(*result);

}

void euclidean_distance_512(fixed_float *result, fixed_float v1[BUFFER_SIZE],
		fixed_float v2[BUFFER_SIZE]) {
#pragma HLS inline
#pragma HLS array_partition variable=v1 complete
#pragma HLS array_partition variable=v2 complete

	fixed_float tmp_buf[BUFFER_SIZE];
#pragma HLS array_partition variable=tmp_buf complete

	COMPUTE_EUCLIDEAN_DISTANCE: for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS UNROLL
		fixed_float tmp = (v1[i] - v2[i]);
		fixed_float tmp2 = tmp * tmp;
		tmp_buf[i] = tmp2;
	}

	*result += reduction_16(tmp_buf);

}

void euclidean_distance_512_ecceding(fixed_float *result, fixed_float v1[BUFFER_SIZE], fixed_float v2[BUFFER_SIZE], index_type_fpga real_elements){
#pragma HLS inline
#pragma HLS array_partition variable=v1 complete
#pragma HLS array_partition variable=v2 complete

	fixed_float tmp_buf[BUFFER_SIZE];
#pragma HLS array_partition variable=tmp_buf complete

	COMPUTE_EUCLIDEAN_DISTANCE_ECCEDING: for(int i = 0; i < BUFFER_SIZE; ++i){
#pragma HLS UNROLL
		fixed_float tmp = (v1[i] - v2[i]) * (i < real_elements);
		fixed_float tmp2 = tmp*tmp;
		tmp_buf[i] = tmp2;
	}

	*result += reduction_16(tmp_buf);

}

void l1_norm_512(fixed_float *result, fixed_float v1[BUFFER_SIZE], fixed_float v2[BUFFER_SIZE]){
#pragma HLS inline
	fixed_float tmp_buf[BUFFER_SIZE];
	COMPUTE_L1_NORM: for(int i = 0; i < BUFFER_SIZE; ++i){
#pragma HLS unroll
		fixed_float tmp = v1[i] - v2[i];
//		index_type_fpga tmp_int = *((index_type_fpga *) &tmp);
//		index_type_fpga abs = tmp_int & 0x7fffffff;
		tmp_buf[i] = tmp;
	}

	*result += reduction_16(tmp_buf);

}

void l1_norm_512_ecceding(fixed_float *result, fixed_float v1[BUFFER_SIZE], fixed_float v2[BUFFER_SIZE], index_type_fpga real_elements){
#pragma HLS inline
	fixed_float tmp_buf[BUFFER_SIZE];
	COMPUTE_L1_NORM: for(int i = 0; i < BUFFER_SIZE; ++i){
#pragma HLS unroll
		fixed_float tmp = (v1[i] - v2[i]) * (i < real_elements);
//		index_type_fpga tmp_int = *((index_type_fpga *) &tmp);
//		index_type_fpga abs = tmp_int & 0x7fffffff;
		tmp_buf[i] = tmp;

	}

	*result += reduction_16(tmp_buf);

}

