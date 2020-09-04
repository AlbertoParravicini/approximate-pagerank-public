#include "axpb.hpp"

void axpb_main(index_type_fpga *dim, fixed_float *a, input_block *block_in,
		fixed_float *b){
//#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem0 depth=1
//#pragma HLS INTERFACE m_axi port = block_in offset = slave bundle = gmem1 depth=1000
//#pragma HLS INTERFACE m_axi port = b offset = slave bundle = gmem0 depth=1
//
//// Ports used for control signals, using AXI slave;
//#pragma HLS INTERFACE s_axilite port = a bundle = control
//#pragma HLS INTERFACE s_axilite port = block_in bundle = control
//#pragma HLS INTERFACE s_axilite port = b bundle = control
//#pragma HLS INTERFACE s_axilite port = return bundle = control

	fixed_float local_x[BUFFER_SIZE];
	fixed_float local_a   = *a;
	fixed_float local_b   = *b;
	index_type_fpga  local_dim = *dim;
// TODO: handle case when dim % BUFFER_SIZE != 0

	PARTITION_LOOP: for (int i = 0; i < local_dim / BUFFER_SIZE; ++i) {
#pragma HLS pipeline
#pragma HLS loop tripcount min=640 max=640 avg=640

		// Unpack the buffer
		input_block local_block = block_in[i];

		read_block_float(local_block, local_x);

		// Perform computation
		axpb_512(local_a, local_x, local_b);

		// Repack the buffer
		input_block tmp_block;
		REPACK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
			unsigned int lower = j * 32;
			unsigned int upper = (j + 1) * 32 - 1;
			unsigned int tmp = *((unsigned int *) &local_x[j]);
			tmp_block.range(upper, lower) = tmp;
		}

		block_in[i] = tmp_block;

	}

}

void axpb_512(fixed_float a, fixed_float x[BUFFER_SIZE], fixed_float b) {
#pragma HLS inline
#pragma HLS array_partition variable=x complete

	COMPUTE_AXPB: for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS UNROLL
		x[i] = a * x[i] + b;
	}
}

void axpb_512_ecceding(fixed_float a, fixed_float x[BUFFER_SIZE], fixed_float b, index_type_fpga real_elements){
	COMPUTE_AXPB_ECCEDING: for(int i = 0; i < BUFFER_SIZE; ++i){
#pragma HLS UNROLL
		x[i] = (a * x[i] + b ) * (i < real_elements);
	}
}
