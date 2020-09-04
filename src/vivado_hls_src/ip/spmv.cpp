/*

#include "spmv.hpp"

/////////////////////////////
/////////////////////////////

void spmv_main(input_block *ptr, input_block *idx, input_block *val,
		index_type_fpga N, index_type_fpga E, input_block *result, input_block *vec) {

// Ports used to transfer data, using AXI master;
#pragma HLS INTERFACE m_axi port = ptr offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = idx offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = val offset = slave bundle = gmem2

#pragma HLS INTERFACE m_axi port = result offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = vec offset = slave bundle = gmem3

// Ports used for control signals, using AXI slave;
#pragma HLS INTERFACE s_axilite register port = N bundle = control
#pragma HLS INTERFACE s_axilite register port = E bundle = control
#pragma HLS INTERFACE s_axilite register port = return bundle = control

	// Allocate a local buffer that contains all the values of "vec";
	fixed_float vec_local[MAX_VERTICES];
#pragma HLS ARRAY_PARTITION variable=vec_local cyclic factor=16

 	// Copy values of "vec" in the local buffer;
	READ_VEC: for (index_type_fpga i = 0; i < N / BUFFER_SIZE; ++i){
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=64 max=64 avg=64
		read_block_float(vec[i], vec_local + i * BUFFER_SIZE);
	}

	// Execute the SPMV;
	spmv(ptr, idx, val, N, E, result, vec_local);
}

*/
