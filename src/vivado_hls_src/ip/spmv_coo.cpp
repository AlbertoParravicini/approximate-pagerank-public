#include "spmv_coo.hpp"

/////////////////////////////
/////////////////////////////

void spmv_coo_with_scatter_main(input_block *x, input_block *y, input_block *val, index_type N,
		index_type E, input_block *res, input_block *vec) {
// Ports used to transfer data, using AXI master;
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = y offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = val offset = slave bundle = gmem1

#pragma HLS INTERFACE m_axi port = res offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = vec offset = slave bundle = gmem3

// Ports used for control signals, using AXI slave;
#pragma HLS INTERFACE s_axilite register port = N bundle = control
#pragma HLS INTERFACE s_axilite register port = E bundle = control
#pragma HLS INTERFACE s_axilite register port = return bundle = control

	// Allocate a local buffer that contains all the values of "res" and "vec";
	fixed_float res_local[MAX_N];
#pragma HLS RESOURCE variable=res_local core=RAM_S2P_BRAM
#pragma HLS array_partition variable=res_local cyclic factor=hls_buffer_size

	fixed_float vec_local[MAX_N];
#pragma HLS RESOURCE variable=vec_local core=XPM_MEMORY uram
#pragma HLS ARRAY_PARTITION variable=vec_local cyclic factor=hls_buffer_size

	std::cout << "start spmv" << std::endl;

	RESET_RES_LOCAL: for (index_type i = 0; i < N / BUFFER_SIZE; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=hls_iterations_v max=hls_iterations_v avg=hls_iterations_v
		for (index_type j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
			res_local[i * BUFFER_SIZE + j] = 0;
		}
	}

	READ_VEC_LOCAL: for (index_type i = 0; i < N / BUFFER_SIZE; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=hls_iterations_v max=hls_iterations_v avg=hls_iterations_v
		read_block_float(vec[i], vec_local + i * BUFFER_SIZE);
	}

	// Execute the SPMV;
	spmv_coo_with_scatter(x, y, val, E, res_local, vec_local);

	// Copy values of "res" to the output;
	WRITE_RES: for (index_type i = 0; i < N / BUFFER_SIZE; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=hls_iterations_v max=hls_iterations_v avg=hls_iterations_v
		input_block tmp_block;
		write_block_float(&tmp_block, &res_local[i * BUFFER_SIZE]);
		res[i] = tmp_block;
	}
}
