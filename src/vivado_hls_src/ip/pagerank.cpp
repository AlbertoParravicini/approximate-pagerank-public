#include "pagerank.hpp"

void pagerank_vector_ops(index_type_fpga dim, fixed_float dangling_scale,
		fixed_float shift_factor, fixed_float alpha, fixed_float max_err,
		input_block *dangling_bitmap, input_block *pr_vec, input_block *tmp_pr) {

	dangling_type local_dangling_bitmap[AP_UINT_BITWIDTH];
#pragma HLS array_partition variable=local_dangling_bitmap cyclic factor=16
	fixed_float local_pr_vec[BUFFER_SIZE];
#pragma HLS array_partition variable=local_pr_vec complete
	fixed_float local_tmp_pr[BUFFER_SIZE];
#pragma HLS array_partition variable=local_tmp_pr complete

	fixed_float dangling_contribution = 0.0;

	DANGLING_CONTRIBUTION: for (int i = 0; i < dim / BUFFER_SIZE + 1; ++i) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=63 max=63 avg=63
#pragma HLS dependence variable=local_dangling_bitmap inter false
#pragma HLS dependence variable=local_pr_vec inter false
#pragma HLS dependence variable=local_dangling_bitmap intra false
#pragma HLS dependence variable=local_pr_vec intra false

		input_block cur_dangling_bitmap = dangling_bitmap[i * BUFFER_SIZE / AP_UINT_BITWIDTH];
		const unsigned int offset = (i * BUFFER_SIZE) % (AP_UINT_BITWIDTH);
		READ_BITMAP: {
			// Note that this is instance dependent
#pragma HLS occurrence cycle=31
		if(offset == 0) read_block_dangling(cur_dangling_bitmap, local_dangling_bitmap);
		}
		read_block_float(pr_vec[i], local_pr_vec);
		dot_product_512(&dangling_contribution, local_pr_vec,
				local_dangling_bitmap + offset);
	}

	fixed_float scaling_factor = shift_factor
			+ (dangling_scale * dangling_contribution);

	AXPB: for (int i = 0; i < dim / BUFFER_SIZE + 1; ++i) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=63 max=63 avg=63
#pragma HLS dependence variable=local_tmp_pr inter false
#pragma HLS dependence variable=local_pr_vec inter false
#pragma HLS dependence variable=local_tmp_pr intra false
#pragma HLS dependence variable=local_pr_vec intra false
		read_block_float(tmp_pr[i], local_tmp_pr);
		axpb_512(alpha, local_tmp_pr, scaling_factor);

		//euclidean_distance_512(&error, local_tmp_pr, local_pr_vec);
		input_block tmp_block1;
		write_block_float(&tmp_block1, local_tmp_pr);
		pr_vec[i] = tmp_block1;
	}

//	// Perform axpb and euclidean distance on ecceding elements
//	read_block_float(tmp_pr[dim / BUFFER_SIZE], local_tmp_pr);
//	axpb_512_ecceding(alpha, local_tmp_pr, scaling_factor, dim % BUFFER_SIZE);
//	//euclidean_distance_512_ecceding(&error, local_tmp_pr, local_pr_vec, dim % BUFFER_SIZE);
//	input_block tmp_block1;
//	write_block_float(&tmp_block1, local_tmp_pr);
//	pr_vec[dim / BUFFER_SIZE] = tmp_block1;

}

void pagerank_main(input_block *ptr, input_block *idx, input_block *val,
		index_type_fpga *N, index_type_fpga *E, input_block *result, input_block *pr_vec,
		input_block *dangling_bitmap, input_block *tmp_pr, fixed_float *max_err,
		fixed_float *alpha, index_type_fpga *max_iter) {

// Ports used to transfer data, using AXI master;
#pragma HLS INTERFACE m_axi port = ptr offset = slave bundle = gmem0 depth=1000
#pragma HLS INTERFACE m_axi port = idx offset = slave bundle = gmem1 depth=1000
#pragma HLS INTERFACE m_axi port = val offset = slave bundle = gmem2 depth=1000
#pragma HLS INTERFACE m_axi port = dangling_bitmap offset = slave bundle = gmem0 depth=1000
#pragma HLS INTERFACE m_axi port = tmp_pr offset = slave bundle = gmem1 depth=1000
#pragma HLS INTERFACE m_axi port = pr_vec offset = slave bundle = gmem2  depth=1000
#pragma HLS INTERFACE m_axi port = N offset = slave bundle = gmem3  depth=1
#pragma HLS INTERFACE m_axi port = E offset = slave bundle = gmem3  depth=1
#pragma HLS INTERFACE m_axi port = max_err offset = slave bundle = gmem3 depth=1
#pragma HLS INTERFACE m_axi port = alpha offset = slave bundle = gmem3 depth=1
#pragma HLS INTERFACE m_axi port = result offset = slave bundle = gmem0  depth=1000

// Ports used for control signals, using AXI slave;
#pragma HLS INTERFACE s_axilite port = ptr bundle = control
#pragma HLS INTERFACE s_axilite port = idx bundle = control
#pragma HLS INTERFACE s_axilite port = val bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = max_err bundle = control
#pragma HLS INTERFACE s_axilite port = alpha bundle = control
#pragma HLS INTERFACE s_axilite port = E bundle = control
#pragma HLS INTERFACE s_axilite port = dangling_bitmap bundle = control
#pragma HLS INTERFACE s_axilite port = tmp_pr bundle = control
#pragma HLS INTERFACE s_axilite port = result bundle = control
#pragma HLS INTERFACE s_axilite port = pr_vec bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

	fixed_float ALPHA = *alpha;
	fixed_float MAX_ERROR = *max_err;
	index_type_fpga MAX_ITER = *max_iter;
	index_type_fpga NUM_VERTICES = *N;
	index_type_fpga NUM_EDGES = *E;
	fixed_float DANGLING_SCALE = ALPHA / NUM_VERTICES;
	fixed_float SHIFT_FACTOR = ((fixed_float) 1.0 - ALPHA) / NUM_VERTICES;

	fixed_float err = 0.0;
	fixed_float dangling_contrib = 0.0;
	unsigned int iter = 0;
	bool converged = false;

	while (iter < MAX_ITER) {
#pragma HLS loop tripcount min=6 max=6 avg=6
		spmv_main(ptr, idx, val, &NUM_VERTICES, &NUM_EDGES, tmp_pr, pr_vec);

		pagerank_vector_ops(NUM_VERTICES, DANGLING_SCALE, SHIFT_FACTOR, ALPHA, MAX_ERROR,
				dangling_bitmap, pr_vec, tmp_pr);

		iter++;
	}

	std::cout << "Pagerank converged after " << iter << " iterations" << std::endl;

	for (int i = 0; i < NUM_VERTICES / BUFFER_SIZE + 1; ++i) {
#pragma HLS loop tripcount min=62 max=62 avg=62
		input_block tmp_block;

		memcpy_buf_to_buf(&tmp_block, &pr_vec[i]);

		result[i] = tmp_block;
	}

}

