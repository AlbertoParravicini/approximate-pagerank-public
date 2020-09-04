#include "pagerank.hpp"

extern "C" {

inline void pagerank_vector_ops(index_type_fpga N, fixed_float dangling_scale,
		fixed_float shift_factor, fixed_float alpha, fixed_float max_err,
		input_block *dangling_bitmap, fixed_float pr[MAX_VERTICES],
		input_block *pr_tmp) {
#pragma HLS ARRAY_PARTITION variable=pr cyclic factor=2

	dangling_type local_dangling_bitmap[AP_UINT_BITWIDTH];
#pragma HLS array_partition variable=local_dangling_bitmap complete
	fixed_float local_pr_tmp[BUFFER_SIZE];
#pragma HLS array_partition variable=local_pr_tmp complete

	fixed_float dot_product_buffer[BUFFER_SIZE];
#pragma HLS array_partition variable=dot_product_buffer complete

	fixed_float dangling_contribution = 0.0;
	index_type num_blocks_n = (N + BUFFER_SIZE - 1) / BUFFER_SIZE;

	DANGLING_CONTRIBUTION: for (unsigned int i = 0; i < num_blocks_n; ++i) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=64 max=64 avg=64

		input_block cur_dangling_bitmap = dangling_bitmap[i * BUFFER_SIZE / AP_UINT_BITWIDTH];
		const unsigned int offset = (i * BUFFER_SIZE) % (AP_UINT_BITWIDTH);
		// Read AP_UINT_BITWIDTH values from the bitmap, but only if offset is 0.
		READ_BITMAP: for (unsigned int j = 0; j < AP_UINT_BITWIDTH; ++j) {
#pragma HLS unroll
			local_dangling_bitmap[j] = offset == 0 ? cur_dangling_bitmap.bit(j) : local_dangling_bitmap[j];
		}

		// Dot product between current chunks of PR values and dangling bitmap;
		DOT_MULTIPLY: for (unsigned int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
			dot_product_buffer[j] = pr[i * BUFFER_SIZE + j] * local_dangling_bitmap[j + offset];
		}

		dangling_contribution += reduction(dot_product_buffer);
	}

	fixed_float scaling_factor = shift_factor + (dangling_scale * dangling_contribution);

	AXPB: for (int i = 0; i < num_blocks_n; ++i) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=64 max=64 avg=64
		read_block_float(pr_tmp[i], local_pr_tmp);
		axpb_512(&alpha, local_pr_tmp, &scaling_factor);
		for (unsigned int j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
			pr[i * BUFFER_SIZE + j] = local_pr_tmp[j];
		}
	}
}

/////////////////////////////
/////////////////////////////

inline void pagerank_vector_ops_gmem(index_type_fpga N,
		fixed_float dangling_scale, fixed_float shift_factor, fixed_float alpha,
		fixed_float max_err, input_block *dangling_bitmap, input_block *pr,
		input_block *pr_tmp) {

	dangling_type local_dangling_bitmap[AP_UINT_BITWIDTH];
#pragma HLS array_partition variable=local_dangling_bitmap complete
	fixed_float local_pr_tmp[BUFFER_SIZE];
#pragma HLS array_partition variable=local_pr_tmp complete

	fixed_float dot_product_buffer[BUFFER_SIZE];
#pragma HLS array_partition variable=dot_product_buffer complete

	fixed_float local_pr[BUFFER_SIZE];
#pragma HLS array_partition variable=local_pr complete

	fixed_float dangling_contribution = 0.0;

	index_type num_blocks_n = (N + BUFFER_SIZE - 1) / BUFFER_SIZE;

	DANGLING_CONTRIBUTION: for (unsigned int i = 0; i < num_blocks_n; ++i) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=128 max=128 avg=128

		input_block cur_dangling_bitmap = dangling_bitmap[i * BUFFER_SIZE / AP_UINT_BITWIDTH];
		const unsigned int offset = (i * BUFFER_SIZE) % (AP_UINT_BITWIDTH);
		// Read AP_UINT_BITWIDTH values from the bitmap, but only if offset is 0.
		READ_BITMAP: for (unsigned int j = 0; j < AP_UINT_BITWIDTH; ++j) {
#pragma HLS unroll
			local_dangling_bitmap[j] = offset == 0 ? cur_dangling_bitmap.bit(j) : local_dangling_bitmap[j];
		}
		// Read a block of PR values;
		read_block_float(pr[i], local_pr);

		// Dot product between current chunks of PR values and dangling bitmap;
		DOT_MULTIPLY: for (unsigned int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
			dot_product_buffer[j] = local_pr[j] * local_dangling_bitmap[j + offset];
		}

		dangling_contribution += reduction_16(dot_product_buffer);
	}

	fixed_float scaling_factor = shift_factor + (dangling_scale * dangling_contribution);

	AXPB: for (int i = 0; i < num_blocks_n; ++i) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=128 max=128 avg=128
		read_block_float(pr_tmp[i], local_pr_tmp);
		axpb_512(&alpha, local_pr_tmp, &scaling_factor);

		// Write a block of PR values;
		input_block tmp_block;
		write_block_float(&tmp_block, local_pr_tmp);
		pr[i] = tmp_block;
	}
}

/////////////////////////////
/////////////////////////////

inline void compute_scaling_factor(index_type_fpga num_blocks_n,
		fixed_float dangling_scale, fixed_float shift_factor,
		input_block *dangling_bitmap, fixed_float pr[MAX_VERTICES],
		fixed_float *scaling_factor) {
#pragma HLS ARRAY_PARTITION variable=pr cyclic factor=16

	dangling_type local_dangling_bitmap[AP_UINT_BITWIDTH];
#pragma HLS array_partition variable=local_dangling_bitmap complete
	fixed_float local_pr_tmp[BUFFER_SIZE];
#pragma HLS array_partition variable=local_pr_tmp complete

	fixed_float dot_product_buffer[BUFFER_SIZE];
#pragma HLS array_partition variable=dot_product_buffer complete

	fixed_float dangling_contribution = 0.0;

	DANGLING_CONTRIBUTION: for (unsigned int i = 0; i < num_blocks_n; ++i) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=64 max=64 avg=64

		input_block cur_dangling_bitmap = dangling_bitmap[i * BUFFER_SIZE / AP_UINT_BITWIDTH];
		const unsigned int offset = (i * BUFFER_SIZE) % (AP_UINT_BITWIDTH);
		// Read AP_UINT_BITWIDTH values from the bitmap, but only if offset is 0.
		READ_BITMAP: for (unsigned int j = 0; j < AP_UINT_BITWIDTH; ++j) {
#pragma HLS unroll
			local_dangling_bitmap[j] = offset == 0 ? cur_dangling_bitmap.bit(j) : local_dangling_bitmap[j];
		}

		// Dot product between current chunks of PR values and dangling bitmap;
		DOT_MULTIPLY: for (unsigned int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
			dot_product_buffer[j] = pr[i * BUFFER_SIZE + j] * local_dangling_bitmap[j + offset];
		}

		dangling_contribution += reduction(dot_product_buffer);
	}

	*scaling_factor = (dangling_scale * dangling_contribution);
}

inline void personalized_pagerank_vector_ops_local_buffer_only(
		index_type_fpga num_blocks_n, fixed_float alpha,
		fixed_float scaling_factor, fixed_float pr[MAX_VERTICES],
		fixed_float pr_write_back[MAX_VERTICES], fixed_float shift_factor,
		index_type_fpga preferred_index) {
#pragma HLS ARRAY_PARTITION variable=pr cyclic factor=16

	AXPB: for (int i = 0; i < num_blocks_n; ++i) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=64 max=64 avg=64
		for (unsigned int j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
			int idx = i * BUFFER_SIZE + j;
			pr[idx] = alpha * pr[idx] + scaling_factor + ((idx == preferred_index) ? shift_factor : (fixed_float) 0);
			pr_write_back[idx] = pr[idx];
		}
	}
}

/////////////////////////////
/////////////////////////////

void pagerank_main(input_block *ptr, input_block *idx, input_block *val,
		index_type_fpga N, index_type_fpga E, input_block *result,
		input_block *pr, input_block *dangling_bitmap, input_block *pr_tmp,
		float max_err, float alpha, index_type_fpga max_iter) {

// Ports used to transfer data, using AXI master;
#pragma HLS INTERFACE m_axi port = ptr offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = idx offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = val offset = slave bundle = gmem2

#pragma HLS INTERFACE m_axi port = result offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = pr offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dangling_bitmap offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = pr_tmp offset = slave bundle = gmem3

// Ports used for control signals, using AXI slave;
#pragma HLS INTERFACE s_axilite register port = N bundle = control
#pragma HLS INTERFACE s_axilite register port = E bundle = control
#pragma HLS INTERFACE s_axilite register port = max_err bundle = control
#pragma HLS INTERFACE s_axilite register port = alpha bundle = control
#pragma HLS INTERFACE s_axilite register port = max_iter bundle = control
#pragma HLS INTERFACE s_axilite register port = return bundle = control

	// Cast to fixed float;
	fixed_float a = (fixed_float) alpha;
	fixed_float max_error = (fixed_float) max_err;

	fixed_float dangling_scale = a / N;
	fixed_float shift_factor = ((fixed_float) 1.0 - a) / N;

	index_type num_blocks_n = (N + BUFFER_SIZE - 1) / BUFFER_SIZE;
	index_type num_vertices_padded = num_blocks_n * BUFFER_SIZE;
	index_type num_blocks_e = (E + BUFFER_SIZE - 1) / BUFFER_SIZE;
	index_type num_edges_padded = num_blocks_e * BUFFER_SIZE;
	index_type num_blocks_bitmap = (N + AP_UINT_BITWIDTH - 1) / AP_UINT_BITWIDTH;

	fixed_float err = 0.0;
	fixed_float dangling_contrib = 0.0;
	unsigned int iter = 0;
	bool converged = false;

	// Allocate a local buffer that contains all the values of "pr";
	fixed_float pr_local[MAX_VERTICES];
#pragma HLS ARRAY_PARTITION variable=pr_local cyclic factor=16

	// Copy values of "pr" in the local buffer;
	READ_PR: for (index_type_fpga i = 0; i < num_blocks_n; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=64 max=64 avg=64
		read_block_float(pr[i], pr_local + i * BUFFER_SIZE);
	}

	// Main loop of PageRank;
	while (iter < max_iter) {
#pragma HLS loop tripcount min=6 max=6 avg=6
		spmv(ptr, idx, val, num_vertices_padded, E, pr_tmp, pr_local);
		pagerank_vector_ops(N, dangling_scale, shift_factor, a, max_error,
				dangling_bitmap, pr_local, pr_tmp);

		iter++;
	}

//	for (int i = 0; i < num_blocks_n; ++i) {
//#pragma HLS PIPELINE II=1
//#pragma HLS loop tripcount min=62 max=62 avg=62
//		result[i] = pr[i];
//	}

	for (int i = 0; i < num_blocks_n; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS loop tripcount min=64 max=64 avg=64
		input_block tmp_block;
		write_block_float(&tmp_block, &pr_local[i * BUFFER_SIZE]);
		result[i] = tmp_block;
	}
}

/////////////////////////////
/////////////////////////////

void personalized_pagerank_coo_main(input_block *start, input_block *end,
		input_block *val, index_type_fpga N, index_type_fpga E,
		input_block *result, input_block *pr, input_block *dangling_bitmap,
		input_block *pr_tmp, float max_err,	float alpha, index_type_fpga max_iter,
		index_type_fpga personalization_vertex) {
// Ports used to transfer data, using AXI master;
#pragma HLS INTERFACE m_axi port = start offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = end offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = val offset = slave bundle = gmem1

#pragma HLS INTERFACE m_axi port = result offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = pr offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = dangling_bitmap offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = pr_tmp offset = slave bundle = gmem2

// Ports used for control signals, using AXI slave;
#pragma HLS INTERFACE s_axilite register port = N bundle = control
#pragma HLS INTERFACE s_axilite register port = E bundle = control
#pragma HLS INTERFACE s_axilite register port = max_err bundle = control
#pragma HLS INTERFACE s_axilite register port = alpha bundle = control
#pragma HLS INTERFACE s_axilite register port = max_iter bundle = control
#pragma HLS INTERFACE s_axilite register port = personalization_vertex bundle = control
#pragma HLS INTERFACE s_axilite register port = return bundle = control

	// Cast to fixed float;
	fixed_float a = (fixed_float) alpha;
	fixed_float max_error = (fixed_float) max_err;

	fixed_float dangling_scale = a / N;
	fixed_float shift_factor = (fixed_float) 1.0 - a;

	index_type num_blocks_n = (N + BUFFER_SIZE - 1) / BUFFER_SIZE;
	index_type num_vertices_padded = num_blocks_n * BUFFER_SIZE;
	index_type num_blocks_e = (E + BUFFER_SIZE - 1) / BUFFER_SIZE;
	index_type num_edges_padded = num_blocks_e * BUFFER_SIZE;
	index_type num_blocks_bitmap = (N + AP_UINT_BITWIDTH - 1) / AP_UINT_BITWIDTH;

	fixed_float err = 0.0;
	fixed_float scaling_factor = 0.0;
	unsigned int iter = 0;
	bool converged = false;

	fixed_float pr_local_write[MAX_VERTICES];
#pragma HLS array_partition variable=pr_local_write cyclic factor=16
#pragma HLS resource variable=pr_local_write core=RAM_1P_BRAM

	// Allocate a local buffer that contains all the values of "pr";
	fixed_float pr_local[MAX_VERTICES];
#pragma HLS array_partition variable=pr_local cyclic factor=16
#pragma HLS resource variable=pr_local core=XPM_MEMORY uram

	// Copy values of "pr" in the local buffer;
	READ_PR: for (index_type_fpga i = 0; i < num_blocks_n; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=64 max=64 avg=64
		read_block_float(pr[i], pr_local + i * BUFFER_SIZE);
	}

	// Main loop of PageRank;
	while (iter < max_iter) {
#pragma HLS loop tripcount min=6 max=6 avg=6

		fixed_float scaling_factor = 0;
		compute_scaling_factor(num_blocks_n, dangling_scale, shift_factor,
				dangling_bitmap, pr_local, &scaling_factor);

		// Reset the local result vector;
		RESET_PR_LOCAL: for (index_type i = 0; i < num_blocks_n; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop tripcount min=64 max=64 avg=64
			for (index_type j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
				pr_local_write[i * BUFFER_SIZE + j] = 0;
			}
		}

		// Execute the SPMV;
		spmv_coo_with_scatter(start, end, val, num_edges_padded, pr_local_write, pr_local);

		// Execute other operations to update the PageRank vector;
		// Additionally, write back from pr_local_write to pr_local
		personalized_pagerank_vector_ops_local_buffer_only(num_blocks_n, a,
				scaling_factor, pr_local_write, pr_local, shift_factor, personalization_vertex);

		iter++;
	}

	for (int i = 0; i < num_blocks_n; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS loop tripcount min=64 max=64 avg=64
		input_block tmp_block;
		write_block_float(&tmp_block, &pr_local[i * BUFFER_SIZE]);
		result[i] = tmp_block;
	}
}

}
