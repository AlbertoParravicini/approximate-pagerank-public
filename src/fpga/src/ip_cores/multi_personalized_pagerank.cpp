#include "multi_personalized_pagerank.hpp"

/////////////////////////////
/////////////////////////////

void compute_scaling_factor(
		index_type num_blocks_n,
		fixed_float dangling_scale, fixed_float shift_factor,
		input_block *dangling_bitmap, fixed_float pr[N_PPR_VERTICES][MAX_VERTICES],
		fixed_float scaling_factor[N_PPR_VERTICES]) {

	fixed_float dot_product_buffer[N_PPR_VERTICES][BUFFER_SIZE];
#pragma HLS array_partition variable=dot_product_buffer complete dim=0

	fixed_float dangling_contribution[N_PPR_VERTICES];
#pragma HLS array_partition variable=dangling_contribution complete

	RESET_DANGLING_CONTRIBUTION: for (unsigned int i = 0; i < N_PPR_VERTICES; i++) {
#pragma HLS unroll
		dangling_contribution[i] = 0;
	}

	DANGLING_CONTRIBUTION: for (unsigned int i = 0; i < num_blocks_n; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=hls_iterations_v max=hls_iterations_v avg=hls_iterations_v

		// Obtain the corresponding dangling block (there are PACKET_ELEMENT_SIZE pr blocks for each dangling_bitmap block);
		index_type dangling_block_index = i / PACKET_ELEMENT_SIZE;
		input_block curr_dangling_bitmap = dangling_bitmap[dangling_block_index];

		// Dot product between current chunks of PR values and dangling bitmap;
		DOT_MULTIPLY: for (unsigned int k = 0; k < N_PPR_VERTICES; k++) {
#pragma HLS unroll
			index_type pr_start_index = i * BUFFER_SIZE;
			index_type dangling_start_index = (i * BUFFER_SIZE) % AP_UINT_BITWIDTH;
			for (unsigned int j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
				dot_product_buffer[k][j] = pr[k][pr_start_index + j] * curr_dangling_bitmap.bit(dangling_start_index + j).to_bool();
			}
			dangling_contribution[k] += reduction(dot_product_buffer[k]);
		}
	}

	for (unsigned int i = 0; i < N_PPR_VERTICES; i++) {
#pragma HLS unroll
		scaling_factor[i] = dangling_scale * dangling_contribution[i];
	}
}

/////////////////////////////
/////////////////////////////

void personalized_pagerank_vector_ops_local_buffer_only(
		index_type num_blocks_n, fixed_float alpha,
		fixed_float scaling_factor[N_PPR_VERTICES],
		fixed_float pr_in[N_PPR_VERTICES][MAX_VERTICES],
		fixed_float pr_out[N_PPR_VERTICES][MAX_VERTICES],
		fixed_float shift_factor, index_type personalization_vertices[N_PPR_VERTICES],
		fixed_error_float errors[N_PPR_VERTICES][MAX_ITERATIONS],
		unsigned int num_iteration) {

		// Use a buffer to compute the error norm;
	fixed_error_float error_buffer[N_PPR_VERTICES][BUFFER_SIZE];
#pragma HLS array_partition variable=error_buffer complete
#pragma HLS array_partition variable=error_buffer complete

	// Initialize the errors buffer;
	for (unsigned int k = 0; k < N_PPR_VERTICES; k++) {
#pragma HLS unroll
		for (unsigned int j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
			error_buffer[k][j] = 0;
		}
	}

	AXPB: for (int i = 0; i < num_blocks_n; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=hls_iterations_v max=hls_iterations_v avg=hls_iterations_v
		for (unsigned int k = 0; k < N_PPR_VERTICES; k++) {
#pragma HLS unroll

			fixed_float curr_scaling = scaling_factor[k];
			index_type curr_pers = personalization_vertices[k];

			for (unsigned int j = 0; j < BUFFER_SIZE; j++) {
	#pragma HLS unroll
				int idx = i * BUFFER_SIZE + j;

				fixed_float pr_curr_in = pr_in[k][idx];
				fixed_float pr_curr_old = pr_out[k][idx];

				// Scale the current PPR values;
				fixed_float updated_pr_value = alpha * pr_curr_in + curr_scaling + ((idx == curr_pers) ? shift_factor : (fixed_float) 0);
				pr_out[k][idx] = updated_pr_value;

				// Keep track of the error L2 norm;
				error_buffer[k][j] += (fixed_error_float) ((updated_pr_value - pr_curr_old) * (updated_pr_value - pr_curr_old));

				// Reset the other PPR buffer;
				pr_in[k][idx] = (fixed_float) 0;
			}
		}
	}

	for (unsigned int k = 0; k < N_PPR_VERTICES; k++) {
#pragma HLS unroll
		errors[k][num_iteration] = reduction(error_buffer[k]);
	}
}

/////////////////////////////
/////////////////////////////

void multi_ppr_main(input_block *start, input_block *end,
		input_block *val, index_type N, index_type E,
		input_block *result, input_block *dangling_bitmap,
		float max_err, float alpha, index_type max_iter, index_type *personalization_vertices, fixed_error_float* errors) {
// Ports used to transfer data, using AXI master;
#pragma HLS INTERFACE m_axi port = start offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = end offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = val offset = slave bundle = gmem2

#pragma HLS INTERFACE m_axi port = result offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = dangling_bitmap offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = personalization_vertices offset slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = errors offset slave bundle = gmem3

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
	fixed_float shift_factor = (fixed_float) 1.0 - a;

	index_type num_blocks_n = (N + BUFFER_SIZE - 1) / BUFFER_SIZE; // Note that the actual number of blocks is num_blocks_n * N_PPR_VERTICES;
	index_type num_vertices_padded = num_blocks_n * BUFFER_SIZE;
	index_type num_blocks_e = (E + BUFFER_SIZE - 1) / BUFFER_SIZE;
	index_type num_edges_padded = num_blocks_e * BUFFER_SIZE;
	index_type num_blocks_bitmap = (N + AP_UINT_BITWIDTH - 1) / AP_UINT_BITWIDTH;

	static fixed_float scaling_factor[N_PPR_VERTICES];
#pragma HLS array_partition variable=scaling_factor complete

	unsigned int iter = 0;

	static index_type local_personalization_vertices[N_PPR_VERTICES];
#pragma HLS array_partition variable=local_personalization_vertices complete

	static fixed_float pr_local_result[N_PPR_VERTICES][MAX_VERTICES];
#pragma HLS array_reshape variable=pr_local_result cyclic factor=hls_buffer_size dim=2
#pragma HLS array_partition variable=pr_local_result complete dim=1
#pragma HLS resource variable=pr_local_result core=XPM_MEMORY uram

	// Allocate a local buffer that contains all the values of "pr";
	static fixed_float pr_local[N_PPR_VERTICES][MAX_VERTICES];
#pragma HLS array_reshape variable=pr_local cyclic factor=hls_buffer_size dim=2
#pragma HLS array_partition variable=pr_local complete dim=1
#pragma HLS resource variable=pr_local core=XPM_MEMORY uram

	// Allocate a local buffer to store the error values;
	static fixed_error_float errors_local[N_PPR_VERTICES][MAX_ITERATIONS];
#pragma HLS array_reshape variable=errors_local cyclic factor=hls_buffer_size dim=2
#pragma HLS array_partition variable=errors_local complete dim=1

	// Reset the local PR buffer;
	READ_PR: for (index_type i = 0; i < num_blocks_n; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=hls_iterations_v max=hls_iterations_v avg=hls_iterations_v
		for (index_type j = 0; j < BUFFER_SIZE; j++) {
			for (index_type k = 0; k < N_PPR_VERTICES; k++) {
				pr_local[k][i * BUFFER_SIZE + j] = (fixed_float) 0;
				pr_local_result[k][i * BUFFER_SIZE + j] = (fixed_float) 0;
			}
		}
	}

	// Reset the local error buffer;
	RESET_ERROR: for (index_type i = 0; i < MAX_ITERATIONS / BUFFER_SIZE; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=hls_max_iter max=hls_max_iter avg=hls_max_iter
		for (index_type j = 0; j < BUFFER_SIZE; j++) {
			for (index_type k = 0; k < N_PPR_VERTICES; k++) {
				errors_local[k][i * BUFFER_SIZE + j] = (fixed_error_float) 0;
			}
		}
	}

	// Copy personalization vertices in local buffer and initialize PR;
	SET_PERSONALIZATION_VERTICES: for (index_type i = 0; i < N_PPR_VERTICES; i++) {
#pragma HLS unroll
		index_type personalization_vertex = personalization_vertices[i];
		pr_local[i][personalization_vertex] = (fixed_float) 1;
		local_personalization_vertices[i] = personalization_vertex;
	}

	// Fix the maximum number of iterations;
	unsigned int num_effective_iter = (max_iter < MAX_ITERATIONS) ? max_iter : MAX_ITERATIONS;

	// Main loop of PageRank;
	while (iter < num_effective_iter) {
#pragma HLS loop tripcount min=hls_iter max=hls_iter avg=hls_iter

		compute_scaling_factor(num_blocks_n, dangling_scale, shift_factor, dangling_bitmap, pr_local, scaling_factor);

		// Execute the SPMV;
		spmv_coo_multi_stream(start, end, val, num_edges_padded, pr_local_result, pr_local);

		// Execute other operations to update the PageRank vector;
		// Additionally, reset the pr_local_result buffer;
		personalized_pagerank_vector_ops_local_buffer_only(num_blocks_n, a, scaling_factor,
				pr_local_result, pr_local, shift_factor, local_personalization_vertices, errors_local, iter);
		iter++;
	}

	WRITE_ERROR: for (index_type i = 0; i < num_effective_iter * N_PPR_VERTICES; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop tripcount min=hls_write_errors_iter max=hls_write_errors_iter avg=hls_write_errors_iter
		index_type j = i / N_PPR_VERTICES;
		index_type q = i % N_PPR_VERTICES;
		errors[i] = errors_local[q][j];
	}

	WRITE_RESULT: for (index_type i = 0; i < N_PPR_VERTICES; i++) {
#pragma HLS unroll
		for (index_type j = 0; j < num_blocks_n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=hls_iterations_v max=hls_iterations_v avg=hls_iterations_v
			input_block tmp_block;
			write_block_float(&tmp_block, &pr_local[i][j * BUFFER_SIZE]);
			result[j + i * num_blocks_n] = tmp_block;
		}
	}
}
