#include "ip/pagerank.hpp"

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 16
#endif

#define ITERATIONS 10240 / BUFFER_SIZE

void fixed_spmv(index_type dim, fixed_float *result, fixed_float *csc_col_val,
		index_type *csc_col_ptr, index_type *csc_col_idx, fixed_float *vec) {

	int begin = csc_col_ptr[0];
	ITERATE_COL: for (int i = 0; i < dim; ++i) {
#pragma HLS pipeline
		const int end = csc_col_ptr[i + 1];
		fixed_float acc = 0.0;

		COMPUTE_SINGLE_VAL: for (int j = begin; j < end; ++j) {
#pragma HLS pipeline
			acc += csc_col_val[j] * vec[csc_col_idx[j]];
		}
		result[i] = acc;
		// The new "begin" is the old "end", we save an array access;
		begin = end;
	}
}

void fixed_dot(index_type dim, fixed_float *result, fixed_float *a,
		index_type *b) {

	*result = 0;
	DOT_LOOP_1: for (int i = 0; i < dim; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=10240 max=10240 avg=10240
		*result += a[i] * b[i];
	}
}

void fixed_dot_2(index_type dim, fixed_float *result, fixed_float *a,
		index_type *b) {

	*result = 0;
	fixed_float buffer_a1[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer_a1 complete dim=1
	fixed_float buffer_b1[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer_b1 complete dim=1
	index_type buffer_a2[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer_a2 complete dim=1
	index_type buffer_b2[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer_b2 complete dim=1
	fixed_float buffer_res[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer_res complete dim=1

	// Buffers containing ecceding values [dim % BUFFER_SIZE]
	fixed_float buffer_ecc_a[BUFFER_SIZE];
	fixed_float buffer_ecc_b[BUFFER_SIZE];
	/*
	 // Initialize the first buffer;
	 for (int j = 0; j < BUFFER_SIZE; ++j) {
	 #pragma HLS unroll
	 buffer_a1[j] = a[j];
	 buffer_b1[j] = b[j];
	 buffer_res[j] = 0;
	 }*/

	// TODO: fix case where dim % BUFFER_SIZE != 0;
	// Done since it is mostly safe to read values from array out of bounds
	DOT_LOOP_2: for (int i = 1; i < (dim / BUFFER_SIZE); ++i) {
#pragma HLS loop_tripcount min=640 max=640 avg=640
#pragma HLS PIPELINE II=1

		// Ping-pong buffers, so that we load new values while reading the previous ones;
		if (i % 2) {
			R1: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
				buffer_a2[j] = a[i * BUFFER_SIZE + j];
				buffer_b2[j] = b[i * BUFFER_SIZE + j];
			}

			E1: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
				buffer_res[j] += buffer_a1[j] * buffer_b1[j];
			}
		} else {
			R2: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
				buffer_a1[j] = a[i * BUFFER_SIZE + j];
				buffer_b1[j] = b[i * BUFFER_SIZE + j];
			}

			E2: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
				buffer_res[j] += buffer_a2[j] * buffer_b2[j];
			}
		}
	}

	// Remaining elements from the buffer
	const int remaining = dim % BUFFER_SIZE;

	for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS unroll
		buffer_ecc_a[i] = a[BUFFER_SIZE * (BUFFER_SIZE - 1) + i];
		buffer_ecc_b[i] = b[BUFFER_SIZE * (BUFFER_SIZE - 1) + i];
	}

	for (int i = 0; i < remaining; ++i) {
#pragma HLS pipeline
		buffer_res[i] += buffer_ecc_a[i] * buffer_ecc_b[i];
	}

	// Add the temporary results (TODO: tree sum, 10 iterations instead of 1024);
	for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS unroll
		*result += buffer_res[i];
	}
}

void fixed_axpb(index_type dim, fixed_float *result, fixed_float a,
		fixed_float b) {

	fixed_float buffer_tmp_result[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer_tmp_result complete dim=1

	for (int i = 0; i < dim / BUFFER_SIZE; ++i) {
#pragma HLS pipeline
		for (int j = 0; j < BUFFER_SIZE; ++j) {
			// In this case unroll should act like flatten
#pragma HLS unroll
			result[j] = a * result[j] + b;
		}
	}

	const int remaining = dim % BUFFER_SIZE;

	for (int i = 0; i < remaining; ++i) {
#pragma HLS pipeline
		result[(BUFFER_SIZE - 1) * BUFFER_SIZE + i] = a
				* result[(BUFFER_SIZE - 1) * BUFFER_SIZE + i] + b;
	}

}

void fixed_euclidean_distance(index_type dim, fixed_float *result,
		fixed_float *a, fixed_float *b) {

	*result = 0;
	fixed_float buffer_a1[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer_a1 complete dim=1
	fixed_float buffer_b1[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer_b1 complete dim=1
	index_type buffer_a2[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer_a2 complete dim=1
	index_type buffer_b2[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer_b2 complete dim=1
	fixed_float buffer_res[BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=buffer_res complete dim=1

	// Buffers containing ecceding values [dim % BUFFER_SIZE]
	fixed_float buffer_ecc_a[BUFFER_SIZE];
	fixed_float buffer_ecc_b[BUFFER_SIZE];

	// Initialize the first buffer;
	for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
		buffer_a1[j] = a[j];
		buffer_b1[j] = b[j];
		buffer_res[j] = 0;
	}

	for (int i = 0; i < dim / BUFFER_SIZE; ++i) {
#pragma HLS loop tripcount min=640 max=640 avg=640
#pragma HLS pipeline

		if (i % 2) {
			// Process odd buffers
			// Copy
			M1: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
				buffer_a1[j] = a[i * BUFFER_SIZE + j];
				buffer_b1[j] = b[i * BUFFER_SIZE + j];
			}

			// Euclidean distance
			E1: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
				fixed_float tmp = buffer_a1[j] - buffer_a2[j];
				tmp *= tmp;
				buffer_res[j] += tmp;
			}

		} else {
			// Process even buffers
			// Copy
			E2: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
				buffer_a2[j] = a[i * BUFFER_SIZE + j];
				buffer_b2[j] = b[i * BUFFER_SIZE + j];
			}

			// Euclidean distance
			M2: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS unroll
				fixed_float tmp = buffer_a2[j] - buffer_a2[j];
				tmp *= tmp;
				buffer_res[j] += tmp;
			}

		}

	}

	// Remaining elements from the buffer
	const int remaining = dim % BUFFER_SIZE;

	ME: for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS unroll
		buffer_ecc_a[i] = a[BUFFER_SIZE * (BUFFER_SIZE - 1) + i];
		buffer_ecc_b[i] = b[BUFFER_SIZE * (BUFFER_SIZE - 1) + i];
	}

	for (int i = 0; i < remaining; ++i) {
#pragma HLS pipeline
		fixed_float tmp = buffer_ecc_a[i] - buffer_ecc_b[i];
		tmp *= tmp;
		buffer_res[i] += tmp;
	}

	// Add the temporary results (TODO: tree sum, 10 iterations instead of 1024);
	ADD: for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS unroll
		*result += buffer_res[i];
	}

	*result = sqrtf(*result);
}

void fixed_pagerank_main(fixed_float *result, // Vector that contains the final values of PR;
		fixed_float *csc_col_val, // Vector that contains the normalized value of each edge;
		index_type *csc_col_ptr, // Vector that contains the indegree of each vertex;
		index_type *csc_col_idx, // Source of each edge;
		fixed_float *pr_vec, // Support vector used to compute PR;
		index_type *dangling_bitmap, // 1 if a vertex is dangling (outdegree 0), 0 else;
		fixed_float *result_tmp, // Another temporary vector where PR is stored, as "result" is write only;
		const fixed_float alpha, // Value of alpha;
		const fixed_float max_err, // Maximum error allowed for convergence;
		const index_type N, // Number of vertices;
		const index_type max_iter // Maximum number of iterations allowed;
		) {
	const fixed_float dangling_scale = alpha / N;
	const fixed_float shift_factor = ((fixed_float) 1.0 - alpha) / N;
	bool converged = false;
	index_type n_iter = 0;

	fixed_float tmp_err = 1.0;
	fixed_float dangling_contrib = 0.0;
	fixed_float dangling_contrib_2 = 0.0;

	PR_MAIN_LOOP: while (!converged && n_iter < max_iter) {
#pragma HLS loop_tripcount min=1 max=100 avg=25
		//fixed_spmv(N, result_tmp, csc_col_val, csc_col_ptr, csc_col_idx, pr_vec);
		fixed_dot(N, &dangling_contrib, pr_vec, dangling_bitmap);
		fixed_dot_2(N, &dangling_contrib_2, pr_vec, dangling_bitmap);
		fixed_float scaling_factor = shift_factor
				+ (dangling_scale * dangling_contrib);
		fixed_axpb(N, result_tmp, alpha, scaling_factor);
		fixed_euclidean_distance(N, &tmp_err, result_tmp, pr_vec);

		converged = false; // tmp_err <= max_err;

		// Copy back only if hasn't converged
		// We add a branch :(
		// We save a memcpy at the last iteration :)
		/*		if (!converged) {
		 MEMCPY_TMP_TO_PR: for (int i = 0; i < N; ++i) {
		 #pragma HLS pipeline
		 pr_vec[i] = result_tmp[i];
		 }
		 }*/
		n_iter++;
	}

	// Final memcpy to the output vector;
	/*	MEMCPY_TMP_TO_RESULT: for (int i = 0; i < N; ++i) {
	 #pragma HLS pipeline
	 result[i] = result_tmp[i];
	 }*/
}

void fixed_pagerank(fixed_float *result, fixed_float *csc_col_val,
		index_type *csc_col_ptr, index_type *csc_col_idx, fixed_float *pr_vec,
		index_type *dangling_bitmap, fixed_float *result_tmp,
		fixed_float *alpha, fixed_float *max_err, index_type *N,
		index_type *max_iter) {

// Ports used to transfer data, using AXI master;
#pragma HLS INTERFACE m_axi port = result offset = slave bundle = gmem2 depth=1000
#pragma HLS INTERFACE m_axi port = csc_col_val offset = slave bundle = gmem0 depth=1000
#pragma HLS INTERFACE m_axi port = csc_col_ptr offset = slave bundle = gmem1 depth=1000
#pragma HLS INTERFACE m_axi port = csc_col_idx offset = slave bundle = gmem2 depth=1000
#pragma HLS INTERFACE m_axi port = pr_vec offset = slave bundle = gmem0 depth=1000
#pragma HLS INTERFACE m_axi port = dangling_bitmap offset = slave bundle = gmem1 depth=1000
#pragma HLS INTERFACE m_axi port = alpha offset = slave bundle = gmem3 depth=1
#pragma HLS INTERFACE m_axi port = max_err offset = slave bundle = gmem3  depth=1
#pragma HLS INTERFACE m_axi port = N offset = slave bundle = gmem3  depth=1
#pragma HLS INTERFACE m_axi port = max_iter offset = slave bundle = gmem3  depth=1
#pragma HLS INTERFACE m_axi port = result_tmp offset = slave bundle = gmem3  depth=1

// Ports used for control signals, using AXI slave;
#pragma HLS INTERFACE s_axilite port = csc_col_val bundle = control
#pragma HLS INTERFACE s_axilite port = csc_col_ptr bundle = control
#pragma HLS INTERFACE s_axilite port = csc_col_idx bundle = control
#pragma HLS INTERFACE s_axilite port = pr_vec bundle = control
#pragma HLS INTERFACE s_axilite port = dangling_bitmap bundle = control
#pragma HLS INTERFACE s_axilite port = alpha bundle = control
#pragma HLS INTERFACE s_axilite port = max_err bundle = control
#pragma HLS INTERFACE s_axilite port = result bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = max_iter bundle = control
#pragma HLS INTERFACE s_axilite port = result_tmp bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

	fixed_float dangling_contrib = 0.0;
	fixed_float dangling_contrib_2 = 0.0;
	fixed_float tmp_err = 0.0;
	//fixed_spmv(N, result_tmp, csc_col_val, csc_col_ptr, csc_col_idx, pr_vec);
	//fixed_dot(*N, result, pr_vec, dangling_bitmap);
	fixed_dot_2(*N, result_tmp, pr_vec, dangling_bitmap);



	//fixed_pagerank_main(result, csc_col_val, csc_col_ptr, csc_col_idx, pr_vec, dangling_bitmap, result_tmp, *alpha,
	//	*max_err, *N, *max_iter);
}

