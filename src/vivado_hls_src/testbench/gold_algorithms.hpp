#pragma once

#include "../ip/coo_fpga.hpp"

template<typename I, typename V>
inline void spmv_gold(I *ptr, I *idx, V *val, I N, V *result, V *vec) {
	I begin = ptr[0];
	for (I i = 0; i < N; ++i) {
		I end = ptr[i + 1];
		V acc = 0.0;

		for (I j = begin; j < end; ++j) {
			acc += val[j] * vec[idx[j]];
		}
		result[i] = acc;
		begin = end;
	}
}

template<typename I, typename V>
inline void multi_spmv_gold(I *ptr, I *idx, V *val, I N, I M, V *result, V *vec) {
	I begin = ptr[0];
	for (I i = 0; i < N; ++i) {
		I end = ptr[i + 1];

		for (I k = 0; k < M; k++) {
			V acc = 0.0;
			for (I j = begin; j < end; ++j) {
				acc += val[j] * vec[idx[j] + k * N];
			}
			result[k * N + i] = acc;
		}
		begin = end;
	}
}


template<typename V>
inline void spmv_coo_gold(coo_fixed_fpga_t &csc, V *result, V *vec) {
	// Scatter "vec" values across a vector of size E;
	std::vector<V> vec_scattered(csc.E);
	for (int i = 0; i < csc.E; i++) {
		vec_scattered[i] = vec[csc.end[i]];
	}

	// Main computation;
	for (int i = 0; i < csc.E; i++) {
		result[csc.start[i]] += csc.val[i] * vec_scattered[i];
		// Result is BRAM with cyclic 16 -> a single packet cant access the same address twice
	}
}

template<typename V>
inline void spmv_coo_gold2(coo_fixed_fpga_t &csc, V *result, V *vec) {
	// Scatter "vec" values across a vector of size E;
	std::vector<V> vec_scattered(csc.E);
	for (int i = 0; i < csc.E; i++) {
		vec_scattered[i] = vec[csc.end[i]];
	}

	// Store temporary results;
	std::vector<V> temp_res(csc.E);

	// Main computation;
	for (int i = 0; i < csc.E; i++) {
		temp_res[i] = csc.val[i] * vec_scattered[i];
	}

	// Gather;
	for (int i = 0; i < csc.E; i++) {
		result[csc.start[i]] += temp_res[i];
	}
}

template<typename V>
inline void spmv_coo_gold3(coo_fixed_fpga_t &csc, V *result, V *vec) {

	int B = 2;

	// Scatter "vec" values across a vector of size E;
	std::vector<V> vec_scattered(csc.E_fixed);
	for (int i = 0; i < csc.E_fixed; i++) {
		vec_scattered[i] = vec[csc.end[i]];
	}

	// Main computation;
	for (int i = 0; i < (csc.E_fixed + B - 1) / B; i++) {

		index_type_fpga local_start[B];
		V res_temp[B];
		V res_fin[B];
		// Reset buffers;
		for (int j = 0; j < B; j++) {
			local_start[j] = 0;
			res_temp[j] = 0;
			res_fin[j] = 0;
		}

		// Compute values for a packet;
		for (int j = 0; j < B; j++) {
			if (i * B + j < csc.E_fixed) {
				local_start[j] = csc.start[i * B + j];
				res_temp[j] = csc.val[i * B + j] * vec_scattered[i * B + j];
			}
		}

		// Use B reductions;
		index_type_fpga min_id = local_start[0];
		for (int j = 0; j < B; j++) {
			index_type_fpga curr = j + min_id; // Identify which vertex is considered in this reduction;
			// Reduction;
			for (int q = 0; q < B; q++) {
				res_fin[j] += res_temp[q] * (curr == local_start[q] ? 1 : 0);
			}
		}

		for (int j = 0; j < B; j++) {
			result[j + min_id] += res_fin[j];
		}
//		for (int j = 0; j < B; j++) {
//			result[local_start[j]] += res_temp[j];
//		}
	}
}

inline void euclidean_distance_gold(unsigned int N, float *result,
		float *a, float *b) {

	*result = 0;
	for (int i = 0; i < N; ++i) {
		float tmp = a[i] - b[i];
		tmp *= tmp;
		*result += tmp;
	}

}

inline void axpb_gold(unsigned int N, float *result, float *a,
		float *x, float *b) {

	float local_a = *a;
	float local_b = *b;

	for (int i = 0; i < N; ++i) {
		result[i] = local_a * x[i] + local_b;
	}
}

inline void dot_product_gold(unsigned int N, float *result, unsigned int *a,
		float *b) {

	*result = 0;

	for (int i = 0; i < N; ++i) {
		*result += a[i] * b[i];
	}
}

inline void pagerank_golden(unsigned int *ptr, unsigned int *idx, float *val,
		unsigned int *N, unsigned int *E, float *result, float *pr_vec,
		unsigned int *dangling_bitmap, float *tmp_pr, float *max_err,
		float *alpha, unsigned int *max_iter, unsigned int *iterations_to_convergence) {

	float ERR = *max_err;
	unsigned int ITER = *max_iter;
	unsigned int NUM_VERTICES = *N;
	unsigned int NUM_EDGES = *E;
	float DANGLING_SCALE = *alpha/ NUM_VERTICES;
	float SHIFT_FACTOR = ((float) 1.0 - *alpha) / NUM_VERTICES;

	float err = 0.0;
	float dangling_contrib = 0.0;
	unsigned int iter = 0;
	bool converged = false;

	while(!converged && iter < ITER){

		spmv_gold(ptr, idx, val, NUM_VERTICES, tmp_pr, pr_vec);
		dot_product_gold(NUM_VERTICES, &dangling_contrib, dangling_bitmap, pr_vec);
		float shifting_factor = SHIFT_FACTOR + (DANGLING_SCALE * dangling_contrib);
		axpb_gold(NUM_VERTICES, tmp_pr, alpha, tmp_pr, &shifting_factor);

		euclidean_distance_gold(NUM_VERTICES, &err, tmp_pr, pr_vec);

		converged = err <= ERR;

		memcpy(pr_vec, tmp_pr, sizeof(float) * NUM_VERTICES);
		iter ++;

	}
	*iterations_to_convergence = iter;
	std::cout << "Pagerank golden converged after " << iter << " iterations." << std::endl;
	memcpy(result, pr_vec, sizeof(float) * NUM_VERTICES);

}

