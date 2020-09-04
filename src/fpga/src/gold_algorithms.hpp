#pragma once

inline void spmv_gold(unsigned int *ptr, unsigned int *idx, float *val,
		unsigned int N, float *result, float *vec) {
	int begin = ptr[0];
	for (unsigned int i = 0; i < N; ++i) {
		const int end = ptr[i + 1];
		float acc = 0.0;

		for (int j = begin; j < end; ++j) {
			acc += val[j] * vec[idx[j]];
		}
		result[i] = acc;
		begin = end;
	}
}

inline void euclidean_distance_gold(unsigned int N, float *result, float *a,
		float *b) {

	*result = 0;
	for (unsigned int i = 0; i < N; ++i) {
		float tmp = a[i] - b[i];
		tmp *= tmp;
		*result += tmp;
	}

}

inline void axpb_gold(unsigned int N, float *result, float *a, float *x,
		float *b) {

	float local_a = *a;
	float local_b = *b;

	for (unsigned int i = 0; i < N; ++i) {
		result[i] = local_a * x[i] + local_b;
	}
}

inline void axpb_personalized_gold(unsigned int N, float *result, float *a, float *x,
		float *b, float shift_factor, unsigned int personalization_vertex) {

	float local_a = *a;
	float local_b = *b;

	for (unsigned int i = 0; i < N; ++i) {
		result[i] = local_a * x[i] + local_b + ((personalization_vertex == i) ? shift_factor : (float) 0);
	}
}

inline void dot_product_gold(unsigned int N, float *result, unsigned int *a,
		float *b) {

	*result = 0;

	for (unsigned int i = 0; i < N; ++i) {
		*result += a[i] * b[i];
	}
}

/////////////////////////////
/////////////////////////////

inline void pagerank_golden(unsigned int *ptr, unsigned int *idx, float *val,
		unsigned int *N, unsigned int *E, float *result, float *pr_vec,
		unsigned int *dangling_bitmap, float *tmp_pr, float *max_err,
		float *alpha, unsigned int *max_iter,
		unsigned int *iterations_to_convergence) {

	float ERR = *max_err;
	unsigned int ITER = *max_iter;
	unsigned int NUM_VERTICES = *N;
	float DANGLING_SCALE = *alpha / NUM_VERTICES;
	float SHIFT_FACTOR = ((float) 1.0 - *alpha) / NUM_VERTICES;

	float err = 0.0;
	float dangling_contrib = 0.0;
	unsigned int iter = 0;
	bool converged = false;

	while (!converged && iter < ITER) {

		spmv_gold(ptr, idx, val, NUM_VERTICES, tmp_pr, pr_vec);
		dot_product_gold(NUM_VERTICES, &dangling_contrib, dangling_bitmap, pr_vec);
		float shifting_factor = SHIFT_FACTOR + (DANGLING_SCALE * dangling_contrib);
		axpb_gold(NUM_VERTICES, tmp_pr, alpha, tmp_pr, &shifting_factor);

//		euclidean_distance_gold(NUM_VERTICES, &err, tmp_pr, pr_vec);
//
//		converged = err <= ERR;

		memcpy(pr_vec, tmp_pr, sizeof(float) * NUM_VERTICES);
		iter++;

	}
	*iterations_to_convergence = iter;
	memcpy(result, pr_vec, sizeof(float) * NUM_VERTICES);

}

/////////////////////////////
/////////////////////////////

inline void personalized_pagerank_golden(unsigned int *ptr, unsigned int *idx, float *val,
		unsigned int *N, unsigned int *E, float *result, float *pr_vec,
		unsigned int *dangling_bitmap, float *tmp_pr, float *max_err,
		float *alpha, unsigned int *max_iter,
		unsigned int *iterations_to_convergence, unsigned int personalization_vertex) {

	float ERR = *max_err;
	unsigned int ITER = *max_iter;
	unsigned int NUM_VERTICES = *N;
	float DANGLING_SCALE = *alpha / NUM_VERTICES;
	float SHIFT_FACTOR = ((float) 1.0 - *alpha);

	float err = 0.0;
	float dangling_contrib = 0.0;
	unsigned int iter = 0;
	bool converged = false;

	while (iter < ITER) {

		spmv_gold(ptr, idx, val, NUM_VERTICES, tmp_pr, pr_vec);
		dot_product_gold(NUM_VERTICES, &dangling_contrib, dangling_bitmap, pr_vec);
		float shifting_factor = (DANGLING_SCALE * dangling_contrib);
		axpb_personalized_gold(NUM_VERTICES, tmp_pr, alpha, tmp_pr, &shifting_factor, SHIFT_FACTOR, personalization_vertex);

//		euclidean_distance_gold(NUM_VERTICES, &err, tmp_pr, pr_vec);
//
//		converged = err <= ERR;

		memcpy(pr_vec, tmp_pr, sizeof(float) * NUM_VERTICES);
		iter++;

	}
	*iterations_to_convergence = iter;
	memcpy(result, pr_vec, sizeof(float) * NUM_VERTICES);

}

/////////////////////////////
/////////////////////////////

inline std::vector<std::vector<float>> multi_personalized_pagerank_golden(unsigned int *ptr, unsigned int *idx, float *val,
		unsigned int *N, unsigned int *E, float *pr_vec,
		unsigned int *dangling_bitmap, float *tmp_pr, float *max_err,
		float *alpha, unsigned int *max_iter,
		unsigned int *iterations_to_convergence, unsigned int *indices, int num_indices) {

	float ERR = *max_err;
	unsigned int ITER = *max_iter;
	unsigned int NUM_VERTICES = *N;
	float DANGLING_SCALE = *alpha / NUM_VERTICES;
	float SHIFT_FACTOR = ((float) 1.0 - *alpha);

	float err = 0.0;
	bool converged = false;

	auto results = std::vector<std::vector<float>>();

	for(int i = 0; i < num_indices; ++i) {

		unsigned int iter = 0;
		float dangling_contrib = 0.0;

		unsigned int cur_idx = indices[i];
		pr_vec[cur_idx] = 1;

		while (iter < ITER) {
			spmv_gold(ptr, idx, val, NUM_VERTICES, tmp_pr, pr_vec);
			dot_product_gold(NUM_VERTICES, &dangling_contrib, dangling_bitmap, pr_vec);
			float shifting_factor = (DANGLING_SCALE * dangling_contrib);
			axpb_personalized_gold(NUM_VERTICES, tmp_pr, alpha, tmp_pr, &shifting_factor, SHIFT_FACTOR, cur_idx);

			memcpy(pr_vec, tmp_pr, sizeof(float) * NUM_VERTICES);
			iter++;
		}
		*iterations_to_convergence = iter;

		auto tmp = std::vector<float>(pr_vec, pr_vec + NUM_VERTICES);
		results.push_back(tmp);

		// reset pr_vec
		for(int j = 0; j < NUM_VERTICES; ++j){
			pr_vec[j] = 0;
		}
	}
	return results;
}
