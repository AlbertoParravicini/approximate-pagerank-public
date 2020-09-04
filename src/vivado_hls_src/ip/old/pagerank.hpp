#pragma once

#include <stdlib.h>

#include "csc_fpga.hpp"

extern "C" {

void fixed_spmv(index_type dim, fixed_float *result, fixed_float *csc_col_val, index_type *csc_col_ptr,
		index_type *csc_col_idx, fixed_float *vec);

void fixed_dot(index_type dim, fixed_float *result, fixed_float *a, index_type *b);

void fixed_dot_2(index_type dim, fixed_float *result, fixed_float *a, index_type *b);

void fixed_axpb(index_type dim, fixed_float *result, fixed_float a, fixed_float b);

void fixed_euclidean_distance(index_type dim, fixed_float *result, fixed_float *a, fixed_float *b);

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
		);

void fixed_pagerank(fixed_float *result, fixed_float *csc_col_val, index_type *csc_col_ptr, index_type *csc_col_idx,
		fixed_float *pr_vec, index_type *dangling_bitmap, fixed_float *result_tmp, fixed_float *alpha,
		fixed_float *max_err, index_type *N, index_type *max_iter);
}
