#pragma once

#include "spmv.hpp"
#include "csc_fpga.hpp"
#include "fpga_utils.hpp"

extern "C" {

/////////////////////////////
/////////////////////////////

// This looks better than 2 (pipeline length 4 instead of 64);
inline void reduction_tree_1(
		fixed_float pointwise_res[BUFFER_SIZE],
		index_type x_local[BUFFER_SIZE],
		fixed_float aggregated_res[2 * BUFFER_SIZE],
		index_type start_in_buffer,
		index_type start_x) {
	REDUCTION_OUTER: for (index_type k = 0; k < BUFFER_SIZE; k++) {
#pragma HLS unroll
		index_type curr_x = start_x + k;
		// Aggregate point-wise multiplications only for "x" values equal to "curr_x";
		fixed_float aggregator = 0;
		REDUCTION_INNER: for (index_type q = 0; q < BUFFER_SIZE; q++) {
#pragma HLS unroll
			aggregator += pointwise_res[q] * (curr_x == x_local[q]);
		}
		aggregated_res[start_in_buffer + k] = aggregator;
	}
}

inline void reduction_tree_2(
		fixed_float pointwise_res[BUFFER_SIZE],
		index_type x_local[BUFFER_SIZE],
		fixed_float aggregated_res[2 * BUFFER_SIZE],
		index_type start_in_buffer,
		index_type start_x) {
	REDUCTION_OUTER: for (index_type q = 0; q < BUFFER_SIZE; q++) {
#pragma HLS unroll
		fixed_float pointwise_temp = pointwise_res[q];
		index_type x_local_temp = x_local[q];
		// Aggregate point-wise multiplications only for "x" values equal to "curr_x";
		REDUCTION_INNER: for (index_type k = 0; k < BUFFER_SIZE; k++) {
#pragma HLS unroll
			aggregated_res[start_in_buffer + k] += pointwise_temp * ((start_x + k) == x_local_temp);
		}
	}
}

/////////////////////////////
/////////////////////////////

inline void inner_spmv_product(
		index_type x[BUFFER_SIZE],
		index_type y[BUFFER_SIZE],
		fixed_float val[BUFFER_SIZE],
		fixed_float vec[MAX_N],
		fixed_float aggregated_res_local[2 * BUFFER_SIZE],
		index_type start_x,
		index_type start_in_buffer) {

    fixed_float pointwise_res_local[BUFFER_SIZE];
#pragma HLS array_partition variable=pointwise_res_local complete dim=1

	// 1. Point-wise multiplication of a chunk of "val" and "scattered_vec";
	POINTWISE: for (index_type k = 0; k < BUFFER_SIZE; k++) {
#pragma HLS unroll
		index_type y_pos = y[k];
		fixed_float val_curr = vec[y_pos];
		pointwise_res_local[k] = val[k] * val_curr;
	}

	// 2. Aggregate point-wise multiplications using BUFFER_SIZE reductions;
	reduction_tree_1(pointwise_res_local, x, aggregated_res_local, start_in_buffer, start_x);
}

/////////////////////////////
/////////////////////////////

inline void spmv_coo_with_scatter(input_block *x, input_block *y, input_block *val,
		index_type E, fixed_float res[MAX_N], fixed_float vec[MAX_N]) {

    index_type x_local[BUFFER_SIZE];
#pragma HLS array_partition variable=x_local complete dim=1
    index_type y_local[BUFFER_SIZE];
#pragma HLS array_partition variable=y_local complete dim=1
    fixed_float val_local[BUFFER_SIZE];
#pragma HLS array_partition variable=val_local complete dim=1

    fixed_float pointwise_res_local[BUFFER_SIZE];
#pragma HLS array_partition variable=pointwise_res_local complete dim=1
    fixed_float aggregated_res_local[2 * BUFFER_SIZE];
#pragma HLS array_partition variable=aggregated_res_local complete dim=1
    fixed_float res_local_to_update[2 * BUFFER_SIZE];
#pragma HLS array_partition variable=res_local_to_update complete dim=1

    // Process the values of the COO matrix in a stream-like fashion, block by block.
    // COO values are 0-padded to have length multiple of BUFFER_SIZE, and it doesn't affect the results;
    OUTER_LOOP: for (index_type i = 0; i < E / BUFFER_SIZE; i++) {
#pragma HLS loop_tripcount min=hls_iterations_e max=hls_iterations_e avg=hls_iterations_e
#pragma HLS pipeline II=1
        // 1. Read chunks of "x", "y" and scatter a block of "val";
        read_block_index(x[i], x_local);
        read_block_index(y[i], y_local);
        read_block_float(val[i], val_local);

        index_type start_x = x_local[0];
		index_type start_in_buffer = start_x % BUFFER_SIZE; // The second half of the buffer is used to overflow values;
		index_type write_start = (start_x / BUFFER_SIZE) * BUFFER_SIZE;

		// Read res values to update;
		READ_RES: for (index_type k = 0; k < 2 * BUFFER_SIZE; k++) {
#pragma HLS unroll
			res_local_to_update[k] = res[write_start + k];
			aggregated_res_local[k] = 0;
		}

		// Perform the main SpMV computation;
		inner_spmv_product(x_local, y_local, val_local, vec, aggregated_res_local, start_x, start_in_buffer);

		// Store the aggregated results in a temporary buffer.
		// We can unroll it as "res" has BUFFER_SIZE cyclic partitions, and we write BUFFER_SIZE adjacent values;
		WRITE_RES_LOCAL: for (index_type k = 0; k < 2 * BUFFER_SIZE; k++) {
#pragma HLS unroll
			res_local_to_update[k] += aggregated_res_local[k];
		}

		// Write updated values to "res";
		WRITE_RES: for (index_type k = 0; k < 2 * BUFFER_SIZE; k++) {
#pragma HLS unroll
			res[write_start + k] = res_local_to_update[k];
		}
    }
}

/////////////////////////////
/////////////////////////////

void spmv_coo_main(input_block *x, input_block *y, input_block *val,
		index_type N, index_type E, input_block *res, input_block *vec, input_block *scattered_vec);

void spmv_coo_with_scatter_main(input_block *x, input_block *y, input_block *val, index_type N,
		index_type E, input_block *res, input_block *vec);

}
