#include "multi_spmv.hpp"
//#include "multi_spmv_2.hpp"
#include "multi_spmv_stream.hpp"

/////////////////////////////
/////////////////////////////

// This looks better than 2 (pipeline length 4 instead of 64);
void reduction_tree_1(
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

void reduction_tree_2(
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

void inner_spmv_product(
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
		pointwise_res_local[k] = val[k] * vec[y[k]];
	}

	// 2. Aggregate point-wise multiplications using BUFFER_SIZE reductions;
	reduction_tree_1(pointwise_res_local, x, aggregated_res_local, start_in_buffer, start_x);
}

/////////////////////////////
/////////////////////////////

void spmv_coo_multi(input_block *x, input_block *y, input_block *val,
		index_type E, fixed_float res[N_PPR_VERTICES][MAX_N], fixed_float vec[N_PPR_VERTICES][MAX_N]) {

    index_type x_local[BUFFER_SIZE];
#pragma HLS array_partition variable=x_local complete dim=1
    index_type y_local[BUFFER_SIZE];
#pragma HLS array_partition variable=y_local complete dim=1
    fixed_float val_local[BUFFER_SIZE];
#pragma HLS array_partition variable=val_local complete dim=1

    fixed_float aggregated_res_local[N_PPR_VERTICES][2 * BUFFER_SIZE];
#pragma HLS array_partition variable=aggregated_res_local complete dim=0

    fixed_float res_local_to_update[N_PPR_VERTICES][2 * BUFFER_SIZE];
#pragma HLS array_partition variable=res_local_to_update complete dim=0

    // Process the values of the COO matrix in a stream-like fashion, block by block.
    // COO values are 0-padded to have length multiple of BUFFER_SIZE, and it doesn't affect the results;
    OUTER_LOOP: for (index_type i = 0; i < E / BUFFER_SIZE; i++) {
#pragma HLS loop_tripcount min=hls_iterations_e max=hls_iterations_e avg=hls_iterations_e
#pragma HLS pipeline II=1

    	// Read chunks of "x", "y" and "val"
        read_block_index(x[i], x_local);
        read_block_index(y[i], y_local);
        read_block_float(val[i], val_local);

		index_type start_x = x_local[0];
		index_type start_in_buffer = start_x % BUFFER_SIZE; // The second half of the buffer is used to overflow values;
        index_type write_start = (start_x / BUFFER_SIZE) * BUFFER_SIZE;

        // Read res values to update;
        INNER_LOOP: for(index_type j = 0; j < N_PPR_VERTICES; j++) {
#pragma HLS unroll

        	READ_RES: for (index_type k = 0; k < 2 * BUFFER_SIZE; k++) {
#pragma HLS unroll
				res_local_to_update[j][k] = res[j][write_start + k];
				aggregated_res_local[j][k] = 0;
			}

        	// Perform the main SpMV computation;
        	inner_spmv_product(x_local, y_local, val_local, vec[j],	aggregated_res_local[j], start_x, start_in_buffer);

        	// Store the aggregated results in a temporary buffer.
        	// We can unroll it as "res" has BUFFER_SIZE cyclic partitions, and we write BUFFER_SIZE adjacent values;
        	WRITE_RES_LOCAL: for (index_type k = 0; k < 2 * BUFFER_SIZE; k++) {
#pragma HLS unroll
        		res_local_to_update[j][k] += aggregated_res_local[j][k];
        	}

        	// Write updated values to "res";
        	WRITE_RES: for (index_type k = 0; k < 2 * BUFFER_SIZE; k++) {
#pragma HLS unroll
        		res[j][write_start + k] = res_local_to_update[j][k];
			}
        }
    }
}

void multi_spmv_main(input_block *x, input_block *y, input_block *val, index_type N,
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
	fixed_float res_local[N_PPR_VERTICES][MAX_N];
#pragma HLS array_partition variable=res_local complete dim=1
#pragma HLS array_partition variable=res_local cyclic factor=hls_buffer_size dim=2
#pragma HLS RESOURCE variable=res_local core=XPM_MEMORY uram

	fixed_float vec_local[N_PPR_VERTICES][MAX_N];
#pragma HLS array_partition variable=vec_local complete dim=1
#pragma HLS array_partition variable=vec_local cyclic factor=hls_buffer_size dim=2
#pragma HLS RESOURCE variable=vec_local core=XPM_MEMORY uram

	// Note that the actual number of blocks is num_blocks_n * N_PPR_VERTICES;
	index_type num_blocks_n = (N + BUFFER_SIZE - 1) / BUFFER_SIZE;

	RESET_RES_LOCAL: for (index_type i = 0; i < num_blocks_n; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=hls_iterations_v max=hls_iterations_v avg=hls_iterations_v
		for (index_type j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
			for (index_type k = 0; k < N_PPR_VERTICES; k++) {
				res_local[k][i * BUFFER_SIZE + j] = 0;
			}
		}
	}

	READ_VEC_LOCAL: for (index_type k = 0; k < N_PPR_VERTICES; k++) {
#pragma HLS unroll
		READ_VEC_LOCAL_INNER: for (index_type i = 0; i < num_blocks_n; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=hls_iterations_v max=hls_iterations_v avg=hls_iterations_v
			read_block_float(vec[i + k * num_blocks_n], vec_local[k] + i * BUFFER_SIZE);
		}
	}

	// Execute the SPMV;
	spmv_coo_multi_stream(x, y, val, E, res_local, vec_local);

	// Copy values of "res" to the output;
	WRITE_RESULT: for (index_type i = 0; i < N_PPR_VERTICES; i++) {
#pragma HLS unroll
		WRITE_RESULT_INNER: for (index_type j = 0; j < num_blocks_n; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=hls_iterations_v max=hls_iterations_v avg=hls_iterations_v
			input_block tmp_block;
			write_block_float(&tmp_block, &res_local[i][j * BUFFER_SIZE]);
			res[j + i * num_blocks_n] = tmp_block;
		}
	}
}
