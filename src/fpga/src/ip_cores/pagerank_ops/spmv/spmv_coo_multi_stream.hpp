#pragma once

#include "hls_stream.h"
#include "../../../csc_fpga/csc_fpga.hpp"
#include "../../../fpga_utils.hpp"
#include "spmv_utils.hpp"

/////////////////////////////
/////////////////////////////

typedef struct reduction_result {
	input_block aggregated_res;
	index_type write_start;
	index_type start_in_buffer;
} reduction_result_t;


void inner_spmv_product_stream(
		hls::stream<input_block> &x,
		hls::stream<input_block> &val,
		hls::stream<input_block> &vec,
		hls::stream<reduction_result_t> &aggregated_res) {

	input_block x_local = x.read();
	input_block val_local = val.read();
	input_block vec_local = vec.read();

	fixed_float pointwise_res_local[BUFFER_SIZE];
#pragma HLS array_partition variable=pointwise_res_local complete dim=1

	input_block aggregated_res_local;

	reduction_result_t result;

	index_type start_x = x_local.range(FIXED_WIDTH - 1, 0);
	index_type start_in_buffer = start_x % BUFFER_SIZE;
	result.write_start = (start_x / BUFFER_SIZE) * BUFFER_SIZE;
	result.start_in_buffer = start_in_buffer;

	// Point-wise multiplication of a chunk of "val" and "scattered_vec";
	POINTWISE: for (index_type k = 0; k < BUFFER_SIZE; k++) {
#pragma HLS unroll
		unsigned int lower_range = FIXED_WIDTH * k;
		unsigned int upper_range = FIXED_WIDTH * (k + 1) - 1;
		unsigned int val_curr = val_local.range(upper_range, lower_range);
		fixed_float val_float = *((fixed_float *) &val_curr);
		unsigned int vec_curr = vec_local.range(upper_range, lower_range);
		fixed_float vec_float = *((fixed_float *) &vec_curr);
		pointwise_res_local[k] = val_float * vec_float;
	}

	REDUCTION_OUTER: for (index_type k = 0; k < BUFFER_SIZE; k++) {
#pragma HLS unroll
		index_type curr_start_x = start_x + k;
		// Aggregate point-wise multiplications only for "x" values equal to "curr_x";
		fixed_float aggregator = 0;
		REDUCTION_INNER: for (index_type q = 0; q < BUFFER_SIZE; q++) {
#pragma HLS unroll
			unsigned int lower_q = FIXED_WIDTH * q;
			unsigned int upper_q = FIXED_WIDTH * (q + 1) - 1;
			fixed_float pointwise_res_curr = pointwise_res_local[q];
			unsigned int x_curr = x_local.range(upper_q, lower_q);
			aggregator += pointwise_res_curr * (curr_start_x == x_curr);
		}
		unsigned int lower_k = FIXED_WIDTH * k;
		unsigned int upper_k = FIXED_WIDTH * (k + 1) - 1;
		aggregated_res_local.range(upper_k, lower_k) = *((unsigned int *) &aggregator);
	}
	result.aggregated_res = aggregated_res_local;
	aggregated_res << result;
}

/////////////////////////////
/////////////////////////////

void spmv_coo_multi_stream_inner(input_block *x, input_block *y, input_block *val,
		index_type E, fixed_float res[N_PPR_VERTICES][MAX_N], fixed_float vec[N_PPR_VERTICES][MAX_N],
		fixed_float last_res_buffer[N_PPR_VERTICES][BUFFER_SIZE], index_type *last_write_start) {

#pragma HLS dataflow

	input_block x_local;
	input_block y_local;
	input_block val_local;
	input_block vec_local[N_PPR_VERTICES];
#pragma HLS array_partition variable=vec_local complete dim=1

    // Define streams;
    hls::stream<input_block> x_stream[N_PPR_VERTICES];
#pragma HLS STREAM variable=x_stream depth=10 dim=1
#pragma HLS array_partition variable=x_stream complete dim=1
    hls::stream<input_block> val_stream[N_PPR_VERTICES];
#pragma HLS STREAM variable=val_stream depth=10 dim=1
#pragma HLS array_partition variable=val_stream complete dim=1
    hls::stream<input_block> vec_stream[N_PPR_VERTICES];
#pragma HLS STREAM variable=vec_stream depth=10 dim=1
#pragma HLS array_partition variable=vec_stream complete dim=1

    hls::stream<input_block> pointwise_stream[N_PPR_VERTICES];
#pragma HLS STREAM variable=pointwise_stream depth=10 dim=1
#pragma HLS array_partition variable=pointwise_stream complete dim=1
    hls::stream<reduction_result_t> aggregated_res_stream[N_PPR_VERTICES];
#pragma HLS STREAM variable=aggregated_res_stream depth=10 dim=1
#pragma HLS array_partition variable=aggregated_res_stream complete dim=1

    fixed_float aggregated_res_local[N_PPR_VERTICES][2 * BUFFER_SIZE];
#pragma HLS array_partition variable=aggregated_res_local complete dim=0

    fixed_float res_local_to_write[N_COLS][BUFFER_SIZE];
#pragma HLS array_partition variable=res_local_to_write complete dim=0
    fixed_float res_local_to_increment[N_COLS][BUFFER_SIZE];
#pragma HLS array_partition variable=res_local_to_increment complete dim=0

    // Reset local storage buffers;
    for (index_type i = 0; i < N_COLS; i++) {
#pragma HLS unroll
    	for (index_type j = 0; j < 2 * BUFFER_SIZE; j++) {
#pragma HLS unroll
    		aggregated_res_local[i][j] = 0;
    	}
    	for (index_type j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
    		res_local_to_write[i][j] = 0;
    		res_local_to_increment[i][j] = 0;
    	}
    }


    // Process the values of the COO matrix in a stream-like fashion, block by block.
    // COO values are 0-padded to have length multiple of BUFFER_SIZE, and it doesn't affect the results;
    OUTER_LOOP: for (index_type i = 0; i < E / BUFFER_SIZE; i++) {
#pragma HLS loop_tripcount min=hls_iterations_e max=hls_iterations_e avg=hls_iterations_e
#pragma HLS pipeline II=1

    	// Read chunks of "x", "y", "val", then scatter values of "vec";
        x_local = x[i];
        y_local = y[i];
        val_local = val[i];
    	SCATTER_VEC: for (index_type j = 0; j < BUFFER_SIZE; j++) {
    #pragma HLS unroll
    		unsigned int lower_range = FIXED_WIDTH * j;
    		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
    		unsigned int y_curr = y_local.range(upper_range, lower_range);
    		SCATTER_VEC_INNER: for(index_type k = 0; k < N_PPR_VERTICES; k++) {
    #pragma HLS unroll
    			fixed_float scattered_vec = vec[k][y_curr];
    			vec_local[k].range(upper_range, lower_range) = *((unsigned int *) &scattered_vec);
    		}
    	}
        ITERATE_COL: for(index_type j = 0; j < N_PPR_VERTICES; j++) {
        #pragma HLS unroll
			x_stream[j] << x_local;
			val_stream[j] << val_local;
			vec_stream[j] << vec_local[j];
        }
    }

    OUTER_LOOP2: for (index_type i = 0; i < E / BUFFER_SIZE; i++) {
    #pragma HLS loop_tripcount min=hls_iterations_e max=hls_iterations_e avg=hls_iterations_e
    #pragma HLS pipeline II=1
        ITERATE_COL2: for(index_type j = 0; j < N_PPR_VERTICES; j++) {
#pragma HLS unroll
			// Perform point-wise products;
        	inner_spmv_product_stream(x_stream[j], val_stream[j], vec_stream[j], aggregated_res_stream[j]);
        }
    }

    // Support array used in the storage FSM.
    // Values written in it are the same for each column,
    //   but we need an array to have independent parallel R/W access;
	index_type write_start_old[N_COLS];
#pragma HLS array_partition variable=write_start_old complete dim=1
	for(index_type j = 0; j < N_COLS; j++) {
#pragma HLS unroll
		write_start_old[j] = 0;
	}

	OUTER_LOOP_3: for (index_type i = 0; i < E / BUFFER_SIZE; i++) {
#pragma HLS loop_tripcount min=hls_iterations_e max=hls_iterations_e avg=hls_iterations_e
#pragma HLS pipeline II=1
		ITERATE_COL_3: for(index_type j = 0; j < N_COLS; j++) {
	#pragma HLS unroll
            reduction_result_t reduction_result_local;

        	// Read the aggregated stream output;
        	aggregated_res_stream[j] >> reduction_result_local;
        	// Dividing-multiplying by BUFFER_SIZE again is required to infer that writes on "res" are BUFFER_SIZE-aligned;
        	index_type write_start = (reduction_result_local.write_start / BUFFER_SIZE) * BUFFER_SIZE;
        	index_type write_start_old_local = (write_start_old[j] / BUFFER_SIZE) * BUFFER_SIZE;
        	index_type start_in_buffer = reduction_result_local.start_in_buffer;

        	// Move BUFFER_SIZE aggregated results to an array of size 2 * BUFFER_SIZE,
        	//   with an offset of "start_in_buffer";
        	read_block_float(reduction_result_local.aggregated_res, aggregated_res_local[j] + start_in_buffer);

        	// Write-back Finite-State Machine;
        	if (write_start == write_start_old_local) {
        		// In this case, we can accumulate values on the current buffers,
        		//   half on a buffer and half on another buffer;
            	INCREMENT_RES_LOCAL: for (index_type k = 0; k < BUFFER_SIZE; k++) {
#pragma HLS unroll
            		res_local_to_write[j][k] += aggregated_res_local[j][k];
            		res_local_to_increment[j][k] += aggregated_res_local[j][k + BUFFER_SIZE];
            	}
        	} else {
        		// Write back the current values on "res_local_to_write",
        		//   and shift values from "res_local_to_increment" to "res_local_to_write";
         		WRITE_RES: for (index_type k = 0; k < BUFFER_SIZE; k++) {
#pragma HLS unroll
         			res[j][write_start_old_local + k] = res_local_to_write[j][k];
         			res_local_to_write[j][k] = res_local_to_increment[j][k] + aggregated_res_local[j][k];
         			res_local_to_increment[j][k] = aggregated_res_local[j][k + BUFFER_SIZE];
				}
        	}
        	// Reset the aggregated results buffer;
        	RESET_AGGREGATED_RES: for (index_type k = 0; k < 2 * BUFFER_SIZE; k++) {
#pragma HLS unroll
        		aggregated_res_local[j][k] = 0;
        	}
        	write_start_old[j] = write_start;
		}
    }
	// Write the last chunk of values that cannot be processed within the dataflow region;
	ITERATE_COL_4: for (index_type j = 0; j < N_COLS; j++) {
#pragma HLS unroll
		INNER_LOOP_4: for (index_type k = 0; k < BUFFER_SIZE; k++) {
#pragma HLS unroll
			last_res_buffer[j][k] = res_local_to_write[j][k];
		}
	}
	*last_write_start = write_start_old[0];
}

/////////////////////////////
/////////////////////////////

void spmv_coo_multi_stream(input_block *x, input_block *y, input_block *val,
		index_type E, fixed_float res[N_PPR_VERTICES][MAX_N], fixed_float vec[N_PPR_VERTICES][MAX_N]) {

    fixed_float last_res_buffer[N_PPR_VERTICES][BUFFER_SIZE];
#pragma HLS array_partition variable=last_res_buffer complete dim=0

    index_type last_write_start;

	spmv_coo_multi_stream_inner(x, y, val, E, res, vec, last_res_buffer, &last_write_start);

	// Need to write the last chunk outside of the main computation,
	// otherwise we write the same output twice in different dataflow regions and it doesn't work;
	OUTER_LOOP_LAST: for(index_type j = 0; j < N_COLS; j++) {
#pragma HLS unroll
		WRITE_RES_LAST: for (index_type k = 0; k < BUFFER_SIZE; k++) {
#pragma HLS unroll
			res[j][last_write_start + k] = last_res_buffer[j][k];
		}
	}
}

