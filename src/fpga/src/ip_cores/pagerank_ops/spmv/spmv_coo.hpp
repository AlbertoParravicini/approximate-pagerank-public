#pragma once

#include "../../../csc_fpga/csc_fpga.hpp"
#include "../../../fpga_utils.hpp"
#include "spmv_utils.hpp"

/////////////////////////////
/////////////////////////////

inline void scatter_vec(input_block *y, index_type E, input_block *vec, input_block *scattered_vec) {
    index_type y_local[BUFFER_SIZE];
#pragma HLS array_partition variable=y_local complete dim=1
    fixed_float vec_local[BUFFER_SIZE];
#pragma HLS array_partition variable=vec_local complete dim=1
    fixed_float scattered_vec_local[BUFFER_SIZE];
#pragma HLS array_partition variable=scattered_vec_local complete dim=1

    index_type num_blocks_e = (E + BUFFER_SIZE - 1) / BUFFER_SIZE;

 	// Scatter the values of "vec" across a vector of size |E|;
	SCATTER_VEC: for (index_type i = 0; i < num_blocks_e; i++) {
#pragma HLS loop_tripcount min=640 max=640 avg=640
        read_block_index(y[i], y_local);
        // Scatter BUFFER_SIZE values;
        SCATTER_VEC_INNER: for (index_type j = 0; j < BUFFER_SIZE; j++) {
            // Locate the position of the "vec" value corresponding to the current "y" value;
            index_type curr_y = y_local[j];
            index_type vec_block_id = curr_y / BUFFER_SIZE;
            index_type vec_block_shift = curr_y % BUFFER_SIZE;
            input_block required_vec_block = vec[vec_block_id];
            unsigned int lower_range = FIXED_WIDTH * vec_block_shift;
            unsigned int upper_range = FIXED_WIDTH * (vec_block_shift + 1) - 1;
            unsigned int required_vec_value = required_vec_block.range(upper_range, lower_range);
            scattered_vec_local[j] =  *((fixed_float *) &required_vec_value);
        }
		// Write a block of scattered "vec" values;
		input_block temp_block;
		write_block_float(&temp_block, scattered_vec_local);
		scattered_vec[i] = temp_block;
	}
}

inline void scatter_vec_local_buffer(input_block *y, index_type E, fixed_float vec[MAX_N], input_block *scattered_vec) {
#pragma HLS array_partition variable=vec cyclic factor=16

    index_type y_local[BUFFER_SIZE];
#pragma HLS array_partition variable=y_local complete dim=1
    fixed_float vec_local[BUFFER_SIZE];
#pragma HLS array_partition variable=vec_local complete dim=1
    fixed_float scattered_vec_local[BUFFER_SIZE];
#pragma HLS array_partition variable=scattered_vec_local complete dim=1

    index_type num_blocks_e = (E + BUFFER_SIZE - 1) / BUFFER_SIZE;

    // Scatter the values of "vec" across a vector of size |E|;
	SCATTER_VEC: for (index_type i = 0; i < num_blocks_e; i++) {
#pragma HLS loop_tripcount min=640 max=640 avg=640
        read_block_index(y[i], y_local);
        // Scatter BUFFER_SIZE values;
        SCATTER_VEC_INNER: for (index_type j = 0; j < BUFFER_SIZE; j++) {
            scattered_vec_local[j] = vec[y_local[j]];
        }
		// Write a block of scattered "vec" values;
		input_block temp_block;
		write_block_float(&temp_block, scattered_vec_local);
		scattered_vec[i] = temp_block;
	}
}

/////////////////////////////
/////////////////////////////

inline void spmv_coo(input_block *x, input_block *y, input_block *val,
		index_type E, fixed_float res[MAX_N], input_block *scattered_vec) {

#pragma HLS array_partition variable=res cyclic factor=16

    index_type x_local[BUFFER_SIZE];
#pragma HLS array_partition variable=x_local complete dim=1
    fixed_float val_local[BUFFER_SIZE];
#pragma HLS array_partition variable=val_local complete dim=1
    fixed_float scattered_vec_local[BUFFER_SIZE];
#pragma HLS array_partition variable=scattered_vec_local complete dim=1

    fixed_float pointwise_res_local[BUFFER_SIZE];
#pragma HLS array_partition variable=pointwise_res_local complete dim=1
    fixed_float aggregated_res_local[2 * BUFFER_SIZE];
#pragma HLS array_partition variable=aggregated_res_local complete dim=1

    // Process the values of the COO matrix in a stream-like fashion, block by block.
    // COO values are 0-padded to have length multiple of BUFFER_SIZE, and it doesn't affect the results;
    OUTER_LOOP: for (index_type i = 0; i < E / BUFFER_SIZE; i++) {
#pragma HLS loop_tripcount min=640 max=640 avg=640
#pragma HLS pipeline II=1
        // 1. Read chunks of "x", "val" and "scattered_vec"
        read_block_index(x[i], x_local);
        read_block_float(val[i], val_local);
        read_block_float(scattered_vec[i], scattered_vec_local);
        // 2. Point-wise multiplication of a chunk of "val" and "scattered_vec";
        POINTWISE: for (index_type j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
            pointwise_res_local[j] = val_local[j] * scattered_vec_local[j];
        }
        // 3. Reset the aggregated result buffer;
        reset_large_buffer(aggregated_res_local);
        // 4. Aggregate point-wise multiplications using BUFFER_SIZE reductions;
        index_type start_x = x_local[0];
        REDUCTION_OUTER: for (index_type j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
			index_type curr_x = start_x + j;
			index_type start_in_buffer = start_x % BUFFER_SIZE; // The second half of the buffer is used to overflow values;
			// Aggregate point-wise multiplications only for "x" values equal to "curr_x";
			REDUCTION_INNER: for (index_type q = 0; q < BUFFER_SIZE; q++) {
#pragma HLS unroll
				aggregated_res_local[start_in_buffer + j] += pointwise_res_local[q] * (curr_x == x_local[q]);
			}
		}

        // 5. Store the aggregated results in "res".
        // We can unroll it as "res" has BUFFER_SIZE cyclic partitions, and we write BUFFER_SIZE adjacent values;
        index_type write_start = (start_x / BUFFER_SIZE) * BUFFER_SIZE;
        WRITE_RES_LOCAL: for (index_type j = 0; j < 2 * BUFFER_SIZE; j++) {
//#pragma HLS dependence variable=res inter false
#pragma HLS unroll
            res[write_start + j] += aggregated_res_local[j];
        }
    }
}

/////////////////////////////
/////////////////////////////

inline void spmv_coo_with_scatter(input_block *x, input_block *y, input_block *val,
		index_type E, fixed_float res[MAX_N], fixed_float res_read[MAX_N]) {

#pragma HLS array_partition variable=res cyclic factor=16

    index_type x_local[BUFFER_SIZE];
#pragma HLS array_partition variable=x_local complete dim=1
    index_type y_local[BUFFER_SIZE];
#pragma HLS array_partition variable=y_local complete dim=1
    fixed_float val_local[BUFFER_SIZE];
#pragma HLS array_partition variable=val_local complete dim=1
    fixed_float scattered_vec_local[BUFFER_SIZE];
#pragma HLS array_partition variable=scattered_vec_local complete dim=1

    fixed_float pointwise_res_local[BUFFER_SIZE];
#pragma HLS array_partition variable=pointwise_res_local complete dim=1
    fixed_float aggregated_res_local[2 * BUFFER_SIZE];
#pragma HLS array_partition variable=aggregated_res_local complete dim=1


    // Process the values of the COO matrix in a stream-like fashion, block by block.
    // COO values are 0-padded to have length multiple of BUFFER_SIZE, and it doesn't affect the results;
    OUTER_LOOP: for (index_type i = 0; i < E / BUFFER_SIZE; i++) {
#pragma HLS loop_tripcount min=640 max=640 avg=640
#pragma HLS pipeline II=1
        // 1. Read chunks of "x", "val" and scatter a block of "val";
        read_block_index(x[i], x_local);
        read_block_float(val[i], val_local);
        read_block_index(y[i], y_local);

        SCATTER_VEC: for(index_type j = 0; j < BUFFER_SIZE; ++j){
#pragma HLS unroll
        	scattered_vec_local[j] = res_read[y_local[j]];
        }

        //read_block_float(scattered_vec[i], scattered_vec_local);
        // 2. Point-wise multiplication of a chunk of "val" and "scattered_vec";
        POINTWISE: for (index_type j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
            pointwise_res_local[j] = val_local[j] * scattered_vec_local[j];
        }
        // 3. Reset the aggregated result buffer;
        reset_large_buffer(aggregated_res_local);
        // 4. Aggregate point-wise multiplications using BUFFER_SIZE reductions;
        index_type start_x = x_local[0];
        REDUCTION_OUTER: for (index_type j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
			index_type curr_x = start_x + j;
			index_type start_in_buffer = start_x % BUFFER_SIZE; // The second half of the buffer is used to overflow values;
			// Aggregate point-wise multiplications only for "x" values equal to "curr_x";
			REDUCTION_INNER: for (index_type q = 0; q < BUFFER_SIZE; q++) {
#pragma HLS unroll
				aggregated_res_local[start_in_buffer + j] += pointwise_res_local[q] * (curr_x == x_local[q]);
			}
		}

        // 5. Store the aggregated results in "res".
        // We can unroll it as "res" has BUFFER_SIZE cyclic partitions, and we write BUFFER_SIZE adjacent values;
        index_type write_start = (start_x / BUFFER_SIZE) * BUFFER_SIZE;
        WRITE_RES_LOCAL: for (index_type j = 0; j < 2 * BUFFER_SIZE; j++) {
#pragma HLS unroll
            res[write_start + j] += aggregated_res_local[j];
        }
    }
}

/////////////////////////////
/////////////////////////////

inline void spmv_coo_main(input_block *x, input_block *y, input_block *val,
		index_type N, index_type E, input_block *res, input_block *vec, input_block *scattered_vec) {
// Ports used to transfer data, using AXI master;
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = y offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = val offset = slave bundle = gmem1

#pragma HLS INTERFACE m_axi port = res offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = vec offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = scattered_vec offset = slave bundle = gmem2

// Ports used for control signals, using AXI slave;
#pragma HLS INTERFACE s_axilite register port = N bundle = control
#pragma HLS INTERFACE s_axilite register port = E bundle = control
#pragma HLS INTERFACE s_axilite register port = return bundle = control

    // Allocate a local buffer that contains all the values of "res";
	fixed_float res_local[MAX_N];
#pragma HLS array_partition variable=res_local cyclic factor=16
    RESET_RES_LOCAL: for (index_type i = 0; i < MAX_N / BUFFER_SIZE; i++) {
#pragma HLS pipeline II=1
        for (index_type j = 0; j < BUFFER_SIZE; j++) {
#pragma HLS unroll
            res_local[i * BUFFER_SIZE + j] = 0;
        }
    }

    // Scatter the values of "vec" across a vector of size |E|;
    scatter_vec(y, E, vec, scattered_vec);

	// Execute the SPMV;
	spmv_coo(x, y, val, E, res_local, scattered_vec);

    // Copy values of "res" to the output;
	WRITE_RES: for (index_type i = 0; i < N / BUFFER_SIZE; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=64 max=64 avg=64
		input_block tmp_block;
		write_block_float(&tmp_block, &res_local[i * BUFFER_SIZE]);
		res[i] = tmp_block;
	}
}
