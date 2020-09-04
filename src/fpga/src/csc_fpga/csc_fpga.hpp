#pragma once

#include <iostream>
#include <vector>
#include <ap_fixed.h>
#include "../../../common/csc_matrix/csc_matrix.hpp"

#define FIXED_WIDTH 26

#define SCALE (FIXED_WIDTH - 1)
#define FIXED_INTEGER_PART (FIXED_WIDTH - SCALE)

#ifndef N_PPR_VERTICES
#define N_PPR_VERTICES 8
#define N_COLS N_PPR_VERTICES
#endif

// The maximum is 1024 but it gives out `Unsupported enormous number of load store instruction`
// Maximum allowed is 512
#define AP_UINT_BITWIDTH 256
#define PACKET_ELEMENT_SIZE 32 // Single values have FIXED_WIDTH bits, but we have to 0-pad the packet to reach AP_UINT_BITWIDTH;
#define BUFFER_SIZE (AP_UINT_BITWIDTH / PACKET_ELEMENT_SIZE) // Each ap_uint has at most 512 bits;

#define MAX_VERTICES (2 << 17) // 2^17

#define MAX_N MAX_VERTICES
#define MAX_N_BUFFER (MAX_N / BUFFER_SIZE)

#define MAX_ITERATIONS 1024

// Values used to specify the trip-count of HLS loops;
const int hls_buffer_size = BUFFER_SIZE;
const int hls_num_vertices = 1000;
const int hls_degree = 10;
const int hls_num_edges = hls_degree * hls_num_vertices;
const int hls_iterations_v = hls_num_vertices / hls_buffer_size;
const int hls_iterations_e = hls_num_edges / hls_buffer_size;
const int hls_iterations_bitmap = hls_num_vertices / AP_UINT_BITWIDTH;
const int hls_max_iter = MAX_ITERATIONS;
const int hls_iter = 6;
const int hls_write_errors_iter = N_PPR_VERTICES * hls_iter;

typedef ap_uint<1> dangling_type;
typedef ap_ufixed<FIXED_WIDTH, FIXED_INTEGER_PART, AP_TRN_ZERO> fixed_float;
//typedef ap_ufixed<FIXED_WIDTH, FIXED_INTEGER_PART, AP_TRN_ZERO, AP_SAT_ZERO> fixed_float;
// typedef float fixed_float;

// Type use for L2 error norm;
typedef ap_ufixed<32, 2, AP_TRN_ZERO> fixed_error_float;
//typedef float fixed_error_float;


typedef ap_uint<AP_UINT_BITWIDTH> input_block;

typedef struct csc_fixed_fpga_t {
    std::vector<fixed_float> col_val;
    std::vector<index_type_fpga> col_ptr;
    std::vector<index_type_fpga> col_idx;
} csc_fixed_fpga_t;

csc_fixed_fpga_t convert_to_fixed_point_fpga(csc_t matrix);
