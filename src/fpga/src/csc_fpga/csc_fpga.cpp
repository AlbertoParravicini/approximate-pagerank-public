#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "csc_fpga.hpp"

csc_fixed_fpga_t convert_to_fixed_point_fpga(csc_t matrix) {
	csc_fixed_fpga_t out;
	std::vector<fixed_float> value;
	std::vector<index_type_fpga> ptr;
	std::vector<index_type_fpga> idx;


	// Pthread cannot be enabled in HLS
//	std::thread t1([&value, &matrix, &out]() {
//		for (float &el : matrix.col_val) {
//			value.push_back(el);
//		}
//		out.col_val = value;
//	});
//
//	std::thread t2([&idx, &matrix, &out]() {
//		for (index_type_fpga &el : matrix.col_idx) {
//			idx.push_back(el);
//		}
//		out.col_idx = idx;
//	});
//
//	std::thread t3([&ptr, &matrix, &out]() {
//		for (index_type_fpga &el : matrix.col_ptr) {
//			ptr.push_back(el);
//		}
//		out.col_ptr = ptr;
//	});
//
//	t1.join();
//	t2.join();
//	t3.join();

	for(int i = 0; i < matrix.col_val.size(); ++i){
		value.push_back(matrix.col_val[i]);
	}

	out.col_val = value;
	for(int i = 0; i < matrix.col_idx.size(); ++i){
		idx.push_back(matrix.col_idx[i]);
	}
	out.col_idx = idx;
	for(int i = 0; i < matrix.col_ptr.size(); ++i){
		ptr.push_back(matrix.col_ptr[i]);
	}
	out.col_ptr = ptr;
	return out;
}
