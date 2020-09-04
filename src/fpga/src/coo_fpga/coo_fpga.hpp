#pragma once

#include <iostream>
#include <vector>
#include <ap_fixed.h>
#include "../csc_fpga/csc_fpga.hpp"
#include "../../../common/coo_matrix/coo_matrix.hpp"

struct coo_fixed_fpga_t {
    std::vector<index_type_fpga> start;
    std::vector<index_type_fpga> end;
    std::vector<fixed_float> val;
    index_type N = 0;
    index_type E = 0;
    // Optionally, the COO contains diagonal value with value 0, to avoid skipping diagonal values;
    index_type E_fixed = 0;
    bool extra_self_loops_added = false;

    coo_fixed_fpga_t(coo_t coo, bool add_missing_loops = false) : coo_fixed_fpga_t(coo.start, coo.end, coo.val, add_missing_loops) {}

    coo_fixed_fpga_t(std::vector<index_type> _start, std::vector<index_type> _end, std::vector<num_type> _val, bool add_missing_loops = false) {

    	E = _start.size();
    	index_type extra_N = 0;
		for (int i = 0; i < _start.size(); ++i){
			start.push_back(_start[i]);
			end.push_back(_end[i]);
			val.push_back(_val[i]);

			N = std::max(N, _start[i]);

			// Add a 0 diagonal value if this row has no non-zero values, to avoid skipping this row;
			if (i < _start.size() - 1 && add_missing_loops) {
				for (int j = _start[i] + 1; j < _start[i + 1]; j++) {
					start.push_back(j);
					end.push_back(j);
					val.push_back(0);
					extra_N++;
				}
			}
		}
		E_fixed = E + extra_N;
		extra_self_loops_added = true;
	}

    // Create a COO from a CSC. Note that the COO will contain the graph transposed, i.e. edges are ingoing;
    coo_fixed_fpga_t(csc_fixed_fpga_t csc, bool add_missing_loops = false) {
    	N = csc.col_ptr.size() - 1;
    	E = csc.col_idx.size();
    	E_fixed = E;
		for (int i = 0; i < csc.col_ptr.size() - 1; i++) {
			int s = csc.col_ptr[i];
			int e = csc.col_ptr[i + 1];
			for (int j = s; j < e; j++) {
				start.push_back(i);
				end.push_back(csc.col_idx[j]);
				val.push_back(csc.col_val[j]);
			}
			// Add a 0 diagonal value if this row has no non-zero values, to avoid skipping this row;
			if (s == e && add_missing_loops) {
				start.push_back(i);
				end.push_back(i);
				val.push_back(0);
				E_fixed++;
			}
		}
		extra_self_loops_added = true;
	}

    void print_coo(bool compact = false, bool transposed = true) {
		if (compact) {


			int n = 0;
			// TODO: need to add transposed;
			index_type last_s = 0;
			index_type curr_e = 0;
			std::vector<index_type> neighbors;
			std::vector<fixed_float> vals;

			std::cout << "N: " << N << ", E: " << E << std::endl;

			for (int i = 0; i < start.size(); i++) {
				index_type curr_s = start[i];
				if (curr_s == last_s) {
					neighbors.push_back(end[curr_e]);
					vals.push_back(val[curr_e++]);
				} else {
					std::cout << n << ") degree: " << neighbors.size() << std::endl;
					std::cout << "  edges: ";
					for (auto e: neighbors) {
						std::cout << e << ", ";
					}
					std::cout << std::endl;
					std::cout << "  vals: ";
					for (auto v: vals) {
//						std::cout << v.to_float() << ", ";
					}
					std::cout << std::endl;

					last_s = curr_s;
					neighbors = { end[curr_e] };
					vals = { val[curr_e++] };
					n = curr_s;
				}
			}
			std::cout << n << ") degree: " << neighbors.size() << std::endl;
			std::cout << "  edges: ";
			for (auto e: neighbors) {
				std::cout << e << ", ";
			}
			std::cout << std::endl;
			std::cout << "  vals: ";
			for (auto v: vals) {
//				std::cout << v.to_float() << ", ";
			}
			std::cout << std::endl;
		} else {
			for (int i = 0; i < start.size(); i++) {
//				std::cout << start[i] << (transposed ? " <- " : " -> ") << end[i] << ": " << val[i].to_float() << std::endl;
			}
		}
	}
};
