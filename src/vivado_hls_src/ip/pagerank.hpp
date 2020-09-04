#include "fpga_utils.hpp"
#include "spmv.hpp"
#include "distance.hpp"
#include "dot_product.hpp"
#include "axpb.hpp"

extern "C" {
	void pagerank_main(input_block *ptr, input_block *idx, input_block *val,
			index_type_fpga *N, index_type_fpga *E, input_block *result, input_block *pr_vec,
			input_block *dangling_bitmap, input_block *tmp_pr,
			fixed_float *max_err, fixed_float *alpha, index_type_fpga *max_iter);
}

