#pragma once

void spmv(float *result, float *csc_val, unsigned int *csc_ptr,
          unsigned int *csc_idx, float *vec, const int DIMV);
