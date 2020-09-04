//
// Created by fra on 26/04/19.
//

#pragma once

#include <set>
#include <stdlib.h>

// Square matrix of size DIMV * DIMV, stored as dense array;
template <typename T>
void generate_sparse_matrix(T *matrix, unsigned DIMV, unsigned min_sparse);

template <typename T>
void fill_spm(T *matrix, unsigned DIMV);

template <typename T>
void transpose(T *out, T *in, unsigned DIMV);

template <typename T>
void to_csc(T *csc_val, int *csc_ptr, int *csc_idx, T *src, unsigned DIMV);

template <typename T>
unsigned int count_non_zero(T *m, unsigned DIMV);
