void spmv(float *result, float *csc_val, unsigned int *csc_ptr,
          unsigned int *csc_idx, float *vec, const int DIMV) {
    for (int i = 0; i < DIMV; ++i) {
        const int begin = csc_ptr[i];
        const int end = csc_ptr[i + 1];
        float acc = 0.0;

        for (int j = begin; j < end; ++j) {
            acc += csc_val[j] * vec[csc_idx[j]];
        }

        result[i] = acc;
    }
}
