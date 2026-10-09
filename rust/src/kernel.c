int __nvvm_read_ptx_sreg_tid_x(void);
int __nvvm_read_ptx_sreg_ntid_x(void);
int __nvvm_read_ptx_sreg_ctaid_x(void);

__attribute__((nvptx_kernel))
void matvec_kernel(
    unsigned int num_elements,
    const float* ke_template,
    const unsigned int* elem_dofs,
    const float* elem_weights,
    const float* p_vector,
    float* y_elem
) {
    unsigned int elem_idx = (unsigned int)__nvvm_read_ptx_sreg_ctaid_x() * (unsigned int)__nvvm_read_ptx_sreg_ntid_x() + (unsigned int)__nvvm_read_ptx_sreg_tid_x();
    if (elem_idx >= num_elements) return;

    float w = elem_weights[elem_idx];
    unsigned int offset = elem_idx * 24;

    float p_local[24];
    for (int i = 0; i < 24; ++i) {
        unsigned int dof = elem_dofs[offset + i];
        p_local[i] = p_vector[dof];
    }

    for (int i = 0; i < 24; ++i) {
        float sum = 0.0f;
        int row_offset = i * 24;
        for (int j = 0; j < 24; ++j) {
            sum += ke_template[row_offset + j] * p_local[j];
        }
        y_elem[offset + i] = w * sum;
    }
}
