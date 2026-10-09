int __nvvm_read_ptx_sreg_tid_x(void);
int __nvvm_read_ptx_sreg_ntid_x(void);
int __nvvm_read_ptx_sreg_ctaid_x(void);
void __syncthreads(void);

// Address space 3 is PTX shared memory (.shared)
__attribute__((address_space(3))) float s_p[8 * 32];
__attribute__((address_space(3))) float s_ke[576];

__attribute__((nvptx_kernel))
void matvec_kernel(
    unsigned int num_elements,
    const float* ke_template,
    const unsigned int* elem_dofs,
    const float* elem_weights,
    const float* p_vector,
    float* y_elem
) {
    unsigned int tid = (unsigned int)__nvvm_read_ptx_sreg_tid_x();
    unsigned int warp_in_block = tid >> 5; // tid / 32
    unsigned int lane_id = tid & 31;       // tid % 32
    unsigned int global_warp_id = ((unsigned int)__nvvm_read_ptx_sreg_ctaid_x() * ((unsigned int)__nvvm_read_ptx_sreg_ntid_x() >> 5)) + warp_in_block;

    // Cache the 24x24 stiffness template (576 floats) into shared memory once per block
    if (tid < 576) {
        s_ke[tid] = ke_template[tid];
    }
    __syncthreads();

    if (global_warp_id >= num_elements) return;

    float w = elem_weights[global_warp_id];
    unsigned int offset = global_warp_id * 24;

    // 100% Coalesced Load: Threads 0..23 in the warp read consecutive 4-byte integers
    if (lane_id < 24) {
        unsigned int dof = elem_dofs[offset + lane_id];
        s_p[warp_in_block * 32 + lane_id] = p_vector[dof];
    }

    // Compute row lane_id in parallel across the warp (24 FMA cycles instead of 576!)
    if (lane_id < 24) {
        float sum = 0.0f;
        unsigned int row_offset = lane_id * 24;
        unsigned int p_base = warp_in_block * 32;
        #pragma unroll
        for (int j = 0; j < 24; ++j) {
            sum += s_ke[row_offset + j] * s_p[p_base + j];
        }
        // 100% Coalesced Write: Threads 0..23 write consecutive 4-byte floats
        y_elem[offset + lane_id] = w * sum;
    }
}
