#include "flash_fwd_arbitrary_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_ARBITRARY

template void run_mha_fwd_arbitrary_<90, cutlass::bfloat16_t, 64, 64, FLASHATTENTION_MAX_NUM_FUNC>(Flash_fwd_params &params, cudaStream_t stream);
template void run_mha_fwd_arbitrary_<90, cutlass::bfloat16_t, 256, 256, FLASHATTENTION_MAX_NUM_FUNC>(Flash_fwd_params &params, cudaStream_t stream);

#endif
