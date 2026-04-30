#include "flash_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_ARBITRARY
#ifndef FLASHATTENTION_DISABLE_BACKWARD

template void run_mha_bwd_<90, cutlass::bfloat16_t, 64, false, FLASHATTENTION_MAX_NUM_FUNC>(Flash_bwd_params &params, cudaStream_t stream);
template void run_mha_bwd_<90, cutlass::bfloat16_t, 256, false, FLASHATTENTION_MAX_NUM_FUNC>(Flash_bwd_params &params, cudaStream_t stream);

#endif
#endif
