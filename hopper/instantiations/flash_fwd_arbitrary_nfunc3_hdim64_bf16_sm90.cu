#include "flash_fwd_arbitrary_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM64
template void run_mha_fwd_arbitrary_<90, cutlass::bfloat16_t, 64, 64, 3>(Flash_fwd_params &params, cudaStream_t stream);
#endif
