#include "flash_fwd_arbitrary_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM256
template void run_mha_fwd_arbitrary_<90, cutlass::bfloat16_t, 256, 256, 3>(Flash_fwd_params &params, cudaStream_t stream);
#endif
