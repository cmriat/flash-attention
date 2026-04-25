// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Explicit arbitrary-mask backward instantiations for Hopper kNFunc=3.
// Keep this aligned with the currently supported arbitrary forward matrix:
// bf16 only, no softcap, head_dim 64 or 256.

#include "flash_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_BACKWARD
#ifndef FLASHATTENTION_DISABLE_ARBITRARY

#ifndef FLASHATTENTION_DISABLE_HDIM64
template void run_mha_bwd_<90, cutlass::bfloat16_t, 64, false, 3>(Flash_bwd_params &params, cudaStream_t stream);
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
template void run_mha_bwd_<90, cutlass::bfloat16_t, 256, false, 3>(Flash_bwd_params &params, cudaStream_t stream);
#endif

#endif
#endif
