/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/kernel_launch.h"

#include "cuda_check.h"
#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel_sm90_arbitrary.h"
#include "mainloop_fwd_sm90_tma_gmma_ws_arbitrary.hpp"
#include "epilogue_fwd.hpp"

using namespace cute;

template <int Arch, int kHeadDim, int kHeadDimV, int kNFunc, typename Element, typename ElementOut, bool Varlen>
void run_flash_fwd_arbitrary(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(Arch == 90, "Gemma arbitrary forward is Hopper-only");
    using ArchTag = cutlass::arch::Sm90;

    static constexpr auto kBlockMN_RS_IntraWGOverlap =
        tile_size_fwd_sm90(kHeadDim, kHeadDimV, false, false, true, sizeof(Element), false, false, false);
    static constexpr int kBlockM = std::get<0>(kBlockMN_RS_IntraWGOverlap);
    static constexpr int kBlockN = std::get<1>(kBlockMN_RS_IntraWGOverlap);
    static constexpr bool MmaPV_is_RS = std::get<2>(kBlockMN_RS_IntraWGOverlap);
    static constexpr bool IntraWGOverlap = std::get<3>(kBlockMN_RS_IntraWGOverlap);
    static constexpr int kStages = 2;

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;
    using ClusterShape = cute::Shape<_1, _1, _1>;
    using CollectiveMainloop = flash::CollectiveMainloopFwdSm90Arbitrary<
        kStages,
        ClusterShape,
        TileShape_MNK,
        kHeadDimV,
        Element,
        float,
        ArchTag,
        false,
        false,
        false,
        Varlen,
        false,
        false,
        false,
        MmaPV_is_RS,
        IntraWGOverlap,
        false,
        false,
        false,
        true,
        kNFunc>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
        TileShape_MNK_PV,
        ClusterShape,
        ElementOut,
        ArchTag,
        CollectiveMainloop::NumMmaThreads,
        Varlen,
        false,
        false,
        false>;
    using Scheduler = std::conditional_t<
        Varlen,
        flash::VarlenDynamicPersistentTileScheduler<
            kBlockM,
            kBlockN,
            CollectiveMainloop::NumMmaThreads,
            CollectiveMainloop::NumProducerThreads,
            false,
            false,
            true,
            false,
            true,
            true>,
        flash::DynamicPersistentTileScheduler<
            CollectiveMainloop::NumMmaThreads,
            CollectiveMainloop::NumProducerThreads,
            false,
            false,
            true>>;
    using AttnKernel =
        flash::enable_sm90_or_later<flash::FlashAttnFwdSm90Arbitrary<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    bool const is_varlen_q = params.cu_seqlens_q;
    bool const is_varlen_k = params.cu_seqlens_k;
    int seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;
    int batch_q = !is_varlen_q ? params.b : 1;
    int batch_k = !is_varlen_k ? (params.kv_batch_idx ? params.b_k : params.b) : 1;

    typename CollectiveMainloop::StrideV v_strides =
        make_stride(params.v_row_stride, _1{}, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0);
    typename CollectiveMainloop::Arguments mainloop_args{
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},
        {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen_q ? params.q_batch_stride : 0},
        static_cast<Element*>(params.k_ptr),
        {!params.page_table ? (!is_varlen_k ? params.seqlen_k : params.total_k) : params.page_size,
         params.d, params.h_k, !params.page_table ? batch_k : params.num_pages},
        {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0},
        static_cast<Element*>(params.v_ptr),
        params.dv,
        v_strides,
        nullptr,
        {0, params.d, params.h_k, 0},
        {0, _1{}, 0, 0},
        nullptr,
        {0, _1{}, 0, 0},
        nullptr,
        {0, _1{}, 0, 0},
        nullptr,
        {0, 0},
        {0, _1{}},
        nullptr,
        {0, _1{}},
        false,
        params.page_table,
        {params.kv_batch_idx ? params.b_k : params.b, !params.page_table ? 0 : params.seqlen_k / params.page_size},
        {params.page_table_batch_stride, _1{}},
        params.scale_softmax,
        nullptr, nullptr, nullptr,
        {0, 0},
        {0, 0},
        {0, 0},
        params.window_size_left, params.window_size_right, params.attention_chunk,
        params.softcap,
        params.num_splits,
        params.kv_batch_idx,
        params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
        params.seqused_q, params.seqused_k,
        params.leftpad_k, params.seqlens_rotary,
        params.mask_func_ptr,
        {params.func_seqlen, params.arbitrary_func_num, params.func_head, params.func_batch},
        {_1{}, params.func_nfunc_stride, params.func_head_stride, params.func_batch_stride}
    };
    typename flash::BlockSparsityArguments block_sparsity_args{
        params.block_sparse_mask_cnt,
        params.block_sparse_mask_offset,
        params.block_sparse_mask_idx,
        params.block_sparse_full_cnt,
        params.block_sparse_full_offset,
        params.block_sparse_full_idx,
        params.block_sparse_num_blocks,
        params.block_sparse_num_heads,
        params.block_sparse_num_batches
    };
    typename CollectiveEpilogue::Arguments epilogue_args{
        static_cast<ElementOut*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q, params.num_splits},
        {params.o_row_stride, _1{}, params.o_head_stride, !is_varlen_q ? params.o_batch_stride : 0, 0},
        static_cast<float*>(params.oaccum_ptr),
        {params.oaccum_row_stride, _1{}, params.oaccum_head_stride, !is_varlen_q ? params.oaccum_batch_stride : 0, params.oaccum_split_stride},
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, 0},
        static_cast<float*>(params.softmax_lseaccum_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, params.h * seqlen_q * batch_q},
        params.h_k,
        params.cu_seqlens_q,
        params.seqused_q
    };

    int num_blocks_m = cutlass::ceil_div(params.seqlen_q, kBlockM);
    num_blocks_m = cutlass::round_up(num_blocks_m, 1);
    typename flash::TileSchedulerArguments scheduler_args{
        num_blocks_m,
        params.h,
        params.b,
        params.num_splits,
        params.h / params.h_k,
        params.seqlen_q,
        params.seqlen_k,
        params.d,
        params.dv,
        sizeof(Element),
        params.tile_count_semaphore,
        params.cu_seqlens_q,
        params.seqused_q,
        params.num_splits_dynamic_ptr,
        params.num_m_blocks_ptr,
        params.varlen_batch_idx_ptr,
        params.num_nheads_in_l2_ptr
    };

    if constexpr (Varlen) {
        if (!params.skip_scheduler_metadata_computation) {
            prepare_varlen_num_blocks(params, stream, false, kBlockM, kBlockN, params.prepare_varlen_pdl);
            CHECK_CUDA_KERNEL_LAUNCH();
        }
    }

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments(
        {mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args, block_sparsity_args});

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    auto kernel = cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    CHECK_CUTLASS(cutlass::kernel_launch<AttnKernel>(
        grid_dims,
        block_dims,
        smem_size,
        stream,
        kernel_params,
        Arch >= 90 && Varlen && !params.skip_scheduler_metadata_computation && params.prepare_varlen_pdl));
}

template <int Arch, typename T, int kHeadDim, int kHeadDimV, int kNFunc>
void run_mha_fwd_arbitrary_(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(Arch == 90, "Gemma arbitrary forward is Hopper-only");
    static_assert(cute::is_same_v<T, cutlass::bfloat16_t>, "Gemma arbitrary forward is bf16-only");
    using T_out = T;
    VARLEN_SWITCH(params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k, Varlen, [&] {
        run_flash_fwd_arbitrary<Arch, kHeadDim, kHeadDimV, kNFunc, T, T_out, Varlen>(params, stream);
    });
}
