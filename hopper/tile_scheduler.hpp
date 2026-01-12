/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

// Host side kernel arguments
struct TileSchedulerArguments {
    // num_head is num_head_q if not PackGQA, else num_head_k
    int const num_blocks, num_head, num_batch, num_splits;
    int const qhead_per_khead;
    int const seqlen;  // Only used if Varlen and cu_seqlens == nullptr and seqused == nullptr
    int const seqlen_k, headdim, headdim_v, element_size;  // Used to calculate L2 swizzling
    int* const tile_count_semaphore = nullptr;
    int const* const cu_seqlens = nullptr;
    int const* const seqused = nullptr;
    // int const* const num_m_blocks_ptr = nullptr;
    int const* const num_splits_dynamic_ptr = nullptr;
    // CP (Context Parallelism) parameters
    int const cp_world_size = 1;
    int const cp_rank = 0;
    // SM splitting for prefill/decode
    float const prefill_sm_percentage = 0.0f;  // Percentage of SMs dedicated to prefill (0.0-1.0)
    int const num_prefill_batches = 0;  // Number of prefill batches (batches are ordered: prefill first, then decode)
};

///////////////////////////////////////////////////////////////////////////////

template<bool Varlen=false, bool Split=false, bool PackGQA=false, int kBlock=128>
class SingleTileScheduler {

public:

    using SharedStorage = int;

    // Device side kernel params
    struct Params {
        int const num_blocks, num_head, num_batch, num_splits;
        int const qhead_per_khead;
        int const seqlen;
        cutlass::FastDivmod nsplits_divmod;
        int const* const cu_seqlens;
        int const* const seqused;
        int const* const num_splits_dynamic_ptr = nullptr;
        int const cp_world_size = 1;
        int const cp_rank = 0;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args, int num_sm = 0) {
        (void)num_sm;  // Unused for this scheduler
        assert(!Split || !Varlen || args.num_splits_dynamic_ptr != nullptr);
        assert(!Split || !Varlen || args.num_splits < (1 << 16)); // We use the top 16 bits to store num_splits
        return {args.num_blocks, args.num_head, args.num_batch, !Split ? 1 : args.num_splits,
                args.qhead_per_khead, args.seqlen,
                cutlass::FastDivmod(!Split ? 1 : args.num_splits),
                !Varlen ? nullptr : args.cu_seqlens, !Varlen ? nullptr : args.seqused,
                args.num_splits_dynamic_ptr,
                args.cp_world_size, args.cp_rank};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.num_blocks), uint32_t((!Split ? 1 : params.num_splits) * params.num_head), uint32_t(params.num_batch)};
    }

    struct WorkTileInfo {
        int block_idx = 0;
        int bidh = 0;
        int bidb = 0;
        int split_idx = 0;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb >= 0;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block_idx, bidh, bidb, !Split ? 0 : split_idx};
        }

    };

    CUTLASS_DEVICE
    SingleTileScheduler(SharedStorage* const smem_scheduler) { }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        WorkTileInfo work_info {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), 0};
        if constexpr (Split) {
            int split_idx;
            work_info.bidh = params.nsplits_divmod.divmod(split_idx, work_info.bidh);
            work_info.split_idx = split_idx;
        }
        bool is_valid_tile = true;
        if constexpr (Varlen) {
            int seqlen = params.seqused
                ? params.seqused[work_info.bidb]
                : (params.cu_seqlens ? params.cu_seqlens[work_info.bidb + 1] - params.cu_seqlens[work_info.bidb] : params.seqlen);
            if constexpr (PackGQA) { seqlen *= params.qhead_per_khead; }
            is_valid_tile = work_info.block_idx * kBlock < seqlen;
        }
        if constexpr (Varlen && Split) {
            int num_splits_dynamic = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[work_info.bidb] : params.num_splits;
            is_valid_tile &= work_info.split_idx < num_splits_dynamic;
            // Use the top 16 bits to store num_splits
            work_info.split_idx |= (num_splits_dynamic << 16);
        }
        work_info.bidb = is_valid_tile ? work_info.bidb : -1;
        return work_info;
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {0, 0, -1, 0};
    }

};

///////////////////////////////////////////////////////////////////////////////

template<bool Split=false>
class StaticPersistentTileScheduler {

public:

    using SharedStorage = int;

    // Device side kernel params
    struct Params {
        int total_blocks;
        cutlass::FastDivmod m_block_divmod, head_divmod;
        cutlass::FastDivmod nsplits_divmod;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args, int num_sm = 0) {
        (void)num_sm;  // Unused for this scheduler
        return {args.num_blocks * args.num_head * args.num_batch * (!Split ? 1 : args.num_splits),
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head * (!Split ? 1 : args.num_splits)),
                cutlass::FastDivmod(!Split ? 1 : args.num_splits)};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
            int split_idx = 0;
            if constexpr (Split) {
                bidh = params.nsplits_divmod.divmod(split_idx, bidh);
            }
            return {block, bidh, bidb, split_idx};
        }

    };

    CUTLASS_DEVICE
    StaticPersistentTileScheduler(SharedStorage* const smem_scheduler) {};

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

};

template<int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp,
        bool Split=false, bool PackGQA=false, bool WarpSpecialized=true>
class DynamicPersistentTileScheduler {

    // This scheduler targets the causal (or local) case where each tile takes different
    // amount of time. We use longest-processing-time-first scheduling:
    // the longest remaining tile is assigned to the first SM that's free.
    // SM indicates they are free by incrementing a semaphore.
    // However, we have to make sure K & V still fit into L2 cache, so we perform scheduling
    // on "sections" of the head & batch dimension, each section consisting of e.g. 8 heads.
    // This is the L2 swizzling part. The size of each section is precomputed based on the
    // size of K & V and the L2 cache size.

    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

public:
    using SharedStorage = int;

protected:
    SharedStorage* const tile_count_smem;

public:

    // Device side kernel params
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod, head_divmod;
        cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
        cutlass::FastDivmod const l2_minor_residual_divmod;
        int const num_hb_quotient;
        int* const tile_count_semaphore;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args, int num_sm = 0) {
        (void)num_sm;  // Unused for this scheduler
        int const size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size * 2;
        int const size_l2 = 32 * 1024 * 1024;  // 32 MB for K & V
        // Swizzle is the size of each "section". Round swizzle to a power of 2
        // If not PackGQA already, the size of each section can increase by qhead_per_khead
        // Need to be careful about the case where only one head will fit
        int const swizzle = (size_l2 < size_one_kv_head ? 1 : (1 << cutlass::find_log2(size_l2 / size_one_kv_head))) * (PackGQA ? 1 : args.qhead_per_khead);
        // If we're in the last section (called residual), we don't want to divide by
        // swizzle. Instead we want to divide by the remainder.
        int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
        int const num_split_blocks = args.num_blocks * (!Split ? 1 : args.num_splits);
        // printf("num_split_blocks = %d, num_head = %d, num_batch = %d, swizzle = %d, PackGQA = %d, qhead_per_khead = %d, num_hb_remainder = %d\n", num_split_blocks, args.num_head, args.num_batch, swizzle, int(PackGQA), args.qhead_per_khead, num_hb_remainder);
        assert(args.tile_count_semaphore != nullptr);
        return {num_split_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * num_split_blocks),
                // don't divide by 0
                cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
                (args.num_head * args.num_batch) / swizzle,
                args.tile_count_semaphore};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            int l2_mod, bidhb, bidhb_residual;
            bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
            // If we're in the last section (called residual), we don't want to divide by
            // swizzle. Instead we want to divide by the remainder.
            if (bidhb < params.num_hb_quotient) {
                block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
            } else {
                block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
            }
            bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
            int split_idx = 0;
            if constexpr (Split) {
                split_idx = params.m_block_divmod.divmod(block, block);
            }
            // Longest-processing-time-first
            block = params.m_block_divmod.divisor - 1 - block;
            return {block, bidh, bidb, split_idx};
        }

    };

    CUTLASS_DEVICE
    DynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : tile_count_smem(smem_scheduler) {};

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        if (WarpSpecialized || cutlass::canonical_warp_idx_sync() > 0) {
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
        }
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // thread 0 already has the right tile_idx, just need to broadcast to the rest of warp 0
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            if (threadIdx.x % NumProducerThreads == 0) {
                *tile_count_smem = current_work.tile_idx;
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return {new_tile_idx};
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            int tile_idx = *tile_count_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            return {tile_idx};
        }
    }

};

template<int kBlock, int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp, bool Split=false, bool PackGQA=false, bool WarpSpecialized=true>
class VarlenDynamicPersistentTileScheduler {

    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

public:
    using SharedStorage = int4;

protected:
    SharedStorage* const work_info_smem;

public:

    // Device side kernel params
    struct Params {
        int num_head, num_batch;
        int const qhead_per_khead;
        int const seqlen;
        cutlass::FastDivmod head_divmod;
        cutlass::FastDivmod nsplits_divmod;
        // Original code (commented out):
        // int* const tile_count_semaphore;
        int* const tile_count_semaphore;  // Kept for backward compatibility, but prefill/decode SMs use their own semaphores
        int* const prefill_tile_count_semaphore;  // Separate semaphore for prefill tiles
        int* const decode_tile_count_semaphore;   // Separate semaphore for decode tiles
        int const* const cu_seqlens;
        int const* const seqused;
        // int* const num_m_blocks_ptr;
        int const* const num_splits_dynamic_ptr;
        float const prefill_sm_percentage;
        int const num_sm;
        int const num_prefill_batches;  // Number of prefill batches (batches are ordered: prefill first, then decode)
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args, int num_sm) {
        // If Split, for the purpose of scheduling, we pretend that instead there are
        // (args.num_splits * args.num_head) number of heads.
        assert(args.tile_count_semaphore != nullptr);
        assert(args.num_head < (1 << 16));  // We use the top 16 bits to store num_splits & split_idx
        assert(!Split || args.num_splits < (1 << 8)); // We use the top 8 bits to store num_splits
        assert(args.prefill_sm_percentage >= 0.0f && args.prefill_sm_percentage <= 1.0f);
        // Original code (commented out):
        // return {args.num_head, args.num_batch,
        //         args.qhead_per_khead, args.seqlen,
        //         cutlass::FastDivmod(args.num_head),
        //         cutlass::FastDivmod(!Split ? 1 : args.num_splits),
        //         args.tile_count_semaphore, args.cu_seqlens, args.seqused,
        //         // args.num_m_blocks_ptr,
        //         args.num_splits_dynamic_ptr,
        //         args.prefill_sm_percentage, num_sm};
        
        // Ni: For partitioned scheduling, we use separate semaphores for prefill and decode
        // They should be allocated as [prefill_semaphore, decode_semaphore] in memory
        int* prefill_semaphore = args.tile_count_semaphore;
        int* decode_semaphore = args.tile_count_semaphore + 1;
        return {args.num_head, args.num_batch,
                args.qhead_per_khead, args.seqlen,
                cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(!Split ? 1 : args.num_splits),
                args.tile_count_semaphore,  // Keep for backward compatibility
                prefill_semaphore,
                decode_semaphore,
                args.cu_seqlens, args.seqused,
                // args.num_m_blocks_ptr,
                args.num_splits_dynamic_ptr,
                args.prefill_sm_percentage, num_sm,
                args.num_prefill_batches};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            // if (blockIdx.x >= 0 && (threadIdx.x == 128 || threadIdx.x == 0)) { printf("blockIdx.x = %d, threadIdx.x = %d, checking valid, bidb = %d, params.num_batch = %d\n", blockIdx.x, threadIdx.x, bidb, params.num_batch); }
            return bidb < params.num_batch;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            if constexpr (!Split) {
                return {block, bidh, bidb, 0 /*split_idx*/};
            } else {
                // the top 8 bits of bidh store num_splits and the next 8 bits store split_idx
                // reinterpret_cast to uint32_t to make sure we're not doing sign extension when we shift
                uint32_t bidh_packed = reinterpret_cast<uint32_t const&>(bidh);
                uint32_t bidh_actual_u = bidh_packed & 0x0000FFFF;
                int bidh_actual = reinterpret_cast<int&>(bidh_actual_u);
                // Use the top 16 bits of split_idx to store num_splits and the next 16 bits to store split_idx
                uint32_t split_idx_u = ((bidh_packed & 0x00FF0000) >> 16) + ((bidh_packed & 0xFF000000) >> 8);
                int split_idx = reinterpret_cast<int&>(split_idx_u);
                // int bidh_actual = params.nsplits_divmod.divmod(split_idx, bidh);
                // if (threadIdx.x == 128) {
                //     printf("blockIdx.x = %d, bidb = %d, bidh = %d, bidh_actual = %d, split_idx = %d\n", blockIdx.x, bidb, bidh, bidh_actual, split_idx);
                // }
                return {block, bidh_actual, bidb, split_idx};
            }
        }
    };

    CUTLASS_DEVICE
    VarlenDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : work_info_smem(smem_scheduler) {};

    // Ni: Helper function to determine if a batch is prefill (seqlen_q > 1) or decode (seqlen_q == 1)
    // Note: When PackGQA is enabled, seqlen is multiplied by qhead_per_khead, so we check
    // the original seqlen_q before the multiplication to correctly identify decode vs prefill.
    CUTLASS_DEVICE
    bool is_prefill_batch(Params const& params, int bidb) const {
        // Ni: If prefill_sm_percentage is 0, use original unified scheduling (no partitioning)
        // In unified scheduling, batches are not partitioned, but we still check sequence length
        // to determine if it's a prefill or decode batch
        int seqlen_q;  // Original sequence length (before PackGQA multiplication)
        if (params.seqused) {
            seqlen_q = bidb < params.num_batch ? params.seqused[bidb] : 0;
        } else if (params.cu_seqlens) {
            int cur_cu_seqlen = bidb <= params.num_batch ? params.cu_seqlens[bidb] : 0;
            int next_cu_seqlen = bidb < params.num_batch ? params.cu_seqlens[bidb + 1] : cur_cu_seqlen;
            seqlen_q = next_cu_seqlen - cur_cu_seqlen;
        } else {
            seqlen_q = params.seqlen;
        }
        // Decode: seqlen_q == 1, Prefill: seqlen_q > 1
        // We check the original seqlen_q, not the PackGQA-multiplied version
        return seqlen_q > 1;
    }

    // Ni: Helper function to check if current SM belongs to prefill group
    CUTLASS_DEVICE
    bool is_prefill_sm(Params const& params) const {
        // Ni: If prefill_sm_percentage is 0, use original unified scheduling (no partitioning)
        // Return false (treat as decode) since we're using unified scheduling
        if (params.prefill_sm_percentage == 0.0f) {
            return false;
        }
        int sm_id = int(blockIdx.x);
        int prefill_sm_count = int(params.num_sm * params.prefill_sm_percentage + 0.5f);  // Round to nearest
        return sm_id < prefill_sm_count;
    }

    // Ni: Get the starting bidb for the current SM's batch group
    CUTLASS_DEVICE
    int get_batch_start(Params const& params) const {
        // If prefill_sm_percentage is 0, use original unified scheduling (no partitioning)
        if (params.prefill_sm_percentage == 0.0f) {
            return 0;
        }
        bool is_prefill_sm_group = is_prefill_sm(params);
        if (is_prefill_sm_group) {
            return 0;  // Prefill batches start at 0
        } else {
            // Decode batches start after all prefill batches
            return params.num_prefill_batches;
        }
    }

    // Ni: Get the ending bidb (exclusive) for the current SM's batch group
    CUTLASS_DEVICE
    int get_batch_end(Params const& params) const {
        // If prefill_sm_percentage is 0, use original unified scheduling (no partitioning)
        if (params.prefill_sm_percentage == 0.0f) {
            return params.num_batch;
        }
        bool is_prefill_sm_group = is_prefill_sm(params);
        if (is_prefill_sm_group) {
            // Prefill batches end where decode batches begin
            return params.num_prefill_batches;
        } else {
            // Decode batches end at num_batch
            return params.num_batch;
        }
    }

    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile(Params const& params, int next_tile_idx, WorkTileInfo const& current_work) const {
        int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
        auto get_num_m_blocks = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            int seqlen = params.seqlen * (!PackGQA ? 1 : params.qhead_per_khead);
            if (seqlen > kBlock) {
                if (params.seqused) {
                    seqlen = batch_idx < params.num_batch ? params.seqused[batch_idx] : 0;
                } else if (params.cu_seqlens) {
                    int cur_cu_seqlen = batch_idx <= params.num_batch ? params.cu_seqlens[batch_idx] : 0;
                    int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
                    seqlen = next_cu_seqlen - cur_cu_seqlen;
                } else {
                    seqlen = params.seqlen;
                }
                if constexpr (PackGQA) { seqlen *= params.qhead_per_khead; }
            }
            return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                ? cute::ceil_div(seqlen, kBlock) : 0;
                // ? params.num_m_blocks_ptr[batch_idx] : 0;
        };

        auto get_num_splits = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                ? (!Split ? 1 : (params.num_splits_dynamic_ptr
                                ? params.num_splits_dynamic_ptr[batch_idx]
                                : params.nsplits_divmod.divisor))
                : 0;
        };

        int num_m_blocks = get_num_m_blocks(current_work.bidb);  // Different for each lane
        int num_splits = get_num_splits(current_work.bidb);
        int num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
        // Cumulative number of blocks for the next 31 batches
        int num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
        // Total number of blocks for the next 31 batches
        int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
        // Only the lower 16 bits are the actual bidh
        int current_bidh = !Split ? current_work.bidh : (current_work.bidh & 0x0000FFFF);
        // Start tile of the current batch + an estimate of the total tiles within the group
        int group_end_tile = current_work.tile_idx - current_work.block - current_bidh * __shfl_sync(0xffffffff, num_split_m_blocks, 0 /*lane*/) + m_blocks_in_group * params.num_head;  // Same for all lanes
        if constexpr (Split) {
            int current_split_idx = (current_work.bidh & 0x00FF0000) >> 16;
            group_end_tile -= current_split_idx * __shfl_sync(0xffffffff, num_m_blocks, 0 /*lane*/);
        }
        bool is_prefill_sm_group = is_prefill_sm(params);
        int batch_start = get_batch_start(params);  // Start of this SM group's batch range
        int batch_end = get_batch_end(params);     // End (exclusive) of this SM group's batch range
        
        // Ni: Assert that current_work.bidb is within this SM group's batch range
        // When prefill_sm_percentage == 0, batch_start=0 and batch_end=num_batch, so this always passes for valid bidb
        // get_initial_work sets the correct initial bidb, and subsequent calls should maintain it
        assert(current_work.bidb >= batch_start && current_work.bidb < batch_end);
        int bidb = current_work.bidb;
        
        // if (blockIdx.x <= 9 && threadIdx.x == 0) {
        //     printf("Before while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, cur tile_idx = %d, cur block = %d, cur bidh = %d, num_split_m_blocks = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, current_work.bidb, num_m_blocks, next_tile_idx, current_work.tile_idx, current_work.block, current_bidh, num_split_m_blocks, group_end_tile, m_blocks_in_group);
        // }
        // if (threadIdx.x == 0 && blockIdx.x == 0) { printf("tile_idx = %d, group_end_tile = %d, num_m_blocks_cumulative = %d, m_blocks_in_group = %d\n", current_work.tile_idx, group_end_tile, num_m_blocks_cumulative, m_blocks_in_group); }
        while (group_end_tile <= next_tile_idx) {
            bidb += cutlass::NumThreadsPerWarp - 1;
            // Ni: If prefill_sm_percentage != 0, check if the next batch group would exceed our SM group's batch range
            // When prefill_sm_percentage == 0.0f, we use unified scheduling and rely on get_num_m_blocks returning 0
            // when bidb >= num_batch, which makes m_blocks_in_group = 0 and stops group_end_tile from advancing
            if (params.prefill_sm_percentage != 0.0f && bidb >= batch_end) {
                // We've searched through all batches in our SM group's range and didn't find the tile
                // This means next_tile_idx is beyond the tiles available for this SM group
                // if (blockIdx.x <= 9 && threadIdx.x == 0) {
                //     printf("Returning early, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
                // }
                return {next_tile_idx, 0, 0, params.num_batch};
            }
            
            // Since batches are already partitioned (prefill first, then decode),
            // we only need to check if batches are within our range
            // All batches in [batch_start, batch_end) should match our SM group
            // Note: get_num_m_blocks and get_num_splits already check lane < 31 internally
            bool batch_matches = false;
            int num_m_blocks_for_lane = 0;
            int num_splits_for_lane = 0;
            int batch_idx = lane + bidb;
            // Check if batch is within our SM group's range
            if (batch_idx >= batch_start && batch_idx < batch_end) {
                batch_matches = true;
                // Compute blocks for this specific batch
                num_m_blocks_for_lane = get_num_m_blocks(bidb);  // get_num_m_blocks uses lane + bidb internally
                num_splits_for_lane = get_num_splits(bidb);      // get_num_splits uses lane + bidb internally
            }
            // Each lane now has its filtered values
            num_m_blocks = num_m_blocks_for_lane;
            if constexpr (Split) {
                num_splits = num_splits_for_lane;
            }
            // Only count tiles from matching batches (zero for non-matching)
            num_split_m_blocks = batch_matches ? (!Split ? num_m_blocks : num_m_blocks * num_splits) : 0;
            num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
            m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
            
            // Only advance group_end_tile if we have matching batches
            if (m_blocks_in_group > 0) {
                group_end_tile += m_blocks_in_group * params.num_head;
            }
            // If no matching batches (m_blocks_in_group == 0), the loop will continue and try the next group
            // if (blockIdx.x <= 9 && threadIdx.x == 0) {
            //     printf("Bottom of while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
            // }
        }
        int group_start_tile = group_end_tile - m_blocks_in_group * params.num_head;
        // The next problem to process is the first one that does not have ending tile position
        // that is greater than or equal to tile index.
        // Only consider batches within our SM group's range (already filtered in the loop above)
        bool tile_in_range = false;
        int batch_idx = lane + bidb;
        // Check if batch is within our SM group's range
        if (batch_idx >= batch_start && batch_idx < batch_end) {
            tile_in_range = (group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx);
        }
        int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, tile_in_range));
        // if (threadIdx.x == 31 || threadIdx.x == 0) { printf("blockIdx.x = %d, tidx %d, group_start_tile = %d, num_m_blocks_cumulative = %d, num_head = %d, next_tile_idx = %d, ballot = %x, batch_idx_in_group = %d\n", blockIdx.x, threadIdx.x, group_start_tile, num_m_blocks_cumulative, params.num_head, next_tile_idx, tmp, batch_idx_in_group); }
        bidb += batch_idx_in_group;
        // Ni: After adding batch_idx_in_group, check if bidb exceeds batch_end
        // This can happen if bidb was near the boundary and batch_idx_in_group pushed it over
        if (bidb >= batch_end) {
            return {next_tile_idx, 0, 0, params.num_batch};
        }
        
        num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
        if constexpr (Split) { num_splits = __shfl_sync(0xffffffff, num_splits, batch_idx_in_group); }
        int mh_block = next_tile_idx - group_start_tile - (batch_idx_in_group == 0 ? 0 : __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1)) * params.num_head;
        int bidh = mh_block / num_m_blocks;
        int block = mh_block - bidh * num_m_blocks;
        if constexpr (Split) {
            int bidh_actual = bidh / num_splits;
            int split_idx = bidh - bidh_actual * num_splits;
            // TODO: idk why this gives wrong answer nondeterministically
            // int bidh_actual, split_idx;
            // split_idx = params.head_divmod.divmod(bidh_actual, bidh);
            // Use the top 8 bits to store num_splits and the next 8 bits to store split_idx
            // reinterpret_cast to uint32_t to make sure we're not doing sign extension when we shift
            uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh_actual) + (reinterpret_cast<uint32_t&>(split_idx) << 16) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
            // if (threadIdx.x == 0) {
            //     printf("blockIdx.x = %d, group_start_tiled = %d, bidb = %d, batch_idx_in_group = %d, mh_block = %d, num_m_blocks = %d, bidh = %d, bidh_actual = %d, split_idx = %d, num_splits = %d, bidh_packed = %d\n", blockIdx.x, group_start_tile, bidb, batch_idx_in_group, mh_block, num_m_blocks, bidh, bidh_actual, split_idx, num_splits, bidh_packed);
            // }
            bidh = reinterpret_cast<int&>(bidh_packed);
        }
        // if (blockIdx.x <= 9 && threadIdx.x == 0) {
        //     printf("Before returning, blockIdx.x = %d, threadIdx.x = %d, group_start_tile = %d, batch_idx_in_group = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d, mh_block = %d, bidh = %d, block = %d\n", blockIdx.x, threadIdx.x, group_start_tile, batch_idx_in_group, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group, mh_block, bidh, block);
        // }
        // Ni Debug: Print tile assignment result (batch_start and batch_end are defined earlier in the function)
        if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
            bool is_valid_tile = bidb < params.num_batch;
            printf("[TILE] SM=%d (group=%s), tile_idx=%d -> (block=%d, bidh=%d, bidb=%d), valid=%d, batch_range=[%d,%d)\n",
                int(blockIdx.x),
                is_prefill_sm_group ? "prefill" : "decode",
                next_tile_idx, block, bidh, bidb,
                is_valid_tile ? 1 : 0,
                batch_start, batch_end);
        }
        return {next_tile_idx, block, bidh, bidb};
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            // Ni: If prefill_sm_percentage is 0, use original unified scheduling (no partitioning)
            if (params.prefill_sm_percentage == 0.0f) {
                WorkTileInfo work_info = tile_idx_to_work_tile(params, int(blockIdx.x), {0, 0, 0, 0});
                if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                    *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
                    // Ni Debug: Print initial tile assignment (unified scheduling)
                    printf("[INIT] SM=%d (group=unified), tile_idx=%d -> (block=%d, bidh=%d, bidb=%d), valid=%d\n",
                        int(blockIdx.x),
                        work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb,
                        work_info.is_valid(params) ? 1 : 0);
                }
                flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
                return work_info;
            }
            
            // Ni: With separate semaphores, each SM group has its own tile index space starting from 0
            // Prefill SMs: use sm_id directly (0 to num_prefill_sm-1)
            // Decode SMs: use sm_id relative to decode group (0 to num_decode_sm-1)
            // This matches the semaphore pattern: atomicAdd(semaphore, 1) + num_sm_for_group
            // where semaphore starts at 0, so first value is num_sm_for_group
            // But for initial work, we want tile_idx relative to the group (0, 1, 2, ...)
            bool is_prefill_sm_group = is_prefill_sm(params);
            int sm_id = int(blockIdx.x);
            int num_prefill_sm = int(params.num_sm * params.prefill_sm_percentage + 0.5f);
            int initial_tile_idx;
            int initial_bidb;
            if (is_prefill_sm_group) {
                // Prefill SMs: blockIdx.x = 0 to num_prefill_sm-1, use directly
                initial_tile_idx = sm_id;
                initial_bidb = 0;  // Prefill batches start at 0
            } else {
                // Decode SMs: blockIdx.x = num_prefill_sm to num_sm-1, convert to relative ID
                int sm_id_in_decode_group = sm_id - num_prefill_sm;
                initial_tile_idx = sm_id_in_decode_group;
                initial_bidb = params.num_prefill_batches;  // Decode batches start after prefill batches
            }
            WorkTileInfo work_info = tile_idx_to_work_tile(params, initial_tile_idx, {0, 0, 0, initial_bidb});
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
                // Ni Debug: Print initial tile assignment
                printf("[INIT] SM=%d (group=%s, sm_in_group=%d), tile_idx=%d -> (block=%d, bidh=%d, bidb=%d), valid=%d\n",
                    int(blockIdx.x),
                    is_prefill_sm_group ? "prefill" : "decode",
                    is_prefill_sm_group ? sm_id : (sm_id - num_prefill_sm),
                    work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb,
                    work_info.is_valid(params) ? 1 : 0);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0});
        }
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        // Don't arrive at the TileCountSmemEmpty barrier here, because get_initial_work will do that
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            // Ni: If prefill_sm_percentage is 0, use original unified scheduling (no partitioning)
            if (params.prefill_sm_percentage == 0.0f) {
                current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
                return;
            }
            
            // Ni:Use separate semaphores for prefill and decode SMs
            bool is_prefill_sm_group = is_prefill_sm(params);
            int* semaphore = is_prefill_sm_group ? params.prefill_tile_count_semaphore : params.decode_tile_count_semaphore;
            int num_sm_for_group = is_prefill_sm_group 
                ? int(params.num_sm * params.prefill_sm_percentage + 0.5f)
                : (params.num_sm - int(params.num_sm * params.prefill_sm_percentage + 0.5f));
            int old_tile_idx = current_work.tile_idx;
            current_work.tile_idx = atomicAdd(semaphore, 1) + num_sm_for_group;
            // Debug: Print when fetching next tile
            printf("[FETCH] SM=%d (group=%s), old_tile_idx=%d -> new_tile_idx=%d (semaphore_val=%d)\n",
                int(blockIdx.x),
                is_prefill_sm_group ? "prefill" : "decode",
                old_tile_idx,
                current_work.tile_idx,
                current_work.tile_idx - num_sm_for_group);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // thread 0 has the next tile_idx, just need to broadcast to the rest of warp 0
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            WorkTileInfo work_info = {__shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/), current_work.block, current_work.bidh, current_work.bidb};
            work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
                // Debug: Print when getting next work tile
                bool is_prefill_sm_group = is_prefill_sm(params);
                printf("[NEXT] SM=%d (group=%s), tile_idx=%d -> (block=%d, bidh=%d, bidb=%d), valid=%d\n",
                    int(blockIdx.x),
                    is_prefill_sm_group ? "prefill" : "decode",
                    work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb,
                    work_info.is_valid(params) ? 1 : 0);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            int4 work_info = *work_info_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w};
        }
    }

};

} // flash
