#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <torch/all.h>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

template <int BLOCK_SIZE>
__global__ void compute_problem_sizes_w4a8(
    const int32_t* __restrict__ topk_ids,
    int32_t* problem_sizes1,
    int32_t* problem_sizes2,
    const int topk_length,
    const int n,
    const int k) {
  int expert_id = blockIdx.x;

  int occurrences = 0;
  // Optimized: vectorized memory access using int4 for better memory bandwidth
  // Process vectorized chunks first
  bool aligned = (reinterpret_cast<uintptr_t>(topk_ids) % 16 == 0);

  if (aligned) {
    const int4* vec_ptr = reinterpret_cast<const int4*>(topk_ids);
    int vec_length = topk_length / 4;

    for (int i = threadIdx.x; i < vec_length; i += BLOCK_SIZE) {
      int4 vec_data = vec_ptr[i];
      occurrences +=
          (vec_data.x == expert_id) + (vec_data.y == expert_id) + (vec_data.z == expert_id) + (vec_data.w == expert_id);
    }

    for (int i = vec_length * 4 + threadIdx.x; i < topk_length; i += BLOCK_SIZE) {
      occurrences += (topk_ids[i] == expert_id);
    }
  } else {
    for (int i = threadIdx.x; i < topk_length; i += BLOCK_SIZE) {
      occurrences += (topk_ids[i] == expert_id);
    }
  }

  using BlockReduce = cub::BlockReduce<int, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int final_occurrences = BlockReduce(temp_storage).Sum(occurrences);

  if (threadIdx.x == 0) {
    problem_sizes1[expert_id * 3] = 2 * n;
    problem_sizes1[expert_id * 3 + 1] = final_occurrences;
    problem_sizes1[expert_id * 3 + 2] = k;
    problem_sizes2[expert_id * 3] = k;
    problem_sizes2[expert_id * 3 + 1] = final_occurrences;
    problem_sizes2[expert_id * 3 + 2] = n;
  }
}

template <int BLOCK_SIZE>
__device__ void
cumsum_block_scan(const int32_t* __restrict__ input, int32_t* __restrict__ output, int n, int input_stride) {
  using BlockScan = cub::BlockScan<int32_t, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_scan_storage;
  __shared__ int32_t s_broadcast_val;

  int tid = threadIdx.x;
  int32_t base_prefix_sum = 0;
  const int num_chunks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int chunk = 0; chunk < num_chunks; chunk++) {
    const int base_idx = chunk * BLOCK_SIZE;
    const int index = base_idx + tid;

    const int32_t val = (index < n) ? input[index * input_stride] : 0;
    int32_t local_prefix_sum;
    BlockScan(temp_scan_storage).InclusiveSum(val, local_prefix_sum);
    const int32_t prefix_sum = local_prefix_sum + base_prefix_sum;
    if (index < n) {
      output[index] = prefix_sum;
    }
    if (chunk < num_chunks - 1) {
      if (tid == BLOCK_SIZE - 1) {
        s_broadcast_val = prefix_sum;
      }
      __syncthreads();
      base_prefix_sum = s_broadcast_val;
    }
  }
}

template <int BLOCK_SIZE>
__global__ void compute_expert_offsets_w4a8_kernel(
    const int32_t* __restrict__ problem_sizes1, int32_t* __restrict__ expert_offsets, int n, int stride) {
  if (threadIdx.x == 0) {
    expert_offsets[0] = 0;
  }
  cumsum_block_scan<BLOCK_SIZE>(problem_sizes1, expert_offsets + 1, n, stride);
}

void compute_expert_offsets_w4a8(
    cudaStream_t stream, const int32_t* problem_sizes1, int32_t* expert_offsets, int n, int stride = 1, int off = 0) {
#define compute_expert_offsets_w4a8_call(BLOCK_SIZE) \
  compute_expert_offsets_w4a8_kernel<BLOCK_SIZE>     \
      <<<1, BLOCK_SIZE, 0, stream>>>(problem_sizes1 + off, expert_offsets, n, stride);

  if (n <= 32) {
    compute_expert_offsets_w4a8_call(32);
  } else if (n <= 64) {
    compute_expert_offsets_w4a8_call(64);
  } else if (n <= 128) {
    compute_expert_offsets_w4a8_call(128);
  } else if (n <= 256) {
    compute_expert_offsets_w4a8_call(256);
  } else if (n <= 512) {
    compute_expert_offsets_w4a8_call(512);
  } else {
    compute_expert_offsets_w4a8_call(1024);
  }
#undef compute_expert_offsets_w4a8_call
}

/// Fused kernel: prefix-sum expert offsets + build double work_table (GEMM1 + GEMM2).
/// Runs as <<<1, BLOCK_SIZE>>>. Phase 1 uses all threads for prefix-sum,
/// Phase 2 uses thread-0 to build both work_tables in a single pass.
template <int BLOCK_SIZE>
__global__ void compute_expert_offsets_w4a8_with_worktable_kernel(
    const int32_t* __restrict__ problem_sizes1,  // [E*3] layout: {N, M_tokens, K} per expert
    int32_t* __restrict__ expert_offsets,         // [E+1] prefix-sum output
    int32_t* __restrict__ n_per_expert_out,       // [E] token counts for GEMV
    int32_t* __restrict__ work_table1,            // [E*3] flattened {expert_id, n_tiles, cta_start} for GEMV1
    int32_t* __restrict__ work_table2,            // [E*3] flattened {expert_id, n_tiles, cta_start} for GEMV2
    int32_t* __restrict__ metadata,               // [3]: {num_active, total_flat_ctas1, total_flat_ctas2}
    int32_t* __restrict__ tile_counter1,          // [1] atomic counter for GEMV1
    int32_t* __restrict__ tile_counter2,          // [1] atomic counter for GEMV2
    int n,                                        // num_experts
    int stride,                                   // stride in problem_sizes1 (=3)
    int off,                                      // offset to token count field (=1)
    int M_tiles1,                                 // M tiles for GEMV1
    int M_tiles2) {                               // M tiles for GEMV2

  // Phase 1: prefix-sum of token counts → expert_offsets (all threads)
  if (threadIdx.x == 0) {
    expert_offsets[0] = 0;
  }
  cumsum_block_scan<BLOCK_SIZE>(problem_sizes1 + off, expert_offsets + 1, n, stride);
  __syncthreads();

  // Phase 2: thread-0 builds n_per_expert + double work_table
  if (threadIdx.x == 0) {
    int num_active = 0;
    int cta_offset1 = 0;
    int cta_offset2 = 0;
    for (int e = 0; e < n; e++) {
      int ne = problem_sizes1[e * stride + off];  // token count for expert e
      n_per_expert_out[e] = ne;
      if (ne > 0) {
        int nt = (ne + 7) / 8;  // N tiles (8 tokens per GEMV tile)
        int base = num_active * 3;
        work_table1[base]     = e;            // expert_id
        work_table1[base + 1] = nt;           // n_tiles
        work_table1[base + 2] = cta_offset1;  // cta_start for GEMV1
        work_table2[base]     = e;
        work_table2[base + 1] = nt;
        work_table2[base + 2] = cta_offset2;  // cta_start for GEMV2
        num_active++;
        cta_offset1 += M_tiles1 * nt;
        cta_offset2 += M_tiles2 * nt;
      }
    }
    metadata[0] = num_active;
    metadata[1] = cta_offset1;   // total_flat_ctas for GEMV1
    metadata[2] = cta_offset2;   // total_flat_ctas for GEMV2
    *tile_counter1 = 0;
    *tile_counter2 = 0;
  }
}

void compute_expert_offsets_w4a8_with_worktable(
    cudaStream_t stream,
    const int32_t* problem_sizes1,
    int32_t* expert_offsets,
    int32_t* n_per_expert_out,
    int32_t* work_table1,
    int32_t* work_table2,
    int32_t* metadata,
    int32_t* tile_counter1,
    int32_t* tile_counter2,
    int n, int stride, int off, int M_tiles1, int M_tiles2) {
#define compute_expert_offsets_w4a8_wt_call(BLOCK_SIZE)                   \
  compute_expert_offsets_w4a8_with_worktable_kernel<BLOCK_SIZE>           \
      <<<1, BLOCK_SIZE, 0, stream>>>(problem_sizes1, expert_offsets,     \
          n_per_expert_out, work_table1, work_table2, metadata,          \
          tile_counter1, tile_counter2, n, stride, off, M_tiles1, M_tiles2);

  if (n <= 32) {
    compute_expert_offsets_w4a8_wt_call(32);
  } else if (n <= 64) {
    compute_expert_offsets_w4a8_wt_call(64);
  } else if (n <= 128) {
    compute_expert_offsets_w4a8_wt_call(128);
  } else if (n <= 256) {
    compute_expert_offsets_w4a8_wt_call(256);
  } else if (n <= 512) {
    compute_expert_offsets_w4a8_wt_call(512);
  } else {
    compute_expert_offsets_w4a8_wt_call(1024);
  }
#undef compute_expert_offsets_w4a8_wt_call
}

void get_cutlass_w4a8_moe_mm_data_caller(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
  auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
  auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());

  constexpr uint64_t BLOCK_SIZE = 512;
  compute_problem_sizes_w4a8<BLOCK_SIZE><<<num_experts, BLOCK_SIZE, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(problem_sizes2.data_ptr()),
      topk_ids.numel(),
      n,
      k);

  compute_expert_offsets_w4a8(
      stream,
      static_cast<const int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(expert_offsets.data_ptr()),
      num_experts,
      3,
      1);
}

void get_cutlass_w4a8_moe_mm_data_with_worktable_caller(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& n_per_expert,
    torch::Tensor& work_table1,
    torch::Tensor& work_table2,
    torch::Tensor& metadata,
    torch::Tensor& tile_counter1,
    torch::Tensor& tile_counter2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k,
    const int64_t M_tiles1,
    const int64_t M_tiles2) {
  auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());

  constexpr uint64_t BLOCK_SIZE = 512;
  compute_problem_sizes_w4a8<BLOCK_SIZE><<<num_experts, BLOCK_SIZE, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(problem_sizes2.data_ptr()),
      topk_ids.numel(),
      n,
      k);

  compute_expert_offsets_w4a8_with_worktable(
      stream,
      static_cast<const int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(expert_offsets.data_ptr()),
      static_cast<int32_t*>(n_per_expert.data_ptr()),
      static_cast<int32_t*>(work_table1.data_ptr()),
      static_cast<int32_t*>(work_table2.data_ptr()),
      static_cast<int32_t*>(metadata.data_ptr()),
      static_cast<int32_t*>(tile_counter1.data_ptr()),
      static_cast<int32_t*>(tile_counter2.data_ptr()),
      num_experts,
      3,    // stride
      1,    // offset to token count in problem_sizes1
      (int)M_tiles1,
      (int)M_tiles2);
}
