/**
 * w4a8_gemv.cu — CUTLASS GEMV kernel wrapper for sgl-kernel (torch::Tensor interface)
 *
 * Provides:
 *   cutlass_w4a8_moe_gemv_sm90           — persistent GEMV launch (CUDA Graph compatible)
 *   cutlass_w4a8_moe_preprocess_weights  — weight interleave + scale pad (model load time)
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/all.h>

#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/array.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass_extensions/gemm/kernel/gemv.h"
#include "cutlass_extensions/gemm/device/gemv.h"

namespace {

// Type definitions — aligned with sglang grouped GEMM conventions
using ElementA   = cutlass::int4b_t;         // Weight type (int4, packed as int8)
using ElementB   = cutlass::float_e4m3_t;    // Activation type (fp8 e4m3)
using ElementC   = cutlass::bfloat16_t;      // Output type (bf16)
using ElementSF  = cutlass::bfloat16_t;      // Scale type (bf16, aligned with sglang)
using ElementAcc = float;                    // Accumulator type

static constexpr int kEPA = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16
static constexpr int kTC  = 128;   // Thread count
static constexpr int kTPR = 8;     // Threads per row
static constexpr int kSKS = 1;     // Split-K slices
static constexpr int kSFV = 128;   // Scale factor vector size (group_size)

using LayoutA  = cutlass::layout::RowMajor;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    ElementC, 4, ElementAcc, ElementAcc>;

// Sorted I/O kernel (expert_offsets-based, for MoE decode)
using GemvKernel = cutlass::gemm::kernel::Gemv<
    ElementA, LayoutA, ElementB, ElementC, ElementAcc, Epilogue,
    kEPA, kTC, kTPR, kSKS, ElementSF, kSFV, /*kSortedIO=*/true>;
using GemvDevice = cutlass::gemm::device::Gemv<GemvKernel>;
using ExpertWork = cutlass::gemm::kernel::ExpertWork;

static constexpr int kBlockY = kTC / kTPR;  // blockDim.y = kThreadCount / kThreadsPerRow = 128 / 8 = 16

}  // namespace


/**
 * Persistent GEMV launch — reads work_table/metadata from device memory.
 * CUDA Graph compatible: all dynamic values read from device pointers.
 *
 * @param d              [total, M] bf16 output (sorted by expert)
 * @param a              [total, K] fp8 activations (sorted by expert)
 * @param b              [E, M, K/2] int8 weights (interleaved int4)
 * @param a_scale        [1] float32 activation scale (fp8 quantization scale)
 * @param b_scale        [E, M, padded_groups] bf16 weight scale (padded)
 * @param expert_offsets  [E+1] int32 prefix-sum of tokens per expert
 * @param n_per_expert   [E] int32 token count per expert
 * @param work_table     [max_active*3] int32 (flattened ExpertWork: expert_id, n_tiles, cta_start)
 * @param tile_counter   [1] int32 atomic counter (reset by preprocessing kernel)
 * @param metadata       [3] int32: [num_active, total_flat_ctas_gemm1, total_flat_ctas_gemm2]
 * @param M              weight rows (output dim per expert)
 * @param K              shared dimension (activation dim)
 * @param num_experts    number of experts E
 * @param metadata_offset 1 for GEMM1, 2 for GEMM2 (selects total_flat_ctas from metadata)
 * @param persistent_grid_size fixed grid size (SM_COUNT * occupancy)
 */
void cutlass_w4a8_moe_gemv_sm90(
    torch::Tensor& d,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scale,
    torch::Tensor const& b_scale,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& n_per_expert,
    torch::Tensor const& work_table,
    torch::Tensor& tile_counter,
    torch::Tensor const& metadata,
    int64_t M,
    int64_t K,
    int64_t num_experts,
    int64_t metadata_offset,
    int64_t persistent_grid_size) {

  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "Activations must be fp8 e4m3");
  TORCH_CHECK(b.scalar_type() == torch::kInt8, "Weights must be packed int4 (stored as int8)");
  TORCH_CHECK(d.scalar_type() == torch::kBFloat16, "Output must be bf16");
  TORCH_CHECK(b_scale.scalar_type() == torch::kBFloat16, "Weight scales must be bf16");
  TORCH_CHECK(a_scale.scalar_type() == torch::kFloat32, "Activation scale must be float32");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32, "expert_offsets must be int32");
  TORCH_CHECK(n_per_expert.scalar_type() == torch::kInt32, "n_per_expert must be int32");
  TORCH_CHECK(work_table.scalar_type() == torch::kInt32, "work_table must be int32");
  TORCH_CHECK(tile_counter.scalar_type() == torch::kInt32, "tile_counter must be int32");
  TORCH_CHECK(metadata.scalar_type() == torch::kInt32, "metadata must be int32");
  TORCH_CHECK(metadata_offset == 1 || metadata_offset == 2, "metadata_offset must be 1 or 2");
  TORCH_CHECK(persistent_grid_size > 0, "persistent_grid_size must be > 0");
  TORCH_CHECK(M % (4 * kBlockY) == 0,
      "M must be a multiple of ", 4 * kBlockY,
      " (mma.sync requires all warp threads to participate; M=", M, ")");

  auto stream = at::cuda::getCurrentCUDAStream(a.device().index());

  int M_tiles = ((int)M / 4 + kBlockY - 1) / kBlockY;

  // Use pointer-based alpha for CUDA Graph compatibility (no D2H sync)
  typename Epilogue::Params epilogue_params;
  epilogue_params.alpha = 0;
  epilogue_params.beta = 0;
  epilogue_params.alpha_ptr = a_scale.data_ptr<float>();
  epilogue_params.beta_ptr = nullptr;

  GemvDevice::Arguments args{
      (int32_t)M,
      static_cast<int32_t*>(n_per_expert.data_ptr()),
      (int32_t)K,
      0,  // max_N (unused in persistent mode)
      (int32_t)num_experts,
      epilogue_params,
      {static_cast<ElementA*>(b.data_ptr()), (int)K},  // weights TensorRef
      a.data_ptr(),                                      // activations (B)
      d.data_ptr(),                                      // output (C)
      d.data_ptr(),                                      // output (D)
      (int64_t)M * K,                                   // batch_stride_A (weights)
      0, 0, 0,                                           // unused strides (sorted I/O)
      static_cast<ElementSF*>(b_scale.data_ptr()),       // weight scales
  };

  // Sorted I/O: use expert_offsets for token layout
  args.expert_offsets = static_cast<int32_t*>(expert_offsets.data_ptr());

  // Persistent mode: work_table + atomic tile counter
  args.work_table = reinterpret_cast<ExpertWork*>(work_table.data_ptr());
  args.tile_counter = static_cast<int32_t*>(tile_counter.data_ptr());
  args.persistent_grid_size = (int32_t)persistent_grid_size;
  args.M_tiles = M_tiles;

  // Device pointer indirection for CUDA Graph compatibility
  // metadata layout: [num_active, total_flat_ctas_gemm1, total_flat_ctas_gemm2]
  args.d_num_active = static_cast<int32_t*>(metadata.data_ptr());
  args.d_total_flat_ctas = static_cast<int32_t*>(metadata.data_ptr()) + metadata_offset;

  // Placeholders (kernel reads from device pointers, not these scalars)
  args.total_flat_ctas = 1;
  args.num_active = 1;

  GemvDevice gemv;
  auto status = gemv.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
      "GEMV kernel cannot implement this problem (K must be divisible by ",
      4 * kEPA * GemvKernel::kUnroll * kSKS, ")");

  gemv.initialize(args);
  gemv.run(stream);
}


/**
 * Weight preprocessing: int4 nibble interleave + 2-row interleave + scale padding.
 * Called once at model load time (process_weights_after_loading).
 *
 * @param w_out   [E, M, K/2] int8 output (interleaved weights)
 * @param s_out   [E, M, padded_groups] bf16 output (padded scales)
 * @param w_in    [E, M, K/2] int8 input (original packed int4)
 * @param s_in    [E, M, groups] bf16 input (original scales, [E, M, K/group_size])
 * @param M       weight rows per expert
 * @param K       weight columns per expert
 */
void cutlass_w4a8_moe_preprocess_weights_sm90(
    torch::Tensor& w_out,
    torch::Tensor& s_out,
    torch::Tensor const& w_in,
    torch::Tensor const& s_in,
    int64_t M,
    int64_t K) {

  TORCH_CHECK(w_in.scalar_type() == torch::kInt8, "Weight input must be int8 (packed int4)");
  TORCH_CHECK(s_in.scalar_type() == torch::kBFloat16, "Scale input must be bf16");
  TORCH_CHECK(w_out.scalar_type() == torch::kInt8, "Weight output must be int8");
  TORCH_CHECK(s_out.scalar_type() == torch::kBFloat16, "Scale output must be bf16");

  int E = (int)w_in.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(w_in.device().index());

  // matrix_A_interleave does int4 nibble interleave + 2-row interleave on weights,
  // and pads scales to 8-aligned groups. It synchronizes the stream internally.
  GemvKernel::matrix_A_interleave(
      static_cast<ElementA*>(w_out.data_ptr()),
      static_cast<ElementA*>(const_cast<void*>(w_in.data_ptr())),
      static_cast<ElementSF*>(s_out.data_ptr()),
      static_cast<ElementSF*>(const_cast<void*>(s_in.data_ptr())),
      E, (int)M, (int)K, stream);
}
