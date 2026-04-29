import torch


def get_cutlass_w4a8_moe_mm_data(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    input_permutation: torch.Tensor,
    output_permutation: torch.Tensor,
    num_experts: int,
    n: int,
    k: int,
):
    """
    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in topk_ids (token-expert mapping) and uses it to
    compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation after the input is sorted with
                      input_permutation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    - input_permutation: Permutation that must be used to shuffle the input
                         before executing the MMs.
    - output_permutation: Permutation that must be used to shuffle the output
                          after executing the MMs.
    """
    torch.ops.sgl_kernel.get_cutlass_w4a8_moe_mm_data.default(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_experts,
        n,
        k,
    )


def cutlass_w4a8_moe_mm(
    d: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    experts_offsets: torch.tensor,
    problem_sizes: torch.tensor,
    a_strides: torch.tensor,
    b_strides: torch.tensor,
    d_strides: torch.tensor,
    s_strides: torch.tensor,
    chunk_size: int = 128,
    topk: int = 8,
):
    """
    Perform grouped matrix multiplication between int4 weights and fp8 activations.

    This function executes multiple GEMM operations in parallel, which is useful for
    scenarios like Mixture of Experts (MoE) where different inputs go through different
    experts. The implementation leverages NVIDIA Hopper architecture features for
    optimal performance with quantized weights.

    Args:
        d: Output matrices of shape [total_m, total_n]
        a: Activation matrices in FP8 (float_e4m3_t) format
           Each tensor should be of shape [total_m, K] in row-major layout
        b: Weight matrices in packed int4 format
           Each tensor should be of shape [E, N, K//2] in column-major layout
           where each byte contains two 4-bit integers
        a_scales: Scale factors for the inputs
        b_scales: Scale factors for the quantized weights
           Each tensor should be of shape [E, K//512, N*8]
        experts_offsets: Tensor containing expert offsets for determining group boundaries
        problem_sizes: with shape [num_experts, 3] (M, N, K for each group) (int32)
        a_strides: Strides information for A matrices
        b_strides: Strides information for B matrices
        d_strides: Strides information for D matrices
        s_strides: Strides information for b_scales matrices
        chunk_size: Number of elements each scale value applies to (K//512), default to 128

    Requirements:
        - All tensors must be on a CUDA device
        - Requires an NVIDIA Hopper GPU (H100)
        - A tensors must be in float8_e4m3fn format
        - B tensors must contain packed int4 values (stored as int8)

    Note:
        The function computes: D = (A * (B * scales))
        for each group of tensors in parallel
    """

    torch.ops.sgl_kernel.cutlass_w4a8_moe_mm.default(
        d,
        a,
        b,
        a_scales,
        b_scales,
        experts_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        d_strides,
        s_strides,
        chunk_size,
        topk,
    )


def cutlass_w4a8_moe_gemv(
    d: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    n_per_expert: torch.Tensor,
    work_table: torch.Tensor,
    tile_counter: torch.Tensor,
    metadata: torch.Tensor,
    M: int,
    K: int,
    num_experts: int,
    metadata_offset: int,
    persistent_grid_size: int,
):
    """
    Persistent GEMV launch for W4A8 MoE decode.
    CUDA Graph compatible: all dynamic values read from device pointers.
    """
    torch.ops.sgl_kernel.cutlass_w4a8_moe_gemv.default(
        d, a, b, a_scale, b_scale, expert_offsets, n_per_expert,
        work_table, tile_counter, metadata,
        M, K, num_experts, metadata_offset, persistent_grid_size,
    )


def cutlass_w4a8_moe_preprocess_weights(
    w_out: torch.Tensor,
    s_out: torch.Tensor,
    w_in: torch.Tensor,
    s_in: torch.Tensor,
    M: int,
    K: int,
):
    """
    Weight preprocessing: int4 nibble interleave + 2-row interleave + scale padding.
    Called once at model load time.
    """
    torch.ops.sgl_kernel.cutlass_w4a8_moe_preprocess_weights.default(
        w_out, s_out, w_in, s_in, M, K,
    )


def get_cutlass_w4a8_moe_mm_data_with_worktable(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    n_per_expert: torch.Tensor,
    work_table1: torch.Tensor,
    work_table2: torch.Tensor,
    metadata: torch.Tensor,
    tile_counter1: torch.Tensor,
    tile_counter2: torch.Tensor,
    input_permutation: torch.Tensor,
    output_permutation: torch.Tensor,
    num_experts: int,
    n: int,
    k: int,
    M_tiles1: int,
    M_tiles2: int,
):
    """
    Fused: compute problem sizes + expert offsets + build double work_table
    for GEMV1 and GEMV2 in a single kernel pass.
    """
    torch.ops.sgl_kernel.get_cutlass_w4a8_moe_mm_data_with_worktable.default(
        topk_ids, expert_offsets, problem_sizes1, problem_sizes2,
        n_per_expert, work_table1, work_table2, metadata,
        tile_counter1, tile_counter2,
        input_permutation, output_permutation,
        num_experts, n, k, M_tiles1, M_tiles2,
    )
