# FP8 → W4A8 checkpoint converters (int4 / mxfp4)

Convert a block-FP8 MoE checkpoint into a W4A8 checkpoint that sglang's `w4afp8`
quant method can load. Only routed-expert `gate_proj`/`up_proj`/`down_proj`
weights are re-quantized (dense/attention layers stay FP8, copied byte-for-byte).
RTN, no calibration; streams per-safetensors-shard, multi-GPU.

- `quantize_fp8_to_wint4.py` — INT4 weight × FP8 act (`moe_weight_format=int4`,
  per-output-channel group-128, packed for the sgl-kernel `cutlass_w4a8` layout).
- `quantize_fp8_to_wmxfp4.py` — MXFP4 weight × FP8 act (`moe_weight_format=mxfp4`,
  group-32 E2M1 values + UE8M0 scales; needs FlashInfer PR #3738 humming kernel at
  runtime). See the module docstring for the exact E2M1/E8M0 numerics.
- `convert_all.sh` — multi-GPU driver over checkpoint shards.

Each source FP8 tensor is dequantized with its 128×128 block scale (accepts F32
**or** BF16 block scales — Qwen3.5 stores BF16), then re-quantized independently.

## Usage

```bash
# single shard (rank r of `parts`)
python3 quantize_fp8_to_wmxfp4.py --model-dir <FP8_CKPT> --output-dir <OUT> \
        --parts 94 --rank 0

# all shards across NG GPUs
bash convert_all.sh quantize_fp8_to_wmxfp4.py <FP8_CKPT> <OUT> 94 8 <LOGDIR>
```

Output ckpt config carries `moe_weight_format` (int4|mxfp4), `group_size`
(128|32) and `activation_scheme=dynamic`, so sglang auto-selects the path.
