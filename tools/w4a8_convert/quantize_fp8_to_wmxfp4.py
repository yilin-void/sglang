#!/usr/bin/env python3
"""Stream a block-FP8 MoE checkpoint into an MXFP4 expert checkpoint.

Only routed expert ``gate_proj``, ``up_proj`` and ``down_proj`` weights are
changed.  Each source FP8 tensor is dequantized with its 128x128 block scale,
then quantized independently to MXFP4.  TensorRT-LLM's CUDA quantizer is used
on SM100+, while SM90 uses a CUDA-compatible PyTorch implementation because
the TensorRT-LLM 1.3.0rc14 FP4 device conversion is disabled before SM100.
All other tensors are copied byte-for-byte.

The output is deliberately stored in the logical checkpoint layout:

* expert ``weight``: packed E2M1 bytes, ``uint8``, shape ``[N, K / 2]``;
* expert ``weight_scale``: raw linear E8M0 bytes, ``uint8``, shape
  ``[N, K / 32]``.

Runtime-specific FlashInfer interleaving and Humming preprocessing belong in
the loader, not in the checkpoint converter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FP8_BLOCK_SIZE = 128
MXFP4_GROUP_SIZE = 32
MXFP4_ROUNDTRIP_MAX_REL_RMSE = 0.20
_ROUNDTRIP_VALIDATED_SHAPES: set[tuple[int, int]] = set()
_REFERENCE_QUANTIZER_LOGGED = False
EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj))\.weight$"
)


@dataclass(frozen=True)
class TensorSpec:
    """A tensor entry read from a safetensors header."""

    dtype: str
    shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class OutputSpec:
    """A tensor entry that will be emitted to an output shard."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class CopyRecord:
    """A source tensor copied without numerical conversion."""

    rank: int
    source_shard: str
    source_name: str
    output: OutputSpec

    @property
    def outputs(self) -> tuple[OutputSpec, ...]:
        return (self.output,)


@dataclass(frozen=True)
class ExpertRecord:
    """One FP8 expert weight and its paired source/output scale tensors."""

    rank: int
    source_shard: str
    source_weight_name: str
    source_scale_name: str
    weight_output: OutputSpec
    scale_output: OutputSpec

    @property
    def outputs(self) -> tuple[OutputSpec, ...]:
        return (self.weight_output, self.scale_output)


Record = CopyRecord | ExpertRecord


@dataclass(frozen=True)
class ConversionPlan:
    """Complete deterministic conversion plan shared by every worker rank."""

    records: tuple[Record, ...]
    output_weight_map: Mapping[str, str]
    total_size: int
    source_specs: Mapping[str, TensorSpec]
    expert_layer_prefixes: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", "--model-dir", required=True)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--parts", type=int, default=16)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate headers and print this rank's plan without loading torch.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Truncated safetensors header length: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > 100_000_000:
            raise ValueError(
                f"Unreasonably large safetensors header ({header_length} bytes): {path}"
            )
        payload = handle.read(header_length)
        if len(payload) != header_length:
            raise ValueError(f"Truncated safetensors header: {path}")
    header = json.loads(payload)
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header in {path}")
    return header


def _tensor_spec(entry: Mapping[str, Any], name: str, shard: str) -> TensorSpec:
    try:
        dtype = str(entry["dtype"])
        shape = tuple(int(dim) for dim in entry["shape"])
        start, end = (int(offset) for offset in entry["data_offsets"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid tensor header for {name} in {shard}: {entry}"
        ) from exc
    if any(dim < 0 for dim in shape) or start < 0 or end < start:
        raise ValueError(f"Invalid tensor shape/offset for {name} in {shard}: {entry}")
    return TensorSpec(dtype=dtype, shape=shape, nbytes=end - start)


def _discover_source(
    model_dir: Path,
) -> tuple[dict[str, str], dict[str, TensorSpec], list[str]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        raw_weight_map = index.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise ValueError(f"Missing weight_map in {index_path}")
        weight_map = {str(name): str(shard) for name, shard in raw_weight_map.items()}
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_paths = sorted(model_dir.glob("*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No model.safetensors.index.json or *.safetensors under {model_dir}"
            )
        shard_names = [path.name for path in shard_paths]
        weight_map = {}

    specs: dict[str, TensorSpec] = {}
    discovered_weight_map: dict[str, str] = {}
    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Source shard does not exist: {shard_path}")
        header = _read_safetensors_header(shard_path)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in specs:
                raise ValueError(f"Tensor {name} appears in more than one source shard")
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid tensor entry for {name} in {shard_path}")
            specs[name] = _tensor_spec(entry, name, shard_name)
            discovered_weight_map[name] = shard_name

    if weight_map:
        missing_from_headers = set(weight_map).difference(specs)
        extra_in_headers = set(specs).difference(weight_map)
        if missing_from_headers or extra_in_headers:
            raise ValueError(
                "Source index/header mismatch: "
                f"missing={len(missing_from_headers)}, extra={len(extra_in_headers)}"
            )
        for name, shard_name in weight_map.items():
            if discovered_weight_map[name] != shard_name:
                raise ValueError(
                    f"Source index maps {name} to {shard_name}, but it is in "
                    f"{discovered_weight_map[name]}"
                )
    else:
        weight_map = discovered_weight_map

    return weight_map, specs, shard_names


def _output_shard_name(rank: int, parts: int) -> str:
    # Keep the zero-based naming convention used by the existing TRT-LLM script.
    return f"model-{rank:05d}-of-{parts:05d}.safetensors"


def _expert_layer_prefix(weight_name: str) -> str:
    before_expert_id, _ = weight_name.split(".experts.", 1)
    return f"{before_expert_id}.experts"


def build_plan(model_dir: Path, parts: int) -> ConversionPlan:
    if parts < 1:
        raise ValueError(f"parts must be positive, got {parts}")

    source_weight_map, source_specs, source_shards = _discover_source(model_dir)
    shard_ordinals = {name: index for index, name in enumerate(source_shards)}
    paired_scale_names: set[str] = set()
    expert_weight_names: set[str] = set()

    for name in source_specs:
        match = EXPERT_WEIGHT_RE.match(name)
        if match is None:
            continue
        scale_name = f"{match.group('prefix')}.weight_scale_inv"
        if scale_name not in source_specs:
            raise ValueError(f"Missing FP8 block scale {scale_name} for {name}")
        paired_scale_names.add(scale_name)
        expert_weight_names.add(name)

    if not expert_weight_names:
        raise ValueError(
            "No routed expert FP8 weights were found in the source checkpoint"
        )

    records: list[Record] = []
    output_weight_map: dict[str, str] = {}
    expert_layer_prefixes: set[str] = set()
    total_size = 0

    ordered_names = sorted(
        source_specs,
        key=lambda name: (shard_ordinals[source_weight_map[name]], name),
    )
    for name in ordered_names:
        if name in paired_scale_names:
            continue
        source_shard = source_weight_map[name]
        rank = shard_ordinals[source_shard] % parts
        output_shard = _output_shard_name(rank, parts)
        source_spec = source_specs[name]

        if name in expert_weight_names:
            match = EXPERT_WEIGHT_RE.match(name)
            assert match is not None
            if source_spec.dtype != "F8_E4M3" or len(source_spec.shape) != 2:
                raise ValueError(
                    f"Expert source weight must be 2D F8_E4M3: {name} has "
                    f"dtype={source_spec.dtype}, shape={source_spec.shape}"
                )
            rows, columns = source_spec.shape
            if rows % FP8_BLOCK_SIZE != 0 or columns % FP8_BLOCK_SIZE != 0:
                raise ValueError(
                    f"Expert FP8 weight shape must be divisible by {FP8_BLOCK_SIZE}: "
                    f"{name} has shape {source_spec.shape}"
                )
            if columns % MXFP4_GROUP_SIZE != 0:
                raise ValueError(
                    f"Expert K must be divisible by {MXFP4_GROUP_SIZE}: "
                    f"{name} has K={columns}"
                )

            source_scale_name = f"{match.group('prefix')}.weight_scale_inv"
            source_scale_spec = source_specs[source_scale_name]
            expected_source_scale_shape = (
                rows // FP8_BLOCK_SIZE,
                columns // FP8_BLOCK_SIZE,
            )
            if (
                source_scale_spec.dtype not in ("F32", "BF16")
                or source_scale_spec.shape != expected_source_scale_shape
            ):
                raise ValueError(
                    f"FP8 block scale {source_scale_name} must be F32/BF16 "
                    f"{expected_source_scale_shape}; got "
                    f"dtype={source_scale_spec.dtype}, "
                    f"shape={source_scale_spec.shape}"
                )
            if source_weight_map[source_scale_name] != source_shard:
                raise ValueError(
                    f"Weight and scale must share a source shard: {name}, "
                    f"{source_scale_name}"
                )

            weight_output = OutputSpec(
                name=name,
                dtype="U8",
                shape=(rows, columns // 2),
                nbytes=rows * columns // 2,
            )
            scale_output = OutputSpec(
                name=f"{match.group('prefix')}.weight_scale",
                dtype="U8",
                shape=(rows, columns // MXFP4_GROUP_SIZE),
                nbytes=rows * columns // MXFP4_GROUP_SIZE,
            )
            record: Record = ExpertRecord(
                rank=rank,
                source_shard=source_shard,
                source_weight_name=name,
                source_scale_name=source_scale_name,
                weight_output=weight_output,
                scale_output=scale_output,
            )
            expert_layer_prefixes.add(_expert_layer_prefix(name))
        else:
            output = OutputSpec(
                name=name,
                dtype=source_spec.dtype,
                shape=source_spec.shape,
                nbytes=source_spec.nbytes,
            )
            record = CopyRecord(
                rank=rank,
                source_shard=source_shard,
                source_name=name,
                output=output,
            )

        for output in record.outputs:
            if output.name in output_weight_map:
                raise ValueError(f"Duplicate output tensor name: {output.name}")
            output_weight_map[output.name] = output_shard
            total_size += output.nbytes
        records.append(record)

    return ConversionPlan(
        records=tuple(records),
        output_weight_map=output_weight_map,
        total_size=total_size,
        source_specs=source_specs,
        expert_layer_prefixes=tuple(sorted(expert_layer_prefixes)),
    )


def _quant_cfg_for_output(model_dir: Path, plan: ConversionPlan) -> dict[str, Any]:
    source_path = model_dir / "quant_cfg.json"
    if source_path.is_file():
        result = _read_json(source_path)
        raw_layers = result.get("quantized_layers", {})
        if not isinstance(raw_layers, dict):
            raise ValueError(f"quantized_layers must be an object in {source_path}")
        layers: dict[str, Any] = dict(raw_layers)
    else:
        result = {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
        }
        layers = {}
        expert_scale_names = {
            record.source_scale_name
            for record in plan.records
            if isinstance(record, ExpertRecord)
        }
        for name in plan.source_specs:
            if not name.endswith(".weight_scale_inv") or name in expert_scale_names:
                continue
            module_name = name[: -len(".weight_scale_inv")]
            layers[module_name] = {"quant_algo": "FP8_BLOCK_SCALES"}

    expert_spec = {
        "quant_algo": "W4A16_MXFP4",
        "weight_format": "mxfp4",
        "group_size": MXFP4_GROUP_SIZE,
        "has_zero_point": False,
        "pre_quant_scale": False,
    }
    for layer_name in list(layers):
        if ".mlp.experts" in layer_name:
            del layers[layer_name]
    for layer_name in plan.expert_layer_prefixes:
        layers[layer_name] = dict(expert_spec)

    result["quant_algo"] = "MIXED_PRECISION"
    result.setdefault("kv_cache_quant_algo", None)
    result["quantized_layers"] = {
        name: layers[name] for name in sorted(layers)
    }
    return result


def _write_auxiliary_files(
    model_dir: Path,
    output_dir: Path,
    plan: ConversionPlan,
) -> None:
    excluded = {
        ".git",
        "config.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
        "quant_cfg.json",
    }
    for source in sorted(model_dir.iterdir()):
        if not source.is_file():
            continue
        if source.suffix == ".safetensors" or source.name in excluded:
            continue
        shutil.copy2(source, output_dir / source.name)

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing source config: {config_path}")
    config = _read_json(config_path)
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        quantization_config = {}
    else:
        quantization_config = dict(quantization_config)
    quantization_config.update(
        {
            "quant_method": "w4afp8",
            "weight_format": "mxfp4",
            "moe_weight_format": "mxfp4",
            "group_size": MXFP4_GROUP_SIZE,
            "activation_scheme": "dynamic",
        }
    )
    config["quantization_config"] = quantization_config
    _atomic_write_json(output_dir / "config.json", config)

    hf_quant_config = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "weight_format": "mxfp4",
            "moe_weight_format": "mxfp4",
            "group_size": MXFP4_GROUP_SIZE,
            "activation_scheme": "dynamic",
        }
    }
    _atomic_write_json(output_dir / "hf_quant_config.json", hf_quant_config)
    _atomic_write_json(
        output_dir / "quant_cfg.json", _quant_cfg_for_output(model_dir, plan)
    )
    _atomic_write_json(
        output_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": plan.total_size},
            "weight_map": {
                name: plan.output_weight_map[name]
                for name in sorted(plan.output_weight_map)
            },
        },
    )


def _safetensors_header(outputs: Iterable[OutputSpec]) -> tuple[bytes, int]:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for output in outputs:
        header[output.name] = {
            "dtype": output.dtype,
            "shape": list(output.shape),
            "data_offsets": [offset, offset + output.nbytes],
        }
        offset += output.nbytes
    payload = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    payload += b" " * ((8 - len(payload) % 8) % 8)
    return payload, offset


def _load_runtime():
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "MXFP4 conversion requires torch and safetensors"
        ) from exc

    # TensorRT-LLM 1.3.0rc14 registers fp4_quantize on Hopper, but its
    # FP16/BF16 -> FP4 device helper returns zero for __CUDA_ARCH__ < 1000.
    # Do not mistake a registered schema for an SM90-capable implementation.
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 10:
        return torch, safe_open, None

    # Importing either package may register the TRT-LLM custom-op namespace,
    # depending on the image/build layout.
    for module_name in ("tensorrt_llm", "flashinfer"):
        try:
            __import__(module_name)
        except ImportError:
            pass
        try:
            fp4_quantize = torch.ops.trtllm.fp4_quantize
        except AttributeError:
            continue
        return torch, safe_open, fp4_quantize
    # The reference path below is slower but has the same logical checkpoint
    # contract, so the converter does not require a private custom op.
    return torch, safe_open, None


def _as_raw_u8(tensor, torch, expected_shape: Sequence[int], label: str):
    tensor = tensor.contiguous()
    raw = tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)
    expected_numel = math.prod(expected_shape)
    if raw.numel() != expected_numel:
        raise ValueError(
            f"{label} has {raw.numel()} raw bytes, expected {expected_numel} "
            f"for shape {tuple(expected_shape)} (source dtype {tensor.dtype})"
        )
    return raw.reshape(tuple(expected_shape)).contiguous()


def _quantize_mxfp4_cuda_compatible(weight, torch):
    """Quantize ``[N, K]`` to linear group-32 MXFP4 on any CUDA arch.

    This follows the TensorRT-LLM CUDA kernel and FlashInfer's formal FP4 test
    helper: group amax/6, round-toward-positive-infinity UE8M0 scales,
    round-to-nearest-even E2M1 values, and low-nibble-first packing.  The PR's
    trace template currently uses floor-log2/ties-down semantics and is not a
    bit-exact oracle for the actual CUDA quantizer.
    """
    rows, columns = weight.shape
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("MXFP4 quantization input contains NaN or Inf")
    blocks = weight.to(torch.float32).reshape(
        rows, columns // MXFP4_GROUP_SIZE, MXFP4_GROUP_SIZE
    )
    block_scale = blocks.abs().amax(dim=-1) / 6.0

    # For positive float32 values, adding all mantissa bits and masking the
    # exponent implements the same round-toward-+inf conversion used by
    # __nv_cvt_float_to_e8m0(..., cudaRoundPosInf).  It also preserves the
    # CUDA zero-block encoding (raw scale byte 0).
    rounded_scale_bits = (
        block_scale.contiguous().view(torch.int32) + 0x007FFFFF
    ) & 0x7F800000
    actual_scale = rounded_scale_bits.view(torch.float32)
    scale_u8 = (rounded_scale_bits >> 23).to(torch.uint8).contiguous()
    reciprocal_scale = torch.where(
        actual_scale != 0,
        actual_scale.reciprocal(),
        torch.zeros_like(actual_scale),
    )
    scaled = blocks * reciprocal_scale.unsqueeze(-1)

    # E2M1 round-to-nearest-even boundaries.  The alternating strict/inclusive
    # comparisons encode the hardware tie choices without an N*K int64
    # bucketize buffer: .25 down, .75 up, 1.25 down, 1.75 up, etc.
    magnitude = scaled.abs()
    magnitude_code = (magnitude > 0.25).to(torch.uint8)
    magnitude_code.add_((magnitude >= 0.75).to(torch.uint8))
    magnitude_code.add_((magnitude > 1.25).to(torch.uint8))
    magnitude_code.add_((magnitude >= 1.75).to(torch.uint8))
    magnitude_code.add_((magnitude > 2.5).to(torch.uint8))
    magnitude_code.add_((magnitude >= 3.5).to(torch.uint8))
    magnitude_code.add_((magnitude > 5.0).to(torch.uint8))
    sign_code = (scaled < 0).to(torch.uint8) << 3
    codes = (magnitude_code | sign_code).reshape(rows, columns)
    packed_u8 = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed_u8, scale_u8


def _validate_mxfp4_sample_roundtrip(
    reference,
    packed_u8,
    scale_u8,
    torch,
    label: str,
) -> None:
    """Catch packed-nibble, E8M0, or op-ABI mismatches on a small row sample."""
    sample_rows = min(8, reference.shape[0])
    reference = reference[:sample_rows].to(torch.float32)
    packed_u8 = packed_u8[:sample_rows]
    scale_u8 = scale_u8[:sample_rows]
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed_u8.device,
    )
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(sample_rows, -1)
    values = lut[codes.to(torch.long)]
    scales = torch.exp2(scale_u8.to(torch.float32) - 127.0).repeat_interleave(
        MXFP4_GROUP_SIZE, dim=-1
    )
    decoded = values * scales
    if decoded.shape != reference.shape or not torch.isfinite(decoded).all():
        raise ValueError(
            f"MXFP4 roundtrip produced invalid values for {label}: "
            f"decoded={tuple(decoded.shape)}, reference={tuple(reference.shape)}"
        )
    error_rms = torch.mean((decoded - reference).square()).sqrt()
    reference_rms = torch.mean(reference.square()).sqrt().clamp_min(1.0e-12)
    relative_rmse = float((error_rms / reference_rms).item())
    if relative_rmse > MXFP4_ROUNDTRIP_MAX_REL_RMSE:
        raise ValueError(
            f"MXFP4 roundtrip check failed for {label}: relative RMSE "
            f"{relative_rmse:.6f} > {MXFP4_ROUNDTRIP_MAX_REL_RMSE:.2f}. "
            "Check the MXFP4 quantizer output, nibble order, and "
            "linear UE8M0 scale layout."
        )
    print(
        f"[mxfp4] roundtrip OK for {label}: relative_rmse={relative_rmse:.6f}",
        flush=True,
    )


def _quantize_expert_weight(
    weight,
    block_scale,
    record: ExpertRecord,
    torch,
    fp4_quantize,
):
    rows, columns = record.weight_output.shape[0], record.weight_output.shape[1] * 2
    if tuple(weight.shape) != (rows, columns):
        raise ValueError(
            f"Loaded shape mismatch for {record.source_weight_name}: "
            f"{tuple(weight.shape)} vs {(rows, columns)}"
        )
    expected_scale_shape = (
        rows // FP8_BLOCK_SIZE,
        columns // FP8_BLOCK_SIZE,
    )
    if tuple(block_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"Loaded scale shape mismatch for {record.source_scale_name}: "
            f"{tuple(block_scale.shape)} vs {expected_scale_shape}"
        )

    weight_cuda = weight.to(device="cuda", non_blocking=False)
    scale_cuda = block_scale.to(device="cuda", dtype=torch.float32)
    # [N, K] -> [N/128, 128, K/128, 128], so every FP8 block gets the
    # corresponding two-dimensional inverse/dequantization scale.
    dequantized = (
        weight_cuda.to(torch.float32).reshape(
            rows // FP8_BLOCK_SIZE,
            FP8_BLOCK_SIZE,
            columns // FP8_BLOCK_SIZE,
            FP8_BLOCK_SIZE,
        )
        * scale_cuda[:, None, :, None]
    ).reshape(rows, columns)
    dequantized = dequantized.to(torch.bfloat16).contiguous()

    # Contract required by the SM90 MXFP4xFP8 path:
    #   global_scale=None, group=32, UE8M0=True, swizzled=False (linear).
    # TRT-LLM's op only performs the conversion on SM100+, despite registering
    # on SM90, so Hopper must use the CUDA-compatible PyTorch implementation.
    if fp4_quantize is not None and torch.cuda.get_device_capability()[0] >= 10:
        if not bool(torch.isfinite(dequantized).all().item()):
            raise ValueError(
                f"MXFP4 quantization input contains NaN or Inf for "
                f"{record.source_weight_name}"
            )
        packed, raw_scale = fp4_quantize(
            dequantized,
            None,
            MXFP4_GROUP_SIZE,
            True,
            False,
        )
        packed_u8 = _as_raw_u8(
            packed, torch, record.weight_output.shape, record.source_weight_name
        )
        scale_u8 = _as_raw_u8(
            raw_scale, torch, record.scale_output.shape, record.source_scale_name
        )
    else:
        global _REFERENCE_QUANTIZER_LOGGED
        if not _REFERENCE_QUANTIZER_LOGGED:
            print(
                "[mxfp4] using the CUDA-compatible PyTorch quantizer "
                "(TensorRT-LLM FP4 conversion requires SM100+)",
                flush=True,
            )
            _REFERENCE_QUANTIZER_LOGGED = True
        packed_u8, scale_u8 = _quantize_mxfp4_cuda_compatible(dequantized, torch)
    logical_shape = (rows, columns)
    if logical_shape not in _ROUNDTRIP_VALIDATED_SHAPES:
        _validate_mxfp4_sample_roundtrip(
            dequantized,
            packed_u8,
            scale_u8,
            torch,
            record.source_weight_name,
        )
        _ROUNDTRIP_VALIDATED_SHAPES.add(logical_shape)
    return packed_u8.cpu(), scale_u8.cpu()


def _write_tensor_bytes(handle, tensor, expected_nbytes: int, torch, name: str) -> None:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    if raw.numel() != expected_nbytes:
        raise ValueError(
            f"Serialized byte count mismatch for {name}: "
            f"{raw.numel()} vs {expected_nbytes}"
        )
    before = handle.tell()
    raw.numpy().tofile(handle)
    written = handle.tell() - before
    if written != expected_nbytes:
        raise IOError(f"Short write for {name}: {written} vs {expected_nbytes}")


def _records_by_shard(records: Iterable[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = {}
    for record in records:
        grouped.setdefault(record.source_shard, []).append(record)
    return grouped


def _write_rank_shard(
    model_dir: Path,
    output_dir: Path,
    rank: int,
    parts: int,
    records: Sequence[Record],
) -> None:
    torch, safe_open, fp4_quantize = _load_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MXFP4 quantization")

    outputs = [output for record in records for output in record.outputs]
    header, expected_data_size = _safetensors_header(outputs)
    output_path = output_dir / _output_shard_name(rank, parts)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    grouped = _records_by_shard(records)
    processed = 0
    try:
        # NumPy's tofile writes through the file descriptor, so keep this
        # stream unbuffered to make its descriptor position match tell().
        with temporary.open("wb", buffering=0) as destination:
            destination.write(struct.pack("<Q", len(header)))
            destination.write(header)
            data_start = destination.tell()

            for shard_name in sorted(grouped):
                source_path = model_dir / shard_name
                with safe_open(
                    str(source_path), framework="pt", device="cpu"
                ) as source:
                    for record in grouped[shard_name]:
                        if isinstance(record, CopyRecord):
                            tensor = source.get_tensor(record.source_name)
                            _write_tensor_bytes(
                                destination,
                                tensor,
                                record.output.nbytes,
                                torch,
                                record.output.name,
                            )
                            del tensor
                        else:
                            weight = source.get_tensor(record.source_weight_name)
                            block_scale = source.get_tensor(record.source_scale_name)
                            packed, raw_scale = _quantize_expert_weight(
                                weight,
                                block_scale,
                                record,
                                torch,
                                fp4_quantize,
                            )
                            _write_tensor_bytes(
                                destination,
                                packed,
                                record.weight_output.nbytes,
                                torch,
                                record.weight_output.name,
                            )
                            _write_tensor_bytes(
                                destination,
                                raw_scale,
                                record.scale_output.nbytes,
                                torch,
                                record.scale_output.name,
                            )
                            del weight, block_scale, packed, raw_scale

                        processed += 1
                        if processed % 100 == 0:
                            print(
                                f"[mxfp4 rank {rank}] processed "
                                f"{processed}/{len(records)} tensors",
                                flush=True,
                            )

            actual_data_size = destination.tell() - data_start
            if actual_data_size != expected_data_size:
                raise IOError(
                    f"Output data size mismatch: {actual_data_size} vs "
                    f"{expected_data_size}"
                )
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.parts < 1:
        raise SystemExit(f"--parts must be positive, got {args.parts}")
    if args.rank < 0 or args.rank >= args.parts:
        raise SystemExit(
            f"--rank must be in [0, {args.parts}), got {args.rank}"
        )

    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if model_dir == output_dir:
        raise SystemExit("--output_dir must differ from --model_dir")
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_dir}")

    print(f"[mxfp4 rank {args.rank}] planning conversion from {model_dir}", flush=True)
    plan = build_plan(model_dir, args.parts)
    rank_records = tuple(record for record in plan.records if record.rank == args.rank)
    rank_outputs = [output for record in rank_records for output in record.outputs]
    rank_size = sum(output.nbytes for output in rank_outputs)
    expert_count = sum(isinstance(record, ExpertRecord) for record in rank_records)
    print(
        f"[mxfp4 rank {args.rank}] records={len(rank_records)} "
        f"expert_weights={expert_count} output_tensors={len(rank_outputs)} "
        f"bytes={rank_size} total_bytes={plan.total_size}",
        flush=True,
    )
    if args.plan_only:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.rank == 0:
        _write_auxiliary_files(model_dir, output_dir, plan)
    _write_rank_shard(
        model_dir,
        output_dir,
        args.rank,
        args.parts,
        rank_records,
    )
    print(
        f"[mxfp4 rank {args.rank}] wrote "
        f"{output_dir / _output_shard_name(args.rank, args.parts)}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # keep per-rank log failures concise and visible
        print(f"[mxfp4] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
