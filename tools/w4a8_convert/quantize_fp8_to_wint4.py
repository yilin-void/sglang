#!/usr/bin/env python3
"""Stream a block-FP8 MoE checkpoint into a W4AFP8 (INT4) expert checkpoint.

Forked from the DSV4 FP8->MXFP4 weight-only converter, changed to produce the
INT4 checkpoint contract that sglang's `w4afp8` (W4AFp8Config, cutlass_w4a8_moe)
loads. Only routed expert gate_proj/up_proj/down_proj weights are converted:

    F8_E4M3 weight [N, K] + 128x128 block weight_scale_inv (F32 or BF16)
        -> dequantize to BF16
        -> per-row, symmetric INT4 with group_size=128 along K
        -> packed int4 weight  I8   [N, K/2]   (low nibble = even index)
        -> group scale F32     [N, K/128]       (dequant multiplier = amax/7)

Activation is quantized to FP8 at runtime (weight-only checkpoint; no input
scales are written -> loader defaults to 1.0 static, i.e. dynamic-ish). Scale
interleaving that sglang does in `process_weights_after_loading` is NOT
serialized: the on-disk scale is the logical [N, K/128] layout.

All non-routed-expert tensors are copied byte-for-byte (attention, linear-attn,
shared experts, router gates, embeddings, norms, lm_head stay in their source
dtype -- typically block-FP8 / bf16).
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
INT4_GROUP_SIZE = 128
INT4_ROUNDTRIP_MAX_REL_RMSE = 0.15
_ROUNDTRIP_VALIDATED_SHAPES: set[tuple[int, int]] = set()
EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj))\.weight$"
)
_SRC_SCALE_DTYPES = {"F32", "BF16"}


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class OutputSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class CopyRecord:
    rank: int
    source_shard: str
    source_name: str
    output: OutputSpec

    @property
    def outputs(self) -> tuple[OutputSpec, ...]:
        return (self.output,)


@dataclass(frozen=True)
class ExpertRecord:
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
    parser.add_argument("--plan-only", action="store_true")
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
            raise ValueError(f"Unreasonably large header ({header_length}): {path}")
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
        raise ValueError(f"Invalid tensor header for {name} in {shard}: {entry}") from exc
    if any(dim < 0 for dim in shape) or start < 0 or end < start:
        raise ValueError(f"Invalid tensor shape/offset for {name} in {shard}")
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
        weight_map = {str(n): str(s) for n, s in raw_weight_map.items()}
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_paths = sorted(model_dir.glob("*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(f"No safetensors under {model_dir}")
        shard_names = [p.name for p in shard_paths]
        weight_map = {}

    specs: dict[str, TensorSpec] = {}
    discovered: dict[str, str] = {}
    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Source shard missing: {shard_path}")
        header = _read_safetensors_header(shard_path)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in specs:
                raise ValueError(f"Tensor {name} in more than one shard")
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid entry for {name} in {shard_path}")
            specs[name] = _tensor_spec(entry, name, shard_name)
            discovered[name] = shard_name

    if weight_map:
        if set(weight_map).difference(specs) or set(specs).difference(weight_map):
            raise ValueError("Source index/header mismatch")
        for name, shard_name in weight_map.items():
            if discovered[name] != shard_name:
                raise ValueError(f"Index maps {name} to {shard_name} but it is in {discovered[name]}")
    else:
        weight_map = discovered
    return weight_map, specs, shard_names


def _output_shard_name(rank: int, parts: int) -> str:
    return f"model-{rank:05d}-of-{parts:05d}.safetensors"


def _expert_layer_prefix(weight_name: str) -> str:
    before, _ = weight_name.split(".experts.", 1)
    return f"{before}.experts"


def build_plan(model_dir: Path, parts: int) -> ConversionPlan:
    if parts < 1:
        raise ValueError(f"parts must be positive, got {parts}")
    source_weight_map, source_specs, source_shards = _discover_source(model_dir)
    shard_ordinals = {n: i for i, n in enumerate(source_shards)}
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
        raise ValueError("No routed expert FP8 weights found")

    records: list[Record] = []
    output_weight_map: dict[str, str] = {}
    expert_layer_prefixes: set[str] = set()
    total_size = 0

    ordered = sorted(source_specs, key=lambda n: (shard_ordinals[source_weight_map[n]], n))
    for name in ordered:
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
                raise ValueError(f"Expert weight must be 2D F8_E4M3: {name} {source_spec}")
            rows, columns = source_spec.shape
            if rows % FP8_BLOCK_SIZE or columns % FP8_BLOCK_SIZE:
                raise ValueError(f"Expert shape must be /{FP8_BLOCK_SIZE}: {name} {source_spec.shape}")
            if columns % INT4_GROUP_SIZE:
                raise ValueError(f"Expert K must be /{INT4_GROUP_SIZE}: {name} K={columns}")
            if columns % 2:
                raise ValueError(f"Expert K must be even for int4 packing: {name}")

            source_scale_name = f"{match.group('prefix')}.weight_scale_inv"
            sss = source_specs[source_scale_name]
            expected = (rows // FP8_BLOCK_SIZE, columns // FP8_BLOCK_SIZE)
            if sss.dtype not in _SRC_SCALE_DTYPES or sss.shape != expected:
                raise ValueError(
                    f"FP8 block scale {source_scale_name} must be {_SRC_SCALE_DTYPES} "
                    f"{expected}; got dtype={sss.dtype} shape={sss.shape}"
                )
            if source_weight_map[source_scale_name] != source_shard:
                raise ValueError(f"Weight/scale must share shard: {name}")

            weight_output = OutputSpec(
                name=name, dtype="I8",
                shape=(rows, columns // 2), nbytes=rows * (columns // 2),
            )
            scale_output = OutputSpec(
                name=f"{match.group('prefix')}.weight_scale_inv", dtype="F32",
                shape=(rows, columns // INT4_GROUP_SIZE),
                nbytes=rows * (columns // INT4_GROUP_SIZE) * 4,
            )
            record: Record = ExpertRecord(
                rank=rank, source_shard=source_shard,
                source_weight_name=name, source_scale_name=source_scale_name,
                weight_output=weight_output, scale_output=scale_output,
            )
            expert_layer_prefixes.add(_expert_layer_prefix(name))
        else:
            output = OutputSpec(
                name=name, dtype=source_spec.dtype,
                shape=source_spec.shape, nbytes=source_spec.nbytes,
            )
            record = CopyRecord(rank=rank, source_shard=source_shard, source_name=name, output=output)

        for output in record.outputs:
            if output.name in output_weight_map:
                raise ValueError(f"Duplicate output tensor: {output.name}")
            output_weight_map[output.name] = output_shard
            total_size += output.nbytes
        records.append(record)

    return ConversionPlan(
        records=tuple(records), output_weight_map=output_weight_map,
        total_size=total_size, source_specs=source_specs,
        expert_layer_prefixes=tuple(sorted(expert_layer_prefixes)),
    )


def _write_auxiliary_files(model_dir: Path, output_dir: Path, plan: ConversionPlan) -> None:
    excluded = {".git", "config.json", "hf_quant_config.json",
                "model.safetensors.index.json", "quant_cfg.json"}
    for source in sorted(model_dir.iterdir()):
        if not source.is_file():
            continue
        if source.suffix == ".safetensors" or source.name in excluded:
            continue
        shutil.copy2(source, output_dir / source.name)

    config = _read_json(model_dir / "config.json")
    qc = config.get("quantization_config")
    qc = dict(qc) if isinstance(qc, dict) else {}
    # keep block-fp8 params for the non-expert (copied) tensors; switch method
    # to w4afp8 and declare INT4 routed-expert format for sglang.
    qc.update({
        "quant_method": "w4afp8",
        "moe_weight_format": "int4",
        "linear_weight_format": "fp8",
        "group_size": INT4_GROUP_SIZE,
        "moe_activation_scheme": "dynamic",
        "linear_activation_scheme": "dynamic",
    })
    config["quantization_config"] = qc
    _atomic_write_json(output_dir / "config.json", config)

    _atomic_write_json(output_dir / "model.safetensors.index.json", {
        "metadata": {"total_size": plan.total_size},
        "weight_map": {n: plan.output_weight_map[n] for n in sorted(plan.output_weight_map)},
    })


def _safetensors_header(outputs: Iterable[OutputSpec]) -> tuple[bytes, int]:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for output in outputs:
        header[output.name] = {
            "dtype": output.dtype, "shape": list(output.shape),
            "data_offsets": [offset, offset + output.nbytes],
        }
        offset += output.nbytes
    payload = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((8 - len(payload) % 8) % 8)
    return payload, offset


def _load_runtime():
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("INT4 conversion requires torch and safetensors") from exc
    return torch, safe_open


def _quantize_wint4_symmetric(weight, torch):
    """[N, K] bf16/f32 -> symmetric per-row group-128 INT4.

    Returns (packed_i8 [N, K/2], scale_f32 [N, K/128]). Packing matches
    sgl-kernel cutlass_w4a8 test `pack_int4_values_to_int8`:
    low nibble = even column, high nibble = odd column, signed 4-bit.
    scale = group amax / 7 (dequant multiplier, stored as weight_scale_inv).
    """
    rows, columns = weight.shape
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("INT4 quantization input contains NaN or Inf")
    w = weight.to(torch.float32).reshape(rows, columns // INT4_GROUP_SIZE, INT4_GROUP_SIZE)
    amax = w.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 7.0).clamp_min(1e-12)
    q = torch.round(w / scale).clamp_(-8, 7).to(torch.int8).reshape(rows, columns)
    low = q[:, 0::2]
    high = q[:, 1::2]
    packed = ((high << 4) | (low & 0x0F)).to(torch.int8).contiguous()
    scale_out = scale.reshape(rows, columns // INT4_GROUP_SIZE).to(torch.float32).contiguous()
    return packed, scale_out


def _validate_wint4_roundtrip(reference, packed_i8, scale_f32, torch, label: str) -> None:
    sample = min(8, reference.shape[0])
    ref = reference[:sample].to(torch.float32)
    packed = packed_i8[:sample]
    scale = scale_f32[:sample]
    # unpack signed int4: low nibble even col, high nibble odd col
    low = (packed & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    high = torch.where(high >= 8, high - 16, high)
    codes = torch.stack((low, high), dim=-1).reshape(sample, -1).to(torch.float32)
    scales = scale.repeat_interleave(INT4_GROUP_SIZE, dim=-1)
    decoded = codes * scales
    if decoded.shape != ref.shape or not torch.isfinite(decoded).all():
        raise ValueError(f"INT4 roundtrip invalid for {label}")
    err = torch.mean((decoded - ref).square()).sqrt()
    ref_rms = torch.mean(ref.square()).sqrt().clamp_min(1e-12)
    rel = float((err / ref_rms).item())
    if rel > INT4_ROUNDTRIP_MAX_REL_RMSE:
        raise ValueError(
            f"INT4 roundtrip failed for {label}: rel RMSE {rel:.4f} > "
            f"{INT4_ROUNDTRIP_MAX_REL_RMSE}. Check nibble order / scale."
        )
    print(f"[wint4] roundtrip OK for {label}: rel_rmse={rel:.6f}", flush=True)


def _quantize_expert_weight(weight, block_scale, record: ExpertRecord, torch):
    rows = record.weight_output.shape[0]
    columns = record.weight_output.shape[1] * 2
    if tuple(weight.shape) != (rows, columns):
        raise ValueError(f"Shape mismatch {record.source_weight_name}: {tuple(weight.shape)} vs {(rows, columns)}")
    expected = (rows // FP8_BLOCK_SIZE, columns // FP8_BLOCK_SIZE)
    if tuple(block_scale.shape) != expected:
        raise ValueError(f"Scale shape mismatch {record.source_scale_name}: {tuple(block_scale.shape)} vs {expected}")

    weight_cuda = weight.to(device="cuda", non_blocking=False)
    scale_cuda = block_scale.to(device="cuda", dtype=torch.float32)
    dequant = (
        weight_cuda.to(torch.float32).reshape(
            rows // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE,
            columns // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE,
        ) * scale_cuda[:, None, :, None]
    ).reshape(rows, columns).to(torch.bfloat16).contiguous()

    packed, scale_out = _quantize_wint4_symmetric(dequant, torch)
    logical = (rows, columns)
    if logical not in _ROUNDTRIP_VALIDATED_SHAPES:
        _validate_wint4_roundtrip(dequant, packed, scale_out, torch, record.source_weight_name)
        _ROUNDTRIP_VALIDATED_SHAPES.add(logical)
    return packed.cpu(), scale_out.cpu()


def _write_tensor_bytes(handle, tensor, expected_nbytes: int, torch, name: str) -> None:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    if raw.numel() != expected_nbytes:
        raise ValueError(f"Byte count mismatch {name}: {raw.numel()} vs {expected_nbytes}")
    before = handle.tell()
    raw.numpy().tofile(handle)
    if handle.tell() - before != expected_nbytes:
        raise IOError(f"Short write for {name}")


def _records_by_shard(records: Iterable[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = {}
    for record in records:
        grouped.setdefault(record.source_shard, []).append(record)
    return grouped


def _write_rank_shard(model_dir, output_dir, rank, parts, records) -> None:
    torch, safe_open = _load_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for INT4 quantization")
    outputs = [o for r in records for o in r.outputs]
    header, expected_data = _safetensors_header(outputs)
    output_path = output_dir / _output_shard_name(rank, parts)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    grouped = _records_by_shard(records)
    processed = 0
    try:
        with temporary.open("wb", buffering=0) as dst:
            dst.write(struct.pack("<Q", len(header)))
            dst.write(header)
            data_start = dst.tell()
            for shard_name in sorted(grouped):
                with safe_open(str(model_dir / shard_name), framework="pt", device="cpu") as src:
                    for record in grouped[shard_name]:
                        if isinstance(record, CopyRecord):
                            t = src.get_tensor(record.source_name)
                            _write_tensor_bytes(dst, t, record.output.nbytes, torch, record.output.name)
                            del t
                        else:
                            w = src.get_tensor(record.source_weight_name)
                            bs = src.get_tensor(record.source_scale_name)
                            packed, scale_out = _quantize_expert_weight(w, bs, record, torch)
                            _write_tensor_bytes(dst, packed, record.weight_output.nbytes, torch, record.weight_output.name)
                            _write_tensor_bytes(dst, scale_out, record.scale_output.nbytes, torch, record.scale_output.name)
                            del w, bs, packed, scale_out
                        processed += 1
                        if processed % 200 == 0:
                            print(f"[wint4 rank {rank}] {processed}/{len(records)} tensors", flush=True)
            if dst.tell() - data_start != expected_data:
                raise IOError(f"Output data size mismatch: {dst.tell()-data_start} vs {expected_data}")
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.parts < 1:
        raise SystemExit(f"--parts must be positive, got {args.parts}")
    if args.rank < 0 or args.rank >= args.parts:
        raise SystemExit(f"--rank must be in [0, {args.parts})")
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if model_dir == output_dir:
        raise SystemExit("--output_dir must differ from --model_dir")
    if not model_dir.is_dir():
        raise SystemExit(f"Model dir does not exist: {model_dir}")

    print(f"[wint4 rank {args.rank}] planning from {model_dir}", flush=True)
    plan = build_plan(model_dir, args.parts)
    rank_records = tuple(r for r in plan.records if r.rank == args.rank)
    rank_outputs = [o for r in rank_records for o in r.outputs]
    experts = sum(isinstance(r, ExpertRecord) for r in rank_records)
    print(
        f"[wint4 rank {args.rank}] records={len(rank_records)} expert_weights={experts} "
        f"outputs={len(rank_outputs)} bytes={sum(o.nbytes for o in rank_outputs)} "
        f"total_bytes={plan.total_size}", flush=True,
    )
    if args.plan_only:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.rank == 0:
        _write_auxiliary_files(model_dir, output_dir, plan)
    _write_rank_shard(model_dir, output_dir, args.rank, args.parts, rank_records)
    print(f"[wint4 rank {args.rank}] wrote {output_dir / _output_shard_name(args.rank, args.parts)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[wint4] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
