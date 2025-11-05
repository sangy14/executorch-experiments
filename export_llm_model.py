#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple, Any, Dict

import torch
from torch.export import Dim
import torch.nn as nn

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from transformers import AutoTokenizer, AutoModelForCausalLM


# Wrapper that simplifies model output for ExecuTorch export.
class ExecuTorchWrapper(nn.Module):
    def __init__(self, model: nn.Module, use_cache: bool = False):
        super().__init__()
        self.model = model
        self.model.eval()
        self.use_cache = use_cache

    def forward(self, input_ids: torch.Tensor):
        outputs = self.model(input_ids=input_ids, return_dict=False, use_cache=self.use_cache)
        return outputs[0]


def load_tokenizer(model_name: str):
    print("[1/6] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    vocab_size = getattr(tok, "vocab_size", None)
    print(f"[OK] Tokenizer loaded (vocab_size={vocab_size})")
    return tok


def load_model(model_name: str, dtype=torch.float32, device_map="cpu", use_cache: bool = False) -> Tuple[nn.Module, float]:
    print("[2/6] Loading model...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    # Ensure the model config matches the requested use_cache behavior
    try:
        model.config.use_cache = use_cache
    except Exception:
        pass
    model.eval()
    elapsed = time.time() - start
    print(f"[OK] Model loaded in {elapsed:.2f}s (use_cache={use_cache})")
    return model, elapsed


def prepare_wrapper(model: nn.Module, vocab_size: int, test_shape=(1, 128), use_cache: bool = False):
    print("[3/6] Wrapping model and running a test inference...")
    wrapped = ExecuTorchWrapper(model, use_cache=use_cache)
    test_input = torch.randint(0, vocab_size, test_shape)
    with torch.no_grad():
        out = wrapped(test_input)
    print(f"[OK] Wrapper test passed: output shape {out.shape}")
    return wrapped, test_input


def export_with_dynamic_shapes(model: nn.Module, example_inputs, seq_max: int, use_batch_dim: bool = False):
    print("[4/6] Exporting with dynamic shapes...")
    batch_dim = Dim("batch_size")
    seq_dim = Dim("seq_len", max=seq_max)
    dynamic_shapes: Dict[str, Dict[int, Any]]
    if use_batch_dim:
        dynamic_shapes = {"input_ids": {0: batch_dim, 1: seq_dim}}
    else:
        dynamic_shapes = {"input_ids": {1: seq_dim}}

    start = time.time()
    with torch._dynamo.config.patch(fake_tensor_cache_enabled=False):
        exported = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    elapsed = time.time() - start
    print(f"[OK] Model exported in {elapsed:.2f}s")
    return exported, elapsed


def convert_to_executorch(exported_program):
    print("[5/6] Converting to ExecuTorch format...")
    start = time.time()
    program = to_edge_transform_and_lower(exported_program, partitioner=[XnnpackPartitioner()]).to_executorch()
    elapsed = time.time() - start
    print(f"[OK] Converted to ExecuTorch in {elapsed:.2f}s")
    return program, elapsed


def save_program(program, out_path: Path):
    print("[6/6] Saving model...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(program.buffer)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Model saved to '{out_path}' ({size_mb:.2f} MB)")
    return size_mb


def parse_args():
    p = argparse.ArgumentParser(description="Export HuggingFace causal LM to ExecuTorch (.pte)")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--seq-max", type=int, default=128)
    p.add_argument("--use-batch-dim", action="store_true", help="Make batch dimension dynamic as well")
    p.add_argument("--use-cache", action="store_true", default=True, help="Set model.use_cache to True during forward (default: False)")
    return p.parse_args()


def sanitize_model_name_for_filename(model_name: str) -> str:
    # Lowercase and replace characters that are problematic in filenames
    sanitized = model_name.lower().replace("/", "_").replace(":", "_").replace(" ", "_")
    return sanitized


def main():
    args = parse_args()
    model_name = args.model

    # Derive output filename from lower-case, sanitized model name (no user-provided output)
    filename = f"{sanitize_model_name_for_filename(model_name)}.pte"
    out_path = Path(filename)

    tokenizer = load_tokenizer(model_name)
    model, load_time = load_model(model_name, use_cache=args.use_cache)
    vocab_size = getattr(tokenizer, "vocab_size", 32000)

    wrapped, test_input = prepare_wrapper(model, vocab_size, test_shape=(1, args.seq_max), use_cache=args.use_cache)
    example_inputs = (test_input,)

    exported, export_time = export_with_dynamic_shapes(wrapped, example_inputs, seq_max=args.seq_max, use_batch_dim=args.use_batch_dim)
    program, convert_time = convert_to_executorch(exported)
    file_size = save_program(program, out_path)

    summary = {
        "model": model_name,
        "load_time_s": load_time,
        "export_time_s": export_time,
        "convert_time_s": convert_time,
        "output_file": str(out_path),
        "file_size_mb": file_size,
        "dynamic_shapes": {
            "seq_len_max": args.seq_max,
            "batch_dynamic": args.use_batch_dim,
        },
        "use_cache": args.use_cache,
    }
    print("\n=== Export Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
