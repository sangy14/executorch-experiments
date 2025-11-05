from pathlib import Path
from typing import List, Optional
import argparse

import torch
from transformers import AutoTokenizer
from executorch.runtime import Runtime


def load_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def load_executorch_forward(model_path: Path):
    rt = Runtime.get()
    prog = rt.load_program(model_path)
    return prog.load_method("forward")


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """
    logits: 1D tensor (vocab,)
    temperature: if 0 -> greedy, else softmax temperature sampling
    returns token id (int)
    """
    if temperature == 0.0:
        return int(torch.argmax(logits).item())
    logits = logits / max(1e-8, temperature)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate(
    prompt: str,
    tokenizer_name: str,
    model_path: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7
) -> str:
    
    tokenizer = load_tokenizer(tokenizer_name)
    forward = load_executorch_forward(Path(model_path))

    # Prepare input tensor
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    generated_ids: List[int] = []
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # ExecuTorch forward returns logits as [batch, seq_len, vocab]
            logits = forward.execute((input_ids,))[0]  # keep as tensor on device
            last_logits = logits[0, -1, :]

            next_id = sample_next_token(last_logits, temperature=temperature)
            generated_ids.append(next_id)

            # stop if EOS
            if eos_id is not None and next_id == eos_id:
                break

            # append next token and continue
            next_id_tensor = torch.tensor([[next_id]], dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, next_id_tensor], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate text using an ExecuTorch model")
    parser.add_argument("--prompt", "-p", type=str, default='Explain what artificial intelligence is', help="Prompt text")
    parser.add_argument("--tokenizer-name", "-t", type=str, default='Qwen/Qwen2.5-1.5B-Instruct', help="Tokenizer name or path")
    parser.add_argument("--max-new-tokens", "-n", type=int, default=90, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 for greedy)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # derive model filename from tokenizer name: downcase and replace slashes with underscores
    derived_model_filename = args.tokenizer_name.lower().replace("/", "_") + ".pte"

    text = generate(
        args.prompt,
        args.tokenizer_name,
        derived_model_filename,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print("Generated:", text)
