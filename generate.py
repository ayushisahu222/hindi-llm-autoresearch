"""
Generate Hindi text from a trained autoresearch-mlx model.

Usage:
    uv run generate.py                                         # interactive mode
    uv run generate.py --prompt "भारत एक"                    # single prompt
    uv run generate.py --prompt "एक बार की बात है" --tokens 200 --temp 0.8
"""

import argparse
import json
import math
import os
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from prepare import Tokenizer

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
MODEL_PATH = os.path.join(CACHE_DIR, "model.npz")
CONFIG_PATH = os.path.join(CACHE_DIR, "config.json")


# --- Model definition (copied from train.py to avoid triggering training on import) ---

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def create_additive_causal_mask(seq_len, dtype=mx.float32):
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def create_sliding_window_mask(seq_len, window_size, dtype=mx.float32):
    indices = mx.arange(seq_len)
    causal = indices[None, :] > indices[:, None]
    too_far = (indices[:, None] - indices[None, :]) >= window_size
    blocked = causal | too_far
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x, ve, mask):
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).reshape(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = norm(self.rope(q))
        k = norm(self.rope(k))
        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        return self.c_proj(mx.maximum(self.c_fc(x), 0) ** 2)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, mask):
        x = x + self.attn(norm(x), ve, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones((config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.zeros((config.n_layer,), dtype=mx.float32)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        }
        self._mask_cache = {}

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = long_window
        return sizes

    def _get_masks(self, seq_len):
        for window_size in set(self.window_sizes):
            key = (seq_len, window_size)
            if key not in self._mask_cache:
                if window_size >= seq_len:
                    self._mask_cache[key] = create_additive_causal_mask(seq_len)
                else:
                    self._mask_cache[key] = create_sliding_window_mask(seq_len, window_size)
        return [self._mask_cache[(seq_len, w)] for w in self.window_sizes]

    def __call__(self, idx):
        _, seq_len = idx.shape
        masks = self._get_masks(seq_len)
        x = norm(self.wte(idx))
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, masks[i])
        logits = 15.0 * mx.tanh(self.lm_head(norm(x)).astype(mx.float32) / 15.0)
        return logits


# --- Inference ---

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No saved model found at {MODEL_PATH}")
        print("Run `uv run train.py` first to train and save the model.")
        raise SystemExit(1)

    with open(CONFIG_PATH) as f:
        config = GPTConfig(**json.load(f))

    model = GPT(config)

    # Custom loader: MLX's load_weights mishandles string-integer dict keys
    # (value_embeds uses {"0": ..., "2": ...}) and the old npz may contain
    # _mask_cache entries. Navigate using the model's own structure so lists
    # get int indices and dicts get string keys. Skip unknown paths.
    from mlx.utils import tree_flatten

    flat_weights = dict(mx.load(MODEL_PATH))
    valid_paths = {path for path, _ in tree_flatten(model.parameters())}

    def set_param(path, value):
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, list):
            obj[int(last)] = value
        elif isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)

    for path, val in flat_weights.items():
        if path in valid_paths:
            set_param(path, val)

    mx.eval(model.parameters())
    model.eval()
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Loaded model ({n_params / 1e6:.1f}M params, vocab={config.vocab_size})")
    return model, config


def generate(model, tokenizer, prompt, max_new_tokens=150, temperature=0.8, top_k=50):
    ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    x = mx.array([ids], dtype=mx.int32)

    print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        x_crop = x[:, -model.config.sequence_len:]
        logits = model(x_crop)
        next_logits = logits[0, -1, :] / temperature

        if top_k > 0:
            threshold = mx.sort(next_logits)[-top_k]
            next_logits = mx.where(next_logits < threshold, mx.array(float("-inf")), next_logits)

        next_token = int(mx.random.categorical(mx.log(mx.softmax(next_logits))).item())
        x = mx.concatenate([x, mx.array([[next_token]], dtype=mx.int32)], axis=1)
        print(tokenizer.decode([next_token]), end="", flush=True)

    print()


def main():
    parser = argparse.ArgumentParser(description="Generate Hindi text from trained model")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--tokens", type=int, default=150, help="Tokens to generate (default: 150)")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (default: 50)")
    args = parser.parse_args()

    model, _ = load_model()
    tokenizer = Tokenizer.from_directory()

    if args.prompt:
        print("\n--- Generated output ---")
        generate(model, tokenizer, args.prompt, args.tokens, args.temp, args.top_k)
    else:
        print("\nInteractive mode. Type a Hindi prompt and press Enter. Ctrl+C to quit.\n")
        while True:
            try:
                prompt = input("Prompt> ").strip()
                if not prompt:
                    continue
                print("\n--- Generated output ---")
                generate(model, tokenizer, prompt, args.tokens, args.temp, args.top_k)
                print()
            except KeyboardInterrupt:
                print("\nBye!")
                break


if __name__ == "__main__":
    main()
