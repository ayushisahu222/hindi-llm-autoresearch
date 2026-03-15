# hindi-llm-autoresearch

An autonomous LLM research loop for Hindi language modeling on Apple Silicon, built on MLX. The AI runs experiments, evaluates results, and iterates — all while you sleep.

**Based on the incredible work by:**
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx)

---

## What it does

Trains a small GPT-style language model on Hindi text ([ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)) using a fixed 5-minute wall-clock budget per experiment. The metric is **bits per byte (val_bpb)** — lower is better and comparable across vocab sizes.

The experiment loop is designed to be fully autonomous: make a change to `train.py`, run it, record the result, keep or discard, repeat. An AI agent can run this loop indefinitely while you're away.

---

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10–3.13
- [uv](https://github.com/astral-sh/uv)

## License

MIT
