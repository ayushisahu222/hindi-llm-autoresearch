"""
Download Hindi data from ai4bharat/sangraha for autoresearch-mlx.
Uses only `requests` (already in deps — no huggingface_hub needed).

Usage:
    uv run download_hindi.py              # download 4 shards (~1.4 GB, good for MacBook)
    uv run download_hindi.py --shards 6  # download all 6 shards (~2.1 GB)
"""

import argparse
import os
import time

import requests

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# Verified Hindi shards from ai4bharat/sangraha (~360 MB each, 6 total)
HF_BASE = "https://huggingface.co/datasets/ai4bharat/sangraha/resolve/refs%2Fconvert%2Fparquet/verified/partial-hin"
TOTAL_SHARDS = 6


def download_shard(shard_index, local_index, total):
    """Download one parquet shard with retries."""
    remote_name = f"{shard_index:04d}.parquet"
    local_name = f"shard_{local_index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, local_name)

    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  [{local_index+1}/{total}] {local_name} already exists ({size_mb:.0f} MB), skipping.")
        return True

    url = f"{HF_BASE}/{remote_name}"
    print(f"  [{local_index+1}/{total}] Downloading {remote_name} -> {local_name} (~360 MB)...")

    for attempt in range(1, 6):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            temp_path = filepath + ".tmp"
            downloaded = 0
            with open(temp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        print(f"\r    {downloaded / 1024 / 1024:.0f} MB downloaded...", end="", flush=True)
            print()
            os.rename(temp_path, filepath)
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"    Saved {local_name} ({size_mb:.0f} MB)")
            return True
        except (requests.RequestException, IOError) as exc:
            print(f"\n    Attempt {attempt}/5 failed: {exc}")
            for p in [filepath + ".tmp", filepath]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            if attempt < 5:
                wait = 2 ** attempt
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)

    print(f"  FAILED to download shard {shard_index} after 5 attempts.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download Hindi data for autoresearch-mlx")
    parser.add_argument(
        "--shards", type=int, default=4,
        help=f"Number of shards to download, max {TOTAL_SHARDS} (default: 4, ~1.4 GB)"
    )
    args = parser.parse_args()

    n = min(args.shards, TOTAL_SHARDS)
    if n < 2:
        print("Need at least 2 shards (1 train + 1 val). Setting --shards 2.")
        n = 2

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Downloading {n} Hindi shards to {DATA_DIR}")
    print(f"  Shards 0..{n-2}  -> training data")
    print(f"  Shard  {n-1}      -> validation data")
    print(f"  Total size: ~{n * 360} MB")
    print()

    ok = 0
    for i in range(n):
        if download_shard(i, i, n):
            ok += 1

    print(f"\nDone: {ok}/{n} shards downloaded.")

    if ok < 2:
        print("ERROR: Need at least 2 shards to proceed.")
        return

    val_idx = n - 1
    print(f"""
--- Next steps ---

1. Edit prepare.py — change these 3 lines near the top:

   MAX_SHARD = {n - 1}
   VAL_SHARD = {val_idx}
   VOCAB_SIZE = 16384

2. Train the tokenizer:
   uv run prepare.py

3. Start training:
   uv run train.py
""")


if __name__ == "__main__":
    main()
