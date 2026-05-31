"""
Download all HuggingFace models/datasets for DS200.Q21 Group 2.

Usage:
    export HF_TOKEN=hf_xxxx   # set token first in terminal
    python3 scripts/download_models.py
"""

import os
from huggingface_hub import snapshot_download

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

REPOS = [
    ("NCPhat2005/vihatet5_reimpl",        "model",   "vihatet5_reimpl"),
    ("NCPhat2005/visobert_labeling",       "model",   "visobert_labeling"),
    ("NCPhat2005/vit5_finetune_hate_only", "model",   "vit5_finetune_hate_only"),
    ("NCPhat2005/vit5_finetune_balanced",  "model",   "vit5_finetune_balanced"),
    ("NCPhat2005/vit5_finetune_multi",     "model",   "vit5_finetune_multi"),
    ("NCPhat2005/focal_loss_exp",          "model",   "vit5_focal_loss_exp"),
    ("NCPhat2005/vit5_pretrain_hate_only", "model",   "vit5_pretrain_hate_only"),
    ("NCPhat2005/vit5_pretrain_balanced",  "model",   "vit5_pretrain_balanced"),
    # voz_hsd_labeled is 12M+ rows — uncomment only if you need it
    # ("NCPhat2005/voz_hsd_labeled",       "dataset", "voz_hsd_labeled"),
]

token = os.environ.get("HF_TOKEN")
if not token:
    print("[WARNING] HF_TOKEN not set — downloads will be slower (rate-limited).")
    print("          Run: export HF_TOKEN=hf_xxxx  then retry.\n")

os.makedirs(MODELS_DIR, exist_ok=True)

for repo_id, repo_type, local_dir in REPOS:
    dest = os.path.join(MODELS_DIR, local_dir)
    # Skip only if model weights already downloaded completely
    weight_file = os.path.join(dest, "model.safetensors")
    if os.path.isfile(weight_file) and os.path.getsize(weight_file) > 1_000_000:
        print(f"[SKIP] {local_dir}/ already has model.safetensors ({os.path.getsize(weight_file) // 1_000_000} MB)")
        continue

    print(f"[Downloading] {repo_id}  -->  models/{local_dir}/")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=dest,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
        print(f"[OK] {local_dir}\n")
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {local_dir} — run script again to resume.")
        break
    except Exception as e:
        print(f"[ERROR] {local_dir}: {e}\n")

print("Done.")
