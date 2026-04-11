#!/usr/bin/env python3
"""Push each subfolder in models/ to its own Hugging Face repo.

Usage:
    python scripts/push_models_to_hf.py --username NCPhat2005 --dry-run
    python scripts/push_models_to_hf.py --username NCPhat2005 --create-repos
    python scripts/push_models_to_hf.py --username NCPhat2005 --include "vihatet5_reimpl,voz_hsd_labeled"
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, upload_folder


def infer_repo_type(folder_path: str) -> str:
    """model if it has model files, else dataset."""
    model_markers = ("model.safetensors", "pytorch_model.bin", "config.json")
    for m in model_markers:
        if os.path.isfile(os.path.join(folder_path, m)):
            return "model"
    return "dataset"


def push_folder(
    api: HfApi,
    folder_path: str,
    username: str,
    *,
    create_repo: bool,
    commit_prefix: str,
    dry_run: bool,
) -> None:
    folder_name = os.path.basename(folder_path)
    repo_type = infer_repo_type(folder_path)
    repo_id = f"{username}/{folder_name}"

    print("------------------------------------------")
    print(f"Folder:     {folder_name}")
    print(f"Repo type:  {repo_type}")
    print(f"Repo id:    {repo_id}")

    if dry_run:
        print(f"[DRY-RUN] would create repo: {repo_id} ({repo_type})")
        print(f"[DRY-RUN] would upload {folder_path} -> {repo_id}")
        return

    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        print(f"Repo ready: {repo_id}")

    url = upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        repo_type=repo_type,
        commit_message=f"{commit_prefix} {folder_name}",
        ignore_patterns=[".*", "gitattributes"],
    )
    print(f"Done: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push model folders to HuggingFace")
    parser.add_argument("--username", required=True, help="HF username or org")
    parser.add_argument("--models-dir", default="models", help="Path to models dir")
    parser.add_argument("--create-repos", action="store_true", help="Create repos if missing")
    parser.add_argument("--include", default="", help="Comma-separated folder names to include")
    parser.add_argument("--exclude", default="", help="Comma-separated folder names to exclude")
    parser.add_argument("--commit-prefix", default="Upload", help="Commit message prefix")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only")
    args = parser.parse_args()

    if not os.path.isdir(args.models_dir):
        print(f"Error: {args.models_dir} does not exist")
        sys.exit(1)

    include = set(filter(None, args.include.split(",")))
    exclude = set(filter(None, args.exclude.split(",")))

    api = HfApi()

    folders = sorted(
        d for d in os.listdir(args.models_dir)
        if os.path.isdir(os.path.join(args.models_dir, d))
    )

    if not folders:
        print(f"No subfolders in {args.models_dir}")
        sys.exit(1)

    print(f"Models dir:  {args.models_dir}")
    print(f"Username:    {args.username}")
    print(f"Create repos: {args.create_repos}")
    print(f"Dry run:     {args.dry_run}")
    print(f"Folders:     {folders}")
    print()

    for name in folders:
        if include and name not in include:
            print(f"Skip (not in include): {name}")
            continue
        if name in exclude:
            print(f"Skip (excluded): {name}")
            continue

        folder_path = os.path.join(args.models_dir, name)
        try:
            push_folder(
                api,
                folder_path,
                args.username,
                create_repo=args.create_repos,
                commit_prefix=args.commit_prefix,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"ERROR pushing {name}: {e}")
            continue

    print("\nAll done.")


if __name__ == "__main__":
    main()
