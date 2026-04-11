#!/bin/bash
# Push each subfolder in models/ to its own Hugging Face repo.
#
# Example:
#   bash scripts/push_models_to_hf.sh --username NCPhat2005 --create-repos
#
# Requirements:
#   - huggingface-cli login completed (or HF_TOKEN is set)
#   - git and git-lfs installed

set -euo pipefail

MODELS_DIR="models"
HF_USERNAME=""
CREATE_REPOS="false"
INCLUDE_PATTERN=""
EXCLUDE_PATTERN=""
DRY_RUN="false"
COMMIT_MESSAGE_PREFIX="Upload"

print_usage() {
    cat <<'EOF'
Usage: bash scripts/push_models_to_hf.sh [options]

Options:
  --username <name>          Hugging Face username or org (required)
  --models_dir <path>        Path to models directory (default: models)
  --create-repos             Create repos if they do not exist
  --include <regex>          Only push folder names matching regex
  --exclude <regex>          Skip folder names matching regex
  --commit-prefix <text>     Commit message prefix (default: Upload)
  --dry-run                  Print actions without pushing
  -h, --help                 Show this help message

Repo type inference:
  - Folder containing model.safetensors / pytorch_model.bin / config.json => model repo
  - Otherwise => dataset repo

Notes:
  - Dataset repos are pushed to: https://huggingface.co/datasets/<username>/<repo>
  - Model repos are pushed to:   https://huggingface.co/<username>/<repo>
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --username)
            HF_USERNAME="$2"
            shift 2
            ;;
        --models_dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --create-repos)
            CREATE_REPOS="true"
            shift 1
            ;;
        --include)
            INCLUDE_PATTERN="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE_PATTERN="$2"
            shift 2
            ;;
        --commit-prefix)
            COMMIT_MESSAGE_PREFIX="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift 1
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ -z "$HF_USERNAME" ]]; then
    echo "Error: --username is required"
    print_usage
    exit 1
fi

if [[ ! -d "$MODELS_DIR" ]]; then
    echo "Error: models directory does not exist: $MODELS_DIR"
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "Error: git is not installed"
    exit 1
fi

if [[ "$DRY_RUN" != "true" ]] && ! command -v git-lfs >/dev/null 2>&1; then
    echo "Error: git-lfs is not installed"
    exit 1
fi

if [[ "$DRY_RUN" != "true" ]] && ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "Error: huggingface-cli is not installed"
    echo "Install with: pip install huggingface_hub"
    exit 1
fi

infer_repo_type() {
    local folder="$1"
    if [[ -f "$folder/model.safetensors" ]] || [[ -f "$folder/pytorch_model.bin" ]] || [[ -f "$folder/config.json" ]]; then
        echo "model"
    else
        echo "dataset"
    fi
}

create_hf_repo_if_needed() {
    local repo_id="$1"
    local repo_type="$2"

    if [[ "$CREATE_REPOS" != "true" ]]; then
        return 0
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] create_repo ${repo_id} (${repo_type})"
        return 0
    fi

    if ! command -v python >/dev/null 2>&1; then
        echo "Error: python is required for --create-repos"
        exit 1
    fi

    python - "$repo_id" "$repo_type" <<'PY'
import sys
from huggingface_hub import create_repo

repo_id = sys.argv[1]
repo_type = sys.argv[2]

create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
print(f"Created or confirmed repo: {repo_id} ({repo_type})")
PY
}

push_single_folder() {
    local folder_path="$1"
    local folder_name
    folder_name="$(basename "$folder_path")"

    if [[ -n "$INCLUDE_PATTERN" ]] && ! [[ "$folder_name" =~ $INCLUDE_PATTERN ]]; then
        echo "Skip (not matched include): $folder_name"
        return 0
    fi

    if [[ -n "$EXCLUDE_PATTERN" ]] && [[ "$folder_name" =~ $EXCLUDE_PATTERN ]]; then
        echo "Skip (matched exclude): $folder_name"
        return 0
    fi

    local repo_type
    repo_type="$(infer_repo_type "$folder_path")"

    local repo_id="${HF_USERNAME}/${folder_name}"
    local remote_url=""
    if [[ "$repo_type" == "dataset" ]]; then
        remote_url="https://huggingface.co/datasets/${repo_id}"
    else
        remote_url="https://huggingface.co/${repo_id}"
    fi

    echo "------------------------------------------"
    echo "Folder:     $folder_name"
    echo "Repo type:  $repo_type"
    echo "Repo id:    $repo_id"
    echo "Remote URL: $remote_url"

    create_hf_repo_if_needed "$repo_id" "$repo_type"

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local repo_dir="$tmp_dir/repo"
    mkdir -p "$repo_dir"

    cp -R "$folder_path"/. "$repo_dir"/

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] git init in $repo_dir"
        echo "[DRY-RUN] git lfs track *.safetensors *.bin *.pt *.ckpt"
        echo "[DRY-RUN] git add . && git commit -m \"${COMMIT_MESSAGE_PREFIX} ${folder_name}\""
        echo "[DRY-RUN] git push -u ${remote_url} main"
        rm -rf "$tmp_dir"
        return 0
    fi

    (
        cd "$repo_dir"
        git init -q
        git lfs install --local >/dev/null
        git lfs track "*.safetensors" "*.bin" "*.pt" "*.ckpt" >/dev/null

        git add .
        if git diff --cached --quiet; then
            echo "No files to commit for: $folder_name"
            exit 0
        fi

        git commit -m "${COMMIT_MESSAGE_PREFIX} ${folder_name}" >/dev/null
        git branch -M main
        git remote add origin "$remote_url"
        git push -u origin main
    )

    rm -rf "$tmp_dir"
    echo "Done: $folder_name"
}

echo "Starting push from: $MODELS_DIR"
echo "Target account:     $HF_USERNAME"
echo "Create repos:       $CREATE_REPOS"
echo "Dry run:            $DRY_RUN"

found_any="false"
for folder in "$MODELS_DIR"/*; do
    if [[ -d "$folder" ]]; then
        found_any="true"
        push_single_folder "$folder"
    fi
done

if [[ "$found_any" != "true" ]]; then
    echo "No subfolders found in: $MODELS_DIR"
    exit 1
fi

echo "All done."
