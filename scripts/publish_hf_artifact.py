#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a model-forge artifact folder to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, for example user/model-name")
    parser.add_argument("--folder", required=True, help="Local model, dataset, or result folder to upload")
    parser.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    parser.add_argument("--revision", default=None, help="Optional branch or revision")
    parser.add_argument("--commit-message", default="Upload model-forge artifact")
    parser.add_argument("--ignore", action="append", default=[], help="Glob to ignore; can be passed multiple times")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before publishing")

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"artifact folder does not exist: {folder}")

    api = HfApi(token=token)
    user = api.whoami(token=token).get("name", "unknown")
    create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
        token=token,
    )
    upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(folder),
        revision=args.revision,
        commit_message=args.commit_message,
        ignore_patterns=args.ignore or None,
        token=token,
    )
    print(f"uploaded {folder} to https://huggingface.co/{args.repo_id} as {user}")


if __name__ == "__main__":
    main()
