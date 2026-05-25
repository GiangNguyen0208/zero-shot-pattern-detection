"""Deploy to HuggingFace Spaces.

Usage:
    python deploy_hf.py --token YOUR_HF_TOKEN
    # or set HF_TOKEN environment variable first
"""

import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi, create_repo


SPACE_NAME = "zero-shot-pattern-detection"

FILES_TO_UPLOAD = [
    "app.py",
    "requirements.txt",
    "src/__init__.py",
    "src/detector.py",
    "src/preprocessing.py",
    "src/template_matching.py",
    "src/feature_matching.py",
    "src/postprocessing.py",
    "src/visualization.py",
    "generate_examples.py",
    "examples/patterns/example1_pattern.png",
    "examples/patterns/example2_pattern.png",
    "examples/patterns/example3_pattern.png",
    "examples/drawings/example1_drawing.png",
    "examples/drawings/example2_drawing.png",
    "examples/drawings/example3_drawing.png",
]

HF_README = """---
title: Zero-Shot Pattern Detection in Technical Drawings
emoji: "\U0001F50D"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: false
license: mit
short_description: Detect arbitrary patterns in BOM technical drawings without training
---

# Zero-Shot Pattern Detection in Technical Drawings

Upload a **pattern image** and a **drawing image** to detect all occurrences of the pattern.

- **Zero-shot**: Works with any pattern, no training needed
- **Multi-scale**: Detects at 0.5x - 2.0x scale
- **Fast**: < 5 seconds on CPU

See the [GitHub repo](https://github.com/GiangNguyen0208/zero-shot-pattern-detection) for full documentation.
"""


def main():
    parser = argparse.ArgumentParser(description="Deploy to HuggingFace Spaces")
    parser.add_argument("--token", default=None, help="HuggingFace API token")
    parser.add_argument("--username", default=None, help="HuggingFace username")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Provide --token or set HF_TOKEN environment variable")
        print("Get your token at: https://huggingface.co/settings/tokens")
        return

    api = HfApi(token=token)
    user_info = api.whoami()
    username = args.username or user_info["name"]
    repo_id = f"{username}/{SPACE_NAME}"

    print(f"Creating/updating Space: {repo_id}")

    try:
        create_repo(repo_id, repo_type="space", space_sdk="gradio", token=token, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(HF_README)
        readme_tmp = f.name

    try:
        api.upload_file(
            path_or_fileobj=readme_tmp,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
            token=token,
        )
    finally:
        os.unlink(readme_tmp)

    for filepath in FILES_TO_UPLOAD:
        full_path = os.path.join(base_dir, filepath)
        if not os.path.exists(full_path):
            print(f"  SKIP (not found): {filepath}")
            continue

        print(f"  Uploading: {filepath}")
        api.upload_file(
            path_or_fileobj=full_path,
            path_in_repo=filepath,
            repo_id=repo_id,
            repo_type="space",
            token=token,
        )

    print(f"\nDone! Space URL: https://huggingface.co/spaces/{repo_id}")
    print("It may take a few minutes for the Space to build and start.")


if __name__ == "__main__":
    main()
