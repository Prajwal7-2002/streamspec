# -*- coding: utf-8 -*-
"""Upload StreamSpec Space to Hugging Face."""

from pathlib import Path
from huggingface_hub import HfApi, whoami

def main():
    api = HfApi()
    user_info = whoami()
    username = user_info["name"]
    print(f"Authenticated as Hugging Face user: {username} ({user_info.get('fullname')})")

    root = Path(__file__).resolve().parent.parent

    # Publish Space
    space_id = f"{username}/streamspec"
    space_dir = root / "hf_space"
    print(f"\nCreating and uploading Space: {space_id}...")
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="static",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=space_id,
        repo_type="space",
        commit_message="feat: publish StreamSpec interactive studio v1.2.0 (Double-Buffer Prefetcher, Zero-Extra-Sweep, Roofline CAD)",
    )
    print(f"✓ Space uploaded: https://huggingface.co/spaces/{space_id}")
    print("\n🎉 StreamSpec Studio successfully published to Hugging Face!")

if __name__ == "__main__":
    main()
