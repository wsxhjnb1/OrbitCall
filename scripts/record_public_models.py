"""Download metadata only; numerical experiments remain on Spark."""

import json
from pathlib import Path

from huggingface_hub import HfApi


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "metadata/public_models.json"
    if destination.exists():
        raise FileExistsError(destination)
    models = []
    for repo in ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"):
        info = HfApi().model_info(repo, files_metadata=True)
        files = []
        for item in info.siblings:
            if item.rfilename.endswith((".json", ".safetensors", ".txt")) or item.rfilename in {"LICENSE", "README.md"}:
                files.append({"path": item.rfilename, "size": item.size,
                              "sha256": item.lfs.sha256 if item.lfs else None,
                              "git_blob_id": item.blob_id})
        models.append({"repo_id": repo, "revision": info.sha,
                       "url": "https://huggingface.co/" + repo,
                       "files": files})
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"models": models}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: item[k] for k in ("repo_id", "revision")} for item in models]))


if __name__ == "__main__":
    main()
