"""Environment block written next to every runner's output, so a result file can be traced
to the software and hardware that produced it.

    write_env(output_path)          -> <output_path>.env.json   (runners that overwrite)
    append_env(output_path, key)    -> <output_path>.env.jsonl  (runners that append rows)

Call these after the run: they query the GPU, which initializes CUDA.
"""
import importlib.metadata
import json
import platform
import subprocess
import sys
import time

import torch

PACKAGES = ("torch", "numpy", "pandas", "matplotlib", "transformers", "tokenizers", "psutil")


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def git_commit():
    """Short HEAD, '-dirty' if tracked files differ from HEAD (untracked files are ignored)."""
    head = _run(["git", "rev-parse", "--short", "HEAD"])
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return head + ("-dirty" if dirty and not dirty.startswith("unavailable") else "")


def env_block():
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    cuda = torch.cuda.is_available()
    return dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=git_commit(),
        python=sys.version.split()[0],
        os=platform.platform(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version() if cuda else None,
        gpu=torch.cuda.get_device_name(0) if cuda else None,
        gpu_memory_mb=round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2, 1) if cuda else None,
        driver_version=_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) if cuda else None,
        driver_model=_run(["nvidia-smi", "--query-gpu=driver_model.current", "--format=csv,noheader"]) if cuda else None,
        argv=sys.argv,
        packages=versions,
    )


def write_env(output_path):
    path = output_path + ".env.json"
    with open(path, "w") as f:
        json.dump(env_block(), f, indent=2)
    print(f"wrote {path}")
    return path


def append_env(output_path, key):
    path = output_path + ".env.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(dict(row=key, **env_block())) + "\n")
    print(f"appended {path}")
    return path
