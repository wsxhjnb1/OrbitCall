"""Flag-only, isolated-process Torch precision round-trip evidence.

This script creates no tensors, loads no checkpoint, and does not initialize
CUDA. It checks configuration restoration, not numerical accuracy or speed.
Run on the study host; output paths must be fresh and immutable.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.receipt, args.log):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite retained output: {path.name}")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {path.parent.name}")

    import torch

    def snapshot():
        return {
            "legacy_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "mkldnn_matmul_fp32_precision": torch.backends.mkldnn.matmul.fp32_precision,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_fp32_precision": torch.backends.cudnn.fp32_precision,
            "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
            "cudnn_rnn_fp32_precision": torch.backends.cudnn.rnn.fp32_precision,
            "cuda_initialized": torch.cuda.is_initialized(),
        }

    def restore(saved):
        # The legacy setter overwrites backend raw policies; it must run first.
        torch.set_float32_matmul_precision(saved["legacy_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = saved["cuda_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = saved["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.fp32_precision = saved["cuda_matmul_fp32_precision"]
        torch.backends.mkldnn.matmul.fp32_precision = saved["mkldnn_matmul_fp32_precision"]

    before = snapshot()
    if before["cuda_initialized"]:
        raise RuntimeError("Flag-only evidence requires an uninitialized CUDA process")
    observations = [{"label": "fresh_process", "flags": before}]
    try:
        torch.set_float32_matmul_precision("highest")
        observations.append({"label": "set_highest", "flags": snapshot()})
        torch.set_float32_matmul_precision("high")
        observations.append({"label": "set_high", "flags": snapshot()})
    finally:
        restore(before)
    restored = snapshot()
    observations.append({"label": "restored_after_legacy_policy", "flags": restored})

    try:
        # Match approximation.py's declared native approximate control.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False
        observations.append({"label": "native_control_inside", "flags": snapshot()})
    finally:
        restore(before)
    restored_control = snapshot()
    observations.append({"label": "restored_after_native_control", "flags": restored_control})

    checks = {
        "legacy_policy_full_snapshot_restored": restored == before,
        "native_control_full_snapshot_restored": restored_control == before,
        "cuda_never_initialized": all(not x["flags"]["cuda_initialized"] for x in observations),
    }
    receipt = {
        "kind": "cpu_only_torch_precision_policy_roundtrip",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Configuration/restoration only; no tensors, checkpoint, CUDA dispatch or timing",
        "source_filename": Path(__file__).name,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "platform": platform.system(),
                    "machine": platform.machine(), "torch": str(torch.__version__),
                    "torch_cuda_build": torch.version.cuda},
        "observations": observations,
        "checks": checks,
        "valid": all(checks.values()),
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    with args.receipt.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    with args.log.open("x", encoding="utf-8") as stream:
        stream.write("CPU-only precision-policy round-trip; no numeric experiment.\n")
        stream.write(rendered)
    print(rendered, end="")
    if not receipt["valid"]:
        raise SystemExit("Precision flags failed exact restoration")


if __name__ == "__main__":
    main()
