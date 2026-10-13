#!/usr/bin/env python3
"""Draw conventional Matplotlib figures from the frozen Agent evidence.

This presentation-only script runs on the original experiment host's CPU.
It performs no inference, new timing, task evaluation or trajectory filtering.
The previous renderer supplies the checked, immutable numerical preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import render_vllm_agent_readme_charts as checked

DEPENDENCY_SHA256 = "5fa093c98cf27f7f20616377b85e8f74716dadb114d934c23c079b7f9177dfa8"
BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
GRAY = "#B8B8B8"
INK = "#303030"


def axes_style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color("#888888")
        axis.spines[side].set_linewidth(.7)
    axis.tick_params(colors=INK, length=3, width=.7)
    axis.set_axisbelow(True)


def save(figure, output, stem, plt):
    for extension in ("png", "svg"):
        path = output / (stem + "." + extension)
        checked.require(not path.exists(), "Never overwrite a completed figure")
        figure.savefig(path, dpi=240, facecolor="white",
                       metadata={"Date": None} if extension == "svg" else None)
    plt.close(figure)


def ratios_plot(output, rows, plt):
    figure, axis = plt.subplots(figsize=(9, 4.2))
    figure.subplots_adjust(left=.105, right=.975, top=.88, bottom=.25)
    colors = (BLUE, ORANGE)
    for metric_index, (kind, label) in enumerate((("agent", "Whole Agent"), ("scoring", "Scoring stage"))):
        color = colors[metric_index]
        for model_index, row in enumerate(rows):
            center = model_index + (-.18 if metric_index == 0 else .18)
            pair = row["pair"]
            median = pair["median_" + kind + "_ratio"]
            low, high = pair[kind + "_range"]
            axis.bar(center, median, width=.32, color=color, alpha=.88,
                     label=label if model_index == 0 else None, zorder=2)
            axis.errorbar(center, median, yerr=[[median-low], [high-median]],
                          fmt="none", ecolor=INK, elinewidth=.9, capsize=3,
                          capthick=.9, zorder=4)
            for block_index, block in enumerate(row["paired_blocks"]):
                # Jitter only the x-coordinate; every measured ratio is intact.
                jitter = -.11 + block_index * .044
                axis.plot(center+jitter, block[kind+"_control_over_optimized"],
                          "o", markersize=3.8, markerfacecolor="white",
                          markeredgecolor=INK, markeredgewidth=.7, zorder=5)
            axis.text(center, high+.038, f"{median:.2f}×", ha="center", va="bottom",
                      fontsize=12, color=INK)
    axis.axhline(1, color="#777777", linestyle=(0, (4, 3)), linewidth=.9, zorder=1)
    axis.set_xlim(-.55, 1.55)
    axis.set_ylim(0, 1.5)
    axis.set_yticks((0, .5, 1., 1.5))
    axis.set_xticks(range(len(rows)), [row["short_name"] for row in rows])
    axis.tick_params(axis="x", labelsize=12, pad=6)
    axis.set_ylabel("Native / OrbitCall wall time", fontsize=12)
    axis.set_title("Paired latency comparison", loc="left", fontsize=14, pad=12)
    axis.grid(axis="y", color="#E7E7E7", linewidth=.6)
    axis.legend(loc="upper left", ncol=2, frameon=False, fontsize=11,
                handlelength=1.3, columnspacing=1.4)
    axes_style(axis)
    figure.text(.105, .12,
        "Six paired blocks · Bars: median · Dots: all blocks · Whiskers: min–max (not CI)",
        fontsize=10, color="#555555")
    figure.text(.105, .057,
        "1.00 = equal latency. Timed work matches; 1.5B separate corpus: 35/36, strict gate failed.",
        fontsize=9.5, color="#555555")
    save(figure, output, "readme_paired_ratios_v3", plt)
    return {"shared_ratio_ylim": [0, 1.5], "bars": "median of all six paired block ratios",
            "dots": "all six measured block ratios, x jitter only",
            "whiskers": "observed minimum and maximum; not confidence intervals",
            "reference": 1, "point_count": 24}


def stages_plot(output, rows, plt):
    figure, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    figure.subplots_adjust(left=.155, right=.975, top=.81, bottom=.22, hspace=.62)
    colors = {"generation": BLUE, "scoring": GREEN,
              "final_generation": ORANGE, "other": GRAY}
    labels = {"generation": "Candidate generation", "scoring": "Scoring",
              "final_generation": "Final answer", "other": "Other"}
    maximum = math.ceil(max(item["mean_wall_seconds"] for row in rows
                           for item in row["stages"].values())) + 1
    for model_index, (axis, row) in enumerate(zip(axes, rows, strict=True)):
        for arm_index, arm in enumerate(checked.ARMS):
            measured = row["stages"][arm]
            left = 0.
            y = 1-arm_index
            for name, value in measured["plot_stage_seconds"].items():
                axis.barh(y, value, left=left, height=.43, color=colors[name],
                          edgecolor="white", linewidth=.4,
                          label=labels[name] if model_index == arm_index == 0 else None,
                          zorder=2)
                left += value
            axis.text(left+.12, y, f"{measured['mean_wall_seconds']:.2f} s",
                      va="center", fontsize=11, color=INK)
        axis.set_yticks((1, 0), ("Native full", "OrbitCall"))
        axis.set_ylim(-.5, 1.6)
        axis.set_xlim(0, maximum)
        axis.set_xticks(range(0, maximum+1, 2))
        axis.set_title(row["short_name"], loc="left", fontsize=12, pad=9)
        axis.grid(axis="x", color="#E7E7E7", linewidth=.6)
        axes_style(axis)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.545, .98),
                  ncol=4, frameon=False, fontsize=10.5, columnspacing=1.3, handlelength=1.5)
    axes[-1].set_xlabel("Mean wall time per eight-episode window (s)", fontsize=12, labelpad=9)
    figure.text(.155, .062,
        "All 12 windows/arm · Other: parsing + tools/history + harness residual",
        fontsize=10, color="#555555")
    figure.text(.155, .022,
        "Baseline: native BF16 full scoring with the adapter installed.",
        fontsize=10, color="#555555")
    save(figure, output, "readme_stage_times_v3", plt)
    return {"shared_seconds_xlim": [0, maximum], "aggregation": "mean of all 12 windows per arm",
            "episodes_per_window": 8, "other_definition": list(checked.OTHER),
            "tiny_segments": "actual widths, no inflation or removed component"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("evidence/vllm_agent/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("evidence/vllm_agent_matplotlib_v3"))
    args = parser.parse_args()
    checked.require(sys.platform == "linux" and platform.machine() == "aarch64",
                    "Render frozen experimental evidence on the original host's CPU")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
                                  text=True, timeout=15).strip().splitlines()
    checked.require(len(gpu) == 1 and "GB10" in gpu[0] and checked.GPU_UUID in gpu[0], "Wrong experiment host")
    start = time.perf_counter()
    source = Path(__file__).resolve()
    dependency = Path(checked.__file__).resolve()
    source_bytes, dependency_bytes = source.read_bytes(), dependency.read_bytes()
    checked.require(hashlib.sha256(dependency_bytes).hexdigest() == DEPENDENCY_SHA256, "Checked preparation source changed")
    input_path, output = (checked.ROOT/args.input).resolve(), (checked.ROOT/args.output).resolve()
    checked.require(input_path.is_relative_to(checked.ROOT) and output.is_relative_to(checked.ROOT)
                    and not output.exists(), "Use the repository input and a fresh output directory")
    data = input_path.read_bytes()
    checked.require(hashlib.sha256(data).hexdigest() == checked.INPUT_SHA256, "Frozen summary changed")
    summary = json.loads(data, object_pairs_hook=checked.unique_keys)
    rows = checked.prepare(summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelcolor": INK, "text.color": INK, "svg.fonttype": "none",
        "svg.hashsalt": hashlib.sha256(source_bytes).hexdigest(), "axes.unicode_minus": False})
    output.mkdir(parents=True, exist_ok=False)
    ratio_style = ratios_plot(output, rows, plt)
    stage_style = stages_plot(output, rows, plt)
    checked.require(input_path.read_bytes() == data and source.read_bytes() == source_bytes
                    and dependency.read_bytes() == dependency_bytes, "Source/input changed during rendering")
    receipt = {"format_version": 1, "status": "complete", "scope": "Conventional Matplotlib styling of frozen measured data",
        "input_file": input_path.relative_to(checked.ROOT).as_posix(), "input_sha256": checked.INPUT_SHA256,
        "script_file": source.relative_to(checked.ROOT).as_posix(), "script_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "dependency_file": dependency.relative_to(checked.ROOT).as_posix(), "dependency_sha256": DEPENDENCY_SHA256,
        "output_directory": output.relative_to(checked.ROOT).as_posix(), "original_gpu_uuid": checked.GPU_UUID,
        "matplotlib_version": matplotlib.__version__, "new_inference_performed": False, "numerical_timing_rerun": False,
        "style": {"layout": "ordinary axes, grouped bars and stacked bars; no cards, badges or oversized typography",
                  "png_width_pixels": 2160, "png_dpi": 240},
        "ratio_style": ratio_style, "stage_style": stage_style,
        "plotted_models": [{"model": row["model"], "paired_points": row["paired_blocks"],
            "median_agent_ratio": row["pair"]["median_agent_ratio"], "agent_range": row["pair"]["agent_range"],
            "median_scoring_ratio": row["pair"]["median_scoring_ratio"], "scoring_range": row["pair"]["scoring_range"],
            "stage_values": row["stages"], "strict_promotion_accepted": row["promotion_accepted"]} for row in rows],
        "output_artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": checked.sha(path)}
                             for path in sorted(output.iterdir()) if path.is_file()},
        "elapsed_cpu_render_seconds": time.perf_counter()-start}
    with (output/"MATPLOTLIB_STYLE_RECEIPT.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "complete", "output": output.relative_to(checked.ROOT).as_posix(),
                      "receipt_sha256": checked.sha(output/"MATPLOTLIB_STYLE_RECEIPT.json")}))


if __name__ == "__main__":
    main()
