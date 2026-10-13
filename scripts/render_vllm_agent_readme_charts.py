#!/usr/bin/env python3
"""Render two README charts from the immutable, measured Agent summary.

Run only on the original Spark CPU. No model, inference, reference evaluation,
CUDA context, timing experiment or trajectory selection is performed. The
six paired ratios and all twelve-window stage means remain unchanged. Other
is exactly parsing + tool/history + harness, including the cache-reset cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4"
INPUT_SHA256 = "7db0016a226af742c00a41327df71a2c34eb1bd8a4a410554f8f438c4655994e"
MODELS = ("Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct")
ARMS = ("native_default", "native_shape_suffix")
STAGES = ("generation", "scoring", "final_generation", "parsing", "tools_and_history", "harness")
OTHER = ("parsing", "tools_and_history", "harness")
NAVY = "#18364B"
TEAL = "#167F83"
CORAL = "#DD795F"
MUTED = "#506674"
GRID = "#DDE5E8"
CARD = "#F5F8F9"
PALE_CORAL = "#FAEBE5"
PALE_TEAL = "#E5F2F0"
OTHER_COLOR = "#B9C7CD"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def finite(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), "Invalid measured value: " + name)
    return value


def near(actual, expected, name):
    finite(actual, name)
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
            "Summary/plot arithmetic disagrees: " + name)


def prepare(summary):
    require(summary["format_version"] == 1 and len(summary["models"]) == 2,
            "Unexpected measured-summary scope")
    by_model = {row["model"]: row for row in summary["models"]}
    require(set(by_model) == set(MODELS), "Exactly the two frozen public models are required")
    rows = []
    for name in MODELS:
        model = by_model[name]
        require(model["retained_windows"] == 24 and len(model["comparisons"]) == 1,
                "Incomplete retained timing population")
        precision = model["precision_scope"]
        require(precision["same_precision_reference"] == "native_default"
                and precision["score_cache_reads"] is False
                and all(precision[key] == "bfloat16" for key in
                        ("generation_weight_backbone_kv", "generation_logit_output",
                         "native_default_score_logit_output", "same_precision_score_logit_output")),
                "Native BF16 precision scope changed")
        pair = model["comparisons"][0]
        blocks = sorted(pair["blocks"], key=lambda row: row["block"])
        require(pair["control"] == ARMS[0] and pair["optimized"] == ARMS[1]
                and [row["block"] for row in blocks] == list(range(6))
                and pair["all_timed_workloads_matched"] is True,
                "Missing paired blocks or changed timed workload disposition")
        for kind in ("agent", "scoring"):
            values = [finite(row[kind + "_control_over_optimized"], kind, positive=True) for row in blocks]
            near(pair["median_" + kind + "_ratio"], statistics.median(values), kind + " median of six ratios")
            require(len(pair[kind + "_range"]) == 2, "Missing frozen range")
            near(pair[kind + "_range"][0], min(values), kind + " range minimum")
            near(pair[kind + "_range"][1], max(values), kind + " range maximum")
        require(model["promotion_accepted"] == pair["same_workload_promotion_accepted"],
                "Frozen promotion disposition changed")
        require(set(model["timed_stage_means"]) == set(ARMS), "Missing measured stage arm")
        stage_rows = {}
        for arm in ARMS:
            raw = model["timed_stage_means"][arm]
            require(raw["windows"] == 12 and raw["episodes_per_window"] == 8
                    and set(raw["mean_stage_seconds"]) == set(STAGES), "Stage denominator changed")
            values = {key: finite(raw["mean_stage_seconds"][key], key) for key in STAGES}
            total = finite(raw["mean_wall_seconds"], "mean online wall", positive=True)
            near(total, math.fsum(values.values()), "complete stage wall sum")
            other = math.fsum(values[key] for key in OTHER)
            stage_rows[arm] = {"windows": 12, "episodes_per_window": 8,
                               "mean_wall_seconds": total, "raw_stage_seconds": values,
                               "other_components_seconds": {key: values[key] for key in OTHER},
                               "plot_stage_seconds": {"generation": values["generation"],
                                   "scoring": values["scoring"], "final_generation": values["final_generation"],
                                   "other": other}}
        rows.append({"model": name, "short_name": name.removesuffix("-Instruct"),
                     "promotion_accepted": model["promotion_accepted"], "pair": pair,
                     "paired_blocks": blocks, "stages": stage_rows})
    require(summary["combined_same_workload_promotion_accepted"] == all(row["promotion_accepted"] for row in rows),
            "Combined frozen disposition changed")
    return rows


def card(figure, bounds, patches, *, color=CARD):
    x, y, width, height = bounds
    figure.add_artist(patches.FancyBboxPatch((x, y), width, height,
        boxstyle="round,pad=0.006,rounding_size=0.018", transform=figure.transFigure,
        facecolor=color, edgecolor="none", zorder=0))


def save(figure, output, stem, plt):
    for extension in ("png", "svg"):
        path = output / (stem + "." + extension)
        require(not path.exists(), "Never overwrite a completed chart")
        figure.savefig(path, dpi=130, facecolor="white", metadata={"Date": None} if extension == "svg" else None)
    plt.close(figure)


def ratios_plot(output, rows, plt, patches, ticker):
    figure = plt.figure(figsize=(14, 8.8), facecolor="white")
    figure.text(.04, .946, "Measured latency. Every paired block.", fontsize=34, weight="bold", color=NAVY)
    figure.text(.04, .884, "Native BF16 full (adapter installed) / OrbitCall suffix", fontsize=24, color=MUTED)
    all_values = [block[key] for row in rows for block in row["paired_blocks"]
                  for key in ("agent_control_over_optimized", "scoring_control_over_optimized")]
    xmin, xmax = min(1., min(all_values)) - .025, max(1., max(all_values)) + .045
    for index, row in enumerate(rows):
        x = .04 + index * .48
        card(figure, (x, .17, .44, .65), patches)
        figure.text(x + .027, .772, row["short_name"], fontsize=28, weight="bold", color=NAVY)
        for offset, kind, label, color in ((.027, "agent", "Whole Agent", TEAL),
                                          (.250, "scoring", "Scoring", CORAL)):
            pair = row["pair"]
            figure.text(x + offset, .718, label, fontsize=24, color=color, weight="bold")
            figure.text(x + offset, .648, f"{pair['median_' + kind + '_ratio']:.2f}×",
                        fontsize=46, color=color, weight="bold")
            limits = pair[kind + "_range"]
            figure.text(x + offset, .601, f"{limits[0]:.3f}–{limits[1]:.3f}", fontsize=24, color=MUTED)
        axis = figure.add_axes((x + .064, .295, .350, .258), facecolor=CARD, zorder=1)
        for block in row["paired_blocks"]:
            y = 6 - block["block"]
            a, b = block["agent_control_over_optimized"], block["scoring_control_over_optimized"]
            axis.plot((a, b), (y, y), color=GRID, linewidth=2.3, zorder=2)
            axis.scatter(a, y, s=94, color=TEAL, edgecolors="white", linewidths=1.2, zorder=3)
            axis.scatter(b, y, s=94, color=CORAL, edgecolors="white", linewidths=1.2, zorder=3)
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(.5, 6.5)
        axis.set_yticks(range(1, 7), ["B6", "B5", "B4", "B3", "B2", "B1"])
        axis.xaxis.set_major_locator(ticker.MultipleLocator(.1))
        axis.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        axis.tick_params(axis="both", labelsize=24, length=0, pad=8, colors=MUTED)
        axis.grid(axis="x", color=GRID, linewidth=1., zorder=0)
        axis.axvline(1., color=NAVY, linewidth=1.6, linestyle=(0, (3, 3)), zorder=1)
        for spine in axis.spines.values():
            spine.set_visible(False)
        passed = row["promotion_accepted"]
        figure.text(x + .027, .204, "Strict gate: " + ("PASSED" if passed else "NOT PASSED"),
                    fontsize=24, weight="bold", color=TEAL if passed else CORAL,
                    bbox={"boxstyle": "round,pad=.22", "facecolor": PALE_TEAL if passed else PALE_CORAL,
                          "edgecolor": "none"})
    figure.text(.04, .112, "Six paired blocks: medians + ranges · 1.00 = equal latency", fontsize=24, color=MUTED)
    figure.text(.04, .066, "Matched timing · 24 windows/model · eight fixed workflows", fontsize=24, color=MUTED)
    figure.text(.04, .020, "1.5B corpus: 35/36 matches · strict gate NOT PASSED", fontsize=24, color=CORAL)
    save(figure, output, "readme_paired_ratios_v2", plt)
    return {"shared_ratio_xlim": [xmin, xmax], "equality_reference": 1.,
            "point_count_per_model": 12, "point_count_per_metric_per_model": 6,
            "ratio_display": "points, never bars from the cropped ratio origin",
            "aggregation": "median and full range of all six paired block ratios"}


def stages_plot(output, rows, plt, patches, ticker):
    figure = plt.figure(figsize=(14, 8.8), facecolor="white")
    figure.text(.04, .946, "Where the online time goes.", fontsize=34, weight="bold", color=NAVY)
    figure.text(.04, .884, "Mean of all 12 windows/arm · 8 episodes/window", fontsize=24, color=MUTED)
    colors = {"generation": NAVY, "scoring": TEAL, "final_generation": CORAL, "other": OTHER_COLOR}
    labels = {"generation": "Candidate gen.", "scoring": "Scoring",
              "final_generation": "Final answer", "other": "Other"}
    for x, stage in zip((.04, .335, .54, .79), colors, strict=True):
        figure.add_artist(patches.Rectangle((x, .824), .022, .020, transform=figure.transFigure,
                                            facecolor=colors[stage], edgecolor="none"))
        figure.text(x + .032, .823, labels[stage], fontsize=24, color=NAVY)
    max_seconds = math.ceil(max(item["mean_wall_seconds"] for row in rows for item in row["stages"].values()) * 1.17)
    for index, row in enumerate(rows):
        bottom = .485 - index * .300
        card(figure, (.04, bottom, .92, .268), patches)
        figure.text(.067, bottom + .213, row["short_name"], fontsize=28, weight="bold", color=NAVY)
        axis = figure.add_axes((.263, bottom + .104, .663, .103), facecolor=CARD, zorder=1)
        for y, arm in zip((1., 0.), ARMS, strict=True):
            measured = row["stages"][arm]
            left = 0.
            for stage, value in measured["plot_stage_seconds"].items():
                axis.barh(y, value, left=left, height=.62, color=colors[stage], edgecolor="none", zorder=3)
                if stage in ("generation", "scoring"):
                    axis.text(left + value / 2., y, f"{value:.2f}", ha="center", va="center",
                              fontsize=24, weight="bold", color="white", zorder=4)
                left += value
            axis.text(left + .15, y, f"{measured['mean_wall_seconds']:.2f}s", fontsize=24,
                      weight="bold", color=NAVY, va="center", zorder=4)
        axis.set_xlim(0, max_seconds)
        axis.set_ylim(-.55, 1.55)
        axis.set_yticks((1., 0.), ("Native full", "OrbitCall"))
        axis.tick_params(axis="y", length=0, pad=15, labelsize=24, colors=NAVY)
        axis.tick_params(axis="x", length=0, pad=8, labelsize=24, colors=MUTED)
        axis.xaxis.set_major_locator(ticker.MultipleLocator(2.))
        axis.grid(axis="x", color=GRID, linewidth=1.2, zorder=0)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
        a, b = (row["stages"][arm]["plot_stage_seconds"] for arm in ARMS)
        figure.text(.067, bottom + .010,
                    f"Native / OrbitCall: final {a['final_generation']:.3f} / {b['final_generation']:.3f}s"
                    f" · other {a['other']:.3f} / {b['other']:.3f}s", fontsize=24, color=MUTED)
    figure.text(.04, .124, "Mean wall time per eight-episode window (seconds)", fontsize=24, color=MUTED)
    figure.text(.04, .069, "Other = parsing + tool/history + harness (including cache reset).", fontsize=24, color=MUTED)
    figure.text(.04, .023, "Native BF16 full baseline: adapter installed. Every stage is included.", fontsize=24, color=MUTED)
    save(figure, output, "readme_stage_times_v2", plt)
    return {"shared_seconds_xlim": [0, max_seconds], "aggregation": "arithmetic mean of all twelve windows per arm",
            "other_definition": "exact sum of parsing + tools_and_history + harness, including cache-reset residual",
            "tiny_segment_policy": "actual widths retained; no minimum-width inflation or omitted component"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("evidence/vllm_agent/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("evidence/vllm_agent_readme_charts_v2"))
    args = parser.parse_args()
    require(sys.platform == "linux" and platform.machine() == "aarch64", "Render only on original Spark CPU")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
                                  text=True, timeout=15).strip().splitlines()
    require(len(gpu) == 1 and "GB10" in gpu[0] and GPU_UUID in gpu[0], "Wrong original rendering host")
    started = time.perf_counter()
    source_path = Path(__file__).resolve()
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    input_path = (ROOT / args.input).resolve()
    output = (ROOT / args.output).resolve()
    require(input_path.is_relative_to(ROOT) and output.is_relative_to(ROOT) and not output.exists(),
            "Use a repository input and a fresh repository output directory")
    input_bytes = input_path.read_bytes()
    require(hashlib.sha256(input_bytes).hexdigest() == INPUT_SHA256, "Immutable measured summary hash changed")
    summary = json.loads(input_bytes, object_pairs_hook=unique_keys,
                         parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    rows = prepare(summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches, ticker
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 24,
                                "svg.fonttype": "none", "svg.hashsalt": source_sha256,
                                "axes.unicode_minus": False, "savefig.facecolor": "white"})
    output.mkdir(parents=True, exist_ok=False)
    ratio_style = ratios_plot(output, rows, plt, patches, ticker)
    stage_style = stages_plot(output, rows, plt, patches, ticker)
    require(input_path.read_bytes() == input_bytes, "Frozen evidence changed during rendering")
    require(source_path.read_bytes() == source_bytes, "Renderer source changed during rendering")
    artifacts = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
                 for path in sorted(output.iterdir()) if path.is_file()}
    receipt = {"format_version": 1, "status": "complete", "operation": "presentation-only redraw of immutable measured figures",
               "input_file": input_path.relative_to(ROOT).as_posix(), "input_sha256": INPUT_SHA256,
               "script_file": source_path.relative_to(ROOT).as_posix(), "script_sha256": source_sha256,
               "output_directory": output.relative_to(ROOT).as_posix(), "original_gpu_uuid": GPU_UUID,
               "matplotlib_version": matplotlib.__version__, "new_inference_performed": False,
               "model_or_cuda_context_loaded": False, "numerical_timing_rerun": False,
               "measured_analysis_sha256": summary["analysis_sha256"],
               "original_renderer_sha256": summary["renderer_source_sha256"],
               "combined_same_workload_promotion_accepted": summary["combined_same_workload_promotion_accepted"],
               "style": {"canvas_inches": [14, 8.8], "png_dpi": 130, "png_width_pixels": 1820,
                         "minimum_font_points": 24, "palette": {"navy": NAVY, "teal": TEAL, "coral": CORAL},
                         "baseline": "native BF16 full scoring with adapter installed; no disabled-adapter arm"},
               "ratio_style": ratio_style, "stage_style": stage_style,
               "plotted_models": [{"model": row["model"], "retained_windows": 24,
                                   "paired_block_count": 6, "paired_points": row["paired_blocks"],
                                   "median_agent_ratio": row["pair"]["median_agent_ratio"],
                                   "agent_range": row["pair"]["agent_range"],
                                   "median_scoring_ratio": row["pair"]["median_scoring_ratio"],
                                   "scoring_range": row["pair"]["scoring_range"],
                                   "strict_promotion_accepted": row["promotion_accepted"],
                                   "separate_corpus_check": row["pair"]["complete_test_corpus_workload_comparison"],
                                   "stage_values": row["stages"]} for row in rows],
               "output_artifacts": artifacts, "elapsed_cpu_render_seconds": time.perf_counter() - started}
    with (output / "CHART_STYLE_RECEIPT.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "readme_measured_charts_complete", "output": output.relative_to(ROOT).as_posix(),
                      "input_sha256": INPUT_SHA256, "receipt_sha256": sha(output / "CHART_STYLE_RECEIPT.json")}, indent=2))


if __name__ == "__main__":
    main()
