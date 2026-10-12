"""Render presentation-only figures from the unchanged Spark analysis JSON.

The original analysis figures remain retained. This command changes labels,
axis formatting and layout only; it does not recompute estimates or select data.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from analyze_study import ARM_LABELS, GPU_UUID, METHOD_LABELS, SCORE_METHODS, save_figure


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("evidence/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("evidence/figures"))
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise RuntimeError("Render study figures on the original DGX Spark")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True, timeout=15
    ).strip()
    if GPU_UUID not in gpu or "GB10" not in gpu:
        raise RuntimeError("Original Spark identity required")
    if args.output.exists():
        raise FileExistsError(args.output)
    summary_hash = sha(args.summary)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    studies = summary["studies"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter
    args.output.mkdir(parents=True)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 8.5), sharey=True)
    positions = np.arange(len(SCORE_METHODS))
    for axis, study in zip(axes, studies, strict=True):
        rows = study["quality"]["accuracy"]["test"]
        values = [rows[method]["supported_accuracy"] for method in SCORE_METHODS]
        axis.barh(positions, [value if value is not None else np.nan for value in values], height=.7,
                  color=["#2166ac" if method == "mass" else "#8f9ca8" for method in SCORE_METHODS])
        for position, method, value in zip(positions, SCORE_METHODS, values, strict=True):
            if value is None:
                continue
            low, high = rows[method]["question_wilson_95_interval"]
            axis.errorbar(value, position, xerr=[[max(0., value-low)], [max(0., high-value)]],
                          fmt="none", ecolor="#263442", capsize=2, linewidth=.8)
            axis.text(1.055, position, f"{rows[method]['successes']}/{rows[method]['supported']}",
                      va="center", ha="left", fontsize=8)
        axis.set_xlim(0, 1.19)
        axis.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        axis.set_title(study["model"].replace("-Instruct", "") +
                       f"\nFrozen test: {study['quality']['test_count']}; unsupported: {rows['mass']['unsupported']}")
        axis.set_xlabel("Restricted scalar test accuracy")
        axis.grid(axis="x", alpha=.2)
    axes[0].set_yticks(positions, [METHOD_LABELS.get(method, method.replace("fixed_", "Fixed "))
                                 for method in SCORE_METHODS])
    axes[0].invert_yaxis()
    figure.suptitle("All frozen controls; approximate question Wilson 95% intervals", fontsize=11)
    figure.tight_layout()
    save_figure(figure, args.output, "test_accuracy")
    plt.close(figure)

    colors = ["#2166ac", "#c75728"]
    for stage in ("scorer", "reducer"):
        pairs = studies[0]["benchmark"][stage + "_pairs"]
        names = [row["pair"] for row in pairs]
        figure, axes = plt.subplots(2, 1, figsize=(12.5, 7.7), sharex=True)
        for metric, axis in zip(("wall", "cuda"), axes, strict=True):
            axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
            for model_index, study in enumerate(studies):
                by_name = {row["pair"]: row for row in study["benchmark"][stage + "_pairs"]}
                for index, name in enumerate(names):
                    row = by_name[name]
                    center = index + (-.12 if model_index == 0 else .12)
                    ratios = [block[metric + "_ratio_a_over_b"] for block in row["blocks"]]
                    axis.scatter([center + delta for delta in (-.045, -.015, .015, .045)], ratios,
                                 color=colors[model_index], alpha=.75, s=25,
                                 label=study["model"].replace("-Instruct", "") if index == 0 else None)
                    median = row[metric]["median_block_ratio"]
                    low, high = row[metric]["empirical_block_bootstrap_median_95_interval"]
                    axis.errorbar(center, median, yerr=[[max(0., median-low)], [max(0., high-median)]],
                                  fmt="D", color=colors[model_index], markersize=5, capsize=4)
            if stage == "reducer":
                axis.set_yscale("log")
                axis.set_ylim(.4, 6.6)
                axis.set_yticks([.5, 1, 2, 4, 6])
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}x"))
                axis.tick_params(axis="y", which="minor", labelleft=False)
            else:
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}x"))
            axis.set_ylabel(("Wall time" if metric == "wall" else "CUDA interval") + " A / B")
            axis.grid(axis="y", alpha=.25)
        axes[0].legend(loc="best", fontsize=9)
        axes[-1].set_xticks(range(len(pairs)), [ARM_LABELS.get(row["arm_a"], row["arm_a"]) + " /\n" +
                                              ARM_LABELS.get(row["arm_b"], row["arm_b"]) for row in pairs])
        title = "Complete FULL scorer: unchanged backbone" if stage == "scorer" else "Resident FULL-logit reducer: backbone and projection excluded"
        figure.suptitle(title + "\nFour paired blocks; A / B > 1 favors B", fontsize=11)
        axes[-1].set_xlabel("Dots: all blocks; diamonds: medians; whiskers: descriptive four-block bootstrap intervals")
        figure.tight_layout()
        save_figure(figure, args.output, stage + "_block_ratios")
        plt.close(figure)
    if sha(args.summary) != summary_hash:
        raise ValueError("Analysis JSON changed during rendering")
    receipt = {"summary_sha256": summary_hash, "renderer_sha256": sha(Path(__file__)),
               "analysis_source_sha256": summary["analysis_source_sha256"],
               "scope": "Presentation-only layout; all estimates, controls, block values and population counts unchanged",
               "host": "original DGX Spark GB10", "matplotlib": matplotlib.__version__,
               "files": {path.name: sha(path) for path in sorted(args.output.iterdir())}}
    with (args.output / "RENDER_RECEIPT.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
