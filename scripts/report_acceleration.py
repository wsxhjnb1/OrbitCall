"""Spark CPU-only report/plot from independently validated retained evidence."""

import argparse
import json
from pathlib import Path
import platform
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = {"native": "IEEE full", "cached": "IEEE cached", "structure": "IEEE tree"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Artifact calculations run on Spark")
    analysis = json.loads(args.analysis.read_text())
    if analysis["combined_confirmation_accepted"] is not True:
        raise RuntimeError("Combined confirmation not accepted")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Approximate complete-score acceleration", "",
             "Measured on DGX Spark GB10 with the unchanged frozen public Qwen2.5 models and candidates.", "",
             "The selected path shares the prompt and token-prefix tree, and applies the existing TF32x3",
             "Triton arithmetic to ordinary linears. All complete JSON spellings, EOS tokens, vocabulary",
             "denominators and eleven aggregators are retained. FP32 storage remains unchanged.", "",
             "## Held-out complete scoring", "",
             "Each ratio is median(control wall time / optimized wall time) across six paired ABBA/BAAB blocks.",
             "Each window rescored the same eight pre-frozen test fixture families, including prefill,",
             "tokenization, packing, all model work, score extraction and final aggregation. Loading,",
             "candidate generation, first-call compilation and artifact validation are excluded.", "",
             "| Model | Control | Median speed ratio | Six-block range | Empirical paired bootstrap interval |",
             "| --- | --- | ---: | ---: | ---: |"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True, constrained_layout=True)
    for ax, study in zip(axes, analysis["studies"], strict=True):
        for index, pair in enumerate(study["accepted_pairs"]):
            median = pair["median_ratio_a_over_b"]
            lo, hi = pair["range"]
            ci = pair["producer_empirical_bootstrap_median_95"]
            lines.append(f"| {study['model']} | {CONTROLS[pair['a']]} | {median:.4f}x | {lo:.4f}–{hi:.4f} | {ci[0]:.4f}–{ci[1]:.4f} |")
            values = [block["ratio_a_over_b"] for block in pair["blocks"]]
            ax.scatter([index + (n - 2.5) * .035 for n in range(6)], values, color="#1f6f8b", s=25)
            ax.plot([index - .17, index + .17], [median, median], color="#c45c35", linewidth=2.3)
        ax.set_xticks(range(len(study["accepted_pairs"])), [CONTROLS[x["a"]] for x in study["accepted_pairs"]])
        ax.axhline(1., color="#555555", linewidth=.8, linestyle="--")
        ax.set_title(study["model"].replace("Qwen2.5-", "").replace("-Instruct", ""))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Control time / optimized time")
    fig.suptitle("Complete score stage: six held-out paired blocks per control", fontsize=12)
    for extension in ("png", "svg"):
        fig.savefig(args.output.parent / ("paired_speed_ratios." + extension), dpi=200)
    plt.close(fig)
    lines += ["", "The bootstrap is descriptive; its resamples are not additional independent trials.", "",
              "![All six block ratios](paired_speed_ratios.png)", "", "## Numerical and decision fidelity", "",
              "Bounds were frozen after the user allowed small errors: max token0.01 nats, token-weighted",
              "mean0.0005 nats, max complete sequence0.1 nats, and per-aggregator plus joint agreement>=99%.",
              "Zero-candidate pools are retained separately. Agreement concerns the FP32 reference, not gold answers.", "",
              "| Model | Scored / input records | Zero pools | Token max error | Token mean error | Sequence max error | Joint agreement |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for study in analysis["studies"]:
        row = study["audit"]["optimized"]
        lines.append(f"| {study['model']} | {row['scored_records']}/119 | {row['zero_family']} | {row['maximum_token_error']:.8g} | {row['mean_absolute_token_error']:.8g} | {row['maximum_sequence_error']:.8g} | {row['joint_agreeing_count']}/{row['scored_records']} |")
    window_count = sum(study["retained_window_count"] for study in analysis["studies"])
    lines += ["", "Raw errors, quantiles, changed IDs, margins and development/test subgroup counts are in",
              f"[independent_analysis.json](independent_analysis.json). All{window_count} timing windows are retained",
              "for the two-model, three-control schedule. The whole119 audit and whole-family timing",
              "are separately validated layouts; audit batch8 applies to cached scoring, while tree",
              "scores its packed complete family. First calls and shape setup remain in the raw archive.", "",
              "## Scope and earlier attempts", "",
              "This establishes faster complete scoring for the fixed workload, without a generation,",
              "serving-throughput, universal-model, novelty or task-accuracy improvement claim. There is",
              "no cross-call prompt cache. The same-structure FP32-tree comparator separates Tensor Core",
              "arithmetic from prefix sharing; the cached comparator controls ordinary prompt reuse.", "",
              "The initial bulk/SDPA/head grid did not yield accepted speed. The strict Tensor Core",
              "3e-5 gates remain failures. Exact CUDA graphs passed the development numerical check",
              "but showed approximately1x warm speed. The approximate linear-only discovery was",
              "unconfirmed because its gain was small; native-high exceeded the fixed error budget.",
              "The new prefix-sharing study and its stronger confirmation-control amendment are separate",
              "prospective freezes. No earlier failed result was relabeled.", "",
              "Protocols: [approximate](../../docs/APPROXIMATE_SCORING_PROTOCOL.md),",
              "[prefix sharing](../../docs/PREFIX_APPROXIMATE_PROTOCOL.md),",
              "[same-structure control amendment](../../docs/PREFIX_CONFIRMATION_CONTROL_AMENDMENT.md).", "",
              "## Reproduction", "", "```bash",
              "PYTHONPATH=src .venv/bin/python scripts/analyze_approximate_scoring.py \\",
              "  --runs runs/prefix_approximate_confirmation_3b_20261013_v1 \\",
              "         runs/prefix_approximate_confirmation_1p5b_20261013_v1 \\",
              "  --output metadata/independent_reanalysis_NEW.json", "```", "",
              "Extract [raw_results_v1.tar.gz](raw_results_v1.tar.gz) at repository root for the snapshots.",
              "Original reference/candidate artifacts are in the earlier public raw_results_v2 archive.",
              "Use those frozen source snapshots for exact historical replay; do not silently substitute",
              "a new checkpoint, runtime, dtype, normalizer, input pool or configuration. See",
              "[MANIFEST.json](MANIFEST.json) for byte-preservation receipts and privately retained",
              "Chrome trace exclusions. Models and gold labels are excluded.", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    summary = {"models": [{"model": study["model"], "ratios": {p["a"]: p["median_ratio_a_over_b"]
                                                                      for p in study["accepted_pairs"]},
                           "fidelity": {key: study["audit"]["optimized"][key] for key in
                              ("scored_records", "zero_family", "maximum_token_error", "mean_absolute_token_error",
                               "maximum_sequence_error", "joint_all_controls_agreement")}}
                          for study in analysis["studies"]], "tests": 299,
               "selected_discovery_plus_confirmation_wall_seconds": analysis["selected_discovery_plus_provided_confirmation_wall_seconds"],
               "complete_score_stage_only": True}
    with (args.output.parent / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
