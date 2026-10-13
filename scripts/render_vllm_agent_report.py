#!/usr/bin/env python3
"""Render a concise report from independently checked vLLM Agent evidence.

Run on the original Spark only, including all arithmetic and plotting. This
script loads no model, evaluates no tool/reference answer, and excludes no
online trajectory. It accepts completed neutral/negative comparisons; a
failed promotion flag remains visible in the report and machine summary.
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
MODELS = ("Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct")
CONTROLS = ("native_default",)
OPTIMIZED = "native_shape_suffix"
SYSTEM_ARMS = (*CONTROLS, OPTIMIZED)
POLICIES = (*SYSTEM_ARMS, "vote4", "greedy1")
STAGES = ("generation", "scoring", "final_generation", "parsing", "tools_and_history", "harness")
LIMITS = {"max_token_absolute_nats": .05, "mean_token_absolute_nats": .005,
          "max_sequence_absolute_nats": .5, "minimum_winner_agreement": .99}
LABELS = {"native_default": "Native BF16 full (adapter installed)", "native_shape_suffix": "Native-shape BF16 suffix",
          "vote4": "Vote 4 (no scoring)", "greedy1": "Greedy 1 (no scoring)"}
SHORT_LABELS = ("Native BF16 full\n(adapter installed)",)


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


def number(value, name, *, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            "Invalid finite value: " + name)
    return value


def near(actual, expected, name):
    number(actual, name, minimum=-1e-7)
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
            "Analysis/report aggregation disagreement: " + name)


def nullable_near(actual, expected, name):
    if expected is None:
        require(actual is None, "Expected unavailable ratio: " + name)
    else:
        near(actual, expected, name)


def compact_audit(row):
    require(type(row["valid"]) is bool, "Missing audit disposition")
    for key in ("nonempty_pools", "zero_pools", "joint_agreeing", "pool_count"):
        require(type(row[key]) is int and row[key] >= 0, "Malformed audit count")
    require(row["nonempty_pools"] + row["zero_pools"] == row["pool_count"] == 72
            and row["joint_agreeing"] <= row["nonempty_pools"], "Incomplete two-round audit population")
    for key in ("max_token_error", "mean_token_error", "max_sequence_error"):
        number(row[key], key)
    return {key: row[key] for key in ("valid", "pool_count", "nonempty_pools", "zero_pools",
            "joint_agreeing", "agreement_counts", "max_token_error", "mean_token_error", "max_sequence_error")}


def compact_workload(row):
    return {"episode_count": row["episode_count"], "matched_count": row["matched_count"],
            "all_matched": row["all_matched"],
            "changed_episodes": [item for item in row["episodes"] if not item["matched_complete_workload"]]}


def workload_totals(rows):
    return {"episode_comparisons": sum(row["episode_count"] for row in rows),
            "matched_comparisons": sum(row["matched_count"] for row in rows),
            "all_matched": all(row["all_matched"] for row in rows)}


def prepare_study(study):
    require(study["development"] is False and study["model"] in MODELS, "Expected confirmation study")
    require(type(study["complete"]) is dict and study["complete"]["valid"] is True
            and study["failure"] is None and len(study["windows"]) == 24
            and not study["raw_only_unaccepted"], "Report requires a complete study; no partial ratios")
    precision = study["precision_scope"]
    require(precision["generation_weight_backbone_kv"] == "bfloat16"
            and precision["generation_logit_output"] == "bfloat16"
            and precision["native_default_score_logit_output"] == "bfloat16"
            and precision["same_precision_score_logit_output"] == "bfloat16"
            and precision["same_precision_reference"] == "native_default"
            and precision["score_cache_reads"] is False, "Scoring precision scope changed")
    require(set(study["policy_quality"]) == set(POLICIES), "Missing task-quality control")
    quality = {}
    for policy in POLICIES:
        row = study["policy_quality"][policy]["summary"]
        require(row["episodes"] == 36 and type(row["task_successes"]) is int
                and 0 <= row["task_successes"] <= 36, "Incomplete task denominator")
        near(row["task_success_rate"], row["task_successes"] / 36, "task success fraction")
        quality[policy] = row
    require(set(study["audit"]) == {OPTIMIZED}, "Missing stock-reference suffix audit")
    audit = {arm: compact_audit(row) for arm, row in study["audit"].items()}
    require(all(row["valid"] for row in audit.values()), "Completed matched-head audit failed")
    comparisons = {row["control"]: row for row in study["timing_comparisons"]}
    require(len(study["timing_comparisons"]) == 1 and set(comparisons) == set(CONTROLS),
            "Expected the prospective stock-native timing control")
    pairs = []
    for control in CONTROLS:
        pair = comparisons[control]
        require(pair["optimized"] == OPTIMIZED and pair["block_count"] == 6
                and pair["window_count"] == 24 and len(pair["blocks"]) == 6,
                "Incomplete paired schedule")
        blocks = []
        for saved in sorted(pair["blocks"], key=lambda row: row["block"]):
            windows = [row for row in study["windows"]
                       if row["control"] == control and row["block"] == saved["block"]]
            require(len(windows) == 4, "Incomplete block windows")
            means, score_means = {}, {}
            for arm in (control, OPTIMIZED):
                selected = [row for row in windows if row["policy"] == arm]
                require(len(selected) == 2, "Paired block requires two windows per arm")
                means[arm] = statistics.mean(number(row["review"]["elapsed_seconds"], "Agent wall") for row in selected)
                score_means[arm] = statistics.mean(number(row["review"]["stage_seconds"]["scoring"], "score wall") for row in selected)
                require(means[arm] > 0, "Zero-duration Agent timing comparison")
                near(saved["wall_seconds_mean_of_two"][arm], means[arm], "paired Agent mean")
                near(saved["disjoint_stage_seconds_mean_of_two"][arm]["scoring"], score_means[arm], "paired score mean")
            agent_ratio = means[control] / means[OPTIMIZED]
            score_ratio = (score_means[control] / score_means[OPTIMIZED]
                           if score_means[OPTIMIZED] > 0 else None)
            near(saved["control_over_optimized"], agent_ratio, "paired Agent ratio")
            nullable_near(saved["scoring_control_over_optimized"], score_ratio, "paired score ratio")
            blocks.append({"block": saved["block"], "agent_control_over_optimized": agent_ratio,
                           "scoring_control_over_optimized": score_ratio,
                           "all_workloads_matched": saved["all_workloads_matched"],
                           "optimized_task_success_drop": saved["optimized_task_success_drop"]})
        require([row["block"] for row in blocks] == list(range(6)), "Missing/duplicate block ID")
        agent_values = [row["agent_control_over_optimized"] for row in blocks]
        score_values = [row["scoring_control_over_optimized"] for row in blocks]
        near(pair["median_control_over_optimized"], statistics.median(agent_values), "Agent median")
        score_available = all(value is not None for value in score_values)
        score_median = statistics.median(score_values) if score_available else None
        score_range = [min(score_values), max(score_values)] if score_available else None
        nullable_near(pair["median_control_over_optimized_scoring"], score_median, "score median")
        require(len(pair["range"]) == 2, "Missing Agent block range")
        near(pair["range"][0], min(agent_values), "Agent range minimum")
        near(pair["range"][1], max(agent_values), "Agent range maximum")
        if score_range is None:
            require(pair["scoring_range"] is None, "Expected unavailable scoring range")
        else:
            require(len(pair["scoring_range"]) == 2, "Missing scoring block range")
            near(pair["scoring_range"][0], score_range[0], "score range minimum")
            near(pair["scoring_range"][1], score_range[1], "score range maximum")
        pairs.append({"control": control, "optimized": OPTIMIZED, "blocks": blocks,
                      "median_agent_ratio": statistics.median(agent_values), "agent_range": [min(agent_values), max(agent_values)],
                      "median_scoring_ratio": score_median, "scoring_range": score_range,
                      "scoring_ratio_unavailable_blocks": [row["block"] for row in blocks
                                                           if row["scoring_control_over_optimized"] is None],
                      "all_timed_workloads_matched": pair["all_timed_workloads_matched"],
                      "timed_workload_comparisons": {
                          "cross_arm": workload_totals([row for block in pair["blocks"]
                                                       for row in block["cross_arm_workload_comparisons"]]),
                          "native_repeat": workload_totals([block["AA_BB_repeatability"][control]
                                                           for block in pair["blocks"]]),
                          "optimized_repeat": workload_totals([block["AA_BB_repeatability"][OPTIMIZED]
                                                              for block in pair["blocks"]])},
                      "complete_test_corpus_all_matched": pair["complete_test_corpus_workload_comparison"]["all_matched"],
                      "complete_test_corpus_workload_comparison": compact_workload(
                          pair["complete_test_corpus_workload_comparison"]),
                      "no_block_task_success_drop": pair["no_block_task_success_drop"],
                      "complete_test_corpus_task_success_drop": pair["complete_test_corpus_task_success_drop"],
                      "same_workload_promotion_accepted": pair["same_workload_promotion_accepted"],
                      "interpretation": pair["interpretation"]})
    stages = {}
    for arm in SYSTEM_ARMS:
        selected = [row["review"] for row in study["windows"] if row["policy"] == arm]
        expected = 12
        require(len(selected) == expected and all(len(row["episode_work"]) == 8 for row in selected),
                "Unexpected timing population/arm allocation")
        for row in selected:
            require(set(row["stage_seconds"]) == set(STAGES) - {"harness"}, "Missing disjoint timer")
            number(row["harness_seconds"], "harness residual", minimum=-1e-7)
            for name, seconds in row["stage_seconds"].items():
                number(seconds, name)
            near(math.fsum(row["stage_seconds"].values()) + row["harness_seconds"], row["elapsed_seconds"], "stage accounting")
        stages[arm] = {"windows": len(selected), "episodes_per_window": 8,
                       "mean_wall_seconds": statistics.mean(row["elapsed_seconds"] for row in selected),
                       "mean_stage_seconds": {name: statistics.mean(
                           row["harness_seconds"] if name == "harness" else row["stage_seconds"][name]
                           for row in selected) for name in STAGES}}
    return {"model": study["model"], "run": study["run"], "precision_scope": precision,
            "promotion_accepted": study["promotion_accepted"], "comparisons": pairs,
            "task_quality": quality, "locked_audit": audit,
            "timed_stage_means": stages,
            "retained_windows": len(study["windows"]), "process_elapsed_seconds": study["process_elapsed_seconds"]}


def status(value):
    return "yes" if value else "no"


def ratio_text(median, interval):
    if median is None:
        require(interval is None, "Unavailable ratio must have no interval")
        return "N/A (zero scoring work)"
    return f"{median:.3f}x [{interval[0]:.3f}, {interval[1]:.3f}]"


def write_markdown(output, summary):
    lines = ["# vLLM Agent integration", "",
             "Measured on DGX Spark GB10 with vLLM 0.29.0 and the pinned Qwen2.5 checkpoints.", "",
             "Weights, backbone, KV cache and ordinary generation remain BF16. The optimized",
             "scorer preserves the stock BF16 vocabulary projection and its native row shape.",
             "Only after that full projection are suffix rows selected for native full-vocabulary",
             "normalization, target-logprob gathering and CPU transfer. Scoring-prefix-cache reads",
             "remain disabled. Every complete candidate spelling, first suffix predictor and EOS is retained.", "",
             "The baseline uses native BF16 full scoring with the adapter installed. Unmarked",
             "requests delegate to the original vLLM method; wrapper overhead is present in both",
             "arms. No separate adapter-disabled engine performance arm was measured.", "",
             "## Complete Agent and scoring latency", "",
             "Ratios are native-full wall time / native-shape-suffix wall time. Values above one",
             "indicate lower measured latency for the candidate. Each value is the median of six paired",
             "ABBA/BAAB block ratios; every window executes the same eight frozen task IDs afresh.",
             "Each model retains all 24 windows. No divergent episode or block is removed.",
             "A zero optimized scoring-stage mean makes that block's scoring ratio unavailable.",
             "If any block is unavailable, the scoring median and range are N/A; all Agent ratios",
             "and every block remain in the evidence. No median is computed from a selected subset.", "",
             "| Model | Control | Whole Agent median [range] | Scoring median [range] | Timed work matches | 36-task work matches | Same-workload promotion |",
             "| --- | --- | ---: | ---: | --- | --- | --- |"]
    for model in summary["models"]:
        for pair in model["comparisons"]:
            agent = ratio_text(pair["median_agent_ratio"], pair["agent_range"])
            scoring = ratio_text(pair["median_scoring_ratio"], pair["scoring_range"])
            corpus = pair["complete_test_corpus_workload_comparison"]
            lines.append(f"| {model['model']} | {LABELS[pair['control']]} | {agent} | {scoring} | {status(pair['all_timed_workloads_matched'])} | {corpus['matched_count']}/{corpus['episode_count']} | {status(pair['same_workload_promotion_accepted'])} |")
    lines += ["", "![Every retained paired block: Agent and scoring ratios](agent_vs_scoring_ratios.png)", "",
              "Timing-workload identity and the separate 36-task corpus check are distinct.",
              "The frozen promotion gate requires both, as well as no task-success drop.",
              "A corpus mismatch alone does not imply differing work in the timed windows.", ""]
    for model in summary["models"]:
        for pair in model["comparisons"]:
            counts = pair["timed_workload_comparisons"]
            count_text = "; ".join(f"{label} {counts[key]['matched_comparisons']}/{counts[key]['episode_comparisons']}"
                                   for key, label in (("cross_arm", "A/B"), ("native_repeat", "A/A"),
                                                      ("optimized_repeat", "B/B")))
            if pair["all_timed_workloads_matched"]:
                timed_scope = ("All timed workload content matches across arms and repeats "
                               f"({count_text}). The timing ratios describe the same eight frozen task IDs.")
            else:
                timed_scope = (f"Timed workload content differs ({count_text}); these timing ratios describe "
                               "observed online policies, not a fixed-workload implementation comparison.")
            corpus = pair["complete_test_corpus_workload_comparison"]
            lines += [f"**{model['model']}:** {timed_scope} Separately, "
                      f"{corpus['matched_count']}/{corpus['episode_count']} corpus episodes have identical complete workload content.", ""]
            for changed in corpus["changed_episodes"]:
                fields = []
                for stage, row in changed["rounds"].items():
                    different = [name.removeprefix("same_").replace("_", " ") for name, matched in row.items() if not matched]
                    if different:
                        fields.append(f"round {stage}: " + ", ".join(different))
                unchanged_outcomes = (all(row["same_selected_call"] and row["same_tool_observation"]
                                          for row in changed["rounds"].values())
                                      and changed["same_final_text"] and changed["same_final_tokens"])
                lines.append(f"Corpus content difference `{changed['id']}` ({'; '.join(fields)}). "
                             + ("Selected calls, actual tool observations, final text and final tokens still match."
                                if unchanged_outcomes else "See the retained per-field differences in the machine summary."))
                lines.append("")
            if not pair["same_workload_promotion_accepted"]:
                reasons = []
                if not pair["all_timed_workloads_matched"]:
                    reasons.append("timed workload content differs")
                if not corpus["all_matched"]:
                    reasons.append("complete 36-task corpus content does not fully match")
                if not pair["no_block_task_success_drop"]:
                    reasons.append("a timed block has lower task success")
                if pair["complete_test_corpus_task_success_drop"]:
                    reasons.append("the corpus has lower task success")
                lines += ["The frozen promotion gate remains **not passed** because " + "; ".join(reasons) + ".", ""]
    lines += ["Task success is reported below; no universal quality claim follows.", "",
              "![Measured disjoint stages in the timed eight-episode windows](stage_wall_times.png)", "",
              "The stacked chart uses arithmetic mean stage times from all retained timing windows:",
              "12 windows per arm per model. It shows seconds per",
              "eight-episode workload, not independent request latency. Parsing, tool/history work",
              "and the measured harness residual remain visible in the accounting even when small.",
              "Cache reset is inside each window's harness residual. Engine initialization is separate;",
              "first-shape work remains in the stage that incurred it, without invented subtraction.", "",
              "## Actual task outcomes", "",
              "Two compulsory tool rounds use real local lookup/arithmetic/calendar outputs, followed",
              "by a model-generated final answer. Every policy retains all 36 held-out tasks, including",
              "empty candidate pools and tool errors. Vote4 and greedy1 use different compute budgets.", "",
              "| Model | Policy | Task success | Valid final JSON | Correct final value |",
              "| --- | --- | ---: | ---: | ---: |"]
    for model in summary["models"]:
        for policy in POLICIES:
            row = model["task_quality"][policy]
            lines.append(f"| {model['model']} | {LABELS[policy]} | {row['task_successes']}/{row['episodes']} | {row['final_json_valid']}/{row['episodes']} | {row['final_correct']}/{row['episodes']} |")
    lines += ["", "## Locked-candidate fidelity", "",
              "The reference here is native full BF16 scoring with the adapter installed. These are",
              "fixed-candidate logprob/decision checks, separate from online task success. There",
              "are 72 round-specific pools per model; only nonempty pools enter agreement rates.", "",
              "| Model | Arm | Nonempty / pools | Token max (nats) | Token mean (nats) | Sequence max (nats) | Joint agreement | Role / disposition |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for model in summary["models"]:
        for arm in (OPTIMIZED,):
            row = model["locked_audit"][arm]
            disposition = "Stock-reference gate passed: " + status(row["valid"])
            lines.append(f"| {model['model']} | {LABELS[arm]} | {row['nonempty_pools']}/{row['pool_count']} | {row['max_token_error']:.7g} | {row['mean_token_error']:.7g} | {row['max_sequence_error']:.7g} | {row['joint_agreeing']}/{row['nonempty_pools']} | {disposition} |")
    lines += ["", "Frozen bounds: token max 0.05 nats, pooled mean 0.005 nats, sequence",
              "max 0.5 nats, and at least 99% agreement for all eleven rules and their joint decision.",
              "Earlier suffix/APC and FP32-output-head development failures remain failures.",
              "Their measurements are not accepted native-shape confirmation results.", "", "## Evidence and scope", "",
              "This is an original deterministic local fixture suite, not an official Agent or BFCL",
              "leaderboard, Internet-service benchmark, or population-level reliability guarantee.",
              "The six repeated timing blocks are not hundreds of independent task trials.", "",
              f"Provided current and retired run intervals total **{summary['cost']['provided_current_plus_retired_wall_lower_bound_seconds']:.3f} seconds**,",
              "a lower bound on distinct recorded process wall time. Nested stages are not added again.",
              "It is not pure GPU-busy time or a complete project budget. Independent analysis and",
              "this rendering step have separate costs. Earlier failed attempts remain in the analysis.", "",
              "[Machine summary](summary.json) · [Independent analysis](independent_analysis.json) ·",
              "[Render receipt](RENDER_RECEIPT.json) · [Native-shape protocol](../../docs/VLLM_AGENT_SHAPE_PROTOCOL.md)", ""]
    with (output / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def plot_results(output, models):
    # Deliberately imported only after the original-host guard in main.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    colors = {"agent": "#ce683f", "scoring": "#168a83"}
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.1), sharey=True)
    visible_ratios = [row[field] for model in models for pair in model["comparisons"]
                      for row in pair["blocks"]
                      for field in ("agent_control_over_optimized", "scoring_control_over_optimized")
                      if row[field] is not None]
    ratio_ymax = max([1., *visible_ratios]) * 1.08
    for axis, model in zip(axes, models, strict=True):
        for index, pair in enumerate(model["comparisons"]):
            for kind, offset in (("agent", -.14), ("scoring", .14)):
                field = kind + "_control_over_optimized"
                values = [row[field] for row in pair["blocks"]]
                available = [(n, value) for n, value in enumerate(values) if value is not None]
                xs = [index + offset + (n - 2.5) * .025 for n, _ in available]
                if available:
                    axis.scatter(xs, [value for _, value in available], s=30,
                                 color=colors[kind], alpha=.8, zorder=3)
                median = pair["median_" + kind + "_ratio"]
                if median is not None:
                    axis.plot([index + offset - .095, index + offset + .095], [median, median],
                              color=colors[kind], linewidth=3, zorder=4)
                else:
                    missing = len(values) - len(available)
                    axis.text(index, .93, f"Scoring: N/A ({missing}/6 blocks have zero scoring work)",
                              transform=axis.get_xaxis_transform(), ha="center", va="top",
                              fontsize=8, color=colors[kind])
            if not pair["same_workload_promotion_accepted"]:
                caption = ("corpus promotion gate not passed" if pair["all_timed_workloads_matched"]
                           and not pair["complete_test_corpus_all_matched"] else "promotion gate not passed")
                axis.text(index, 1.02, caption, transform=axis.get_xaxis_transform(),
                          ha="center", fontsize=8, color="#a24429")
        axis.set_xticks(range(len(CONTROLS)), SHORT_LABELS)
        axis.axhline(1, color="#65738b", linestyle="--", linewidth=1)
        axis.grid(axis="y", alpha=.18)
        axis.set_title(model["model"].replace("-Instruct", ""), pad=28)
    # Set the shared scale only after both panels exist. An early set_ylim()
    # disables autoscaling and can otherwise clip the later model's score dots.
    axes[0].set_ylim(0, ratio_ymax)
    axes[0].set_ylabel("Native full / native-shape-suffix wall time")
    legend = [Line2D([0], [0], marker="o", color=colors[k], label=label)
              for k, label in (("agent", "Whole Agent"), ("scoring", "Scoring stage"))]
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .89), ncol=2, frameon=False)
    figure.suptitle("All paired blocks · complete Agent versus scoring stage", fontsize=15, y=.98)
    figure.text(.5, .015, "All six blocks retained; unavailable scoring ratios marked N/A. Medians require six ratios. Adapter installed in both arms.",
                ha="center", fontsize=9, color="#536178")
    figure.tight_layout(rect=(0, .06, 1, .84))
    for extension in ("png", "svg"):
        figure.savefig(output / ("agent_vs_scoring_ratios." + extension), dpi=200)
    plt.close(figure)

    stage_colors = {"generation": "#4778b6", "scoring": "#168a83", "final_generation": "#db9c3e",
                    "parsing": "#ac84b6", "tools_and_history": "#7c9873", "harness": "#bac3ce"}
    stage_labels = {"generation": "Candidate generation", "scoring": "Scoring", "final_generation": "Final generation",
                    "parsing": "Parsing", "tools_and_history": "Tools + history", "harness": "Harness residual"}
    figure, axes = plt.subplots(2, 1, figsize=(13.4, 6.8), sharex=True)
    labels = [LABELS[arm] for arm in SYSTEM_ARMS]
    for axis, model in zip(axes, models, strict=True):
        for y, arm in enumerate(SYSTEM_ARMS):
            row, left = model["timed_stage_means"][arm], 0.
            for stage in STAGES:
                value = row["mean_stage_seconds"][stage]
                axis.barh(y, value, left=left, height=.63, color=stage_colors[stage],
                          label=stage_labels[stage] if y == 0 else None)
                left += value
            axis.text(left, y, f"  {row['mean_wall_seconds']:.3f}s · n={row['windows']}", va="center", fontsize=9)
        axis.set_yticks(range(len(SYSTEM_ARMS)), labels)
        axis.invert_yaxis()
        axis.set_title(model["model"].replace("-Instruct", ""), loc="left", fontsize=12)
        axis.grid(axis="x", alpha=.18)
        axis.set_axisbelow(True)
        axis.margins(x=.2)
    axes[-1].set_xlabel("Mean measured wall seconds per eight-episode window")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .915), ncol=3, frameon=False)
    figure.suptitle("Where time went · actual disjoint stages, all timed windows", fontsize=15, y=.985)
    figure.text(.5, .015, "Each arm uses all 12 windows, with the adapter installed. No workload filtering, cost subtraction, or invented tool latency.",
                ha="center", fontsize=9, color="#536178")
    figure.tight_layout(rect=(0, .04, 1, .83))
    for extension in ("png", "svg"):
        figure.savefig(output / ("stage_wall_times." + extension), dpi=200)
    plt.close(figure)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evidence/vllm_agent"))
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise RuntimeError("All report calculations and plots run only on the original DGX Spark")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
                                  text=True, timeout=15).strip().splitlines()
    require(len(gpu) == 1 and GPU_UUID in gpu[0] and "GB10" in gpu[0], "Wrong live report host")
    started = time.perf_counter()
    analysis_path = (ROOT / args.analysis).resolve() if not args.analysis.is_absolute() else args.analysis.resolve()
    output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    require(analysis_path.is_relative_to(ROOT) and output.is_relative_to(ROOT) and not output.exists(),
            "Use an existing repo analysis and a fresh repo output directory")
    analysis_bytes = analysis_path.read_bytes()
    analysis = json.loads(analysis_bytes, object_pairs_hook=unique_keys,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    require(analysis["format_version"] == 1 and analysis["limits"] == LIMITS
            and analysis["both_confirmation_models_present"] is True
            and GPU_UUID in analysis["live_gpu_identity"], "Unexpected independent analysis scope")
    require(analysis["analysis_source_sha256"] == sha(ROOT / "scripts/analyze_vllm_agent_shape.py"),
            "Independent analyzer source changed; inspect the source snapshot before rendering")
    confirmations = [row for row in analysis["studies"] if not row["development"]]
    require(len(confirmations) == 2 and {row["model"] for row in confirmations} == set(MODELS),
            "Exactly one completed confirmation per public model is required")
    by_model = {row["model"]: row for row in confirmations}
    models = [prepare_study(by_model[name]) for name in MODELS]
    require(analysis["combined_same_workload_promotion_accepted"] == all(row["promotion_accepted"] for row in models),
            "Combined promotion flag differs from model dispositions")
    cost_keys = ("provided_run_process_wall_seconds", "provided_retired_run_process_wall_seconds",
                 "provided_current_plus_retired_wall_lower_bound_seconds", "independent_analysis_wall_seconds")
    cost = {key: number(analysis[key], key) for key in cost_keys}
    near(cost["provided_current_plus_retired_wall_lower_bound_seconds"],
         cost["provided_run_process_wall_seconds"] + cost["provided_retired_run_process_wall_seconds"], "distinct cost subtotal")
    summary = {"format_version": 1, "analysis_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
               "analysis_source_sha256": analysis["analysis_source_sha256"], "renderer_source_sha256": sha(Path(__file__)),
               "selection_provenance": analysis["selection_provenance"],
               "models": models, "limits": analysis["limits"], "cost": cost, "cost_scope": analysis["cost_scope"],
               "combined_same_workload_promotion_accepted": analysis["combined_same_workload_promotion_accepted"],
               "claim_limits": analysis["claim_limits"], "retired_runs": [
                   {key: row[key] for key in ("run", "disposition", "process_elapsed_seconds", "accepted_speed_ratios")}
                   for row in analysis["retired_failed_run_history"]]}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "independent_analysis.json").open("xb") as stream:
        stream.write(analysis_bytes)
    write_json(output / "summary.json", summary)
    write_markdown(output, summary)
    plot_results(output, models)
    require(analysis_path.read_bytes() == analysis_bytes, "Analysis changed during report rendering")
    artifacts = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
                 for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output / "RENDER_RECEIPT.json", {"status": "complete", "host": gpu[0],
               "analysis_file": analysis_path.relative_to(ROOT).as_posix(), "analysis_sha256": summary["analysis_sha256"],
               "renderer_source_sha256": summary["renderer_source_sha256"], "artifacts": artifacts,
               "elapsed_seconds": time.perf_counter() - started, "model_or_reference_evaluation_performed": False,
               "scope": "presentation of validated analysis; paired scorer ratios and stage means cross-checked; no trajectory selection"})
    print(json.dumps({"status": "vllm_agent_report_complete", "output": output.relative_to(ROOT).as_posix(),
                      "combined_same_workload_promotion_accepted": summary["combined_same_workload_promotion_accepted"]}, indent=2))


if __name__ == "__main__":
    main()
