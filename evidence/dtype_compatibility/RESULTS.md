# Model-native dtype compatibility

The plugin follows the loaded model dtype. The portable demo defaults to `--dtype auto`;
an explicit `--dtype float16` selects the FP16 path. The same source is used for both.

This development-only check used pinned Qwen2.5-1.5B-Instruct on DGX Spark GB10,
vLLM 0.29.0. Comparisons are within each dtype against its own native full scorer.

| Requested | Resolved | Locked token error | Agent token error | Agent traces |
| --- | --- | ---: | ---: | --- |
| auto | torch.bfloat16 | 0 | 0 | equal for both development episodes |
| float16 | torch.float16 | 0 | 0 | equal for both development episodes |

Regression suite: 449 passed, zero failures/errors/skips.

Locked two- and three-argument families retain full JSON/EOS scoring and all eleven
decision rules. Mixed marked/native requests pass. Complete Agent checks retain
all four rounds per dtype: candidate token IDs, score vectors, selected calls,
actual tool observations and final token IDs agree with their native controls.

This is compatibility evidence, not an FP16 speed benchmark or a held-out task score.
Historical BF16 speed ratios and scientific files remain unchanged.

[Frozen protocol](../../docs/VLLM_DTYPE_PROTOCOL.md) ·
[Infrastructure amendments](../../docs/VLLM_DTYPE_VALIDATION_AMENDMENT.md) ·
[Summary](summary.json) · [Raw runs](raw_results_v1.tar.gz) ·
[Export validation](RELEASE_VALIDATION.json)
