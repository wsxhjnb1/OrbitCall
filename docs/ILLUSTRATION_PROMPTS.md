# README illustration provenance

These two conceptual illustrations were generated with ChatGPT's built-in `image_gen` tool on 2026-10-14. They are explanatory artwork, not experimental evidence or exact API examples. The raw generated PNG files were copied without editing. No CLI or API fallback was used.

| Illustration | File | Meaning |
| --- | --- | --- |
| Agent loop | `assets/agent_loop_illustrated.png` | Candidate generation, validation, scoring, selection, real tool execution and observation feedback. |
| Scoring optimization | `assets/native_suffix_illustrated.png` | Full native BF16 projection followed by suffix-only probability processing and transfer. |

The Agent cards contain illustrative JSON-like notation. Consult the runnable example for the actual request envelope. The loop repeats for two tool rounds in the measured fixture, then generates a final answer. In the scoring illustration, matrix sizes and row indices are schematic: the shared full projection remains intact, and retained predictor rows include the first suffix predictor (`P-1`) and every suffix token through EOS. Every retained row keeps the full vocabulary. The scoring path does not reuse prefix KV caches or shorten the model or LM-head computation.

Measured charts in `evidence/vllm_agent/` remain standard plots of the retained benchmark data and were not regenerated with AI. The prior `RELEASE_VALIDATION_v2.json` refers to the earlier presentation; `ILLUSTRATION_VALIDATION.json` records this presentation-only update.

## Final prompt: Agent loop

```text
Use case: infographic-diagram.
Asset type: vivid conceptual illustration for the English GitHub README of OrbitCall, a tool-calling LLM systems project.
Create a NEW polished wide landscape illustration, roughly 16:9, showing an actual tool-using Agent loop. Make it lively and memorable with elegant softly lit isometric 3D objects, tactile paper JSON cards, a small friendly computational assistant embodied in a GPU module, a sorting gate, a scoring instrument, a calculator and calendar workbench, and a returned observation card. These objects should form an understandable sequence, not a sterile row of rectangular UI boxes. Clean off-white background, teal and coral accents matching a research project, refined shadows, professional editorial quality rather than childish cartoon.
Composition: six clearly separated stages flowing left to right across a gently curved orbital path. Use readable directional arrows in precisely this order: Generate -> Validate -> Score -> Select -> Execute -> Observe. A distinct return arrow travels along the bottom from Observe back to Generate. The GPU module produces several candidate JSON cards; validation filters formatting errors; scoring considers a small finite family of JSON argument key-order spellings; one card is selected and passed to real calculator/calendar tools; the actual returned observation feeds the next turn. The selected card has a gold border, not a correctness checkmark.
Text: title exactly "OrbitCall". Use only these six large stage labels, spelled exactly: "Generate", "Validate", "Score", "Select", "Execute", "Observe". On the GPU module use the short labels "vLLM" and "BF16". Cards may show simple brace/key glyphs with colored token tiles instead of tiny prose. Keep lettering crisp, prominent and legible at GitHub README size.
Constraints: this is a conceptual illustration, not experimental data. No charts, performance ratios, invented numbers, awards, benchmark claims, real brand logos or stock watermarks. Do not show web browsing or invented network services; the actual examples use local executable tools. Show a loop using real observations, not training/fine-tuning or magical guaranteed correct answers. Keep all elements inside generous margins. No additional explanatory paragraphs.
```

## Final prompt: Native BF16 scoring

```text
Use case: scientific-educational.
Asset type: vivid conceptual illustration for the English GitHub README of OrbitCall, matching a softly lit isometric 3D Agent-loop illustration.
Create a NEW wide landscape scientific illustration, roughly 16:9, that clearly explains a postprojection suffix-scoring optimization. Use elegant tactile 3D matrix tiles, a substantial GPU/module with circuitry, flowing data ribbons and two parallel lanes. Professional editorial style, off-white background, teal and coral accents, crisp simple labels; lively physical metaphor, not plain rectangular flowchart boxes.
Scientifically critical layout: on the left, ONE shared native BF16 computation path first passes through a full model/backbone GPU object, then a complete full vocabulary projection matrix. BOTH lanes branch ONLY AFTER the complete projection. Make the full projection visually identical and completely present for both lanes. The upper lane, "Native full", processes ALL predictor rows, displayed as a large tall matrix of tokens. The lower lane, "Suffix only", selects only the trailing suffix predictor rows AFTER projection, displayed as a shorter strip, then performs normalization/gather/CPU transfer only for those selected rows. In both lanes, the matrix width remains exactly the same: the FULL vocabulary, not a smaller vocabulary. Upper lane and lower lane each end in a CPU-side score tray. Use ghosted unrequested prefix rows to visually indicate omitted downstream work, with no implication that model/backbone/head computation was skipped.
Text: title exactly "Same projection. Less scoring work." Shared-path labels exactly "BF16 model" and "Full LM head". Lane labels exactly "Native full" and "Suffix only". After selection place the label "Full vocabulary" across the unchanged-width suffix strip, and the end-stage label "CPU scores". Mark the retained first suffix-predictor boundary with "P-1" and the retained last target with "EOS". All text should be short, clear and legible at README width; no paragraphs.
Constraints: this is a conceptual illustration only, not a benchmark chart. No speed values or numerical performance claims. No prefix-cache reuse, no shorter full LM-head projection, no quantization, no FP32 override, no sparse/restricted vocabulary and no suffix tree. The operation being saved is downstream probability normalization, rank/gather and transfer on unrequested prefix positions. Preserve complete suffix coverage including its first predictor and EOS. Keep arrows causal and unambiguous, no branch before projection. No decorative charts, trophies, stock logos or watermarks. Generous margins.
```
