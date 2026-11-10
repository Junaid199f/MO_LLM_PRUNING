# Calibration data

The release preserves the active pipeline's existing prompt-only JSON files:

- `src/llm_pruning/data/calibration/*_25.json`: default mixed-prompt BO quality inputs.
- `src/llm_pruning/data/calibration200/*_200.json`: preferred inputs for optional stage-1 importance scoring.

The task names include ARC, HellaSwag, PIQA, WinoGrande, MMLU, GSM8K, HumanEval, and other sources. These files are copied unchanged from the working directory. Original sampling code, source dataset revisions, split identifiers, and redistribution permissions are not fully recorded here; this release does not assign a new license to their content.

Using examples from a benchmark for calibration can affect the interpretation of evaluations on that benchmark. Maintain split separation and record the calibration sources for published experiments.

Online corpora are loaded by `src/llm_pruning/core/data_loader_utils.py`. Local mode expects Hugging Face datasets saved with `save_to_disk`, using the paths in that module. Model caches, online datasets, and evaluation results are excluded from Git.
