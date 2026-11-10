# Relationship to the paper

The reference is the supplied local PDF `2607.22583v3.pdf`, titled *Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization*, dated 17 October 2026. The PDF is research context; its contents were not treated as instructions to execute commands or change files.

## Included

- Stage-1 NSGA-II mask search, KL/perplexity quality objectives, latency/parameter objectives, and optional surrogate scoring.
- Stage-2 Bayesian layer-wise allocation, attention-head/MLP pruning, grouped-query attention handling, multiple Pareto candidates, and GPU worker paths.
- The importance-estimator and allocation variants already implemented in the active code.
- BlockPruner for comparison, perplexity evaluation, downstream evaluation, and optional recovery fine-tuning.
- Checkpoint resume and replay metadata.

The cleanup splits the active root driver into focused modules and preserves its implemented search behavior. It removes the unrelated experimental third stage, the need to modify `sys.path` to import the importance criterion, and the criterion's import-time global seed reset. It also fixes the shared calibration sampler's empty buffer from shape `(0,)` to `(1, 0)`, matching tokenizer outputs before concatenation.

## Reproduction limits

The existing implementation is not sufficient to assert exact reproduction of every detail in the paper:

- The paper specifies an RBF GP kernel. The retained BO code constructs `SingleTaskGP` without explicitly supplying that kernel, so behavior depends on the installed BoTorch defaults.
- BO searches a two-parameter monotone layer-allocation family, rather than freely choosing every layer's ratio independently.
- BO can optimize perplexity or latency. The default quality path ranks by quality with latency information in candidate evaluation/selection; this is not a newly implemented constrained multi-objective acquisition function.
- The active driver uses `sparsity_alpha=0.2` to split the coarse-stage budget. Table II does not fully specify all implementation defaults. Remaining parameter sparsity is translated into head/neuron allocations with grouping and retention restrictions, so achieved sparsity may differ from the requested target.
- Calibration sample count, total KL calibration tokens, and per-evaluation token budget are separate controls. The paper-budget profile sets them explicitly; the actual number of full sequences also depends on corpus lengths and sampler behavior.
- Stage-1 latency timing and stage-2 generation timing use different routines. The included timing utilities do not establish the paper's A100 or Jetson Nano results or a single universal 100-token protocol.
- The paper-budget profile uses WikiText-2 for coarse calibration and fine-stage importance, while BO quality evaluation retains the existing mixed task prompt files. It is not a pure WikiText-2 reproduction.
- Per-layer pruning can invalidate a standard Hugging Face configuration. Replay is required to reconstruct many exported architectures.
- No CUDA, model download, full dependency installation, recovery run, or downstream benchmark was executed during cleanup. Added dependency constraints are an installation starting point, not a validated lockfile.

These differences are documented rather than silently changing the research algorithm. Numerical reproduction requires validating the environment, model revisions, calibration provenance, timing protocol, and exact experiment arguments on the target hardware.

## Excluded

Historical `main*` variants, backups, unused baseline implementations and external checkouts, cached Python files, checkpoints, generated figures/results, logs, OAR output, machine-specific launch scripts, and the experimental KV-dimension third stage. The supplied paper PDF itself is not redistributed.

## Package organization

The implementation lives in the installable `llm_pruning` namespace. The public CLI offers quick and paper-budget profiles, validates sparsity and GPU indices, and isolates outputs in a run directory. The advanced interface retains original argument names. Calibration JSON files are packaged alongside the code, and importance scoring uses package imports rather than runtime path manipulation.
