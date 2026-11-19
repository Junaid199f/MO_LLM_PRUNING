The datasets include ARC, HellaSwag, PIQA, WinoGrande, MMLU, GSM8K, HumanEval, and others. These files were copied from the working directory without modification.

Online datasets are loaded using src/llm_pruning/core/data_loader_utils.py. For local use, datasets should be saved using Hugging Face's save_to_disk function and stored in the paths specified in the module. Model caches, downloaded datasets, and evaluation results are not included in the Git repository.
