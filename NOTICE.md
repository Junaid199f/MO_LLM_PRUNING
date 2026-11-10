# Source attribution

This folder reorganizes the supplied research workspace. The parent workspace's README identifies its foundation as **2SSP: A Two-Stage Framework for Structured Pruning of LLMs**, Fabrizio Sandri, Elia Cunegatti, and Giovanni Iacca (TMLR 2025), https://github.com/FabrizioSandri/2SSP. Its GPL v3 license text is retained. This cleanup does not claim authorship of inherited code.

The new README identifies the supplied Ali, Niar, and Talbi paper instead of presenting the inherited 2SSP README as this project's paper.

Preserved source comments identify adapted scoring and evaluation code from **EvoPress**, https://github.com/IST-DASLab/EvoPress. Evaluation uses **Language Model Evaluation Harness**, https://github.com/EleutherAI/lm-evaluation-harness, installed as a dependency. The workspace's structured-pruning backends include implementations of Wanda-SP and FLAP-style scoring and WActiGrad-style importance; their method names are preserved. The source provenance of these adaptations is not fully documented in the supplied workspace.

SliceGPT, AMP, WActiGrad, and evaluation-harness repository checkouts are not redistributed in this folder. Calibration examples and downloaded datasets remain subject to the terms of their original sources. No model weights are included.
