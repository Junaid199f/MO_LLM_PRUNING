"""Run the paper-budget profile directly from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from llm_pruning.cli import main

if __name__ == "__main__":
    main(["--profile", "paper", *sys.argv[1:]])
