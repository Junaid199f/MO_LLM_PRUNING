"""Run the full research CLI, including ablation and recovery options."""


def main():
    from .advanced_cli import parse_args

    args = parse_args()
    from .pipeline import main as run

    run(args)


if __name__ == "__main__":
    main()
