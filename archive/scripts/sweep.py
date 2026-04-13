import os
import logging
import itertools
import sys
from argparse import ArgumentParser, Namespace

# pyrefly: ignore  # import-error
from train import parse_args, train
from src.logging import rank_zero_logger

logger = rank_zero_logger(logging.getLogger(__name__))


def get_configurations() -> list[dict]:
    configs = []

    # Base parameters
    randomize_base = ["no", "yes"]  # Put first, starts with "no"
    architectures = ["encoder-only", "sequence-only", "classifier-only", "all"]
    learning_rates = [1e-5, 1e-4, 1e-3]
    frozen_options = ["yes", "no"]

    # Generate all combinations, with filtering rules
    for rand, arch, lr, frzn in itertools.product(
        randomize_base, architectures, learning_rates, frozen_options
    ):
        # Skip sequence-only, encoder-only, and all with unfrozen base encoder
        if arch in ["sequence-only", "encoder-only", "all"] and frzn == "no":
            continue

        # Skip all configurations where randomize_base=yes except where frzn=yes and architecture=all
        if rand == "yes" and not (frzn == "yes" and arch == "all"):
            continue

        configs.append(
            {
                "randomize_base": rand,
                "architecture": arch,
                "learning_rate": lr,
                "base_encoder_frozen": frzn,
                "token_embedding_dim": 128 if arch == "all" else 512,
                "head_encoder_layers": 8,
            }
        )

    return configs


def get_sweep_args(args: list[str]) -> tuple[Namespace, int, list[str]]:
    """Get sweep-specific arguments and remaining train.py arguments"""
    parser = ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument(
        "--configuration-index",
        type=int,
        help="Index of the configuration to run (0 to num_configs-1). "
        "Defaults to SLURM_ARRAY_TASK_ID if not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sweep_results",
        help="Base output directory for all sweep results",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="sweep_01",
        help="Prefix for the run name (default: sweep_01)",
    )

    # Parse known args to separate sweep args from train args
    sweep_args, remaining_args = parser.parse_known_args(args)

    # Get configuration index from args or environment
    config_idx = sweep_args.configuration_index
    if config_idx is None:
        slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_id is None:
            raise ValueError(
                "Must provide either --configuration-index argument or "
                "set SLURM_ARRAY_TASK_ID environment variable"
            )
        config_idx = int(slurm_id)

    # Validate configuration index
    configs = get_configurations()
    if config_idx >= len(configs):
        raise ValueError(
            f"Configuration index {config_idx} is out of range (0-{len(configs) - 1})"
        )

    return sweep_args, config_idx, remaining_args


def get_run_name(sweep_args, config_idx, config):
    """Generate a run name from configuration parameters"""
    return "__".join(
        [
            sweep_args.run_name,
            f"cfg_{config_idx:03d}",
            f"rand_{config['randomize_base']}",
            f"arch_{config['architecture']}",
            f"frzn_{config['base_encoder_frozen']}",
            f"lr_{config['learning_rate']:.0e}",
        ]
    )


def cmd_run(args):
    """Run a single configuration from the sweep"""
    # Parse sweep-specific and training arguments
    sweep_args, config_idx, train_args = get_sweep_args(args)

    # Get all configurations and select the requested one
    configs = get_configurations()
    logger.info(f"Running configuration {config_idx} of {len(configs)}")
    config = configs[config_idx]

    # Create run name and output directory
    run_name = get_run_name(sweep_args, config_idx, config)
    output_dir = os.path.join(sweep_args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Add output directory and run name to train args before parsing
    train_args.extend(["--output-dir", output_dir, "--run-name", run_name])
    base_args = parse_args(train_args)

    # Update arguments with current configuration
    args_dict = vars(base_args)
    args_dict.update(config)

    # Run training
    logger.info(f"Starting run {config_idx} with config: {config}")
    train(Namespace(**args_dict))
    logger.info(f"Completed run {config_idx}")


def cmd_show_configs(args):
    """Display all configurations and their run names"""
    configs = get_configurations()

    # Use the provided run_name as prefix
    sweep_args = Namespace(run_name=args.run_name)

    print(f"Total configurations: {len(configs)}")
    print("=" * 80)

    for idx, config in enumerate(configs):
        run_name = get_run_name(sweep_args, idx, config)
        print(f"Config {idx:3d} | {run_name}")
        for key, value in config.items():
            print(f"  {key:20s}: {value}")
        print("-" * 80)


def setup_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Hyperparameter sweep CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a single configuration")
    run_parser.add_argument(
        "--configuration-index",
        type=int,
        help="Index of the configuration to run (0 to num_configs-1). "
        "Defaults to SLURM_ARRAY_TASK_ID if not provided.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default="sweep_results",
        help="Base output directory for all sweep results",
    )
    run_parser.add_argument(
        "--run-name",
        type=str,
        default="sweep_01",
        help="Prefix for the run name (default: sweep_01)",
    )

    # Show configs command with run-name argument
    show_parser = subparsers.add_parser(
        "show_configs", help="Display all configurations"
    )
    show_parser.add_argument(
        "--run-name",
        type=str,
        default="sweep_01",
        help="Prefix for the run name (default: sweep_01)",
    )

    return parser


def main() -> None:
    parser = setup_parser()
    args, _ = parser.parse_known_args()

    if args.command == "run":
        cmd_run(sys.argv[2:])  # Skip 'sweep.py run'
    elif args.command == "show_configs":
        cmd_show_configs(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
