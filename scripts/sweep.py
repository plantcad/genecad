import os
import logging
import itertools
import sys
from argparse import ArgumentParser, Namespace
from train import parse_args, train

logger = logging.getLogger(__name__)

def get_configurations() -> list[dict]:
    configs = []
    
    # Base parameters
    architectures = ['encoder-only', 'sequence-only', 'classifier-only', 'all']
    learning_rates = [1e-5, 1e-4, 1e-3]
    frozen_options = ['yes', 'no']
    
    # Generate all combinations, skipping sequence-only with unfrozen base encoder
    for arch, lr, frzn in itertools.product(architectures, learning_rates, frozen_options):
        if arch in ['sequence-only', 'encoder-only', 'all'] and frzn == 'no':
            continue

        configs.append({
            'architecture': arch,
            'learning_rate': lr,
            'base_encoder_frozen': frzn,
            'token_embedding_dim': 128 if arch == "all" else 512,
            'head_encoder_layers': 8,
        })
    
    return configs

def get_sweep_args(args: list[str]) -> tuple[Namespace, int, list[str]]:
    """Get sweep-specific arguments and remaining train.py arguments"""
    parser = ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument('--configuration-index', type=int,
                       help='Index of the configuration to run (0 to num_configs-1). '
                            'Defaults to SLURM_ARRAY_TASK_ID if not provided.')
    parser.add_argument('--output-dir', type=str, default='sweep_results',
                       help='Base output directory for all sweep results')
    parser.add_argument('--run-name', type=str, default='sweep_01',
                       help='Prefix for the run name (default: sweep_01)')
    
    # Parse known args to separate sweep args from train args
    sweep_args, remaining_args = parser.parse_known_args(args)
    
    # Get configuration index from args or environment
    config_idx = sweep_args.configuration_index
    if config_idx is None:
        slurm_id = os.environ.get('SLURM_ARRAY_TASK_ID')
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
            f"Configuration index {config_idx} is out of range (0-{len(configs)-1})"
        )
    
    return sweep_args, config_idx, remaining_args

def main() -> None:
    # Parse sweep-specific and training arguments
    sweep_args, config_idx, train_args = get_sweep_args(sys.argv[1:])
    
    # Get all configurations and select the requested one
    configs = get_configurations()
    logger.info(f"Running configuration {config_idx} of {len(configs)}")
    logger.info(f"Configurations:\n{configs}")
    config = configs[config_idx]
    
    # Create run name and output directory
    run_name = "__".join([
        sweep_args.run_name,
        f"cfg_{config_idx:03d}",
        f"arch_{config['architecture']}",
        f"frzn_{config['base_encoder_frozen']}",
        f"lr_{config['learning_rate']:.0e}"
    ])
    output_dir = os.path.join(sweep_args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Add output directory and run name to train args before parsing
    train_args.extend(['--output-dir', output_dir, '--run-name', run_name])
    base_args = parse_args(train_args)
    
    # Update arguments with current configuration
    args_dict = vars(base_args)
    args_dict.update(config)
    
    # Run training
    logger.info(f"Starting run {config_idx} with config: {config}")
    train(Namespace(**args_dict))
    logger.info(f"Completed run {config_idx}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main() 