import argparse
import sys
import os
from src.analysis import get_token_class_weights
from src.modeling import TOKEN_CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute plant-style class weights from a Zarr dataset. "
            "Matches the mathematical logic in src/analysis.py and provides "
            "diagnostic output for verification."
        )
    )
    parser.add_argument("--dataset", required=True, help="Path to the training Zarr dataset")
    parser.add_argument(
        "--split", 
        type=str, 
        default="train", 
        help="Split name (used if --dataset is a directory containing shards)"
    )
    args = parser.parse_args()

    # Use the core analysis function to compute weights
    # This automatically handles both single Zarr directories and sharded datasets.
    # We pass the path directly; get_token_class_weights handles the logic.
    dataset, weights = get_token_class_weights(
        path=args.dataset,
        split=args.split,
        class_names=TOKEN_CLASS_NAMES
    )

    # --- Diagnostic Output (to stderr so bash scripts ignore it) ---
    
    print("\n--- Class Frequency Table ---", file=sys.stderr)
    print(weights[["label_name", "label_freq", "label_weight"]].to_string(index=False), file=sys.stderr)
    
    print("\n--- TOKEN_CLASS_FREQUENCIES dictionary (for src/modeling.py) ---", file=sys.stderr)
    for _, row in weights.iterrows():
        # Format identical to src/modeling.py
        label_key = row['label_name'].upper().replace('-', '_')
        print(f"    TokenBiluoClass.{label_key}.value: {row['label_freq']:e},", file=sys.stderr)
    
    print("\n------------------------------\n", file=sys.stderr)

    # --- Bash Compatibility Output (to stdout) ---
    
    # Print as a single space-separated string of weights
    # Note: Bash scripts like train_animal_multispecies.sh grab this via CLASS_WEIGHTS=$(...)
    print(" ".join([f"{w:.6f}" for w in weights["label_weight"].values]))


if __name__ == "__main__":
    main()