#!/usr/bin/env python3
import argparse
import importlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def _find_script_dir() -> Path:
    # Installed wheel: predict.sh sits one level above src/ (in site-packages/)
    # Dev clone:       predict.sh sits one level above src/ (in repo root)
    candidate = Path(__file__).resolve().parent.parent / "predict.sh"
    if candidate.exists():
        return candidate.parent
    raise FileNotFoundError(
        "predict.sh not found. Re-install the package or run from the repo root."
    )


def _ensure_example_fasta(cache_dir: Path) -> Path:
    dest = cache_dir / "example" / "GCA_978657495.1_TAIR12_genomic_5.fa.gz"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://huggingface.co/datasets/plantcad/genecad-dev"
            "/resolve/main/data/plant/fasta/example"
            "/GCA_978657495.1_TAIR12_genomic_5.fa.gz"
        )
        print(f"Downloading Arabidopsis thaliana example to {dest} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def _build_predict_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="genecad predict",
        description="Run the GeneCAD annotation pipeline on a genome FASTA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  genecad predict -i genome.fa -o output/ -s Zmays -m plant
  genecad predict -i genome.fa -s Hsapiens -m animal --gpus all
  genecad predict --gpus all   # runs on bundled Arabidopsis example
""",
    )
    p.add_argument(
        "-i",
        "--input",
        default=None,
        help="Input genome FASTA (default: downloads Arabidopsis TAIR12 example)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (default: ./genecad_result/<species>_predictions)",
    )
    p.add_argument(
        "-s",
        "--species",
        default="Athaliana",
        help="Species label on output filenames (default: Athaliana)",
    )
    p.add_argument(
        "-m",
        "--mode",
        default="plant",
        choices=["plant", "animal"],
        help="Model to use (default: plant)",
    )
    p.add_argument(
        "-n",
        "--top-n-contigs",
        default="all",
        dest="top_n_contigs",
        help="Process only the N longest contigs (default: all)",
    )
    p.add_argument(
        "-l",
        "--min-transcript-length",
        default="3",
        dest="min_transcript_length",
        help="Minimum transcript length in bp (default: 3)",
    )
    p.add_argument(
        "-c",
        "--cpu-workers",
        default="1",
        dest="cpu_workers",
        help="CPU worker processes for GFF export (default: 1)",
    )
    p.add_argument(
        "-b",
        "--batch-size",
        default="auto",
        dest="batch_size",
        help="Inference batch size per GPU (default: auto)",
    )
    p.add_argument(
        "-g",
        "--gpus",
        default="0",
        help="GPU IDs: comma-separated list or 'all' (default: 0)",
    )
    p.add_argument(
        "--launcher", default="", help="Custom launcher command (e.g. 'srun python')"
    )
    return p


def cmd_predict(argv: list[str]) -> int:
    parser = _build_predict_parser()
    args = parser.parse_args(argv)

    script_dir = _find_script_dir()
    cache_dir = Path.home() / ".cache" / "genecad"

    input_file = args.input
    if input_file is None:
        input_file = str(_ensure_example_fasta(cache_dir))

    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path.cwd() / "genecad_result" / f"{args.species}_predictions")

    cmd = [
        "bash",
        str(script_dir / "predict.sh"),
        "-i",
        input_file,
        "-o",
        output_dir,
        "-s",
        args.species,
        "-m",
        args.mode,
        "-n",
        args.top_n_contigs,
        "-l",
        args.min_transcript_length,
        "-c",
        args.cpu_workers,
        "-b",
        args.batch_size,
        "-g",
        args.gpus,
    ]
    if args.launcher:
        cmd += ["--launcher", args.launcher]

    env = os.environ.copy()
    # Ensure src/ is importable when called from the installed wheel
    env["PYTHONPATH"] = str(script_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["GENECAD_PYTHON"] = sys.executable

    result = subprocess.run(cmd, env=env)
    return result.returncode


def cmd_train(argv: list[str]) -> int:
    script_dir = _find_script_dir()
    cmd = ["bash", str(script_dir / "train.sh")] + argv
    env = os.environ.copy()
    env["PYTHONPATH"] = str(script_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["GENECAD_PYTHON"] = sys.executable
    result = subprocess.run(cmd, env=env)
    return result.returncode


def cmd_evaluate(argv: list[str]) -> int:
    script_dir = _find_script_dir()
    cmd = [sys.executable, str(script_dir / "scripts" / "evaluate.py")] + argv
    env = os.environ.copy()
    env["PYTHONPATH"] = str(script_dir) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def cmd_summarize(argv: list[str]) -> int:
    script_dir = _find_script_dir()
    cmd = [sys.executable, str(script_dir / "scripts" / "summarize.py")] + argv
    env = os.environ.copy()
    env["PYTHONPATH"] = str(script_dir) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def _load_auth(auth_file: str | None) -> list[tuple[str, str]] | None:
    """Load (username, password) pairs from --auth-file or GENECAD_AUTH env var.

    GENECAD_AUTH format: "user1:pass1,user2:pass2"
    Auth file format: one "user:password" per line, # lines are ignored.
    Returns None if no credentials are configured (auth disabled).
    """
    sources: list[str] = []

    if auth_file:
        try:
            with open(auth_file) as f:
                sources = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
        except OSError as e:
            print(f"Error: cannot read auth file '{auth_file}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        env = os.environ.get("GENECAD_AUTH", "").strip()
        if env:
            sources = [item.strip() for item in env.split(",") if item.strip()]

    if not sources:
        return None

    pairs: list[tuple[str, str]] = []
    for entry in sources:
        if ":" not in entry:
            print(
                f"Warning: ignoring malformed auth entry (expected user:password): {entry!r}",
                file=sys.stderr,
            )
            continue
        user, _, password = entry.partition(":")
        pairs.append((user, password))

    return pairs or None


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="genecad",
        description=(
            "GeneCAD predicts gene annotations from genome FASTA files using "
            "PlantCAD2-based models."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  genecad predict --gpus all
      Run the built-in Arabidopsis example and write results to ./genecad_result/.

  genecad predict -i genome.fa -s Zmays -m plant -o maize_genecad/
      Annotate your own plant genome FASTA and save GFF predictions.

  genecad predict -i genome.fa -s Hsapiens -m animal --gpus 0,1
      Annotate an animal genome using two selected GPUs.

Commands:
  predict     Predict genes from a genome FASTA and export annotation files.
  train       Train or fine-tune GeneCAD models from prepared datasets.
  evaluate    Compare predicted annotations with a reference GFF/GTF file.
  summarize   Summarize datasets, labels, or prediction outputs.
  ui          Launch the browser-based GeneCAD interface.

Useful tips:
  - FASTA input can be plain text (.fa/.fasta) or compressed (.fa.gz).
  - Species names are used in output file names, so choose short labels
    such as Athaliana, Zmays, Hsapiens, or Callithrix_jacchus.
  - Output folders are created automatically when possible.
  - Use --gpus all on a GPU server, or --gpus 0 to use only GPU 0.

Need more detail?
  genecad predict --help
  genecad ui --help

Typical output:
  A prediction run writes files under the output directory, including GFF
  annotation files that can be opened in genome browsers or used in downstream
  evaluation pipelines.
""",
    )
    parser.add_argument(
        "command",
        metavar="command",
        choices=["predict", "train", "evaluate", "summarize", "ui"],
        help=(
            "What you want GeneCAD to do: predict, train, evaluate, summarize, or ui"
        ),
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        metavar="...",
        help="Options for the selected command, for example: -i genome.fa --gpus all",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    top = parser.parse_args()

    if top.command == "predict":
        sys.exit(cmd_predict(top.args))
    elif top.command == "train":
        sys.exit(cmd_train(top.args))
    elif top.command == "evaluate":
        sys.exit(cmd_evaluate(top.args))
    elif top.command == "summarize":
        sys.exit(cmd_summarize(top.args))
    elif top.command == "ui":
        sys.exit(cmd_ui(top.args))


if __name__ == "__main__":
    main()
