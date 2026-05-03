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


def cmd_ui(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="genecad ui",
        description="Launch the interactive Web UI.",
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio tunnel link"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the UI on (default: 7860)"
    )
    parser.add_argument(
        "--auth-file",
        default=None,
        metavar="FILE",
        help=(
            "Path to a credentials file (one 'user:password' per line). "
            "Alternatively set GENECAD_AUTH='user1:pass1,user2:pass2' in the environment."
        ),
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Explicitly disable authentication. Only use on trusted local networks.",
    )
    args = parser.parse_args(argv)

    try:
        import gradio as gr
    except ImportError:
        print(
            "Error: gradio is not installed. Please install it using 'uv pip install gradio'."
        )
        return 1

    auth = _load_auth(args.auth_file)
    if auth is None:
        if not args.no_auth:
            print(
                "Error: authentication is required to run the GeneCAD web UI.\n"
                "\n"
                "  Option 1 — credentials file:\n"
                "    genecad ui --auth-file /path/to/credentials.txt\n"
                "    (one 'username:password' per line; # lines are comments)\n"
                "\n"
                "  Option 2 — environment variable:\n"
                "    export GENECAD_AUTH='alice:secret,bob:other'\n"
                "    genecad ui\n"
                "\n"
                "  Option 3 — disable auth (local/trusted networks only):\n"
                "    genecad ui --no-auth",
                file=sys.stderr,
            )
            return 1
        print("Warning: running without authentication (--no-auth).", file=sys.stderr)

    # Import dynamically so the headless CLI doesn't strictly depend on Gradio early on.
    ui_module = importlib.import_module("src.ui")
    create_ui = getattr(ui_module, "create_ui")
    ui_css = getattr(ui_module, "UI_CSS")

    demo = create_ui()

    # We use server_name="0.0.0.0" to ensure it's accessible over network
    # We use queue() to enable streaming output
    demo.queue(max_size=10, default_concurrency_limit=1).launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.port,
        allowed_paths=[os.getcwd()],
        auth=auth,
        auth_message="Enter your GeneCAD credentials to continue.",
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.gray,
        ).set(
            block_border_width="0px",
            panel_border_width="0px",
            button_border_width="0px",
            input_border_width="0px",
            checkbox_border_width="0px",
        ),
        css=ui_css,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="genecad",
        description="GeneCAD: end-to-end genome annotation powered by PlantCAD2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["predict", "train", "evaluate", "summarize", "ui"],
        help="Sub-command to run",
    )
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments forwarded to the sub-command"
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
