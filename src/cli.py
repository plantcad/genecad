#!/usr/bin/env python3
import argparse
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
    p.add_argument("-i", "--input", default=None,
                   help="Input genome FASTA (default: downloads Arabidopsis TAIR12 example)")
    p.add_argument("-o", "--output", default=None,
                   help="Output directory (default: ./genecad_result/<species>_predictions)")
    p.add_argument("-s", "--species", default="Athaliana",
                   help="Species label on output filenames (default: Athaliana)")
    p.add_argument("-m", "--mode", default="plant", choices=["plant", "animal"],
                   help="Model to use (default: plant)")
    p.add_argument("-n", "--top-n-contigs", default="all", dest="top_n_contigs",
                   help="Process only the N longest contigs (default: all)")
    p.add_argument("-l", "--min-transcript-length", default="3",
                   dest="min_transcript_length",
                   help="Minimum transcript length in bp (default: 3)")
    p.add_argument("-c", "--cpu-workers", default="1", dest="cpu_workers",
                   help="CPU worker processes for GFF export (default: 1)")
    p.add_argument("-b", "--batch-size", default="auto", dest="batch_size",
                   help="Inference batch size per GPU (default: auto)")
    p.add_argument("-g", "--gpus", default="0",
                   help="GPU IDs: comma-separated list or 'all' (default: 0)")
    p.add_argument("--launcher", default="",
                   help="Custom launcher command (e.g. 'srun python')")
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
        "bash", str(script_dir / "predict.sh"),
        "-i", input_file,
        "-o", output_dir,
        "-s", args.species,
        "-m", args.mode,
        "-n", args.top_n_contigs,
        "-l", args.min_transcript_length,
        "-c", args.cpu_workers,
        "-b", args.batch_size,
        "-g", args.gpus,
    ]
    if args.launcher:
        cmd += ["--launcher", args.launcher]

    env = os.environ.copy()
    # Ensure src/ is importable when called from the installed wheel
    env["PYTHONPATH"] = str(script_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["GENECAD_PYTHON"] = sys.executable

    result = subprocess.run(cmd, env=env)
    return result.returncode


def cmd_ui(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="genecad ui",
        description="Launch the interactive Web UI.",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio tunnel link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the UI on (default: 7860)")
    args = parser.parse_args(argv)

    try:
        import gradio as gr
    except ImportError:
        print("Error: gradio is not installed. Please install it using 'uv pip install gradio'.")
        return 1

    # Importing dynamically so the headless CLI doesn't strictly depend on Gradio early on
    from src.ui import create_ui, UI_CSS
    demo = create_ui()
    
    # We use server_name="0.0.0.0" to ensure it's accessible over network
    # We use queue() to enable streaming output
    demo.queue().launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.port,
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.gray,
        ).set(
            block_border_width="0px",
            panel_border_width="0px",
            button_border_width="0px",
            input_border_width="0px",
            checkbox_border_width="0px"
        ),
        css=UI_CSS,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="genecad",
        description="GeneCAD: end-to-end genome annotation powered by PlantCAD2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["predict", "ui"],
                        help="Sub-command to run")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                        help="Arguments forwarded to the sub-command")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    top = parser.parse_args()

    if top.command == "predict":
        sys.exit(cmd_predict(top.args))
    elif top.command == "ui":
        sys.exit(cmd_ui(top.args))


if __name__ == "__main__":
    main()
