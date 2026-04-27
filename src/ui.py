import os
import re
import subprocess
import urllib.request
from pathlib import Path
import gradio as gr

_TQDM_RE = re.compile(
    r'\[GPU\s+(\d+)[^\]]*\]:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)'
)


UI_CSS = """
/* ══════════════════════════════════════════════════════
   GeneCAD — Ultimate Apple-style UI
   ══════════════════════════════════════════════════════ */

/* Force Apple Fonts */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
}

/* Light Mode Variables (Apple) */
.gradio-container {
    --body-background-fill: #f5f5f7 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f2f2f7 !important;
    --block-background-fill: #ffffff !important;
    --card-bg: #ffffff !important;
    --block-border-width: 1px !important;
    --block-shadow: 0 4px 24px rgba(0,0,0,0.04) !important;
    --block-radius: 20px !important;
    --input-background-fill: #f2f2f7 !important;
    --input-border-width: 0px !important;
    --input-radius: 12px !important;
    --color-accent: #0071e3 !important;
    --body-text-color: #1d1d1f !important;
    --body-text-color-subdued: #86868b !important;
}

/* Dark Mode Variables (Apple) */
.dark, .dark .gradio-container {
    --body-background-fill: #000000 !important;
    --background-fill-primary: #000000 !important;
    --background-fill-secondary: #1c1c1e !important;
    --block-background-fill: #1c1c1e !important;
    --card-bg: #1c1c1e !important;
    --block-border-width: 1px !important;
    --block-border-color: rgba(255,255,255,0.08) !important;
    --block-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
    --input-background-fill: #2c2c2e !important;
    --input-border-width: 0px !important;
    --color-accent: #0a84ff !important;
    --body-text-color: #f5f5f7 !important;
    --body-text-color-subdued: #86868b !important;
}

/* Base typography */
.gradio-container * {
    box-sizing: border-box;
}

/* Hero Section */
.gene-hero {
    text-align: center;
    padding: 4rem 1.5rem 2rem;
    max-width: 1100px;
    margin: 0 auto;
}
.gene-eyebrow {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    background: rgba(0, 113, 227, 0.1);
    color: #0071e3;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.dark .gene-eyebrow {
    background: rgba(10, 132, 255, 0.2);
    color: #0a84ff;
}
.gene-title {
    margin: 0 0 0.8rem 0;
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.05;
}
.gene-subtitle {
    font-size: 1.15rem;
    line-height: 1.6;
    color: var(--body-text-color-subdued);
    max-width: 54rem;
    margin: 0 auto;
}

/* Quick-start Card */
.gene-qs-outer {
    max-width: 1100px;
    margin: 0 auto 1.5rem;
    padding: 0 1.5rem;
}

.gene-qs-badge {
    display: inline-block;
    background: #34c759;
    color: #ffffff;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}
.dark .gene-qs-badge {
    background: #32d74b;
    color: #000000;
}
.gene-qs h3 {
    margin: 0 0 0.4rem 0;
    font-size: 1.1rem;
    font-weight: 600;
}
.gene-qs p {
    margin: 0;
    font-size: 0.9rem;
    color: var(--body-text-color-subdued);
    line-height: 1.6;
}
.gene-qs code {
    font-family: "SF Mono", ui-monospace, monospace;
    background: var(--input-background-fill);
    padding: 0.15em 0.4em;
    border-radius: 6px;
    font-size: 0.85em;
}

/* Group Cards */
.gradio-container .group,
.gene-card {
    background: var(--card-bg) !important;
    border-radius: 20px !important;
    border: 1px solid var(--block-border-color) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.03) !important;
    overflow: hidden !important;
}
.gradio-container .group .form,
.gradio-container .group [class*="form"],
.gene-card .form,
.gene-card [class*="form"],
.gene-card .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Step Labels */
.gene-step {
    display: block;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--color-accent);
    margin: 0 0 0.25rem 0;
}
.gene-head {
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin: 0 0 0.3rem 0;
    color: var(--body-text-color);
}
.gene-hint {
    font-size: 0.9rem;
    color: var(--body-text-color-subdued);
    margin: 0 0 1rem 0;
    line-height: 1.5;
}

/* Layout */
.gene-layout {
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 1.5rem 3rem;
}

/* Inputs and Forms */
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea {
    background: var(--input-background-fill) !important;
    border: 2px solid transparent !important;
    border-radius: var(--input-radius) !important;
    color: var(--body-text-color) !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    padding: 0.75rem 1rem !important;
    box-shadow: none !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container input[type="number"]:focus,
.gradio-container textarea:focus {
    border-color: var(--color-accent) !important;
    background: var(--block-background-fill) !important;
}

/* Hide Gradio's ugly label backgrounds */
.gradio-container .label-wrap > span,
.gradio-container .block > label > span:first-child {
    background: transparent !important;
    color: var(--body-text-color-subdued) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0 !important;
    margin-bottom: 0.4rem !important;
    border: none !important;
}

/* Compact example checkbox row */
.gene-qs-outer {
    max-width: 1100px !important;
    margin: 0 auto 0.5rem !important;
    padding: 0 1.5rem !important;
    gap: 0 !important;
}
.gene-qs-outer > *,
.gene-qs-outer .gap {
    gap: 0 !important;
}
#example_checkbox,
.gradio-container #example_checkbox,
.gradio-container .block.example-checkbox {
    margin: 0 !important;
    padding: 0.4rem 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    min-height: unset !important;
}
#example_checkbox label.checkbox-container,
.gradio-container #example_checkbox label.checkbox-container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Logs */
#logs_box textarea {
    background: #000000 !important;
    color: #32d74b !important; /* Terminal green */
    border-radius: 12px !important;
    font-family: "SF Mono", ui-monospace, monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
    padding: 1rem !important;
}
.light #logs_box textarea {
    background: #f2f2f7 !important;
    color: #1d1d1f !important;
}

/* Buttons */
.gradio-container button {
    border-radius: 99px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    min-height: 44px !important;
    transition: all 0.2s ease !important;
    border: none !important;
}
.gradio-container button.primary {
    background: var(--color-accent) !important;
    color: #ffffff !important;
}
.gradio-container button.primary:hover {
    filter: brightness(1.1) !important;
    transform: scale(1.02) !important;
}
.gradio-container button.primary:active {
    transform: scale(0.98) !important;
}
.gradio-container button.secondary {
    background: var(--input-background-fill) !important;
    color: var(--body-text-color) !important;
}
.gradio-container button.secondary:hover {
    filter: brightness(0.95) !important;
    transform: scale(1.02) !important;
}
.dark .gradio-container button.secondary:hover {
    filter: brightness(1.2) !important;
}
.gradio-container button.secondary:active {
    transform: scale(0.98) !important;
}
#run_pipeline {
    margin-top: 1.5rem !important;
    min-height: 52px !important;
    font-size: 1.05rem !important;
}

/* Tiny buttons */
.gradio-container .textbox button,
.gradio-container [data-testid="textbox"] button,
.gradio-container .copy-btn {
    border-radius: 8px !important;
    background: var(--input-background-fill) !important;
    min-height: 32px !important;
    min-width: 32px !important;
}

/* File Upload */
#upload_box, #download_box,
.gradio-container .file-preview,
.gradio-container [data-testid="file-upload"],
.gradio-container .upload-container,
#upload_box .upload-container,
#upload_box [style*="dashed"],
#download_box [style*="dashed"] {
    background: var(--input-background-fill) !important;
    border: none !important;
    border-style: none !important;
    border-width: 0px !important;
    border-radius: 16px !important;
    transition: all 0.2s ease !important;
}
.gradio-container .border-dashed {
    border: none !important;
    border-style: none !important;
}
#upload_box *,
#download_box *,
.gradio-container [data-testid="file-upload"] *,
.gradio-container .file-preview *,
.gradio-container .upload-container * {
    border-style: none !important;
    border-color: transparent !important;
}
.gradio-container [data-testid="file-upload"] button,
.gradio-container [data-testid="file-upload"] > div {
    border: none !important;
}
#upload_box:hover, #download_box:hover,
.gradio-container [data-testid="file-upload"]:hover,
.gradio-container .upload-container:hover {
    background: rgba(0, 113, 227, 0.05) !important;
}
.dark #upload_box:hover, .dark #download_box:hover,
.dark .gradio-container [data-testid="file-upload"]:hover,
.dark .gradio-container .upload-container:hover {
    background: rgba(10, 132, 255, 0.05) !important;
}

/* GFF preview box — fixed height, scrollable, monospace */
#gff_preview_box textarea {
    font-family: "SF Mono", ui-monospace, monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
    resize: none !important;
}

/* Accordion */
.advanced-accordion {
    background: transparent !important;
    border: none !important;
}
.gradio-container details.advanced-accordion,
.advanced-accordion details {
    background: var(--input-background-fill) !important;
    border-radius: 14px !important;
    border: none !important;
    margin-top: 1rem !important;
}
/* Quick Start Section */
.gene-qs {
    background: var(--card-bg) !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    border: 1px solid var(--block-border-color) !important;
}
.gradio-container details summary {
    padding: 1rem 1.2rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: var(--body-text-color) !important;
}
"""

EXAMPLE_FASTA_URL = (
    "https://huggingface.co/datasets/plantcad/genecad-dev"
    "/resolve/main/data/plant/fasta/example"
    "/GCA_978657495.1_TAIR12_genomic_5.fa.gz"
)
EXAMPLE_FASTA_PATH = (
    Path.home() / ".cache" / "genecad" / "example"
    / "GCA_978657495.1_TAIR12_genomic_5.fa.gz"
)


def ensure_example_fasta() -> str:
    if EXAMPLE_FASTA_PATH.exists():
        return str(EXAMPLE_FASTA_PATH)
    EXAMPLE_FASTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(EXAMPLE_FASTA_URL, EXAMPLE_FASTA_PATH)
    return str(EXAMPLE_FASTA_PATH)


def on_use_example_change(use_ex: bool):
    if use_ex:
        return (
            gr.update(value="Athaliana"),
            "plant",
            gr.update(value="genecad_result/Athaliana_predictions"),
            gr.update(variant="primary"),    # plant_btn active
            gr.update(variant="secondary"),  # animal_btn inactive
        )
    return (
        gr.update(value="MySpecies"),
        "plant",
        gr.update(value="genecad_result"),
        gr.update(variant="primary"),
        gr.update(variant="secondary"),
    )


def select_plant():
    return "plant", gr.update(variant="primary"), gr.update(variant="secondary")


def select_animal():
    return "animal", gr.update(variant="secondary"), gr.update(variant="primary")


def run_genecad(
    fasta_upload, fasta_path, output_dir, species, domain,
    top_n_contigs, min_transcript_length, cpu_workers,
    batch_size, gpus, use_example,
    progress=gr.Progress(),
):
    input_file = fasta_path.strip() if fasta_path else ""

    _nop = (gr.update(), gr.update(), gr.update(), gr.update())

    if use_example:
        yield "Preparing built-in Arabidopsis chr5 example FASTA...\n", *_nop
        try:
            input_file = ensure_example_fasta()
        except Exception as exc:
            yield f"Error: Failed to download example: {exc}", *_nop
            return
        if not species or species == "MySpecies":
            species = "Athaliana"
        domain = "plant"
        if not output_dir or output_dir.strip() == "genecad_result":
            output_dir = "genecad_result/Athaliana_predictions"
    elif not input_file and fasta_upload is not None:
        input_file = fasta_upload.name

    if not input_file:
        yield "Error: Please provide a FASTA file, or enable the built-in example.", *_nop
        return
    if not os.path.exists(input_file):
        yield f"Error: Input file not found: {input_file}", *_nop
        return

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    predict_sh = os.path.join(script_dir, "predict.sh")
    if not os.path.exists(predict_sh):
        yield f"Error: predict.sh not found at {predict_sh}", *_nop
        return

    cmd = [
        "bash", predict_sh,
        "-i", input_file, "-o", output_dir,
        "-s", species, "-m", domain,
    ]
    if top_n_contigs != "all":
        cmd.extend(["-n", str(top_n_contigs)])
    if min_transcript_length not in (None, "", "3"):
        cmd.extend(["-l", str(min_transcript_length)])
    if cpu_workers not in (None, "", "1"):
        cmd.extend(["-c", str(cpu_workers)])
    if batch_size not in (None, "", "auto"):
        cmd.extend(["-b", str(batch_size)])
    if gpus and gpus.strip().lower() != "all":
        cmd.extend(["--gpus", gpus.strip()])

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    log = f"--- GeneCAD Started ---\nCommand: {' '.join(cmd)}\n\n"
    yield log, *_nop

    gpu_progress: dict[str, float] = {}
    for line in iter(process.stdout.readline, ""):
        # tqdm uses \r to overwrite lines in a TTY; split on both \r and \n
        for segment in re.split(r'[\r\n]', line):
            m = _TQDM_RE.search(segment)
            if m:
                gpu_id, pct_str, cur, total = m.group(1), m.group(2), m.group(3), m.group(4)
                gpu_progress[gpu_id] = int(pct_str) / 100.0
                avg = sum(gpu_progress.values()) / len(gpu_progress)
                progress(avg, desc=f"GPU {gpu_id}: {cur}/{total} batches")
        log += line
        yield log, *_nop

    process.stdout.close()
    rc = process.wait()

    if rc == 0:
        log += "\n\n--- Completed successfully ---\n"
        abs_out = os.path.abspath(output_dir)
        log += f"Results: {abs_out}\n"
        gff_files = [
            os.path.join(abs_out, f)
            for f in os.listdir(abs_out) if f.endswith(".gff")
        ] if os.path.exists(abs_out) else []
        paths = {}
        raw_gff = next((f for f in gff_files if f.endswith("_raw.gff")), None)
        final_gff = next((f for f in gff_files if f.endswith("_final.gff")), None)
        if raw_gff:
            paths["Raw GFF"] = raw_gff
        if final_gff:
            paths["Final GFF"] = final_gff
        choices = list(paths.keys())
        default = "Final GFF" if "Final GFF" in paths else (choices[0] if choices else None)
        preview = ""
        if default:
            with open(paths[default]) as fh:
                preview = "".join(fh.readline() for _ in range(100))
        yield (
            log,
            preview,
            paths,
            gr.update(choices=choices, value=default, visible=bool(choices)),
            gr.update(value=paths.get(default), visible=bool(default)),
        )
    else:
        log += f"\n\n--- Failed (exit {rc}) ---\n"
        yield log, *_nop


def create_ui():
    with gr.Blocks(title="GeneCAD — Genome Annotation", css=UI_CSS) as demo:

        # ── Hero ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="gene-hero">
            <div class="gene-eyebrow">GeneCAD · Browser App</div>
            <h1 class="gene-title">Genome Annotation</h1>
            <p class="gene-subtitle">
                Plant or animal genome annotation, right in your browser.
                Upload a FASTA file or paste a cluster path — then hit Run.
            </p>
        </div>
        """)


        with gr.Row(elem_classes=["gene-qs-outer"]):
            use_example = gr.Checkbox(
                label="Use built-in Arabidopsis chr5 example — auto-fills the form below",
                value=False,
                elem_id="example_checkbox",
                elem_classes=["example-checkbox"],
            )

        # ── Main two-column layout ─────────────────────────────────────────
        # NOTE: no gr.HTML div wrappers around Gradio components.
        # The Column blocks themselves become white cards via
        # --block-background-fill: #ffffff and --block-border-color.
        with gr.Row(equal_height=False, elem_classes=["gene-layout"]):

            # ── Left column: settings ──────────────────────────────────────
            with gr.Column(scale=1):

                with gr.Group(elem_classes=["gene-card"]):
                    # Step 1
                    gr.HTML("""
                    <span class="gene-step">Step 1</span>
                    <p class="gene-head">Input genome</p>
                    <p class="gene-hint">
                      Paste an absolute path on your cluster, or upload a file.
                    </p>
                    """)
                    fasta_path = gr.Textbox(
                        label="FASTA path on cluster",
                        placeholder="/data/genomes/my_genome.fa.gz",
                        elem_id="fasta_path",
                    )
                    fasta_upload = gr.File(label="Upload FASTA", elem_id="upload_box")

                with gr.Group(elem_classes=["gene-card"]):
                    # Step 2
                    gr.HTML("""
                    <span class="gene-step">Step 2</span>
                    <p class="gene-head">Configure the run</p>
                    <p class="gene-hint">
                      Species label is only used for output file names.
                    </p>
                    """)
                    species = gr.Textbox(
                        label="Species name",
                        value="MySpecies",
                        placeholder="e.g. Athaliana",
                    )
                    gr.HTML("""
                    <p class="gene-hint" style="margin-top:0.8rem;margin-bottom:0.4rem">
                      Choose the model that matches your organism.
                    </p>
                    """)

                    # Plant / Animal
                    domain = gr.State("plant")
                    with gr.Row():
                        plant_btn = gr.Button(
                            "🌿  Plant",
                            variant="primary",
                            elem_id="plant_btn",
                        )
                        animal_btn = gr.Button(
                            "🐾  Animal",
                            variant="secondary",
                            elem_id="animal_btn",
                        )

                    output_dir = gr.Textbox(
                        label="Output directory",
                        value="genecad_result",
                    )

                    with gr.Accordion("Advanced options", open=False, elem_classes=["advanced-accordion"]):
                        top_n_contigs = gr.Dropdown(
                            choices=["all", 10, 25, 50, 100],
                            value="all",
                            label="Top contigs to process",
                        )
                        min_transcript_length = gr.Number(
                            value=3, precision=0,
                            label="Minimum transcript length (bp)",
                        )
                        cpu_workers = gr.Number(
                            value=1, precision=0, label="CPU workers"
                        )
                        batch_size = gr.Dropdown(
                            choices=["auto", 8, 16, 32, 64],
                            value="auto",
                            label="Batch size per GPU",
                        )
                        gpus = gr.Textbox(
                            label="GPUs",
                            placeholder="0   or   0,1   or   all",
                        )

                run_btn = gr.Button(
                    "Run GeneCAD →",
                    variant="primary",
                    elem_id="run_pipeline",
                )

            # ── Right column: output ───────────────────────────────────────
            with gr.Column(scale=2):

                with gr.Group(elem_classes=["gene-card"]):
                    gr.HTML("""
                    <span class="gene-step">Pipeline log</span>
                    <p class="gene-head">Output</p>
                    <p class="gene-hint">
                      Progress streams here in real time.
                      Click the copy icon to save the full log.
                    </p>
                    """)
                    logs = gr.Textbox(
                        label="",
                        show_label=False,
                        lines=22,
                        max_lines=36,
                        buttons=["copy"],
                        interactive=False,
                        elem_id="logs_box",
                    )

                with gr.Group(elem_classes=["gene-card"]):
                    gr.HTML("""
                    <span class="gene-step">Results</span>
                    <p class="gene-head">GFF preview</p>
                    <p class="gene-hint">
                      Select a file to preview and download.
                    </p>
                    """)
                    gff_paths_state = gr.State({})
                    gff_radio = gr.Radio(
                        choices=[],
                        label="Select file",
                        visible=False,
                        interactive=True,
                    )
                    gff_preview = gr.Textbox(
                        label="",
                        show_label=False,
                        lines=14,
                        max_lines=14,
                        interactive=False,
                        elem_id="gff_preview_box",
                    )
                    download_btn = gr.DownloadButton(
                        label="⬇  Download selected GFF",
                        variant="primary",
                        visible=False,
                    )

        # ── Events ────────────────────────────────────────────────────────
        plant_btn.click(
            fn=select_plant,
            outputs=[domain, plant_btn, animal_btn],
        )
        animal_btn.click(
            fn=select_animal,
            outputs=[domain, plant_btn, animal_btn],
        )
        use_example.change(
            fn=on_use_example_change,
            inputs=[use_example],
            outputs=[species, domain, output_dir, plant_btn, animal_btn],
        )
        def _on_radio_change(choice, paths):
            path = paths.get(choice, "")
            preview = ""
            if path and os.path.exists(path):
                with open(path) as fh:
                    preview = "".join(fh.readline() for _ in range(100))
            return preview, gr.update(value=path or None, visible=bool(path))

        run_btn.click(
            fn=run_genecad,
            inputs=[
                fasta_upload, fasta_path, output_dir, species, domain,
                top_n_contigs, min_transcript_length, cpu_workers,
                batch_size, gpus, use_example,
            ],
            outputs=[logs, gff_preview, gff_paths_state, gff_radio, download_btn],
        )
        gff_radio.change(
            fn=_on_radio_change,
            inputs=[gff_radio, gff_paths_state],
            outputs=[gff_preview, download_btn],
        )

    return demo
