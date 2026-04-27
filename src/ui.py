import os
import subprocess
import urllib.request
from pathlib import Path
import gradio as gr


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

/* Checkbox (Standard Apple Toggle look) */
.gene-qs-outer .form,
.gene-qs-outer [class*="form"],
#example_checkbox,
.gradio-container #example_checkbox,
.gradio-container .block.example-checkbox {
    margin-top: 1rem !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#example_checkbox label.checkbox-container,
.gradio-container #example_checkbox label.checkbox-container,
.gradio-container .block.example-checkbox label.checkbox-container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.gradio-container input[type="checkbox"]:checked + span,
.gradio-container input[type="checkbox"]:checked ~ span:not(:last-of-type) {
    background: #34c759 !important; /* Apple Green */
    border-color: #34c759 !important;
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
):
    input_file = fasta_path.strip() if fasta_path else ""

    if use_example:
        yield "Preparing built-in Arabidopsis chr5 example FASTA...\n", None
        try:
            input_file = ensure_example_fasta()
        except Exception as exc:
            yield f"Error: Failed to download example: {exc}", None
            return
        if not species or species == "MySpecies":
            species = "Athaliana"
        domain = "plant"
        if not output_dir or output_dir.strip() == "genecad_result":
            output_dir = "genecad_result/Athaliana_predictions"
    elif not input_file and fasta_upload is not None:
        input_file = fasta_upload.name

    if not input_file:
        yield "Error: Please provide a FASTA file, or enable the built-in example.", None
        return
    if not os.path.exists(input_file):
        yield f"Error: Input file not found: {input_file}", None
        return

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    predict_sh = os.path.join(script_dir, "predict.sh")
    if not os.path.exists(predict_sh):
        yield f"Error: predict.sh not found at {predict_sh}", None
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
    yield log, None

    for line in iter(process.stdout.readline, ""):
        log += line
        yield log, None

    process.stdout.close()
    rc = process.wait()

    if rc == 0:
        log += "\n\n--- Completed successfully ---\n"
        out_path = os.path.join(output_dir, f"{species}_predictions")
        log += f"Results: {out_path}\n"
        files = [
            os.path.join(out_path, f)
            for f in os.listdir(out_path) if f.endswith(".gff")
        ] if os.path.exists(out_path) else []
        yield log, files
    else:
        log += f"\n\n--- Failed (exit {rc}) ---\n"
        yield log, None


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

        # ── Quick-start card (pure HTML, no Gradio components inside) ────
        gr.HTML("""
        <div class="gene-qs-outer">
          <div class="gene-qs">
            <div class="gene-qs-badge">New here? Start here</div>
            <h3>Try the built-in Arabidopsis example</h3>
            <p>
              No file needed. Tick the checkbox below and click
              <strong>Run GeneCAD</strong>.<br>
              GeneCAD downloads a small <em>Arabidopsis thaliana</em>
              chr&nbsp;5 genome (~30&nbsp;MB) and fills in
              <strong>Species&nbsp;=&nbsp;Athaliana</strong>,
              <strong>Model&nbsp;=&nbsp;Plant</strong>, and
              <strong>Output&nbsp;=&nbsp;<code>genecad_result/Athaliana_predictions</code></strong>.
              These values guide you on what format to use for your own run.
            </p>
          </div>
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
                    <p class="gene-head">Download files</p>
                    <p class="gene-hint">
                      Annotation GFF files appear here when the pipeline finishes.
                    </p>
                    """)
                    download_files = gr.File(
                        label="Generated GFF files", interactive=False, elem_id="download_box"
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
        run_btn.click(
            fn=run_genecad,
            inputs=[
                fasta_upload, fasta_path, output_dir, species, domain,
                top_n_contigs, min_transcript_length, cpu_workers,
                batch_size, gpus, use_example,
            ],
            outputs=[logs, download_files],
        )

    return demo
