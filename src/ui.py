import os
import subprocess
import urllib.request
from pathlib import Path
import gradio as gr


UI_CSS = """
/* ═══════════════════════════════════════
   GeneCAD — Modern Clean UI
   Palette: Slate + Indigo
   ═══════════════════════════════════════ */

/* ── Fonts ── */
.gradio-container {
    font-family: "Inter", -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif !important;
}

/* ── Light mode tokens ── */
.gradio-container {
    --gc-bg:        #F8FAFC;
    --gc-card:      #FFFFFF;
    --gc-input:     #F1F5F9;
    --gc-border:    rgba(148, 163, 184, 0.25);
    --gc-text:      #0F172A;
    --gc-muted:     #64748B;
    --gc-accent:    #6366F1;
    --gc-accent-h:  #4F46E5;
    --gc-accent-bg: rgba(99, 102, 241, 0.08);
    --gc-term-bg:   #0F172A;
    --gc-term-fg:   #86EFAC;

    --body-background-fill:   var(--gc-bg)    !important;
    --background-fill-primary: var(--gc-card) !important;
    --background-fill-secondary: var(--gc-input) !important;
    --block-background-fill:  var(--gc-card)  !important;
    --block-border-width:     1px             !important;
    --block-border-color:     var(--gc-border)!important;
    --block-shadow:           0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04) !important;
    --block-radius:           16px            !important;
    --input-background-fill:  var(--gc-input) !important;
    --input-border-width:     1px             !important;
    --input-radius:           10px            !important;
    --color-accent:           var(--gc-accent)!important;
    --body-text-color:        var(--gc-text)  !important;
    --body-text-color-subdued:var(--gc-muted) !important;
}

/* ── Dark mode tokens ── */
.dark, .dark .gradio-container {
    --gc-bg:        #0F172A;
    --gc-card:      #1E293B;
    --gc-input:     #0F172A;
    --gc-border:    rgba(148, 163, 184, 0.12);
    --gc-text:      #F1F5F9;
    --gc-muted:     #94A3B8;
    --gc-accent:    #818CF8;
    --gc-accent-h:  #6366F1;
    --gc-accent-bg: rgba(129, 140, 248, 0.12);
    --gc-term-bg:   #020617;
    --gc-term-fg:   #86EFAC;

    --body-background-fill:   var(--gc-bg)    !important;
    --background-fill-primary: var(--gc-card) !important;
    --background-fill-secondary: var(--gc-input) !important;
    --block-background-fill:  var(--gc-card)  !important;
    --block-border-color:     var(--gc-border)!important;
    --block-shadow:           none            !important;
    --input-background-fill:  var(--gc-input) !important;
    --color-accent:           var(--gc-accent)!important;
    --body-text-color:        var(--gc-text)  !important;
    --body-text-color-subdued:var(--gc-muted) !important;
}

/* ── Base ── */
.gradio-container * { box-sizing: border-box; }

/* ── Hero ── */
.gene-hero {
    text-align: center;
    padding: 3rem 1.5rem 1.5rem;
    max-width: 1100px;
    margin: 0 auto;
}
.gene-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    background: var(--gc-accent-bg);
    color: var(--gc-accent);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    border: 1px solid var(--gc-border);
}
.gene-title {
    margin: 0 0 0.75rem;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.08;
    color: var(--gc-text);
}
.gene-subtitle {
    font-size: 1.05rem;
    line-height: 1.65;
    color: var(--gc-muted);
    max-width: 50rem;
    margin: 0 auto;
}

/* ── Cards ── */
.gradio-container .group,
.gene-card {
    background: var(--gc-card) !important;
    border-radius: 16px !important;
    border: 1px solid var(--gc-border) !important;
    box-shadow: var(--block-shadow) !important;
    overflow: hidden !important;
}
.gradio-container .group .form,
.gradio-container .group [class*="form"],
.gene-card .form, .gene-card [class*="form"],
.gene-card .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Section labels ── */
.gene-step {
    display: block;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gc-accent);
    margin: 0 0 0.2rem;
}
.gene-head {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem;
    color: var(--gc-text);
}
.gene-hint {
    font-size: 0.875rem;
    color: var(--gc-muted);
    margin: 0 0 1rem;
    line-height: 1.55;
}

/* ── Layout ── */
.gene-layout {
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 1.5rem 3rem;
}

/* ── Inputs ── */
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea {
    background: var(--gc-input) !important;
    border: 1px solid var(--gc-border) !important;
    border-radius: var(--input-radius) !important;
    color: var(--gc-text) !important;
    font-size: 0.925rem !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    padding: 0.65rem 0.9rem !important;
    box-shadow: none !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container input[type="number"]:focus,
.gradio-container textarea:focus {
    border-color: var(--gc-accent) !important;
    box-shadow: 0 0 0 3px var(--gc-accent-bg) !important;
    outline: none !important;
}

/* ── Labels ── */
.gradio-container .label-wrap > span,
.gradio-container .block > label > span:first-child {
    background: transparent !important;
    color: var(--gc-muted) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0 !important;
    margin-bottom: 0.35rem !important;
    border: none !important;
}

/* ── Compact checkbox row ── */
.gene-qs-outer {
    max-width: 1100px !important;
    margin: 0 auto 0.25rem !important;
    padding: 0 1.5rem !important;
    gap: 0 !important;
    overflow: hidden !important;
}
.gene-qs-outer > *, .gene-qs-outer .gap { gap: 0 !important; }
.gene-qs-outer > div { width: 100% !important; min-width: 0 !important; }
#example_checkbox,
.gradio-container #example_checkbox,
.gradio-container .block.example-checkbox {
    margin: 0 !important;
    padding: 0.25rem 0 !important;
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
    color: var(--gc-text) !important;
}
/* Remove green highlight when checkbox is checked */
#example_checkbox label.checkbox-container:has(input:checked),
.gradio-container #example_checkbox label.checkbox-container:has(input:checked) {
    background: transparent !important;
    color: var(--gc-text) !important;
}
#example_checkbox input[type="checkbox"],
.gradio-container #example_checkbox input[type="checkbox"] {
    accent-color: var(--gc-accent) !important;
}

/* ── Terminal log ── */
#logs_box textarea {
    background: var(--gc-term-bg) !important;
    color: var(--gc-term-fg) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(134,239,172,0.15) !important;
    font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.65 !important;
    padding: 1rem !important;
}

/* ── Buttons ── */
.gradio-container button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    min-height: 42px !important;
    transition: all 0.15s ease !important;
    border: none !important;
    letter-spacing: 0.01em !important;
}
.gradio-container button.primary {
    background: var(--gc-accent) !important;
    color: #fff !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1), 0 0 0 0 var(--gc-accent-bg) !important;
}
.gradio-container button.primary:hover {
    background: var(--gc-accent-h) !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.35) !important;
    transform: translateY(-1px) !important;
}
.gradio-container button.primary:active { transform: translateY(0) !important; }
.gradio-container button.secondary {
    background: var(--gc-input) !important;
    color: var(--gc-text) !important;
    border: 1px solid var(--gc-border) !important;
}
.gradio-container button.secondary:hover {
    border-color: var(--gc-accent) !important;
    color: var(--gc-accent) !important;
    background: var(--gc-accent-bg) !important;
}
#run_pipeline {
    margin-top: 1.25rem !important;
    min-height: 50px !important;
    font-size: 1rem !important;
    border-radius: 12px !important;
    width: 100% !important;
}

/* ── Tiny toolbar buttons ── */
.gradio-container .textbox button,
.gradio-container [data-testid="textbox"] button {
    border-radius: 6px !important;
    background: var(--gc-input) !important;
    border: 1px solid var(--gc-border) !important;
    min-height: 28px !important;
    min-width: 28px !important;
}

/* ── File upload ── */
#upload_box,
.gradio-container [data-testid="file-upload"],
.gradio-container .upload-container {
    background: var(--gc-input) !important;
    border: 1px dashed var(--gc-border) !important;
    border-radius: 12px !important;
    transition: border-color 0.15s ease, background 0.15s ease !important;
}
.gradio-container .border-dashed { border-style: dashed !important; }
#upload_box *,
.gradio-container [data-testid="file-upload"] *,
.gradio-container .upload-container * {
    border-color: transparent !important;
}
.gradio-container [data-testid="file-upload"] button,
.gradio-container [data-testid="file-upload"] > div { border: none !important; }
#upload_box:hover,
.gradio-container [data-testid="file-upload"]:hover,
.gradio-container .upload-container:hover {
    border-color: var(--gc-accent) !important;
    background: var(--gc-accent-bg) !important;
}

/* ── GFF preview ── */
#gff_preview_box textarea {
    font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.6 !important;
    resize: none !important;
}

/* ── Accordion ── */
.advanced-accordion { background: transparent !important; border: none !important; }
.gradio-container details.advanced-accordion,
.advanced-accordion details {
    background: var(--gc-input) !important;
    border-radius: 10px !important;
    border: 1px solid var(--gc-border) !important;
    margin-top: 0.75rem !important;
}
.gradio-container details summary {
    padding: 0.75rem 1rem !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    color: var(--gc-text) !important;
}

/* ── Plant/Animal pill buttons ── */
#plant_btn, #animal_btn {
    border-radius: 10px !important;
    min-height: 40px !important;
    font-size: 0.875rem !important;
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
            gr.update(variant="primary"),
            gr.update(variant="secondary"),
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

    for line in iter(process.stdout.readline, ""):
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
        with gr.Row(equal_height=False, elem_classes=["gene-layout"]):

            # ── Left column: settings ──────────────────────────────────────
            with gr.Column(scale=1):

                with gr.Group(elem_classes=["gene-card"]):
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
