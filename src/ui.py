import os
import subprocess
import gradio as gr

def run_genecad(fasta_upload, fasta_path, output_dir, species, domain, gpus):
    # Determine the actual input file
    input_file = fasta_path.strip()
    if not input_file and fasta_upload is not None:
        input_file = fasta_upload.name
    
    if not input_file:
        yield "Error: Please provide a FASTA file by uploading one or entering a path.", None
        return

    if not os.path.exists(input_file):
        yield f"Error: Input file does not exist at {input_file}", None
        return

    # Base command
    # We call the bash script directly because it sets up the pipeline smoothly
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    predict_sh = os.path.join(script_dir, "predict.sh")
    
    if not os.path.exists(predict_sh):
        yield f"Error: Cannot find predict.sh at {predict_sh}", None
        return

    cmd = [
        "bash", predict_sh,
        "-i", input_file,
        "-o", output_dir,
        "-s", species,
        "-m", domain
    ]
    
    if gpus and gpus.strip().lower() != "all":
        cmd.extend(["--gpus", gpus.strip()])

    yield f"Running command: {' '.join(cmd)}\n\n", None

    # Run subprocess and stream output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    log_output = f"--- GeneCAD Annotation Started ---\nCommand: {' '.join(cmd)}\n\n"
    yield log_output, None

    for line in iter(process.stdout.readline, ''):
        log_output += line
        yield log_output, None

    process.stdout.close()
    return_code = process.wait()

    if return_code == 0:
        log_output += f"\n\n--- GeneCAD Annotation Completed Successfully! ---\n"
        out_path = os.path.join(output_dir, f"{species}_predictions")
        log_output += f"Results saved to {out_path}\n"
        
        # Gather resulting files
        output_files = []
        if os.path.exists(out_path):
            for f in os.listdir(out_path):
                if f.endswith(".gff"):
                    output_files.append(os.path.join(out_path, f))
        
        yield log_output, output_files
    else:
        log_output += f"\n\n--- GeneCAD Annotation Failed (Exit Code {return_code}) ---\n"
        yield log_output, None

def create_ui():
    with gr.Blocks(title="GeneCAD UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            <div align="center">
            
            # GeneCAD Foundation Model Annotation 🧬
            
            *No-Code Web Interface*
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Select Input Genome")
                gr.Markdown("For massive genomes (e.g., >500MB), enter the absolute path on the cluster to avoid browser upload limits.")
                fasta_path = gr.Textbox(label="Path to FASTA on Cluster", placeholder="/path/to/genome.fa")
                gr.Markdown("*OR*")
                fasta_upload = gr.File(label="Upload FASTA from your computer")
                
                gr.Markdown("### 2. Configuration")
                species = gr.Textbox(label="Species Name (Prefix)", value="MySpecies", placeholder="e.g. Athaliana")
                domain = gr.Radio(choices=["plant", "animal"], value="plant", label="Domain / Model Family")
                output_dir = gr.Textbox(label="Output Directory", value="genecad_result")
                gpus = gr.Textbox(label="GPUs to use (leave empty for 'all', or specify e.g. '0,1')", placeholder="all")
                
                run_btn = gr.Button("🚀 Run GeneCAD Pipeline", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### Pipeline Logs")
                logs = gr.Textbox(label="Output", lines=25, max_lines=35, show_copy_button=True, interactive=False)
                
                gr.Markdown("### Download Results")
                download_files = gr.File(label="Generated GFF Files", interactive=False)
        
        # Link the button to the function
        run_btn.click(
            fn=run_genecad,
            inputs=[fasta_upload, fasta_path, output_dir, species, domain, gpus],
            outputs=[logs, download_files]
        )
        
    return demo
