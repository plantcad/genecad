import os
import sys
import re
from pathlib import Path

def patch_file(path, search_pattern, replacement, description):
    if not os.path.exists(path):
        print(f"Skipping {description}: File not found at {path}")
        return False
    
    with open(path, "r") as f:
        content = f.read()
    
    if replacement.strip() in content:
        print(f"Already patched {description}")
        return True
    
    new_content = re.sub(search_pattern, replacement, content, flags=re.MULTILINE)
    
    if new_content == content:
        print(f"Failed to patch {description}: Pattern not found")
        return False
    
    with open(path, "w") as f:
        f.write(new_content)
    print(f"Successfully patched {description}")
    return True

def main():
    print("Running GeneCAD environment compatibility patcher...")

    # 1. Patch mamba_ssm
    try:
        import mamba_ssm
        mamba_path = Path(mamba_ssm.__file__).parent / "utils" / "generation.py"
        
        search_mamba = r"from transformers\.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput"
        replace_mamba = """try:
    from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
except ImportError:
    from transformers.generation import GenerateDecoderOnlyOutput
    GreedySearchDecoderOnlyOutput = SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput"""
        
        patch_file(mamba_path, search_mamba, replace_mamba, "mamba_ssm imports")
        
        # Patch output_cls logic in mamba_ssm
        search_output = r"output_cls = GreedySearchDecoderOnlyOutput if not sample else SampleDecoderOnlyOutput"
        replace_output = "output_cls = GreedySearchDecoderOnlyOutput"
        patch_file(mamba_path, search_output, replace_output, "mamba_ssm output_cls logic")
        
    except ImportError:
        print("mamba_ssm not found in current environment.")

    # 2. Patch HNet models in HF cache
    hf_module_path = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    if hf_module_path.exists():
        # Find mixer_seq.py in any subdirectory
        for mixer_path in hf_module_path.glob("**/mixer_seq.py"):
            print(f"Found model file: {mixer_path}")
            
            # Patch post_init()
            search_init = r"(self\.lm_head = nn\.Linear\(d_embed, vocab_size, bias=False, \*\*factory_kwargs\)\s+self\.tie_weights\(\))"
            replace_init = r"\1\n        self.post_init()"
            patch_file(mixer_path, search_init, replace_init, f"post_init in {mixer_path.parent.name}")
            
            # Patch tie_weights() signature
            search_tie = r"def tie_weights\(self\):"
            replace_tie = "def tie_weights(self, *args, **kwargs):"
            patch_file(mixer_path, search_tie, replace_tie, f"tie_weights signature in {mixer_path.parent.name}")

    print("Patching complete.")

if __name__ == "__main__":
    main()
