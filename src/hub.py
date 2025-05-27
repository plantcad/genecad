import torch
from transformers import AutoModel, AutoConfig

def load_base_model(path: str, revision: str | None = None, dtype=torch.bfloat16) -> AutoModel:
    config = AutoConfig.from_pretrained(path, revision=revision, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(path, config=config, revision=revision, trust_remote_code=True, dtype=dtype)
    return base_model
