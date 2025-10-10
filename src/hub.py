import torch
from transformers import AutoModel, AutoConfig


def load_base_model(
    path: str,
    revision: str | None = None,
    dtype=torch.bfloat16,
    randomize: bool = False,
) -> AutoModel:
    """Load a Hugging Face base encoder with optional random initialization.

    Parameters
    ----------
    path : str
        Repository identifier or local path of the model to load.
    revision : str or None, optional
        Specific revision of the model to load, by default ``None`` which
        selects the latest revision.
    dtype : torch.dtype, optional
        Torch data type to use for the returned model, by default
        ``torch.bfloat16``.
    randomize : bool, optional
        If ``True``, instantiate the model from configuration without
        pretrained weights. By default ``False``.

    Returns
    -------
    transformers.AutoModel
        The instantiated model ready for inference or fine-tuning.
    """
    config = AutoConfig.from_pretrained(path, revision=revision, trust_remote_code=True)
    if randomize:
        base_model = AutoModel.from_config(config, dtype=dtype, trust_remote_code=True)
    else:
        base_model = AutoModel.from_pretrained(
            path, config=config, revision=revision, trust_remote_code=True, dtype=dtype
        )
    return base_model
