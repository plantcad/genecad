"""Exercise the training model/loss without downloading a backbone or using CUDA."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from scripts.train import parse_args  # noqa: E402
from src.modeling import Batch, GeneClassifier, GeneClassifierConfig, Source  # noqa: E402


def tiny_model(seed=11):
    torch.manual_seed(seed)
    return GeneClassifier(
        GeneClassifierConfig(
            architecture="classifier-only",
            base_encoder_dim=16,
            token_embedding_dim=16,
            max_sequence_length=8,
            dropout=0.0,
            enable_visualization=False,
            train_eval_frequency=None,
        )
    )


def batch_for(logits, masks):
    return Batch(
        source=Source(split="train", index=0),
        logits=logits,
        labels=torch.arange(8).reshape(1, 8),
        masks=masks,
        index=torch.tensor([0]),
    )


def test_all_masked_batch_has_finite_zero_loss_and_zero_gradient():
    model = tiny_model()
    logits = torch.randn(1, 8, 17, requires_grad=True)
    loss = model._compute_loss(batch_for(logits, torch.zeros(1, 8, dtype=torch.bool)))
    assert torch.isfinite(loss), "an excluded-only batch must not produce NaN"
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_excluded_labels_and_logits_cannot_affect_loss_or_gradient():
    model = tiny_model()
    masks = torch.tensor([[True, True, False, False, False, False, True, True]])
    logits = torch.randn(1, 8, 17, requires_grad=True)
    batch = batch_for(logits, masks)
    original = model._compute_loss(batch)
    original.backward()
    assert torch.count_nonzero(logits.grad[~masks]) == 0
    changed = logits.detach().clone()
    changed[~masks] = 1000 * torch.randn_like(changed[~masks])
    labels = batch.labels.clone()
    labels[~masks] = 16
    assert torch.equal(
        original, model._compute_loss(replace(batch, logits=changed, labels=labels))
    )


def test_cli_accepts_shared_initial_state_and_fixed_optimizer_steps():
    args = parse_args(
        [
            "--train-dataset",
            "train.zarr",
            "--val-dataset",
            "valid.zarr",
            "--output-dir",
            "run",
            "--initial-state",
            "seed11.pt",
            "--max-steps",
            "1753",
        ]
    )
    assert args.initial_state == "seed11.pt"
    assert args.max_steps == 1753


def test_four_conditions_load_identical_initial_weights(tmp_path):
    from src.training_protocol import (
        read_initial_state,
        restore_initial_state,
        save_initial_state,
        state_digest,
    )

    path = tmp_path / "seed11.pt"
    reference = tiny_model(11)
    expected = save_initial_state(reference, path, seed=11)
    payload = read_initial_state(path, seed=11)
    hashes = []
    for condition, construction_seed in zip("UBMR", [1, 2, 3, 4]):
        model = tiny_model(construction_seed)
        assert state_digest(model.state_dict()) != expected, condition
        hashes.append(restore_initial_state(model, payload))
        assert all(
            torch.equal(a, model.state_dict()[k])
            for k, a in reference.state_dict().items()
        )
    assert hashes == [expected] * 4
    with pytest.raises(ValueError, match="seed"):
        read_initial_state(path, seed=22)
    with pytest.raises(FileExistsError):
        save_initial_state(reference, path, seed=11)


def test_shared_initialization_rejects_per_condition_bias_changes(tmp_path):
    from src.training_protocol import (
        read_initial_state,
        restore_initial_state,
        save_initial_state,
    )

    path = tmp_path / "seed11.pt"
    save_initial_state(tiny_model(), path, seed=11)
    model = tiny_model()
    model.config.token_class_frequencies = {
        k: 1 / 17 for k in model.config.token_class_names
    }
    with pytest.raises(ValueError, match="token_class_frequencies"):
        restore_initial_state(model, read_initial_state(path, seed=11))


def test_shared_initialization_rejects_auto_class_weights():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--train-dataset",
                "train.zarr",
                "--val-dataset",
                "valid.zarr",
                "--output-dir",
                "run",
                "--initial-state",
                "seed11.pt",
                "--auto-class-weights",
                "yes",
            ]
        )


def test_real_lightning_optimizer_updates_with_an_excluded_only_batch():
    import lightning as L
    from torch.utils.data import DataLoader

    model = tiny_model()
    before = model.classifier.Wo.weight.detach().clone()
    examples = []
    for i in range(4):
        examples.append(
            {
                "input_ids": torch.arange(8),
                "inputs_embeds": torch.eye(16)[:8],
                "tag_labels": torch.arange(8),
                "label_mask": torch.full((8,), i != 1, dtype=torch.bool),
                "sample_index": torch.tensor(i),
            }
        )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        max_epochs=-1,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    trainer.fit(model, train_dataloaders=DataLoader(examples, batch_size=1))
    assert trainer.global_step == 2
    assert torch.isfinite(trainer.callback_metrics["train/loss_step"])
    assert not torch.equal(before, model.classifier.Wo.weight)
