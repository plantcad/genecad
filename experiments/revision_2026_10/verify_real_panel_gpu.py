"""Run actual 8192-bp revision-panel data through CUDA, loss, and AdamW.

This bounded check exercises real U/B/M/R1 masks, both strands, and eight species.
Its small effective batch is for correctness; the separate CLI test checks 384.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import lightning as L
import torch
from torch.utils.data import DataLoader

from scripts.train import sample_transform
from src.dataset import XarrayDataset
from src.modeling import GeneClassifier, GeneClassifierConfig, Source
from src.training_protocol import (
    read_initial_state,
    restore_initial_state,
    state_digest,
)


def main(panel, initial, output):
    output.mkdir(parents=True, exist_ok=False)
    data_report = json.loads((panel / "verification.json").read_text())
    assert data_report["result"] == "PASS"
    payload = read_initial_state(initial, seed=11)
    L.seed_everything(11, workers=True)
    model = (
        GeneClassifier(GeneClassifierConfig(**payload["config"]), learning_rate=2e-4)
        .cuda()
        .train()
    )
    model.log = lambda *args, **kwargs: None
    report = {
        "result": "RUNNING",
        "initial_state_sha256": payload["state_sha256"],
        "window_size": 8192,
        "gpu": torch.cuda.get_device_name(),
        "data_role": data_report["role"],
        "effective_batch": 4,
        "conditions": {},
    }
    start = time.monotonic()
    for condition in ["U", "B", "M", "R1"]:
        restored = restore_initial_state(model, payload)
        L.seed_everything(11, workers=True)
        base = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "base_encoder" in n
        ]
        head = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "base_encoder" not in n
        ]
        optimizer = torch.optim.AdamW(
            [{"params": head, "lr": 2e-4}, {"params": base, "lr": 2e-5}],
            weight_decay=0.01,
        )
        base_named = [
            (n, p)
            for n, p in model.named_parameters()
            if p.requires_grad and "base_encoder" in n
        ]
        probes = [
            (n, p.detach().flatten()[:8192].cpu().clone()) for n, p in base_named[:16]
        ]
        before_head = model.classifier.Wo.weight.detach().cpu().clone()
        loader = DataLoader(
            XarrayDataset(
                str(panel / condition / "train.zarr"),
                sample_transform=sample_transform,
                chunk_size=1,
                consolidated=True,
            ),
            batch_size=1,
            num_workers=0,
        )
        optimizer.zero_grad(set_to_none=True)
        losses = []
        updates = 0
        invariance = False
        for i, batch in enumerate(loader):
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prepared = model._prepare_batch(batch, Source(split="train", index=i))
                prepared.logits.retain_grad()
                loss = model._compute_loss(prepared)
                assert torch.isfinite(loss)
                if not invariance and (~prepared.masks).any():
                    with torch.no_grad():
                        logits = prepared.logits.detach().clone()
                        labels = prepared.labels.clone()
                        logits[~prepared.masks] = 1000
                        labels[~prepared.masks] = 16
                        changed = model._compute_loss(
                            replace(prepared, logits=logits, labels=labels)
                        )
                        assert torch.equal(loss.detach(), changed)
                    invariance = True
            group_size = min(4, len(loader) - (i // 4) * 4)
            (loss / group_size).backward()
            assert torch.count_nonzero(prepared.logits.grad[~prepared.masks]) == 0
            losses.append(float(loss))
            if (i + 1) % 4 == 0 or i + 1 == len(loader):
                assert all(
                    torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.grad is not None
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            print(
                "PASS",
                condition,
                i,
                "loss",
                float(loss),
                "masked",
                int((~prepared.masks).sum()),
                flush=True,
            )
        parameters = dict(model.named_parameters())
        assert probes, "no backbone parameter probes"
        base_changed = any(
            not torch.equal(before, parameters[name].detach().flatten()[:8192].cpu())
            for name, before in probes
        )
        head_changed = not torch.equal(
            before_head, model.classifier.Wo.weight.detach().cpu()
        )
        assert base_changed and head_changed, (condition, base_changed, head_changed)
        assert invariance, (condition, "no masked loss invariance probe")
        # A real full-length forward/backward with every label excluded must be finite zero.
        optimizer.zero_grad(set_to_none=True)
        batch["label_mask"].zero_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prepared = model._prepare_batch(batch, Source(split="train", index=i + 1))
            loss = model._compute_loss(prepared)
        assert loss.item() == 0 and torch.isfinite(loss)
        loss.backward()
        assert all(
            torch.count_nonzero(p.grad) == 0
            for p in model.parameters()
            if p.grad is not None
        )
        report["conditions"][condition] = {
            "result": "PASS",
            "initial_state_sha256": restored,
            "windows": len(losses),
            "optimizer_updates": updates,
            "loss_min": min(losses),
            "loss_max": max(losses),
            "backbone_updated": base_changed,
            "head_updated": head_changed,
            "masked_logits_gradient_zero": True,
            "masked_loss_perturbation_invariant": invariance,
            "all_masked_loss_and_gradient_zero": True,
            "final_state_sha256": state_digest(model.state_dict()),
        }
        (output / "verification.json").write_text(json.dumps(report, indent=2))
        del optimizer
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    report.update(
        result="PASS",
        elapsed_s=time.monotonic() - start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
    )
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    print("ALL PASS", json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.panel, a.initial, a.output)
