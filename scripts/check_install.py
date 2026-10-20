"""Check native dependencies and CUDA kernels without downloading model weights."""

import argparse
import json
import platform
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.attention import SDPBackend, sdpa_kernel


@triton.jit
def _add_kernel(x, y, z, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(
        z + offsets,
        tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask),
        mask=mask,
    )


def _check_backward(output, inputs):
    for gradient in torch.autograd.grad(output.float().square().mean(), inputs):
        if not torch.isfinite(gradient).all() or gradient.abs().max() == 0:
            raise RuntimeError(
                "Native kernel backward produced invalid or zero gradients"
            )


def check_install(imports_only=False):
    # Import GeneCAD as well: native kernels alone do not prove its dependencies
    # (including Lightning and Transformers) can be imported together.
    from src import modeling  # noqa: F401
    from src.cli import _genecad_version

    report: dict[str, object] = {
        "status": "imports_only" if imports_only else "passed",
        "architecture": platform.machine(),
        "source_version": _genecad_version(),
        "versions": {
            name: version(name)
            for name in [
                "genecad",
                "torch",
                "triton",
                "flash-attn",
                "mamba-ssm",
                "causal-conv1d",
                "transformers",
            ]
        },
    }
    if imports_only and not torch.cuda.is_available():
        # Triton 3.3.1 initializes a CUDA driver when Mamba's autotuners are
        # imported. A CPU build can check GeneCAD and installed versions, but
        # must defer native imports as well as kernel execution to a GPU node.
        report["native_cuda_imports"] = "deferred_no_gpu"
        return report
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable; run this check on an allocated GPU node"
        )
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError(
            "The FlashAttention 2 backend requires an Ampere or newer GPU"
        )
    # Keep CUDA extensions runtime-only so CPU development/CI installations
    # can type-check the repository without building those optional packages.
    causal_conv1d_fn = import_module("causal_conv1d").causal_conv1d_fn
    flash_attn_func = import_module("flash_attn").flash_attn_func
    scan = import_module("mamba_ssm.ops.selective_scan_interface")
    selective_scan_fn, selective_scan_ref = (
        scan.selective_scan_fn,
        scan.selective_scan_ref,
    )
    report["native_cuda_imports"] = "passed"
    if imports_only:
        return report
    torch.manual_seed(42)
    x, y = (torch.randn(4096, device="cuda") for _ in range(2))
    z = torch.empty_like(x)
    _add_kernel[(4,)](x, y, z, 4096, BLOCK=1024)
    torch.testing.assert_close(z, x + y)

    q, k, v = (
        torch.randn(
            2, 256, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    )
    actual = flash_attn_func(q, k, v, dropout_p=0, causal=False)
    with sdpa_kernel(SDPBackend.MATH):
        reference = F.scaled_dot_product_attention(
            q.float().transpose(1, 2),
            k.float().transpose(1, 2),
            v.float().transpose(1, 2),
        ).transpose(1, 2)
    torch.testing.assert_close(actual.float(), reference, atol=0.015, rtol=0.03)
    flash_error = (actual.float() - reference).abs().max().item()
    _check_backward(actual, (q, k, v))

    x = torch.randn(2, 16, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(16, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual = causal_conv1d_fn(x, w, b, activation="silu")
    reference = F.silu(
        F.conv1d(x.float(), w.float().unsqueeze(1), b.float(), padding=3, groups=16)[
            ..., :256
        ]
    )
    torch.testing.assert_close(actual.float(), reference, atol=0.08, rtol=0.03)
    conv_error = (actual.float() - reference).abs().max().item()
    _check_backward(actual, (x, w, b))

    u = torch.randn(2, 16, 128, device="cuda", requires_grad=True)
    delta = torch.randn_like(u)
    a = -torch.rand(16, 8, device="cuda")
    b, c = (torch.randn(2, 8, 128, device="cuda") for _ in range(2))
    d = torch.randn(16, device="cuda")
    actual = selective_scan_fn(u, delta, a, b, c, d, delta_softplus=True)
    reference = selective_scan_ref(u, delta, a, b, c, d, delta_softplus=True)
    torch.testing.assert_close(actual, reference, atol=0.0002, rtol=0.0002)
    scan_error = (actual - reference).abs().max().item()
    _check_backward(actual, (u,))
    torch.cuda.synchronize()
    report.update(
        gpu=torch.cuda.get_device_name(),
        compute_capability=list(torch.cuda.get_device_capability()),
        cuda=torch.version.cuda,
        max_absolute_errors={
            "flash_attention_bf16": flash_error,
            "causal_conv1d_bf16": conv_error,
            "selective_scan_fp32": scan_error,
        },
        triton_jit="passed",
        native_kernel_backward="passed",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imports-only", action="store_true", help="Skip CUDA execution"
    )
    parser.add_argument("--output", type=Path, help="Also save the JSON report")
    args = parser.parse_args()
    report = check_install(args.imports_only)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
