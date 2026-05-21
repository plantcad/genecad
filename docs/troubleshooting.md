# Troubleshooting: Common Problems

## No CUDA GPUs
```
RuntimeError: No CUDA GPUs are available
```
The main model requires a CUDA GPU to be run for both inference
and training - it cannot be used on a CPU alone. This problem is
often caused by incompatibility between CUDA and pytorch versions, or
by pytorch installing without CUDA drivers. Run the following script
in python:

```
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
```

If `torch.version.cuda` is `None` or an empty string, you have installed a version of torch
that does not support GPU acceleration. This can happen if the GPUs are not
visible to `pip` during the installation process.

If `torch.version.cuda` returns a version number but `torch.cuda.is_available()` is false,
there is either an incompatibility between the driver and pytorch, or you do not have access to
your machine's GPU. On HPCs using SLURM, this can happen if you request a GPU node with `salloc`. You can call `srun` from within `salloc` : `srun --pty bash -i` for
an interactive session. Alternatively, your pytorch CUDA version may be incompatible with the
driver installed. Check this using the command `nvidia-smi` or `nvcc --version`. The version should
be the same or a minor version higher than `torch.version.cuda`.

In most situations, the solution is to remove the current environment and retry installation.
If you are using an HPC with multiple nodes, you must run the installation process on a node with a GPU.
