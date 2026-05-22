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


## Assertion Error during detect_intervals.py

```
Traceback (most recent call last):
  File "/local/workdir/ahb232/genecad/scripts/predict.py", line 1541, in <module>
    main()
  File "/local/workdir/ahb232/genecad/scripts/predict.py", line 1533, in main
    detect_intervals(args)
  File "/local/workdir/ahb232/genecad/scripts/predict.py", line 830, in detect_intervals
    sequence_predictions = merge_prediction_datasets(
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/local/workdir/ahb232/genecad/.venv/lib/python3.12/site-packages/src/prediction.py", line 65, in merge_prediction_datasets
    assert np.array_equal(
           ^^^^^^^^^^^^^^^
AssertionError
```
This can be caused if there was an interruption in the earlier predict.py step
and that step was restarted. Certain data may have been written to file twice. Remove the
affected directory and rerun predict.py. This may only affect one chromosome in a multi-chromosome
run, in which case only the affected chromosome needs to be redone.
