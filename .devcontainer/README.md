# Dev container

CUDA 12.4 / Ubuntu 22.04 dev container for CarDreamer with CARLA 0.9.15.
Open the repo in VS Code and run **Dev Containers: Rebuild and Reopen in Container**.

## What the container provides

| | |
|---|---|
| User | `vscode`, non-root, passwordless `sudo`. The UID/GID is synced to your host user at startup, so files in `/workspace` stay writable. |
| Python | Miniconda in `~/miniconda` with a `cardreamer` env (Python 3.10 + flit). It is activated automatically in new shells. `uv` is in `/usr/local/bin`. |
| CARLA | Host folder `/home/mkurz/carla-sim/CARLA_0.9.15` mounted at `/opt/carla`. `CARLA_ROOT` and `PYTHONPATH` are already set. |
| GPU / resources | All GPUs, 16 CPUs, 128 GB RAM, 32 GB shared memory. Vulkan is configured for `CarlaUE4.sh -RenderOffScreen`. |
| Ports | 9000 forwarded |

Nothing runs after the container is created. Do the one-time setup below yourself.

## One-time setup (inside the container)

1. Install car_dreamer into the `cardreamer` env:

   ```bash
   conda activate cardreamer
   cd /workspace
   flit install --symlink
   ```

   `--symlink` links to the mounted repo, so code changes apply immediately. Re-run
   it only when the dependencies in `pyproject.toml` change.

2. Install the DreamerV3 dependencies (from [dreamerv3/README.md](../dreamerv3/README.md)):

   ```bash
   cd /workspace/dreamerv3
   conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
   pip install -r requirements.txt
   pip install "jax[cuda12_pip]==0.4.34" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

   Everything lives in `~/miniconda`, so it survives restarts but **not** a container rebuild.

## Training

In every new shell, set the CUDA library paths before training:

```bash
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export CUSOLVER_PATH=$(dirname $(python -c "import nvidia.cusolver;print(nvidia.cusolver.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CUSOLVER_PATH/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Then start training. The arguments are the CARLA port and the GPU index. The script starts CARLA itself:

```bash
cd /workspace
bash train_dm3.sh 2000 0 --task carla_four_lane --dreamerv3.logdir ./logdir/carla_four_lane
```

## Notes

- **Different machine or user:** the CARLA host path in the `mounts` section of
  [devcontainer.json](devcontainer.json) is hard-coded. Change it to wherever your
  CARLA 0.9.15 is extracted.
- **Changing the image:** add system packages in the Dockerfile *before* the
  `USER ${USERNAME}` line, or run `sudo apt-get install ...` in the container for one-off installs.
