# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DiffusionNet ("DiffusionNet: Discretization Agnostic Learning on Surfaces", Sharp et al.) — a PyTorch library for deep learning on triangle meshes and point clouds, plus scripts reproducing the paper's experiments. There is no test suite, linter config, or build system (no `setup.py`/`pyproject.toml`).

## Setup

- Environment: `conda env create --name diffusion_net -f environment.yml` (pinned old versions: Python 3.8, torch 1.8.1, numpy 1.20; the file is Linux-specific, so on Windows install the pip packages manually).
- Needs a CUDA GPU; the code assumes `device = torch.device('cuda')`.
- The package lives in `src/diffusion_net` and is not pip-installable. Experiment scripts import it via a `sys.path` insertion pointing at `../../src/`, so run them from inside their experiment directory.
- Core native deps: `robust_laplacian` (Laplacian/mass), `potpourri3d` (geodesics), `scipy`/`sklearn`. Note `geometry.py` must import `scipy` before `torch` or it crashes on some platforms — keep that import order.

## Running experiments

Each directory under `experiments/` has its own README with dataset download steps and commands. Scripts are run from within the experiment folder, e.g.:

```
cd experiments/functional_correspondence
python functional_correspondence.py --train_dataset=faust --input_features=xyz
python functional_correspondence.py --test_dataset=faust --input_features=xyz --load_model=pretrained_models/faust_xyz.pth   # evaluate pretrained (see --evaluate)
```

Pretrained `.pth` models are committed in each experiment's `pretrained_models/`. Datasets go in each experiment's `data/` (gitignored); computed operators are cached on disk under the data dir.

## Architecture

`src/diffusion_net/` (`__init__.py` re-exports everything from all three modules via `import *`):

- `geometry.py` — all geometry processing, done on CPU with numpy/scipy then converted to torch.
  - `compute_operators` builds, per shape: tangent frames, lumped mass vector, Laplacian `L`, eigendecomposition (`evals`, `evecs`, `k_eig` default 128), and the two sparse gradient matrices `gradX`/`gradY`. Meshes and point clouds share the interface (point clouds are signaled by empty `faces`; they use kNN-based normals/gradients).
  - `get_operators` wraps it with an on-disk `.npz` cache keyed by a hash of verts/faces (stored float32, returned with the dtype/device of `verts`). `get_all_operators` does this for a list of shapes in parallel. Precompute once in the dataset constructor, not per step.
  - Also: HKS features (`compute_hks[_autoscale]`), `normalize_positions`, spectral `to_basis`/`from_basis`, kNN, farthest point sampling, and geodesic error evaluation (`get_all_pairs_geodesic_distance`, expensive; cached).
- `layers.py` — `LearnedTimeDiffusion` (per-channel learned diffusion time, `spectral` via eigenbasis or `implicit_dense`), `SpatialGradientFeatures`, `MiniMLP`, `DiffusionNetBlock` (diffusion → gradient features → MLP → residual), and `DiffusionNet` (first linear, `N_block` blocks, last linear, optional `outputs_at` of `vertices`/`edges`/`faces`/`global_mean`, optional `last_activation`).
- `utils.py` — numpy/torch helpers, random rotations, hashing for the cache, label-smoothing loss.

Model forward signature: `model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)`. Tensors are `[N,C]` or batched `[B,N,C]`; N may vary between calls. Note the batch loop in `DiffusionNetBlock` is manual (`torch.mm` on sparse gradients), so batches are mostly used with B=1.

`experiments/<name>/` each contain a `*_dataset.py` (loads data, calls `get_operators`, caches operators in memory) and a training script (argparse flags, train/eval loop, saves to `saved_models/` or `data/saved_models/`). Experiments: SHREC11 classification, RNA mesh segmentation, human segmentation, FAUST/SCAPE functional correspondence (`fmaps_model.py` adds a functional-maps layer on top of DiffusionNet), and sampling invariance.

## Usage tips (from README)

- For a template mesh, precompute operators on the reference pose and feed deformed xyz as features.
- Apply data augmentation (e.g. random rotation) after computing operators, not before.
- Set `with_gradient_rotations=False` if surface normals/orientation aren't consistent.
