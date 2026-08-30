# RAE AFHQ Gaussian generator

This experiment fine-tunes the official RAE DiTDH-S stage-2 model as a
source-only Gaussian-to-image generator for the known AFHQ cat and dog classes.
The RAE encoder and decoder stay frozen. Inference uses only Gaussian noise and
a class label; it does not retrieve or condition on an image endpoint.

## External assets

The jobs expect the official RAE repository at `/home/afrenk/RAE`, revision
`a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041`, with these upstream assets:

- DINOv2-with-registers base RAE decoder and normalization statistics
- ImageNet-256 `DiTDH-S_ep20` stage-2 checkpoint
- AFHQ under `/home/afrenk/twisted-maps/data/afhq`

`slurm_scripts/turing/setup_rae_managed_python.sbatch` creates a UV-managed
Python environment because Turing's system Python does not include `Python.h`,
which Triton needs for `torch.compile`.

## Data preparation

The jobs use repo-local ImageFolder views so only cat and dog are visible and
the upstream validation split stays separate. Create them before submission:

```bash
REPO=/home/afrenk/brown-bridge-human-real
AFHQ=/home/afrenk/twisted-maps/data/afhq
rm -rf "$REPO/data/afhq_cat_dog_train" \
       "$REPO/data/afhq_cat_dog_validation"
mkdir -p "$REPO/data/afhq_cat_dog_train" \
         "$REPO/data/afhq_cat_dog_validation"
ln -s "$AFHQ/train/cat" "$REPO/data/afhq_cat_dog_train/cat"
ln -s "$AFHQ/train/dog" "$REPO/data/afhq_cat_dog_train/dog"
ln -s "$AFHQ/val/cat" "$REPO/data/afhq_cat_dog_validation/cat"
ln -s "$AFHQ/val/dog" "$REPO/data/afhq_cat_dog_validation/dog"
```

Audit split membership before training. This command must print nothing:

```bash
comm -12 \
    <(find "$AFHQ/train/cat" "$AFHQ/train/dog" -type f -printf '%f\n' | sort) \
    <(find "$AFHQ/val/cat" "$AFHQ/val/dog" -type f -printf '%f\n' | sort)
```

Submit jobs from an existing working directory. Slurm writes
`slurm-<job-name>-<job-id>.out` and `.err` there, so a tracked log directory is
not required.

## Training

`configs/rae_afhq_dit_s.yaml` keeps the 501.93M-parameter stage-1 RAE frozen and
fine-tunes the 196.46M-parameter DiTDH-S velocity model. The AFHQ ImageFolder
mapping is `cat=0`, `dog=1`; the original 1,000-class embedding table and null
label 1,000 remain intact for checkpoint compatibility and classifier-free
guidance.

The completed run used 5,153 cat and 4,739 dog training images, a global batch
of 32 on four H100s, BF16, and 50 epochs. It completed 15,450 updates in 18m41s.
The final EMA checkpoint is written under the ignored `outputs/` tree.

Run the jobs in this order:

```bash
sbatch slurm_scripts/turing/setup_rae_managed_python.sbatch
sbatch slurm_scripts/turing/rae_afhq_dit_s_full_h100.sbatch
```

## Source-only sampling

`experiments/sample_rae_afhq_generator.py` overrides the saved config's
ImageNet checkpoint with the final AFHQ checkpoint. The official checkpoint
loader selects its EMA weights. It supports:

- 21-state Euler probability-flow ODE samples
- titleless per-class and same-noise cat/dog trajectory GIFs
- bulk class-balanced PNG generation using native Torch PNG writes
- classifier-free guidance and deterministic seeds

The Linear-path upstream `SBDM` diffusion can become negative. The SDE
trajectory therefore uses the finite positive `sigma` diffusion form, lowercase
Euler dispatch, no terminal correction, and shared Brownian innovations for the
same-noise cat/dog comparison.

The full evaluation sample command is encoded in
`slurm_scripts/turing/rae_afhq_generate_evalset_h100.sbatch`. It generated 5,000
cats and 5,000 dogs with 21-step ODE sampling and CFG 1.5.

`experiments/render_rae_ot_sde_roundtrip.py` is a separate endpoint-coupled
entropic-OT Brownian bridge diagnostic, not a source-only generator.

## Held-out known-class evaluation

`experiments/evaluate_rae_afhq_generator.py` encodes the saved RGB outputs and
real images through the same frozen RAE. The official AFHQ validation cat and
dog images are never used for training. This evaluates held-out images of known
classes, not semantic generalization to unseen classes.

The evaluator uses a train-fitted PCA-128 projection, matched 500-image shards,
and a Torch CUDA eigensolver. Healthy covariance matrices use a fast float32
path. A symmetric float64 CPU solve with a scale-aware ridge is only used as a
failure fallback.

The completed evaluation produced:

| Metric | Cat | Dog |
|---|---:|---:|
| RAE-FD128 (lower) | 20.543 | 27.921 |
| Sinkhorn divergence (lower) | 0.277 | 0.350 |
| Unbiased energy distance (lower) | 0.060 | 0.048 |
| Generated/validation diversity ratio | 0.770 | 0.821 |
| Generated/held-out nearest-train ratio | 1.398 | 1.511 |

A train-only linear probe reached 100% on the 1,000 held-out validation images
and 99.34% target-label agreement on the 10,000 generated images. No near-exact
generated/training matches were found in the pooled RAE feature audit. The
sub-one diversity ratios show that this small model still contracts modes.

Generated datasets, checkpoints, GIFs, feature caches, and metric products live
under `outputs/` and remain excluded from Git.
