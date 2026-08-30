# Brown Bridge

A lean research repository for optimal transport and RAE-based AFHQ cat/dog
generation. The retained code has two clear roles:

- a learned, source-only generator that maps Gaussian noise and a cat/dog label
  to an image with a frozen RAE decoder; and
- small endpoint-coupled OT diagnostics and baselines.

The two roles are not interchangeable. The RAE generator needs no target image
at inference. The OT round-trip renderer selects real endpoints and is only a
path diagnostic.

## Repository map

```text
configs/                    RAE AFHQ fine-tuning configuration
experiments/                Six maintained experiment entry points
slurm_scripts/turing/       RAE setup, training, generation, and evaluation jobs
src/brown_bridge/           Small tested OT and bridge kernel
tests/                      Focused tests for the retained kernel
docs/                       Current runbook and research decisions
```

Maintained experiments:

| Entry point | Purpose |
| --- | --- |
| `experiments/sample_rae_afhq_generator.py` | Source-only Gaussian + label RAE sampling |
| `experiments/evaluate_rae_afhq_generator.py` | Held-out known-class RAE evaluation |
| `experiments/render_rae_ot_sde_roundtrip.py` | Endpoint-coupled dog → cat → dog diagnostic |
| `experiments/mnist_mps_ot.py` | Exact displacement paths from an entropic MNIST coupling |
| `experiments/toy_mps_xm_ot.py` | Small bidirectional best-of-K OT model |
| `experiments/text_wikitext_ot_diffusion.py` | WikiText topic transport proof-of-life |

Retired prototypes remain available in Git history. They are not shipped as
active code.

## Install and verify

The core package depends only on PyTorch:

```bash
uv sync --dev
uv run pytest
uvx ruff check src experiments tests
uvx black --check src experiments tests
```

Install the optional plotting and torchvision dependencies for the small demos:

```bash
uv sync --dev --extra demos
```

The MNIST and toy demos require Apple Metal MPS:

```bash
uv run --extra demos python experiments/mnist_mps_ot.py --download
uv run --extra demos python experiments/toy_mps_xm_ot.py
```

The WikiText baseline is CPU compatible:

```bash
uv run --extra demos python experiments/text_wikitext_ot_diffusion.py --download
```

## RAE AFHQ workflow

The RAE jobs use a separate managed environment and the official RAE checkout.
They are Turing-specific and use cluster paths recorded in the runbook. From the
repository root, submit:

```bash
sbatch slurm_scripts/turing/setup_rae_managed_python.sbatch
sbatch slurm_scripts/turing/rae_afhq_dit_s_full_h100.sbatch
sbatch slurm_scripts/turing/rae_afhq_generate_evalset_h100.sbatch
sbatch slurm_scripts/turing/rae_afhq_evaluate_h100.sbatch
```

See [`docs/rae_afhq_generator.md`](docs/rae_afhq_generator.md) for assets, data
views, configuration, and interpretation. The completed CFG 1.5 evaluation
reached 99.34% generated class agreement. Cat/dog RAE-FD128 was 20.543/27.921.
Diversity ratios of 0.770/0.821 show remaining mode contraction.

The titleless endpoint-conditioned round-trip GIF is produced separately with:

```bash
sbatch slurm_scripts/turing/rae_ot_sde_roundtrip_h100.sbatch
```

## Scientific scope

- AFHQ validation measures held-out images from trained classes. It is not an
  unseen-class generalization result.
- Endpoint-coupled interpolation does not prove source-only generation.
- Frozen-RAE feature metrics are useful audits, not independent human realism
  scores.
- Generated outputs, checkpoints, datasets, logs, and caches are excluded from
  Git.

The current image-space distinction and the longer-term bijective-map direction
are summarized in [`docs/cat_dog_ot.md`](docs/cat_dog_ot.md) and
[`docs/static_ot_atlas_ssl.md`](docs/static_ot_atlas_ssl.md).
