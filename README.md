# Brown bridge: cat-to-dog transport with DINO

The primary goal is a measure-valued class-to-class bridge in DINO feature
space. The same machinery applies to cat→dog or any pair of feature measures.
The project supports two related views:

1. distribution transport: estimate an empirical cat/dog OT coupling and move
   coupling particles from time zero to one;
2. feature editing (secondary comparison): use OT only to propose pairs, then
   learn a source-anchored residual edit for an individual feature.

The notebook prototype has been separated into:

- `src/brown_bridge/data.py`: datasets and batch preprocessing;
- `src/brown_bridge/models.py`: DINO feature/classifier adapters;
- `src/brown_bridge/training.py`: classifier fine-tuning;
- `src/brown_bridge/ot.py`: stable Sinkhorn coupling and divergence;
- `src/brown_bridge/paths.py`: coupling particles, spherical/ambient paths,
  Brownian bridges, and the `n` by `T` time grid;
- `src/brown_bridge/schrodinger.py`: high-level empirical path-space bridge
  with the consistent `epsilon = 4 * diffusivity` convention;
- `src/brown_bridge/flow_matching.py`: OT-CFM targets and ODE integration;
- `src/brown_bridge/editing.py`: residual feature matching and a generic
  FlowEdit inference solver;
- `src/brown_bridge/visualization.py`: PCA projection only;
- `experiments/cat_dog.py`: end-to-end experiment entry point;
- `docs/mathematical_formulation.md`: assumptions, derivations, and references.

## Run

Install the locked environment, then run the ambient empirical Schrödinger
bridge. With `epsilon=0.05`, the consistent diffusivity is `epsilon/4=0.0125`:

```bash
uv sync --group dev
uv run python experiments/cat_dog.py \
  --data-root data \
  --macro-steps 10 \
  --microsteps 4 \
  --particles 1024 \
  --geometry ambient \
  --cost sqeuclidean \
  --epsilon 0.05 \
  --bridge-diffusivity 0.0125
```

This creates `n*T + 1 = 41` path states. To load a production-fine-tuned DINO
checkpoint instead of training in this script, add `--checkpoint path/to/model.pt`.

Train the source-anchored residual editor from the resulting coupling:

```bash
uv run python experiments/residual_feature_edit.py \
  --transport outputs/cat_dog/transport.pt \
  --macro-steps 10 \
  --microsteps 4
```

For the deterministic spherical/geodesic comparator:

```bash
uv run python experiments/cat_dog.py \
  --data-root data \
  --geometry spherical \
  --cost angular_squared \
  --epsilon 0.05
```

The latter stays on the DINO unit sphere, but is not the exact intrinsic
spherical Brownian bridge; see the heat-kernel discussion in the theory note.

## Important boundary

DINO maps images to features; it does not invert arbitrary features back to
images. The transport artifact therefore contains feature trajectories. Image
generation requires a separate conditional decoder or diffusion renderer.

FlowEdit additionally requires pretrained source- and target-conditioned
generative velocity fields. A static DINO encoder supplies neither field; in
this repository DINO is the representation space, while `editing.flow_edit` is
an integration primitive for when such a generative flow is added.

## Test

```bash
uv run pytest
```
