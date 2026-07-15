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
- `src/brown_bridge/reciprocal.py`: shared cat↔dog stochastic fields,
  species-direction constraints, and shared-noise cycle integration;
- `src/brown_bridge/editing.py`: residual feature matching and a generic
  FlowEdit inference solver;
- `src/brown_bridge/stargan.py`: MPS-safe StarGAN-v2 generator and reference
  style encoder used as a native-resolution image observation model;
- `src/brown_bridge/visualization.py`: PCA projection only;
- `experiments/cat_dog.py`: end-to-end experiment entry point;
- `experiments/reciprocal_cat_dog.py`: content-aware OT plus held-out
  bidirectional/reciprocal ablation;
- `experiments/render_dino_stargan_bridge.py`: five-particle, 256-pixel
  cat→dog rendering with DINO content coupling and trajectory diagnostics;
- `experiments/render_full_dataset_demo.py`: seeded, uncurated AFHQ test
  trajectories with a 21-frame latent chord and an independent AFHQ DINO
  linear probe;
- `docs/mathematical_formulation.md`: assumptions, derivations, and references.

## Run

Large local assets are intentionally outside version control. By default,
AFHQ/StarGAN-v2 is expected at `data/stargan-v2/`, and checkpoints plus
generated figures are written beneath `outputs/`. These paths are ignored by
Git; every experiment also accepts explicit path arguments when assets live
elsewhere.

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

Run the reciprocal experiment with spatial DINO coupling and the two
regularizers (`lambda_content` and `lambda_reciprocal`):

```bash
uv run python experiments/reciprocal_cat_dog.py \
  --data-root data \
  --classifier outputs/cat_dog/classifier.pt \
  --output-dir outputs/reciprocal_cat_dog \
  --steps 1500
```

The independent species probe is fitted on validation features only. Reported
target alignment, monotonicity, content displacement, and cycle errors use the
disjoint held-out test partition.

Run the MPS augmented-Lagrangian continuation, which treats endpoint,
content, and reciprocal errors as explicit inequalities:

```bash
uv run python experiments/augmented_reciprocal_cat_dog.py \
  --data-root data \
  --classifier outputs/cat_dog/classifier.pt \
  --reference-summary outputs/reciprocal_cat_dog/summary.json \
  --output-dir outputs/augmented_reciprocal_cat_dog \
  --steps 2500
```

The script fails fast if MPS was requested by the experiment but is not
available, instead of silently falling back to CPU.

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

## Native-resolution visual bridge

After downloading the official StarGAN-v2 AFHQ EMA checkpoint, render the
five supplied held-out cats against a one-to-one set of DINO-matched dog
references:

```bash
uv run python experiments/render_dino_stargan_bridge.py
```

This experiment requires MPS and fails rather than silently switching to CPU.
It outputs both a uniformly parameterized style bridge, which is the clearest
visual account of continuous movement, and a DINO-reparameterized comparator.
DINO supplies the patch-aware coupling and all species-coordinate diagnostics;
StarGAN-v2 supplies only the map from source content and style to 256×256 RGB.

The renderer is adapted from NAVER's official StarGAN-v2 implementation and
inherits its CC BY-NC 4.0 license. It is a practical observation model, not a
claim that the StarGAN style line is itself the learned Schrödinger bridge.

To test explicit conditioning on the realized reciprocal state, run:

```bash
uv run python experiments/condition_stargan_on_reciprocal_sde.py
```

This loads the trained augmented reciprocal velocity and score, simulates 60
MPS steps with nonzero diffusivity, and pins the stochastic residual path to a
reachable generated dog endpoint. DINO and StarGAN remain frozen. Only an
endpoint-anchored style correction is optimized against each displayed
reciprocal state, with patch-content and temporal-curvature regularizers.

For the low-rank coarse-to-fine extension, run:

```bash
uv run python experiments/condition_stargan_layerwise.py
```

The generator remains backward compatible with its original shared style, but
also accepts one style per decoder block. A rank-4 SVD basis warm-started from
the global correction is held fixed while six block-specific coefficient
schedules are optimized. Endpoint anchoring and the same two regularizers are
retained.

Finally, train the bounded image-space decoder on the intermediate reciprocal
states:

```bash
uv run python experiments/train_sde_refiner.py
```

This 819k-parameter module predicts a small spatial warp and RGB residual from
`(source image, base render, Z_0, Z_t, Z_1, t)`. The deformation is multiplied
by `t(1-t)`, making the refiner exactly the identity at both endpoints. This is
a proof-of-concept fit on the five demonstration paths, not a held-out
generalization result.

## Test

```bash
uv run pytest
```

## Full-dataset demo

After training the full AFHQ autoencoder, render the commit-ready held-out
demo without retraining it:

```bash
uv run python experiments/render_full_dataset_demo.py
```

The animation is explicitly labeled as a decoded latent chord. DINO chooses
content-compatible OT endpoints, while a frozen-feature linear probe fitted
on AFHQ train and evaluated on AFHQ test supplies the semantic diagnostics.
No learned SDE drift is integrated by this command.
