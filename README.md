# Brown Bridge

Bidirectional stochastic transport between two unpaired image distributions.
The working example is cat ↔ dog; the intended application is human ↔ AI
imagery. DINO supplies semantic geometry, entropic OT supplies an endpoint
coupling, and a stochastic interpolant supplies a continuous reciprocal path.

This repository deliberately separates two questions:

1. **Which endpoints belong together?** Solve a regularized OT problem in a
   DINO representation after suppressing the domain direction.
2. **How should probability move between them?** Learn the current velocity
   and score of a Brownian bridge mixture. The same two fields define a
   deterministic probability-flow ODE and forward/backward Itô diffusions.

The image decoder is not the transport model. The earlier StarGAN/VAE/refiner
experiments were removed because they made this distinction hard to see and
introduced observation-model artifacts into the path.

## Start here

The executable explanation is [`viz/viz_ot.ipynb`](viz/viz_ot.ipynb). It moves
two unpaired 2-D empirical measures, derives the regression targets, checks the
endpoint marginals, and visualizes the effect of bridge noise.

```bash
uv sync --group dev
uv run pytest
uv run python experiments/cat_dog.py
uv run python experiments/reciprocal_cat_dog.py
```

No package exports are hidden in `__init__.py`; experiments import the module
that owns each object.

## Mathematical object

Let `mu_0` and `mu_1` be the two unpaired laws and let `pi` be a semantic
endpoint coupling. With `(X_0, X_1) ~ pi` and `Z ~ N(0, I)`, use

```text
I_t = (1-t) X_0 + t X_1 + sqrt(epsilon t(1-t)) Z.
```

Regress its pathwise derivative and conditional Gaussian score:

```text
v_t(x) = E[d_t I_t | I_t=x],
s_t(x) = grad log rho_t(x).
```

Then

```text
probability-flow ODE:  dX_t = v_t(X_t) dt
forward SDE:           dX_t = [v_t + epsilon s_t/2](X_t) dt + sqrt(epsilon) dW_t
backward SDE:          dX_t = [v_t - epsilon s_t/2](X_t) dt + sqrt(epsilon) dWbar_t.
```

The two stochastic drifts satisfy Nelson's relation
`b_t^+ - b_t^- = epsilon grad log rho_t`. See
[`docs/stochastic_bridge.md`](docs/stochastic_bridge.md) for the measure-level
formulation and [`docs/research_positioning.md`](docs/research_positioning.md)
for the 2023–26 comparison.

## Code map

```text
src/brown_bridge/
  ot.py                       balanced entropic OT / Sinkhorn divergence
  stochastic_interpolants.py simulation-free score + flow targets
  flow_matching.py            deterministic OT-CFM baseline
  schrodinger.py              empirical Brownian Schrödinger bridge
  reciprocal.py               reciprocal SDE training utilities
  light_sb.py                 lightweight parametric SB baseline
  features.py                 DINO semantic/content coordinates
experiments/
  cat_dog.py                  smallest DINO + OT experiment
  reciprocal_cat_dog.py       bidirectional feature-space SDE experiment
  train_bidirectional_kl_vae_rectified_flow.py
                              held-out AFHQ image-space demonstration
viz/
  viz_ot.ipynb                executable mathematical walkthrough
```

Data, checkpoints, and generated outputs are ignored by git. The removed
renderer experiments remain available in git history; their conclusions are
summarized in [`docs/legacy_experiments.md`](docs/legacy_experiments.md).

## Image-space result

The strongest fully amortized AFHQ demonstration is a single invertible rectified-flow
ODE in the latent coordinates of a frozen pretrained KL-VAE. Rounded entropic
OT in domain-projected DINO space supplies 4,672 cat/dog training pairs. For

```text
X_t = (1-t) X_0 + t X_1,
```

the field minimizes

```text
E[||v_theta(X_t,t) - (X_1-X_0)||^2]
```

plus low-frequency velocity and spatial-gradient penalties. The same field is
integrated from 0 to 1 for cat → dog and from 1 to 0 for dog → cat; a target
image is never passed at inference.

```bash
uv run --extra image python \
  experiments/train_bidirectional_kl_vae_rectified_flow.py \
  --steps 12000 --batch-size 24 --model-width 96 \
  --transformer-heads 8 --eval-examples 64 --examples 6
```

On an independent random 64-image held-out draw, the DINO probe's mean dog
probability moved from 0.112 to 0.831 for cat → dog and from 0.819 to 0.325
for dog → cat. Source-content cosine was about 0.92 in both directions. The
script writes sparse contact sheets, dense timestep GIFs, and aggregate path
metrics.

### Why U-ViT instead of a pure DiT?

The vector field is a compact U-Net with a transformer bottleneck. Attention
provides global shape mixing, while convolutional skip paths preserve pose,
background, and high-frequency locality. At 128 px with roughly five thousand
examples per domain and MPS training, a pure DiT would have weaker spatial
inductive bias and substantially higher token/memory cost. A larger DiT becomes
attractive only with more data or a pretrained latent transformer. For this
benchmark the U-ViT hybrid is the better architectural point on the
fidelity/compute curve.

## SDE coherence result

The score/current-velocity hypothesis succeeds when heavily overfit to one OT
pair. One shared U-ViT reaches the memorized endpoint with SDE latent MSE
`5.1e-4` cat → dog and `4.6e-4` dog → cat. The decoded paths are visibly
coherent in both directions. Run it with:

```bash
uv run --extra image python \
  experiments/overfit_kl_vae_stochastic_interpolant.py \
  --pair-index 0 --steps 6000 --microsteps 4
```

The pair is excluded from subsequent full-distribution training. Amortizing
thousands of bridges into a single source-anchored field still produces
faceted hybrids: the learned field, rather than the KL-VAE or individual OT
chords, is now the observed bottleneck.

The current visual coherence ceiling is therefore nonparametric. Given a
held-out source, spatial DINO content plus coarse KL layout retrieves a unique
endpoint from the opposite-domain training support. The path is sampled from
the exact finite-grid transition law of the pinned Brownian SDE

```text
dX_t = (Y-X_t)/(1-t) dt + sqrt(epsilon) dW_t,
X_0 = x,  X_1 = Y.
```

No user or held-out target is supplied, but the method does use a retrieved
training-support endpoint and is documented as a coherence ceiling rather
than a generalization claim:

```bash
uv run --extra image python \
  experiments/render_empirical_spatial_ot_sde.py \
  --eval-examples 64 --bridge-epsilon 0.01 --microsteps 4
```

The empirical kernel can now be distilled into a source-only terminal
diffusion conditioned on the KL latent, spatial-DINO descriptor, and transport
direction. The learned terminal is joined to the source by the exact pinned
Brownian SDE above. The experiment supports both direct terminal diffusion and
the translated random variable `R = Y - X`:

```bash
uv run --extra image python \
  experiments/train_distilled_terminal_sde.py \
  --target-mode residual --overfit-sources 4 --steps 6000 \
  --condition-dropout 0 --guidance-scale 1
```

This is also the strongest smoke test for the hypothesis. With four sources
per domain and deterministic top-1 teachers, residual diffusion reaches
terminal latent MSE `2.8e-3` cat → dog and `5.4e-3` dog → cat; decoded samples
are visually indistinguishable from their teacher endpoints. The deliberately
memorized historical pair remains excluded from both source and support pools.

Full-data inference never retrieves a training endpoint. It moves held-out
laws in the correct direction and preserves source DINO content, but one small
shared U-ViT still produces off-manifold endpoints on difficult geometries.
Residual diffusion improves dog → cat more than cat → dog. This separates the
remaining problem from the SDE construction: the next scale-up should use
direction-specific terminal score models (or a substantially pretrained
conditional latent DiT) and preserve the same empirical-kernel teacher and
pinned-SDE evaluation.
