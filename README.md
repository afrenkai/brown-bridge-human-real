# Brown Bridge

Bidirectional transport between two empirical distributions. The active
examples use entropic optimal transport to choose endpoint couplings and then
either render exact displacement chords or learn a stochastic Brownian-bridge
diffusion. No rectified-flow or flow-matching objective is part of the active
path.

The central distinction is:

1. OT decides which mass in distribution A is coupled to mass in distribution B.
2. A diffusion bridge models a stochastic path between those coupled endpoint
   laws.
3. A decoder or nearest-neighbor lookup turns the continuous path back into
   observations.

## Start here

```bash
uv sync --group dev
uv run pytest -q

# Small image-feature OT example
uv run python experiments/cat_dog.py

# Bidirectional feature-space bridge
uv run python experiments/reciprocal_cat_dog.py

# Exact 784-D pixel OT: MNIST 9 ↔ 1 on Metal
uv run python experiments/mnist_mps_ot.py --download

# Real text: WikiText-2 topic slices, OT Brownian bridge, XM candidates
uv run python experiments/text_wikitext_ot_diffusion.py --download \
  --steps 1500 --rollout-steps 256

# Exact cat ↔ dog KL-VAE latent chords rendered to pixels
uv run --extra image python \
  experiments/render_kl_vae_ot_pair_interpolations.py --cost spatial
```

Generated data, checkpoints, and plots are ignored by git. The historical
entry points moved out of the active tree are listed in
[`deprecated/README.md`](deprecated/README.md).

## Mathematical object

Let `mu_0` and `mu_1` be empirical laws and let `pi` be an entropic OT coupling.
For `(X_0, X_1) ~ pi`, the exact ambient displacement chord is

```text
X_t = (1 - t) X_0 + t X_1.
```

For a stochastic bridge, use

```text
I_t = (1 - t) X_0 + t X_1
      + sqrt(epsilon * t * (1 - t)) * Z.
```

The active diffusion experiments learn bridge score/drift quantities from
these analytically constructed noisy states and integrate a stochastic
differential equation. This is distinct from fitting a deterministic
rectified-flow velocity field.

## Code map

```text
src/brown_bridge/
  ot.py                       balanced entropic OT / Sinkhorn
  paths.py                    exact chords and Brownian bridge paths
  stochastic_interpolants.py Brownian bridge regression targets
  schrodinger.py              empirical Schrödinger bridge utilities
  reciprocal.py               bidirectional SDE utilities
  light_sb.py                 lightweight stochastic-bridge baseline
  exploration.py              XM best-of-K candidate objective

experiments/
  cat_dog.py                  smallest DINO + OT experiment
  reciprocal_cat_dog.py      bidirectional feature-space bridge
  mnist_mps_ot.py             exact 784-D MNIST 9 ↔ 1 OT check
  text_wikitext_ot_diffusion.py
                              WikiText-2 OT bridge with XM candidates
  render_kl_vae_ot_pair_interpolations.py
                              exact KL-VAE OT chords, forward and reverse
  toy_mps_xm_ot.py            small MPS OT + XM sanity check
```

## Text direction

`text_wikitext_ot_diffusion.py` uses real WikiText-2 paragraphs, forms two
weakly topic-conditioned corpus slices, maps them to a continuous TF-IDF/PCA
latent, and solves entropic OT. It then trains a bidirectional Brownian-bridge
diffusion with four candidate drift/score predictions per noisy state. XM
selects the lowest-loss candidate during training; rollout selection scores
candidate states against the destination empirical distribution.

This is a transport proof-of-life, not yet a semantic text editor. TF-IDF/PCA
is a deliberately transparent placeholder for a decoder-compatible JEPA- or
ELF-style continuous text representation. ELF is relevant to the interface—its
model remains in continuous embedding space until late token decoding—but its
published objective is continuous-time flow matching, so it is not used as the
transport objective here.

The next text step is to replace TF-IDF/PCA with a pretrained continuous text
autoencoder or self-supervised latent, then stream a fixed OpenWebText shard.
