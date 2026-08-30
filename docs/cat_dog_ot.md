# Cat/dog transport: current contracts

This note records the distinction that guides the retained image experiments.

## Distributional goal

Cat-to-dog transport is a distribution problem. Unpaired datasets do not define
a unique dog for each cat. An empirical optimal-transport plan supplies one
model-dependent coupling under a chosen feature cost. It is not ground-truth
identity correspondence.

A useful system must report more than target class accuracy. It should also
audit target-distribution distance, diversity, source-content retention,
training-set proximity, and reverse behavior when a reverse map exists.

## Two retained image workflows

### Source-only generation

`experiments/sample_rae_afhq_generator.py` starts from fresh Gaussian RAE latents and a
cat/dog class label. It loads the EMA weights of the fine-tuned official
DiTDH-S stage-2 model. No real cat or dog endpoint is retrieved at inference.

`experiments/evaluate_rae_afhq_generator.py` compares generated samples with held-out AFHQ
validation images of the two trained classes. The current result has strong
class control but contracted diversity. See `rae_afhq_generator.md` for the
full metrics and limits.

### Endpoint-coupled diagnostic

`experiments/render_rae_ot_sde_roundtrip.py` encodes pools of real dogs and cats, removes
the empirical class-mean direction from its content descriptor, solves an
entropic coupling, and samples a pinned latent Brownian bridge. The reverse
half reuses the innovations to return to the same dog endpoint.

This produces the requested dog → cat → same dog visualization, but it is not a
source-only generator. Its selected cat endpoint is part of the construction.

## Small controls

- `experiments/mnist_mps_ot.py` makes sampled empirical couplings and displacement paths
  visible on a simple dataset.
- `experiments/toy_mps_xm_ot.py` tests whether best-of-K training can preserve multiple
  plausible endpoint modes in a small shared bidirectional model.
- `experiments/text_wikitext_ot_diffusion.py` is a compact text-space dynamic baseline. It
  is exploratory and does not claim decoder-compatible language generation.

## Next research step

The next model should make invertibility a structural contract rather than an
after-the-fact cycle penalty. It should consume only a source state and a target
condition, share parameters across directions, and expose exact inverse and
composition tests. Image-space RAE work provides the generative baseline. The
static-atlas decision record describes the language-facing direction.
