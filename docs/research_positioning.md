# Research positioning, July 2026

## Recommendation

Use a latent stochastic interpolant / Schrödinger bridge as the main model.
Treat Iterative Flow Matching as an optional outer path-correction loop after a
first bridge works. Do not return to a GAN observation model.

The reason is structural. The task gives two endpoint measures and asks for one
bidirectional stochastic law between them. A Schrödinger bridge represents
exactly that object. Rectified flow gives an efficient deterministic transport,
but it does not by itself model conditional multiplicity. IFM improves a
chosen generative path iteratively; it does not solve the missing semantic
coupling or define reciprocal forward/backward dynamics.

## Closest work

- [Stochastic Interpolants](https://arxiv.org/abs/2303.08797) supplies the
  flow/score regression and the ODE/SDE family with common marginals.
- [[SF]2M](https://proceedings.mlr.press/v238/tong24a.html) learns stochastic
  dynamics from unpaired samples using static entropic OT without path
  simulation. This is the closest clean baseline to the proposed core.
- [Schrödinger Bridge Flow](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bb3cfcb0284642a973dd631ec9184f2f-Abstract-Conference.html)
  iterates directly on path measures and targets the SB as its unique stationary
  point, avoiding a freshly trained diffusion model at every iteration.
- [Diffusion Bridge Implicit Models](https://openreview.net/forum?id=eghAocvqBk)
  provides deterministic and non-Markovian fast samplers for diffusion bridge
  models. It is relevant when the demo's number of function evaluations matters.
- [Iterative Flow Matching](https://arxiv.org/abs/2502.16445) motivates path
  correction to reduce hallucinations. It is best positioned here as refinement,
  not as the base mathematical object.
- [Exploring the Design Space of Diffusion Bridge Models](https://openreview.net/forum?id=9dEU1gq3v9)
  unifies diffusion bridges and stochastic interpolants with endpoint
  conditioning, preconditioning, and improved sampling. This is the strongest
  2025 implementation template.
- [Fractional Diffusion Bridge Models](https://openreview.net/forum?id=Hhc5McwASX)
  extends bridge noise beyond Markov Brownian motion and reports unpaired image
  translation. Fractional noise is an interesting later ablation, not a sensible
  first MPS experiment.
- [Bridge Matching Sampler](https://openreview.net/forum?id=gUbNUNuJbO) (ICML
  2026) reframes diffusion matching as a fixed-point iteration rooted in
  Nelson's relation. Its forward/backward consistency is directly relevant.
- [Self-Supervised Semantic Bridge](https://arxiv.org/abs/2602.16664) (2026)
  injects self-supervised visual priors into diffusion bridges for spatially
  faithful unpaired translation. This is the most important collision with a
  simple “DINO + bridge” novelty claim.
- [Bridging Day and Night](https://arxiv.org/abs/2602.15383) (2026) explicitly
  suppresses target-class hallucinations along an SB-based unpaired translation
  trajectory. It supports path-local semantic regularization, but also raises
  the novelty bar for such a term.

## What is and is not novel enough

“DINO features + entropic OT + latent diffusion bridge” is probably not, by
itself, a defensible 2026 novelty claim because self-supervised semantic bridges
and simulation-free OT/SB objectives already exist.

A workshop paper becomes plausible if the contribution is a coherent answer to
**bidirectional provenance transport**:

1. one learned reciprocal law for human ↔ AI rather than two translators;
2. a semantic coupling that explicitly projects out the domain/provenance
   direction while retaining instance geometry;
3. Nelson/time-reversal consistency enforced in one direction-conditioned
   latent model;
4. path-level off-manifold and hallucination diagnostics, not only endpoint FID;
5. a study of stochastic conditional multiplicity: several plausible AI
   counterparts for one human image and vice versa.

That package is more distinctive than the cat↔dog demo. The cat/dog experiment
is still useful as a controlled development benchmark.

## Minimum experimental matrix

Compare against:

- independent-pair flow matching and minibatch OT-CFM;
- `[SF]2M` or an equivalent simulation-free SB;
- Schrödinger Bridge Flow;
- IFM/ReFlow path correction;
- a modern endpoint-conditioned diffusion bridge / DBIM sampler.

Report both endpoint and path properties:

- target FID/KID and precision/recall;
- DINO/LPIPS content preservation;
- forward/backward marginal error and time-reversal error;
- k-NN manifold-tube distance over time;
- path speed/curvature and classifier progression;
- conditional diversity at fixed source;
- held-out provenance classifier behavior and human evaluation.

Randomly choosing a held-out cat and dog remains useful only as a stress-test.
The actual conditional demo should sample the source uniformly, then sample its
target from the held-out semantic coupling.
