# Moving a Cat Distribution to a Dog Distribution with Optimal Transport

*A research note on the Brown Bridge experiment, and on how Explorative
Modeling could make the next version more expressive.*

## The question

Suppose we have a collection of cat images and a separate collection of dog
images. There is no identity-preserving label saying which dog belongs with
which cat. The goal is therefore not to memorize a paired image translator.
The goal is to learn a law that moves the whole cat distribution to the whole
dog distribution—and that can run in the reverse direction as well.

Write the two image distributions as \(\mathsf P_{cat}\) and
\(\mathsf P_{dog}\). A useful transport system should have three properties:

1. its samples should begin in the cat distribution and end in the dog
   distribution;
2. its correspondence should preserve useful instance information, rather
   than pairing every cat with an arbitrary dog;
3. the reverse process should be part of the same model, not an unrelated
   dog-to-cat network.

The repository treats this as a distributional problem. It does not claim that
there is a unique “correct” dog for a particular cat.

## What optimal transport contributes

Optimal transport (OT) chooses a coupling \(\pi(x,y)\) whose marginals are
the cat and dog distributions while minimizing a ground cost:

\[
  \min_{\pi\in\Pi(\mu_0,\mu_1)}
  \mathbb E_{(x,y)\sim\pi}[c(x,y)].
\]

In the experiment, the images are first represented by a frozen DINO encoder.
The cost is computed in a domain-projected semantic coordinate system: a
direction associated with “cat versus dog” is suppressed, while the remaining
geometry is used to find plausible content correspondences. Entropic
regularization makes the empirical Sinkhorn problem stable and gives the
coupling some controlled softness.

This coupling is a modeling assumption, not ground-truth correspondence. A
low-cost pair means “similar under the representation and cost we chose,” not
“these two images depict the same individual.” That distinction is central to
the project.

Once an endpoint coupling has been chosen, the code constructs either an exact
displacement chord or a Brownian-bridge diffusion between paired endpoints. The
main stochastic construction is

\[
  I_t=(1-t)X_0+tX_1+\sqrt{\varepsilon t(1-t)}Z,
\]

where \((X_0,X_1)\sim\pi\) and \(Z\) is standard Gaussian noise. Score and
bridge-drift regression produce reciprocal forward/backward SDEs. The endpoint
coupling remains the OT object; the SDE only supplies a stochastic realization
of the path.

The repository also contains a source-only latent-transport formulation. It
fits a Light Schrödinger Bridge to independent cat and dog latent samples and
uses OT-based or learned endpoint structure for evaluation and comparison.
That is a stronger test of distributional transport than simply decoding a
chord between a selected cat and a selected dog.

## What the current result does—and does not—show

The local implementation is real and tested: the OT solver, stochastic
interpolants, latent models, bridge utilities, and experiments have unit tests,
and the full test suite passes. The important scientific caveat is that a
successful endpoint classifier score is not enough. A model can make an image
look dog-like while losing source content, collapsing diversity, or relying on
an evaluation-time dog reference.

The useful evaluation set is therefore plural:

- target-distribution quality and DINO Sinkhorn distance;
- source-content cosine or LPIPS preservation;
- diversity of several dog outcomes for one cat;
- classifier progression and path monotonicity;
- forward/backward marginal error and time-reversal consistency;
- distance from the union of the endpoint manifolds along the path.

The strongest image-space runs in the repository use a frozen latent autoencoder
for decoding and DINO for semantic geometry. The decoder is an observation
model; it is not itself evidence that the transport law is correct.

## Enter Explorative Modeling

[Explorative Modeling (XM)](https://arxiv.org/abs/2607.27372) proposes a simple
answer to a familiar generative-modeling failure: if one prediction is trained
against many valid targets, squared error tends toward their average. The
average may be blurry and may not lie on the data manifold.

Forward XM samples \(K\) candidate outputs for one training example and keeps
only the candidate with the lowest reconstruction loss:

```python
losses = []
for _ in range(K):
    candidate = model(source, sample_noise())
    losses.append(loss(candidate, target))
min(losses).backward()
```

The extra computation happens during training. At inference, the model can
sample one candidate—or a small number of candidates—instead of unrolling a
long diffusion trajectory. The XM authors report that increasing exploration
improves multimodal image, video, and language models, and that their end-to-end
models can match diffusion-style control results with much lower inference
compute. Those are claims from a very new preprint and should be treated as
promising rather than established consensus. The authors provide [a project
page](https://explorative-modeling.github.io/) and [PyTorch
code](https://github.com/alexiglad/XM).

XM is especially relevant here because it is not itself an OT solver. The paper
argues that directly computing a global coupling is expensive and that
minibatch OT can be a biased proxy. XM instead searches over the model's own
candidate latents and keeps the best match. That solves a different problem:

| Question | Best tool in this project |
| --- | --- |
| Which endpoint mass must move to which endpoint distribution? | Balanced entropic OT / a Schrödinger endpoint coupling |
| How can one source support several plausible target images? | XM's best-of-\(K\) candidate search |
| How should the mass move continuously? | Stochastic interpolants, a bridge, or a flow |
| How do we turn a latent trajectory into pixels? | A frozen latent autoencoder |

The combination is more interesting than either method alone. OT gives XM a
global distributional contract. XM gives OT-based conditional transport a way
to avoid turning multiple valid targets into one conditional mean.

## A concrete XM+OT experiment

The repository now has the first small XM+OT hook. The existing bidirectional
latent diffusion trainer already consumes expected minibatch-OT cat/dog pairs;
passing `--exploration-candidates K` makes each diffusion training example
explore K noise candidates and backpropagate only through the best one. The
default `K=1` is the original baseline.

For example:

```bash
uv run python experiments/train_bidirectional_kl_vae_diffusion.py \
  --exploration-candidates 4 \
  --steps 12000
```

This is intentionally a training-time integration, not a claim that XM has
already solved the image-space transport problem.

### 1. Freeze the semantic geometry

Use the existing DINO encoder, domain projection, and held-out splits. Solve an
entropic OT plan between cat and dog training features. Sample training pairs
from that plan, rather than pairing cats and dogs independently.

### 2. Train a conditional latent generator

Use the source image's frozen VAE latent, a direction bit, and a noise vector as
inputs. Predict a target latent in one or a few steps. For each OT training pair
\((x,y)\), draw \(K\) noise candidates and score each candidate with a joint
loss:

\[
  L_k =
  L_{latent}(\hat y_k,y)
  + \lambda_{sem}L_{DINO}(\hat y_k,y)
  + \lambda_{content}L_{content}(\hat y_k,x).
\]

Backpropagate only through the candidate with the smallest \(L_k\). The latent
term keeps the optimization efficient; the semantic term keeps the selected
candidate in the target domain; and the content term prevents the model from
ignoring its source image.

### 3. Share the reverse model

Train cat→dog and dog→cat with one direction-conditioned network. Use the same
architecture and enforce the existing reciprocal/time-reversal identities
where the parameterization allows it. This gives us one bidirectional law and
lets us test whether a model that succeeds in one direction also returns mass
to the original distribution.

### 4. Compare the right baselines

The active ablation should compare exact OT chords, OT Brownian bridges without
XM, and OT Brownian bridges with XM-\(K\), for several values of \(K\).

Keep the architecture, data, and compute budget visible. XM spends more
training compute by design, so the fair comparison reports both quality at a
fixed number of updates and quality at a fixed FLOP budget.

### 5. Test distributional correctness, not just pretty samples

For each held-out cat, sample several dog candidates. Report target marginal
quality, diversity, source-content preservation, DINO semantic distance, and
the reverse-cycle distributional error. Also report the transport plan's row
and column marginal errors. If XM improves visual sharpness but the generated
dog distribution misses modes or the reverse law fails, it is not an
improvement to the transport problem.

## The main lesson

The cat-to-dog problem has two different kinds of uncertainty. OT handles
*where the distribution should go*. XM may help decide *which of several
plausible outputs should explain a particular training target*. A bridge or
flow handles *the path between them*. A decoder handles *how that path is
observed as an image*.

Keeping those roles separate prevents a common mistake: treating a visually
convincing, endpoint-conditioned morph as proof that a source-only distribution
has been transported. The useful research direction is an XM-augmented,
semantically coupled, bidirectional latent transport model whose success is
measured by both endpoint marginals and provenance-preserving paths.

## Text extension: WikiText-2 before OpenWebText

The same separation gives a small text proof-of-life. The experiment in
`experiments/text_wikitext_ot_diffusion.py` uses real WikiText-2 paragraphs,
selects two weakly topic-conditioned corpus slices, builds a continuous TF-IDF
PCA representation, and solves entropic OT between the two empirical measures.
It then trains a bidirectional Brownian-bridge diffusion model in that latent
space. The model predicts the stochastic bridge drift and score; it does not
train a flow-matching or rectified-flow velocity field.

The XM hook gives each noisy bridge state several candidate drift/score
predictions and backpropagates through the candidate with the smallest bridge
loss. At sampling time, candidate states are selected against the destination
empirical distribution. This is a useful exploratory mechanism, but the
WikiText-2 representation is intentionally only a placeholder for a JEPA- or
ELF-style decoder-compatible latent. ELF is a strong architectural reference
for continuous text embeddings and late token decoding, while its own core
objective remains flow matching; the pure-OT experiment keeps that interface
separate from the diffusion transport objective.

Run it with:

```bash
uv run python experiments/text_wikitext_ot_diffusion.py --download \
  --steps 1500 --rollout-steps 256
```

WikiText-2 is small enough for a reproducible toy. The next scale-up should
stream a fixed OpenWebText shard and replace TF-IDF PCA with a pretrained
continuous text autoencoder or JEPA-like representation.

## References

- [Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation](https://arxiv.org/abs/2607.27372)
- [Explorative Modeling project page](https://explorative-modeling.github.io/)
- [XM reference implementation](https://github.com/alexiglad/XM)
- [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797)
- [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://proceedings.mlr.press/v238/tong24a.html)
- [A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport](https://leonard.perso.math.cnrs.fr/papers/2014-Leonard-A%20survey%20of%20the%20Schrodinger%20problem%20and%20some%20of%20its%20connections%20with%20optimal%20transport.pdf)
