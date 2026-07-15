# Full non-quarantined AFHQ latent bridge

## Data and quarantine

The model is initialized from scratch after the single-pair sanity check. Its
training measures contain 5,064 cats and 4,677 dogs. The memorized cat and dog
are absent by SHA256 from descriptor extraction, augmented training, and
held-out evaluation.

## Mini-batch semantic endpoint coupling

For every training image, frozen DINO supplies a compact species-orthogonal
descriptor containing its global token and a 2x2 pooling of its 4x4 patch-token
grid. At each update, independent cat and dog minibatches define empirical
measures (a_B,b_B), and the coupling is

\[
\pi_B=\arg\min_{\pi\in\Pi(a_B,b_B)}
\langle C_{DINO},\pi\rangle
+\varepsilon\operatorname{KL}(\pi\Vert a_B\otimes b_B),
\qquad \varepsilon=0.1.
\]

Pairs sampled from (\pi_B) determine which latent endpoints share a
conditional path. Across logged batches, expected OT content cost remains
below the independent random-pair cost.

## Joint endpoint and path training

The observation model is an 8x32x32 deterministic latent autoencoder. For a
sampled pair, the stochastic midpoint used during training is

\[
Z_t=(1-t)E_\theta(X_0)+tE_\theta(X_1)
+\sqrt{2\tau t(1-t)}\,\xi,
\qquad \tau=5\times10^{-4}.
\]

The loss is

\[
\mathcal L=\mathcal L_{endpoint}
+\lambda_{cycle}\lVert E_\theta(G_\theta(Z_t))-Z_t\rVert_1
+\lambda_{acc}\lVert X_{t+h}-2X_t+X_{t-h}\rVert_1,
\]

where (h=0.05), (\lambda_{cycle}=0.05), and
(\lambda_{acc}=0.02). Endpoint loss combines pixel L1, edge L1, and two
coarser image scales. Path regularization is ramped over the first 1,000 of
4,000 updates.

## Result

- held-out reconstruction L1: 0.02591
- held-out edge L1: 0.02982
- five untouched test paths end with DINO dog probabilities between 0.948 and
  0.955
- mean adjacent displayed-frame L1: 0.04761
- 48 tests pass

Visually, all five untouched cats become distinct paired dogs with coherent
eyes, muzzles, ears, coat colors, and backgrounds. The paths have neither the
colored residual-diffusion artifacts nor the near-identity failure of the
learned marginal velocity field.

## Interpretation

This scales the successful single-pair capacity check to the endpoint
distributions, but it does not memorize a global one-to-one assignment.
Instead it repeatedly samples balanced local entropic couplings. The displayed
trajectory is the conditional mean of one held-out coupling component; adding
the pinned Brownian term produces stochastic variants around that component.

For the public demo, `experiments/render_full_dataset_demo.py` samples test
indices from a recorded seed, renders 21 frames, and evaluates them with a
linear probe trained only on frozen AFHQ-train DINO features. Its contact sheet
uses a nine-frame subset for readability. The animation remains the decoded
latent chord

\[
\widehat X_t=G_\theta((1-t)E_\theta(X_0)+tE_\theta(X_1)),
\]

not an Euler--Maruyama trajectory of a learned drift. This distinction is
recorded in the output summary and should be retained in presentations.
