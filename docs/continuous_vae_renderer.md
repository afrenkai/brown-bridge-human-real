# Manifold VAE observation kernel for reciprocal feature bridges

Let `E` be the frozen DINO encoder. The reciprocal model defines a process
`Z_t` in DINO feature space with endpoint laws

\[
Z_0\sim\mu_{\rm cat},\qquad Z_1\sim\mu_{\rm dog},
\]

and an endpoint coupling chosen by entropic optimal transport. An image
renderer is an observation kernel. It is useful only if its image law remains
near the natural-image manifold while its semantic evolution agrees with the
feature process. Linear StarGAN style interpolation fails this condition: its
endpoints are valid, but mixed-domain styles were never trained.

## Accepted spatial latent bridge

The AFHQv2 VAE encoder produces a diagonal Gaussian posterior on an `8 x 8`
latent field,

\[
q_\phi(h\mid x)=\mathcal N\!\left(m_\phi(x),
\operatorname{diag}\exp\ell_\phi(x)\right).
\]

For a coupled cat/dog pair, rendering uses the marginal form of an ambient
Brownian bridge,

\[
H_\alpha=(1-\alpha)H_0+\alpha H_1
+\sqrt{2\kappa\alpha(1-\alpha)}\,\xi,
\qquad \xi\sim\mathcal N(0,I).
\]

It is endpoint-pinned almost surely. DINO supplies a monotone reciprocal-SDE
species clock `a_t`; uniform time regularizes it:

\[
\alpha_t=(1-\rho)t+\rho a_t.
\]

The accepted decoder is deliberately manifold-only,

\[
X_t=G_\theta(H_{\alpha_t}).
\]

DINO chooses the endpoint coupling, defines the clock, and evaluates the path,
but no DINO feature tensor or DINO gradient enters `G_theta`.

## Objective and two regularizers

Real AFHQv2 cat and dog images supply endpoint reconstruction supervision:

\[
\mathcal L(\theta,\phi)=
\mathcal L_{\rm rec}
+\lambda_{\rm edge}\mathcal L_{\rm edge}
+\lambda_{\rm KL}
D_{\rm KL}\!\left(q_\phi(h\mid x)\,\|\,\mathcal N(0,I)\right).
\]

The final run used `lambda_edge = 0.2` and `lambda_KL = 0.001`. The edge term
retains face structure; the KL term makes endpoint posteriors bridgeable. The
`8 x 8` field is the architectural compromise: `4 x 4` was smooth but reduced
animals to coarse blobs, while the larger field reconstructs recognizable
faces at 1.63M parameters.

## Rejected direct-DINO model

A 2.81M-parameter conditional VAE was also trained with global and patch DINO
states plus a DINO cycle loss. It achieved good numerical feature alignment but
decoded colored, nonsemantic classifier preimages. Scaling its DINO condition
to zero after training did not recover animals because the bad geometry was
already present in its latent space. This negative control motivates the strict
separation between transport dynamics and image formation above.

## Relation to other approaches

- Residual flow matching learns a deterministic velocity field for samples.
- Flow editing or steering modifies an existing generative trajectory.
- The reciprocal SDE here governs the feature-space law and visual clock.
- The VAE supplies a cheap MPS-compatible observation kernel, but its L1/Gaussian
  likelihood produces softer detail than a latent diffusion decoder would.

Held-out AFHQ reconstruction tests the observation model. DINO scores are
reported only as diagnostics because the classifier was trained on CIFAR-10
and can misclassify real AFHQ dogs.

Run with:

```bash
uv run python experiments/train_continuous_vae_renderer.py \
  --latent-grid 8 --latent-channels 32 \
  --lambda-edge 0.2 --lambda-kl 0.001 \
  --lambda-bridge 0 --conditioning-scale 0
```
