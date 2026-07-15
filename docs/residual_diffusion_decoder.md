# Conditional residual diffusion decoder

## Separation of transport and rendering

Let (m_t) be the reciprocal cat-to-dog path constructed in the frozen DINO
feature space and let (G_\theta(m_t)) be the accepted manifold-VAE image.
The experiment keeps this transport fixed and learns only an AFHQ image
formation residual.  For a real image (X),

\[
B = G_\theta(E_\theta(X)), \qquad R_0 = \operatorname{clip}
\left(\frac{X-B}{s},-1.25,1.25\right), \qquad s=0.25.
\]

No DINO feature enters the residual decoder and no decoder gradient enters the
transport.  This avoids the classifier-preimage failure of the earlier direct
DINO-conditioned renderer.

## VP-SDE and score objective

The forward residual process is the variance-preserving SDE

\[
dR_\tau=-\tfrac12\beta(\tau)R_\tau\,d\tau
          +\sqrt{\beta(\tau)}\,dW_\tau,
\]

with the cosine cumulative schedule

\[
\bar\alpha_\tau =
\frac{\cos^2\!\left(\frac{\tau+0.008}{1.008}\frac{\pi}{2}\right)}
     {\cos^2\!\left(\frac{0.008}{1.008}\frac{\pi}{2}\right)}.
\]

Thus

\[
R_\tau=\sqrt{\bar\alpha_\tau}R_0+
\sqrt{1-\bar\alpha_\tau}\,\varepsilon,
\quad \varepsilon\sim\mathcal N(0,I),
\]

and a 581k-parameter conditional U-Net minimizes

\[
\mathcal L_{\rm DSM}=
\mathbb E\lVert\varepsilon-\varepsilon_\psi(R_\tau,B,\tau)\rVert_2^2.
\]

Two regularizers are evaluated where (\bar\alpha_\tau\ge 0.05):

\[
\mathcal L = \mathcal L_{\rm DSM}
+\lambda_{\rm LF}\lVert K_9*\widehat R_0\rVert_2^2
+\lambda_{\rm range}\lVert (|\widehat R_0|-1)_+\rVert_2^2,
\]

with both lambdas set to (0.02).  The first discourages residual color drift;
the second discourages isolated saturated impulses.

## Sampling modes

Full-noise DDIM sampling produced non-semantic colored texture on bridge
interiors.  A truncated reverse process starts instead at (\tau_0<1) from
the zero-residual prior

\[
R_{\tau_0}=\sqrt{1-\bar\alpha_{\tau_0}}\,\varepsilon.
\]

Sharing (\varepsilon) along all frames of a particle greatly improves
temporal coherence, but the texture remains non-semantic.  Setting
(\varepsilon=0) gives a deterministic probability-flow-like conditional
mode.  It is clean but restores little detail.  Real cat and dog endpoints are
always pinned after refinement.

## AFHQv2 result

- 9,743 training images and 984 held-out images (cat and dog combined)
- 2,000 MPS updates in 5m13s
- denoising MSE: 1.096 at the first update, 0.102 at update 2,000
- full-noise held-out edge energy: 0.0564, versus 0.0414 for real images and
  0.0126 for the VAE base
- full-noise held-out L1: 0.1062, versus 0.0677 for the VAE base
- deterministic bridge adjacent-frame L1: 0.0527, versus 0.0525 for the VAE

The stochastic decoder therefore adds high-frequency energy but not
recoverable animal detail.  The deterministic decoder avoids artifacts but
does not solve blur.  This is a useful negative result rather than a renderer
upgrade.

## Architectural consequence

The missing detail is not conditionally identifiable from a blurred,
off-manifold intermediate image.  A stronger next model should make image
formation part of the bridge itself:

\[
dX_t=b_\phi(X_t,t,m_t)\,dt+\sigma_t\,dW_t,
\]

where (m_t) is the reciprocal DINO state and (b_\phi) is trained by an
endpoint-only Schrödinger-bridge/IPF objective on real AFHQ marginals.  A
32x32 learned image latent would keep this feasible on MPS.  Unlike the
residual decoder, that model learns intermediate image marginals rather than
being asked to invent a unique high-frequency target for an unseen VAE morph.
