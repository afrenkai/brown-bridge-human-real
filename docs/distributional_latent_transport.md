# Source-only distributional latent transport

## Problem

Let \(E:\mathcal X\to\mathbb R^d\) and \(D:\mathbb R^d\to\mathcal X\) be the
frozen AFHQ autoencoder. The endpoint measures are

\[
\mu_0=E_\#\mathsf P_{\rm cat},\qquad
\mu_1=E_\#\mathsf P_{\rm dog}.
\]

Inference receives only \(x\sim\mathsf P_{\rm cat}\). A displayed dog is an
evaluation-only content reference, never a conditioning endpoint. This is
therefore different from the earlier latent-chord demo
\(D((1-t)E(x)+tE(y))\), which requires a particular dog \(y\).

The full-latent experiment has \(d=8\cdot32\cdot32=8192\). Pooled ablations
use \(c=Pz\), \(z=U(Pz)+h\), and

\[
z_t=U(c_t)+\rho(t)h_0,
\qquad
\rho(t)=\rho_1+(1-\rho_1)(1-t)^p.
\]

For \(\rho_1=0\), the residual solves the deterministic equation
\(dh_t=-\kappa(t)h_tdt\), with
\(\kappa(t)=-\partial_t\log\rho(t)\). Pooling exposed a real tradeoff:
\(\rho=1\) preserved cat anatomy, while \(\rho_1=0\) exposed the lifting grid.
The best run therefore transports the complete latent, for which \(h=0\).

## LightSB endpoint law

For Brownian reference

\[
dZ_t=\sqrt\varepsilon\,dW_t,
\]

the diagonal LightSB potential gives

\[
Z_1\mid Z_0=z
\sim \sum_{k=1}^K w_k(z)
\mathcal N(r_k+S_kz,\varepsilon S_k),
\]

where \(S_k\) is positive diagonal and

\[
w_k(z)\propto \alpha_k
\exp\left(\frac{z^\top S_kz+2z^\top r_k}{2\varepsilon}\right).
\]

The potential is fitted from independent marginal samples by minimizing

\[
\mathcal L_{\rm SB}
=\mathbb E_{\mu_0}[\log C_\theta(Z_0)]
-\mathbb E_{\mu_1}[\log v_\theta(Z_1)].
\]

No empirical cat/dog pairing enters this loss. The implementation adds two
explicit regularizers after dividing the objective by \(d\):

\[
\frac1d\mathcal L_{\rm SB}
+\lambda_S\frac1{Kd}\sum_k
\lVert\log S_k-\log S_{\rm init}\rVert^2
+\lambda_w\mathrm{KL}(\bar w\Vert\mathrm{Unif}_K),
\]

with \(\bar w=\mathbb E_{\mu_0}w(Z_0)\). Dimension normalization matters:
without it, the regularizers were negligible and all mass collapsed to one
potential. The first failed trace reached
\(\mathrm{KL}(\bar w\Vert\mathrm{Unif}_{10})=\log 10\).

For a sampled endpoint \(Y\), the displayed stochastic trajectory is one
temporally coherent conditional Brownian bridge,

\[
Z_t=(1-t)Z_0+tY+\sqrt\varepsilon\,(W_t-tW_1),
\quad
\operatorname{Cov}(Z_t\mid Z_0,Y)=\varepsilon t(1-t)I.
\]

The LightSB mean display uses
\(Y=\mathbb E[Z_1\mid Z_0]\) and omits bridge noise. It is a conditional-mean
visualization, not a stochastic sample from the endpoint law.

## Rectified-flow comparator

The comparator uses 32,768 pairs drawn from 256 independently resampled
128-by-128 entropic minibatch-OT plans. This approximates the expected batch
transport plan instead of treating one small batch as the population
coupling. Given a sampled pair \((X,Y)\), training uses

\[
Z_t=(1-t)X+tY,\qquad
\min_v\mathbb E\lVert v_\theta(t,Z_t)-(Y-X)\rVert^2.
\]

The full-latent rectified MLP was narrowed to width 256 for MPS. It underfit
relative to its 2048-dimensional version, whereas diagonal LightSB scaled to
8192 dimensions with only \(O(Kd)\) parameters.

## 2026 context

The experiment incorporates three useful lessons from newer work:

- expected minibatch OT is an actual batch-size-dependent coupling, not merely
  an implementation detail;
- low-frequency image transport can reduce flow curvature, but a decodable
  projection must retain enough spatial information for the observation model;
- geometry-aware flow matching suggests spherical/directional structure as a
  next test, rather than assuming Euclidean latent coordinates are optimal.

Relevant primary sources:

- [Light Schrödinger Bridge](https://arxiv.org/abs/2310.01174)
- [Light and Optimal Schrödinger Bridge Matching](https://arxiv.org/abs/2402.03207)
- [LightSBB-M: Bridging Schrödinger and Bass for Generative Diffusion Modeling](https://arxiv.org/abs/2601.19312)
- [Expected Batch Optimal Transport Plans and Consequences for Flow Matching](https://arxiv.org/abs/2605.12174)
- [Optimal Transport Flow Matching by Design](https://arxiv.org/abs/2606.04092)
- [Geometry-Aware Image Flow Matching](https://arxiv.org/abs/2605.25294)

## Held-out result

All numbers use 64 seeded AFHQ test cats and 64 untouched AFHQ test dogs. The
AFHQ DINO linear probe has balanced test accuracy 0.9166. Dense path metrics
use the first five seeded cats; endpoint-law metrics use all 64.

| model | mean endpoint P(dog) | monotone steps | monotone paths | DINO Sinkhorn to dogs | source-content cosine | diversity / dogs |
|---|---:|---:|---:|---:|---:|---:|
| full-latent LightSB mean | 0.5004 | 0.9700 | 0.6000 | 0.4298 | 0.9618 | 1.3706 |
| full-latent LightSB SDE | 0.3990 | 0.8300 | 0.2000 | 0.4341 | 0.9602 | 1.3621 |
| full-latent OT rectified flow | 0.2112 | 0.9400 | 0.8000 | 0.4619 | 0.9615 | 1.2838 |

The full-latent LightSB mean is the strongest faithful visualization. One of
the five displayed black cats develops a clear dog muzzle and folded ears;
other examples remain cat-like. The experiment therefore demonstrates a real,
source-only movement of part of the empirical mass, but not a solved
cat-to-dog map. A probe mean near 0.5 is not evidence that every cat became a
dog.

Increasing timesteps alone cannot repair an endpoint law that remains between
manifolds. The follow-up below instead changes the observation model while
leaving the source-only marginal-transport claim intact.

## Recovering the earlier doglike-cat result

The sharp earlier `full_dataset_latent_bridge` and
`latent_schrodinger_bridge` artifacts were traced to an endpoint-conditioned
operation. At evaluation they selected a dog with a new DINO assignment,
encoded that exact dog, and decoded a chord to its latent. The learned
reciprocal marginal field was a recorded negative result that stayed close to
the identity. Those images therefore established that the decoder could draw
the path, not that a source-only terminal law had been learned.

The successful high-resolution mechanism was the frozen AFHQ StarGAN-v2
observation model. Let (E_c,E_d:\mathcal X\to\mathbb R^{64}) be its
domain-specific style heads, (G(x,s)) its generator, and let (A) be the
invertible coordinate standardizer fitted to both style marginals. The new
endpoint measures are

\[
\eta_0=(A\circ E_c)_\#\mathsf P_{\rm cat},\qquad
\eta_1=(A\circ E_d)_\#\mathsf P_{\rm dog}.
\]

LightSB is fitted to independent samples from \(\eta_0\) and \(\eta_1\).
For a held-out cat (x), inference is

\[
S_0=A(E_c(x)),\quad S_1\sim\pi_\theta(\cdot\mid S_0),\quad
X_t=G\!\left(x,A^{-1}(S_t)\right).
\]

Conditional on (S_0,S_1), the reciprocal component is the pinned diffusion

\[
dS_t=\frac{S_1-S_t}{1-t}\,dt+\sqrt\varepsilon\,dW_t,
\qquad 0\le t<1.
\]

The source image (x) remains in the generator's content stream for every
time. A dog image is used only to estimate the training marginal or to compute
held-out population metrics; no dog is chosen at inference. This separates
the two responsibilities cleanly: LightSB transports the endpoint law, while
the frozen domain-trained generator supplies a decodable observation chart.

On 64 seeded AFHQ test cats, after fitting all 5,064 permitted cats and 4,677
permitted dogs:

| style-space method | mean endpoint P(dog) | dog endpoints | monotone steps | DINO Sinkhorn to dogs | source-content cosine | diversity / dogs |
|---|---:|---:|---:|---:|---:|---:|
| LightSB conditional mean | **0.8658** | 0.8906 | 0.8405 | **0.0276** | 0.9363 | 1.2323 |
| LightSB reciprocal SDE | 0.8439 | **0.9062** | 0.8164 | 0.0491 | 0.9397 | 1.3180 |
| expected-minibatch-OT rectified flow | 0.8449 | 0.8594 | **0.8542** | 0.0519 | **0.9420** | **1.3440** |

This is the first source-only experiment in the repository whose endpoints
are consistently recognizable dogs. The LightSB mean gives the closest
terminal distribution; stochastic endpoint sampling gives the highest dog
coverage. Rectified flow preserves marginally more DINO content but shows
more cross-domain color artifacts in the visual path.

The contact sheet selects five paths *within the already evaluated set* using
only their source/trajectory diagnostics: high terminal dog probability and
low backward semantic variation. It does not change the population metrics
and does not consult a dog reference. The selected row indices and dense-time
indices are recorded in `summary.json`.

## Uniform physical time and the layerwise observation controller

The selected sheet above is useful for endpoint quality but is a poor account
of time: its per-path semantic subsampling often chooses several indices near
the domain switch. The follow-up experiment therefore uses only uniform
physical times and displays the first five seeded test cats without ranking or
selection.

The LightSB endpoint (S_1) remains frozen. For StarGAN decoder block
\(\ell\), a shared controller predicts a monotone clock

\[
h_\ell(t;c)=\sigma\!\left(a_\ell(c)\operatorname{logit}(t)+b_\ell(c)\right),
\qquad a_\ell(c)>0,
\]

where (c=(S_0,S_1-S_0,F(x))). The implementation explicitly defines
\(h_\ell(0;c)=0\) and \(h_\ell(1;c)=1\), then supplies

\[
s_t^{(\ell)}=(1-h_\ell(t;c))s_0+h_\ell(t;c)s_1
\]

to decoder block \(\ell\). Thus every block stays on the endpoint style
segment, every clock is monotone, and both endpoint images are exactly the
original LightSB endpoint images. Only the interior observation map changes.

The controller is trained on AFHQ train cats with frozen StarGAN, DINO, and
LightSB. Its loss combines DINO geodesic tracking, linear species-coordinate
progress, content projected orthogonally to the species direction, constant
DINO speed, excess total variation, and a weak identity-clock penalty.

On the same 64 held-out cats, evaluated at 25 uniform physical times:

| observation path | largest P(dog) step | DINO speed CV | active probability steps | intermediate-frame fraction | endpoint max error |
|---|---:|---:|---:|---:|---:|
| shared global style chord | 0.4739 | 1.2232 | 8.3594 | 0.2937 | 0 |
| learned layerwise clocks | **0.4423** | **1.1391** | **8.5156** | **0.3275** | **0** |

This is a modest but honest improvement: the largest semantic jump falls by
6.7%, per-path speed variation falls by 6.9%, and the proportion of visibly
intermediate frames rises by 11.5% relative. Terminal-law metrics are
identical by construction. The controller does not eliminate the colored
off-manifold StarGAN phase; that is now the clearest remaining renderer
limitation rather than an endpoint-transport failure.
