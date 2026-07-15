# Cat-to-dog transport in DINO feature space

## 1. Measures, not two point clouds

Let \(P_c\) and \(P_d\) be dataset-dependent image distributions and let
\(\phi:\mathcal X\to\mathbb R^D\) be a fixed or fine-tuned DINO encoder. The
feature measures are the pushforwards

\[
\mu_0=\phi_\#P_c,\qquad \mu_1=\phi_\#P_d.
\]

The fact that individual cats and dogs differ is part of the measure, not a
violation of the model. If \(\xi\) collects pose, breed, background, texture,
lighting, and acquisition variables, then schematically

\[
\mu_k=\int \mu_k^{\xi}\,\eta_k(d\xi),\qquad k\in\{0,1\}.
\]

No component labels or one-to-one identities are required. The coupling is a
joint law on all of this heterogeneity. What it regards as corresponding
variation is determined by the DINO representation and the ground cost. This
is why CLS-only, patch-aware, and multi-layer costs should be treated as
different statistical models rather than interchangeable implementation
details.

For samples \(x_i\) and \(y_j\), normalize the feature rows and form

\[
\widehat\mu_0=\sum_i a_i\delta_{z_i},\qquad
\widehat\mu_1=\sum_j b_j\delta_{w_j},\qquad
z_i,w_j\in\mathbb S^{D-1}.
\]

DINO is an encoder, not a decoder. Everything below produces a feature-space
path. Producing intermediate images requires a separate renderer or generative
decoder conditioned on those features.

## 2. Endpoint coupling

The balanced coupling polytope is

\[
\Pi(a,b)=\{\pi\ge 0:\pi\mathbf 1_m=a,\ \pi^\top\mathbf 1_n=b\},
\]

up to the chosen row/column convention. This implementation solves

\[
\pi_\varepsilon
=\arg\min_{\pi\in\Pi(a,b)}
\langle\pi,C\rangle
+\varepsilon\operatorname{KL}(\pi\Vert a\otimes b)
\]

with log-domain Sinkhorn iterations and an explicit marginal tolerance.

Three ground costs are exposed:

- `sqeuclidean`: \(C_{ij}=\lVert z_i-w_j\rVert^2\), for ambient OT and the
  Brownian reference model;
- `cosine`: \(1-\langle z_i,w_j\rangle\), equal to half the squared chordal
  distance on normalized features;
- `angular_squared`: \(\arccos(\langle z_i,w_j\rangle)^2\), the squared
  geodesic distance on the unit sphere.

The regularized objective is biased as a discrepancy, so distribution
comparison uses the debiased Sinkhorn divergence. The transport *plan* itself
is still used to couple endpoints.

## 3. Why the old barycentric movie is not the transported distribution

The forward barycentric projection is

\[
\bar T(z_i)=\frac{\sum_j\pi_{ij}w_j}{\sum_j\pi_{ij}}.
\]

Its conditional-mean interpretation is useful, but generally
\(\bar T_\#\widehat\mu_0\ne\widehat\mu_1\). Entropic plans are diffuse, so
barycenters can contract modes and land between valid target features.

The corrected construction retains the whole joint law. Draw
\((Z_0,Z_1)\sim\pi_\varepsilon\), then move each coupled pair. Because the
coupling has the prescribed marginals, the resulting path has the exact source
law at \(t=0\) and target law at \(t=1\) (up to Monte Carlo error if particles
are sampled).

## 4. Deterministic paths

### Ambient displacement interpolation

For squared Euclidean cost,

\[
Z_t=(1-t)Z_0+tZ_1,\qquad
\mu_t=(Z_t)_\#\pi.
\]

When \(\pi\) is an unregularized optimal coupling, \((\mu_t)_{t\in[0,1]}\) is
a constant-speed \(W_2\) geodesic. With an entropic coupling it retains the
right endpoints but is an entropically softened, not exactly action-minimal,
interpolation. It also leaves the unit sphere at intermediate times.

### Spherical displacement interpolation

For normalized DINO features, the geometry-preserving alternative is the
great-circle path. With

\[
\theta=\arccos\langle Z_0,Z_1\rangle,
\]

\[
Z_t=
\frac{\sin((1-t)\theta)}{\sin\theta}Z_0+
\frac{\sin(t\theta)}{\sin\theta}Z_1.
\]

The implementation uses an equivalent tangent-vector form that remains stable
for coincident and antipodal pairs. Its conditional velocity is the analytic
derivative of this curve. Pair `angular_squared` cost with `spherical`
interpolation; pair `sqeuclidean` cost with `ambient` interpolation.

## 5. Macro timesteps and microsteps

Let \(n\) be the number of macro intervals and \(T\) the number of numerical
microsteps in every macro interval. The grid is

\[
t_\ell=\frac{\ell}{nT},\qquad \ell=0,\ldots,nT.
\]

Thus macro boundaries are \(0,1/n,\ldots,1\), while the ODE solver or direct
geodesic evaluator exposes \(nT+1\) states. `make_time_grid(n, T)` implements
this convention exactly.

## 6. Dynamic OT and conditional flow matching

The Benamou--Brenier formulation seeks a density/velocity pair satisfying the
continuity equation

\[
\partial_t\mu_t+\nabla\cdot(\mu_t v_t)=0
\]

while minimizing kinetic action

\[
\int_0^1\int\lVert v_t(z)\rVert^2\,d\mu_t(z)\,dt.
\]

Rather than storing a separate path for every empirical pair, OT conditional
flow matching amortizes the conditional paths into a vector field. Sample

\[
(Z_0,Z_1)\sim\pi_\varepsilon,\qquad t\sim U[0,1],
\]

evaluate the analytic conditional state \(Z_t\) and velocity \(U_t\), then fit

\[
\mathcal L_{\mathrm{OT-CFM}}(\theta)
=\mathbb E\left[\lVert v_\theta(Z_t,t)-U_t\rVert^2\right].
\]

The population minimizer is the conditional expectation of pairwise velocity
given \((Z_t,t)\), which induces the same marginal continuity equation under
the usual regularity assumptions. `make_ot_cfm_batch` creates simulation-free
training targets; `integrate_flow` provides Euler, Heun, and RK4 integration
with the requested macro/micro grid. For spherical geometry, predicted
velocities are projected onto the tangent space and states are retracted to the
sphere after each numerical step.

## 7. Schrödinger bridge on path space

Let \(E=\mathbb R^D\), \(\Omega=C([0,1],E)\), and let \(R\in\mathcal P(\Omega)\)
be a reference Markov path measure. The dynamic Schrödinger problem is

\[
P^*=\arg\min_{P\in\mathcal P(\Omega)}
H(P\mid R)
\quad\text{subject to}\quad
P_0=\mu_0,\;P_1=\mu_1,
\]

where \(P_t=(X_t)_\#P\). This is the primary class-to-class object: a law on
whole feature trajectories whose endpoint marginals are exactly the cat and
dog measures.

### 7.1 Static endpoint problem and disintegration

Disintegrate the reference with respect to its endpoints,

\[
R(d\omega)=\int R^{xy}(d\omega)\,R_{01}(dx,dy),
\]

where \(R^{xy}=R(\cdot\mid X_0=x,X_1=y)\) is the reciprocal bridge. Entropy
projection changes the endpoint law but not these conditional bridges:

\[
P^*(d\omega)=\int R^{xy}(d\omega)\,\pi^*(dx,dy),
\]

\[
\pi^*=\arg\min_{\pi\in\Pi(\mu_0,\mu_1)}H(\pi\mid R_{01}).
\]

For empirical endpoint weights \(a_i,b_j\), define

\[
K_{ij}=e^{-c(z_i,w_j)/\varepsilon},\qquad
R_{01}(i,j)=\frac{a_i b_jK_{ij}}{\sum_{k,\ell}a_kb_\ell K_{k\ell}}.
\]

Then

\[
\varepsilon H(\pi\mid R_{01})
=\langle\pi,C\rangle
+\varepsilon H(\pi\mid a\otimes b)+\text{constant},
\]

so the log-domain Sinkhorn plan in `ot.py` is precisely the endpoint entropy
projection for this discrete reference.

### 7.2 Euclidean Brownian reference

The implemented ambient reference is

\[
dX_t=\sqrt{2\tau}\,dW_t,
\qquad L=\tau\Delta,
\]

on a unit time horizon. Its transition density contains

\[
p_1(x,y)\propto
\exp\!\left(-\frac{\lVert x-y\rVert^2}{4\tau}\right).
\]

Consequently, with `sqeuclidean` cost,

\[
\boxed{\varepsilon=4\tau}.
\]

Conditional on \((X_0,X_1)=(x,y)\), the reciprocal bridge solves the singular
drift SDE

\[
dX_t=\frac{y-X_t}{1-t}\,dt+\sqrt{2\tau}\,dW_t,
\qquad X_0=x,
\]

and has the explicit representation

\[
X_t=(1-t)x+ty+\sqrt{2\tau}(W_t-tW_1).
\]

Therefore the intermediate law of the empirical bridge is the Gaussian
mixture

\[
\mu_t
=\sum_{i,j}\pi^*_{ij}\,
\mathcal N\!\left(
(1-t)z_i+tw_j,
2\tau t(1-t)I_D
\right),\qquad 0<t<1.
\]

The endpoints remain the prescribed discrete measures because the conditional
variance vanishes at zero and one. `EmpiricalSchrodingerBridge` implements this
construction, and `sample_paths` samples one temporally correlated Brownian
bridge for every endpoint pair drawn from \(\pi^*\). It does not inject
independent frame noise.

### 7.3 Forward drift, current velocity, and Fokker--Planck equation

Under the usual absolute-continuity hypotheses, the Schrödinger system admits
potentials \(f,g\) with

\[
\frac{dP^*}{dR}=f(X_0)g(X_1).
\]

Writing \(g_t=P_{1-t}g\) for the backward heat-semigroup propagation, the
forward Föllmer drift is

\[
b_t^+(x)=2\tau\nabla\log g_t(x),
\]

and the marginal density solves

\[
\partial_t\rho_t
=-\nabla\!\cdot(b_t^+\rho_t)+\tau\Delta\rho_t.
\]

Define the backward drift and Nelson current/osmotic velocities by

\[
b_t^-=b_t^+-2\tau\nabla\log\rho_t,
\]

\[
v_t=\frac{b_t^++b_t^-}{2}
=b_t^+-\tau\nabla\log\rho_t,
\qquad
u_t=\frac{b_t^+-b_t^-}{2}
=\tau\nabla\log\rho_t.
\]

The Fokker--Planck equation is equivalently the continuity equation

\[
\partial_t\rho_t+\nabla\cdot(v_t\rho_t)=0.
\]

This separates advective class movement from diffusive spreading. In the
small-noise limit \(\tau\downarrow0\), entropic interpolations converge to
displacement interpolations under standard hypotheses; the current velocity
converges to the dynamic-OT velocity while the osmotic term vanishes.

### 7.4 Intrinsic spherical bridge is a different SDE

L2-normalized DINO features lie on \(\mathbb S^{D-1}\). Intrinsic spherical
Brownian motion with generator \(\tau\Delta_{\mathbb S}\) can be written in
Stratonovich form

\[
dZ_t=\sqrt{2\tau}\,(I-Z_tZ_t^\top)\circ dW_t,
\]

or in Itô form

\[
dZ_t=-\tau(D-1)Z_t\,dt
+\sqrt{2\tau}(I-Z_tZ_t^\top)dW_t.
\]

The bridge to endpoint \(y\) adds the Doob-transform drift

\[
2\tau\nabla_{\mathbb S}\log p^{\mathbb S}_{1-t}(Z_t,y),
\]

and the static Sinkhorn kernel must be the spherical heat kernel
\(p^{\mathbb S}_1(z_i,w_j)\). The currently implemented squared-angular kernel

\[
\exp\!\left(-d_{\mathbb S}(z_i,w_j)^2/(4\tau)\right)
\]

is the short-time heat-kernel/geodesic approximation, not the exact global
spherical heat kernel. Likewise, slerp is an exact deterministic geodesic but
not a sample from intrinsic spherical Brownian bridge. The ambient bridge is
therefore the exact SDE baseline; an intrinsic manifold bridge is a distinct
next implementation step.

## 8. Distribution transport versus feature editing

The three formulations answer different questions.

| Method | Learned or solved object | Endpoint guarantee | Per-source preservation | Required model |
|---|---|---|---|---|
| Coupling displacement / OT-CFM | A joint cat/dog coupling, optionally amortized as a marginal velocity field | Exact empirical marginals for the full coupling path; approximate after particle sampling | Only what is encoded in the ground cost | DINO features; optional new vector field |
| Residual feature matching | A correction \(R_t\) conditioned on the original source feature | No exact dog-marginal guarantee | Explicit source anchor; naturally expresses “change this cat” | DINO features plus a newly trained residual field |
| FlowEdit | Difference of two pretrained condition-conditional generative velocity fields | Maps between the model's source/target conditional laws, but does not solve empirical DINO OT | Designed for low-cost, source-faithful edits | A pretrained conditional rectified-flow generator |

So “less distributional matching and more feature editing” is a good
description of the operational shift, with one caveat: FlowEdit still derives
an ODE between source and target *distributions* and evaluates transport cost.
It simply does not estimate a Sinkhorn coupling between the observed DINO point
clouds.

### Source-anchored residual feature matching

“Residual flow matching” is used for several related residual-coordinate or
correction-flow constructions rather than one universally fixed objective. The
definition below is the one used by this repository: a flow on DINO residuals,
conditioned on the unedited source anchor.

For an OT-proposed pair \((z,w)\), define an ambient residual \(r_1=w-z\), or
on the sphere define the tangent residual

\[
r_1=\operatorname{Log}_{z}(w)\in T_z\mathbb S^{D-1}.
\]

Train a source-conditioned field in residual coordinates:

\[
r_t=t r_1,\qquad
\mathcal L_{\mathrm{res}}(\theta)
=\mathbb E\lVert f_\theta(r_t,t;z)-r_1\rVert^2.
\]

At inference, initialize \(r_0=0\), integrate \(\dot r=f_\theta(r,t;z)\), and
decode \(z_t=\operatorname{Exp}_{z}(r_t)\). The original source \(z\) is
available at every step, unlike marginal OT-CFM. OT is now a pairing prior for
training rather than the definition of the final population path. It is a
secondary comparator here because the stated goal requires exact class
marginals, not merely plausible source-conditioned edits.

### FlowEdit

FlowEdit assumes a pretrained conditional rectified-flow model with data at
\(t=0\) and Gaussian noise at \(t=1\). Bind the same model to source and target
conditions to obtain \(V^{src}\) and \(V^{tar}\). Given the clean source
\(Z^{src}_0\), draw

\[
\widehat Z^{src}_t=(1-t)Z^{src}_0+tN_t,
\]

construct the co-located target probe

\[
\widehat Z^{tar}_t=Z^{FE}_t+\widehat Z^{src}_t-Z^{src}_0,
\]

and integrate backward using the averaged delta velocity

\[
\frac{dZ^{FE}_t}{dt}
=\mathbb E\left[
V^{tar}(\widehat Z^{tar}_t,t)
-V^{src}(\widehat Z^{src}_t,t)
\mid Z^{src}_0
\right].
\]

This is inversion-free and training-free *given the pretrained conditional
flow*. `editing.flow_edit` implements the paper's Euler update and `n` by `T`
grid. A DINO encoder alone cannot run FlowEdit because it has no generative
velocity, noise prior, or decoder. The clean architecture is therefore:

\[
\text{generative latent FlowEdit}
\longrightarrow \text{decoded image}
\xrightarrow{\text{DINO}} \text{structure/semantic metric}.
\]

If no conditional generative flow exists, use the residual DINO-feature editor
and treat its output as a representation-level result, not an image edit.

## 9. Identifiability and the reciprocal content-aware bridge

Marginal matching alone does not identify an instance-level semantic map. If
(T_#\mu_c=\mu_d), then composition with any measure-preserving automorphism
of (\mu_d) has the same endpoint law. A dog classifier and a small terminal
Sinkhorn divergence therefore do not establish that source information was
preserved.

The reciprocal experiment first estimates the empirical species direction

\[
d=\frac{\mathbb E_{\mu_d}Z-\mathbb E_{\mu_c}Z}
        {\lVert\mathbb E_{\mu_d}Z-\mathbb E_{\mu_c}Z\rVert},
\qquad P_\perp=I-dd^\top.
\]

For global token (g(x)) and spatial patch tokens (p_k(x)), its OT ground
cost is

\[
c_{\rm content}(x,y)
=\alpha\lVert \widehat{P_\perp g(x)}-\widehat{P_\perp g(y)}\rVert^2
+\frac{\beta}{K}\sum_{k=1}^K
 \lVert\widehat{P_\perp p_k(x)}-\widehat{P_\perp p_k(y)}\rVert^2.
\]

Thus OT proposes correspondences using spatial content after removing the
first-order class mean direction. It is still a modeling assumption, but it is
strictly more informative than global class-feature distance.

One velocity field (v_\theta(z,t,q)) and one score field
(s_\psi(z,t,q)) are shared by directions (q\in\{-1,+1\}). Training draws
both orientations of the same content-aware endpoint coupling and the noisy
interpolant

\[
Z_t=(1-t)Z_0+tZ_1+\sqrt{2\tau t(1-t)}\,\epsilon.
\]

The base score-and-flow objective is augmented by

\[
\mathcal L
=\mathcal L_{\rm SF2M}+\lambda_{\rm content}\mathcal L_{\rm content}
+\lambda_{\rm reciprocal}\mathcal L_{\rm reciprocal}
+\lambda_{\rm terminal}\mathcal L_{\rm terminal}.
\]

For a forward sample and its reverse driven by reversed, negated Brownian
increments,

\[
\mathcal L_{\rm content}
=\mathbb E\lVert P_\perp(Z_1-Z_0)\rVert^2,
\qquad
\mathcal L_{\rm reciprocal}
=\mathbb E\lVert \overleftarrow T^{-W}
  (\overrightarrow T^W(Z_0))-Z_0\rVert^2.
\]

The species term is a one-sided monotonicity penalty

\[
\mathbb E\left[\max\{0,-q\langle v_\theta(Z_t,t,q),d\rangle\}^2\right].
\]

It does not reward arbitrarily large class displacement. An earlier softplus
margin continued pushing samples after they reached the target class and
produced classifier-confident but distributionally worse endpoints. The
terminal distribution loss and the class monotonicity constraint must remain
separate for exactly this reason.

### 9.1 Augmented inequality constraints

A larger fixed endpoint coefficient can overfit a finite set of sliced
projections and overwhelm content preservation. The augmented experiment uses
three stochastic inequality constraints instead:

\[
g_e(\theta)=D_{\rm SW}(\rho_1^\theta,\mu_1)-\delta_e\le 0,
\]

\[
g_c(\theta)=\mathbb E\lVert P_\perp(Z_1-Z_0)\rVert^2-\delta_c\le0,
\qquad
g_r(\theta)=\mathcal L_{\rm reciprocal}-\delta_r\le0.
\]

For constraint (g_k\), multiplier (\lambda_k\ge0\), and penalty
(\rho_k>0\), the Powell--Hestenes--Rockafellar term is

\[
\mathcal A_k(\theta,\lambda_k)
=\frac{
\max\{0,\lambda_k+\rho_k g_k(\theta)\}^2-\lambda_k^2
}{2\rho_k},
\]

with projected dual ascent

\[
\lambda_k\leftarrow
\max\{0,\lambda_k+\rho_k g_k(\theta)\}.
\]

Both directions enter every content and reciprocal constraint batch. This is
necessary: constraining only cat→dog produced a visibly asymmetric dog→cat
failure even though the endpoint objective itself was bidirectional.

Failure to approach the thresholds while multipliers continue growing is a
diagnostic result. It means either the requested constraint set is infeasible
for the current field/representation, or the stochastic sliced-Wasserstein
constraint is insufficiently aligned with the held-out Sinkhorn metric. It
should not be hidden by selecting a classifier-confident endpoint.

## 10. What to measure

At minimum, report:

1. Sinkhorn row and column marginal errors and solver convergence;
2. endpoint discrepancy, including a held-out Sinkhorn divergence;
3. feature norm along the path (exactly one for spherical paths);
4. kinetic action or path length;
5. classifier probability and nearest-neighbor class composition across time;
6. sensitivity to \(\varepsilon\), number of particles, sample split, DINO
   layer, and random seed.

Do not treat a 2D PCA animation as evidence that transport succeeded. PCA is a
lossy visualization fitted after the path is constructed.

## 11. Recommended progression

1. Start with pretrained or production-fine-tuned DINO features. A randomly
   initialized ViT with a DINO-shaped architecture is not a DINO semantic
   representation.
2. Fit the ambient empirical Schrödinger bridge first. It is the exact
   path-space entropy projection implemented today and gives a clear
   \(\mu_0\to\mu_1\) law for heterogeneous classes.
3. Treat \(\varepsilon\) as \(4\tau\), not only as a numerical blur parameter;
   sweep it against endpoint correspondence and intermediate dispersion.
4. Add an intrinsic spherical heat-kernel bridge only if remaining on the
   normalized DINO sphere is substantively important.
5. Train OT-CFM or simulation-free score/flow matching when the bridge must
   generalize out of sample without recomputing empirical couplings.
6. Keep residual editing as a per-source comparator; it does not replace the
   class-marginal bridge.
7. Use FlowEdit at the latent level only after adding a pretrained conditional
   rectified-flow generator; retain DINO for semantic/structure evaluation.

## Primary references

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [DINOv2 official implementation and model cards](https://github.com/facebookresearch/dinov2)
- [A computational fluid mechanics solution to the Monge--Kantorovich mass transfer problem](https://doi.org/10.1007/s002110050002)
- [A Convexity Principle for Interacting Gases](https://doi.org/10.1006/aima.1997.1634)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://openreview.net/forum?id=CD9Snc73AW)
- [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://proceedings.mlr.press/v238/tong24a.html)
- [A survey of the Schrödinger problem and some of its connections with optimal transport](https://arxiv.org/abs/1308.0215)
- [Interpolating between Optimal Transport and MMD using Sinkhorn Divergences](https://arxiv.org/abs/1810.08278)
- [FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](https://openaccess.thecvf.com/content/ICCV2025/html/Kulikov_FlowEdit_Inversion-Free_Text-Based_Editing_Using_Pre-Trained_Flow_Models_ICCV_2025_paper.html)
- [FlowEdit official implementation](https://github.com/fallenshock/FlowEdit)
