# Stochastic bridge formulation

## 1. Path-space problem

Let `Omega = C([0,1], R^d)` with canonical process `X_t`, and let `R^epsilon`
be the law of

```text
dX_t = sqrt(epsilon) dW_t.
```

For endpoint laws `mu_0` and `mu_1`, the Schrödinger problem is

```text
P* = argmin KL(P || R^epsilon)
     subject to P_0 = mu_0 and P_1 = mu_1.
```

Under the usual finite-entropy assumptions this is a strictly convex problem.
Its endpoint law `pi* = P*_{01}` solves the static Schrödinger problem, and the
path law disintegrates as

```text
P*(domega) = integral R^epsilon_xy(domega) pi*(dx,dy).
```

This identity is the clean separation used in the code: choose an endpoint
coupling, then mix reference bridges conditional on those endpoints. It also
shows why independently random cat/dog pairs are not neutral: they impose the
product coupling, which throws away content correspondence.

As `epsilon -> 0`, the entropic interpolation approaches quadratic optimal
transport under standard conditions. For positive `epsilon`, it is a stochastic
optimal-control problem and retains conditional diversity.

## 2. Semantic endpoint coupling

Let `phi(x)` be a frozen self-supervised representation. A raw DINO distance
contains both content and the cat/dog domain direction. Estimate a domain
direction `q` and form a content coordinate

```text
c(x) = normalize((I - q q^T) phi(x)).
```

The empirical coupling solves

```text
min_pi  <pi, C> + tau KL(pi || a tensor b),
C_ij = d(c(x_i), c(y_j))^2,
pi 1 = a,  pi^T 1 = b.
```

This is a modeling assumption, not ground truth correspondence. It should be
ablated against the product coupling, raw-DINO OT, and learned coupling
refinement. A workshop claim needs evidence that the projected representation
retains identity/geometry while discarding only domain style.

## 3. Simulation-free stochastic interpolant

For `(X_0,X_1) ~ pi`, `Z ~ N(0,I)`, and
`gamma(t) = sqrt(epsilon t(1-t))`, define

```text
I_t = (1-t)X_0 + tX_1 + gamma(t)Z.
```

It has exact endpoints. For `t` away from zero and one,

```text
d_t I_t = X_1 - X_0
          + epsilon(1-2t) Z / (2 gamma(t)),

grad_x log p(I_t=x | X_0,X_1) = -Z / gamma(t).
```

Squared-loss regression gives conditional expectations:

```text
v_t(x) = E[d_t I_t | I_t=x],
s_t(x) = E[-Z/gamma(t) | I_t=x] = grad log rho_t(x).
```

No learned trajectory is simulated while forming these targets. In practice,
sample `t` in `[delta,1-delta]` or use a time-weighted score loss because the
conditional score variance diverges at the endpoints.

## 4. Itô and Nelson descriptions

The marginal law `rho_t` satisfies the continuity equation

```text
partial_t rho_t + div(rho_t v_t) = 0.
```

It also satisfies forward and backward Fokker–Planck equations. With diffusion
rate `alpha_t >= 0`, the corresponding drifts are

```text
b_t^+ = v_t + alpha_t s_t / 2,
b_t^- = v_t - alpha_t s_t / 2.
```

Thus `b_t^+ - b_t^- = alpha_t grad log rho_t`. The processes

```text
dX_t = b_t^+(X_t) dt + sqrt(alpha_t) dW_t,
dX_t = b_t^-(X_t) dt + sqrt(alpha_t) dWbar_t  (backward filtration)
```

and the probability-flow ODE `dX_t = v_t(X_t)dt` share the same one-time
marginals when the fields are exact. Their path laws differ. This distinction
matters for the demo: the ODE gives a stable single morph; the SDE represents
the fact that one source image can have several plausible target images.

For one reciprocal, direction-conditioned network, impose

```text
v_reverse(x,t) = -v_forward(x,1-t),
s_reverse(x,t) =  s_forward(x,1-t).
```

This is stronger and more interpretable than training unrelated cat→dog and
dog→cat maps and adding a pixel cycle loss.

## 5. Regularization

Two primary lambda terms are justified:

```text
L = L_flow + lambda_score L_score
    + lambda_TR L_time-reversal
    + lambda_content L_content.
```

`L_time-reversal` enforces the reciprocal identities above.
`L_content` penalizes displacement in protected DINO content coordinates along
the path, not just at the endpoint. An optional tube penalty measures distance
to a local k-NN estimate of the union of endpoint manifolds; unlike hard pixel
clipping it is representation-aware and differentiable.

IFM/path correction belongs outside this first solve: train a bridge, estimate
where its generated path leaves the data-supported tube, update the endpoint or
path coupling, and refit. It is a refinement algorithm, not the definition of
the reciprocal transport problem.

## 6. Image realization

For MPS-scale training, use a frozen diffusion VAE `E,D`:

```text
Z_0 = E(X_0),  Z_1 = E(X_1),
learn the bridge in latent tensors,
X_t = D(Z_t).
```

DINO should choose/regularize semantic correspondences; it should not be the
only decoded state. A compact latent U-Net can preserve spatial information
that a global DINO vector loses. Decode all displayed frames independently from
the same latent path and use sufficiently small numerical steps. This avoids
the layerwise GAN color artifacts observed in the retired experiments.

## Foundations

- Christian Léonard, [A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport](https://leonard.perso.math.cnrs.fr/papers/2014-Leonard-A%20survey%20of%20the%20Schroedinger%20problem%20and%20some%20of%20its%20connections%20with%20optimal%20transport.pdf).
- Michael Albergo, Nicholas Boffi, and Eric Vanden-Eijnden, [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797).
- Alexander Tong et al., [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://proceedings.mlr.press/v238/tong24a.html).
