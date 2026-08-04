# Static OT Atlas: Research Plan and Decision Record

## Decision

The central hypothesis is deliberately narrow: a shared family of **static**
K=1 Monge maps can move a decoder-compatible continuous state between labelled
or discovered distributions without an inference-time ODE or target retrieval.
The decoder and representation are separate engineering problems.  We do not
call a result a transport success unless all three pieces work on held-out
examples:

1. a finite OT coupling is solved correctly on the training support;
2. the learned static map moves a held-out source toward the target
   distribution while preserving the source-specific residual;
3. the decoded endpoint is valid and semantically credible.

The current evidence says that (1) is easy, (2) is dataset-dependent, and (3)
is the present bottleneck.  The immediate plan is therefore to make the text
codec generalize before scaling the atlas.  Decoder-path dynamic OT is a
separate, CUDA-only phase, not a Metal experiment.

## Non-negotiable pure model

Let a continuous codec provide \(z=E(x)\in\mathbb R^d\) and \(\hat x=D(z)\).
An invertible chart separates a transportable factor from retained state:

\[
H(z)=(s,r),\qquad s\in\mathbb R^{d_s},\quad r\in\mathbb R^{d_r}.
\]

For each condition \(a\), an invertible canonicalizer \(P_a\) maps its style
factor to a shared canonical support.  Every ordered pair is then produced by
composition, not by a pair-specific network:

\[
T_{a\to b}(z)=H^{-1}\!\left(P_b^{-1}(P_a(s)),r\right).
\tag{1}
\]

Thus an atlas with \(K\) conditions contains \(K\) learned branches and yields
all \(K^2\) maps.  The diagonal is exactly the identity by construction.  For
any fixed residual, the implementation satisfies

\[
T_{b\to a}\circ T_{a\to b}=I,\qquad
T_{b\to c}\circ T_{a\to b}=T_{a\to c}.
\tag{2}
\]

These are architectural facts, not a claim that the learned map is the
population Brenier map.

### K=1 training supports

For equal-size style supports \(\mathcal S_a=\{s_i^a\}_{i=1}^n\) and
\(\mathcal S_b=\{s_j^b\}_{j=1}^n\), the empirical quadratic OT problem has a
permutation solution:

\[
\pi_{a\to b}\in\arg\min_{\pi\in S_n}
\frac1n\sum_{i=1}^n\|s_i^a-s^b_{\pi(i)}\|_2^2.
\tag{3}
\]

For \(K>2\), alternate exact assignments to a common finite barycenter and
barycentric updates.  This trains the \(K\) canonicalizer branches with
shared support.  It is coordinate descent for a finite multi-marginal
objective; it is not a claim of a global population barycenter optimum.

The held-out map consumes only \((z,a,b)\).  Held-out target data may be used
to calculate an evaluation OT cost, never to select the endpoint.

## What has actually worked and failed

| Setting | What worked | What failed / conclusion |
| --- | --- | --- |
| MNIST 9 \(\leftrightarrow\) 1 | Clean PCA clusters, exact static cycles, direct latent transport. | The best current proof-of-life for the pure atlas. |
| MNIST prototype atlas | 97.24% prototype accuracy, centroid ratio 6.09, cycle MSE \(3.64\times10^{-12}\). | Shows the shared-branch construction can produce a full digit matrix. |
| AFHQ cat \(\leftrightarrow\) dog | Chart-label accuracy 98.63%; coordinate cycle MSE \(8.35\times10^{-13}\). | Decoded DINO OT worsened by 0.0385: frozen latent decoder endpoints are off manifold. |
| PACS domain transport | Exact coordinate cycle MSE \(3.08\times10^{-13}\). | Held-out DINO OT worsened by 0.0303 and decoded images had artifacts. |
| SST-2 compact T5 slots | Native T5-state interface is valid; static inverse cycle MSE \(\approx10^{-13}\). | The 8-fold compact codec overfit: held-out slot CE 4.33; endpoint OT worsened in both directions. |

The SST-2 baseline is implemented in
[`experiments/text_sst2_t5_slot_monge_ot.py`](experiments/text_sst2_t5_slot_monge_ot.py).
Its held-out artifacts are in
[`outputs/text_sst2_t5_slot_monge_ot_1024`](outputs/text_sst2_t5_slot_monge_ot_1024).
This result is a useful failure: it separates a sound static map from an
insufficient decoder-compatible representation.

## Execution plan

### Stage 0 — freeze the baselines

**Hardware:** Metal MPS or CPU.  **Goal:** make the negative and positive
controls reproducible before new modeling.

- Preserve MNIST 9 \(\leftrightarrow\) 1 and the factorized digit atlas as the
  positive static-OT controls.
- Preserve AFHQ, PACS, and SST-2 as explicit decoder/representation negative
  controls.  Do not tune visual samples until their held-out metrics improve.
- Report, for every run: training-support assignment cost, held-out endpoint
  OT cost, target-factor probe, residual/content preservation, decode quality,
  and pure-map cycle/composition error.
- Never present coordinate-cycle precision as evidence of decoded transport.

**Exit criterion:** all artifacts, seeds, and metrics have an immutable
summary.  This criterion is already met for the runs above.

### Stage 1 — general-purpose continuous text codec

**Hardware:** Metal MPS.  **Data:** a fixed, versioned 50k--100k subset of
OpenWebText; SST-2 is held back for the downstream transport test.  **Goal:**
learn a continuous state that remains decodable off the reconstruction support.

Use the frozen T5 encoder/decoder boundary used by ELF, but do not use ELF's
flow-matching generator.  The codec is a learned slot compressor \(Q\) and
expander \(R\):

\[
x\xrightarrow{E_{\rm T5}}h\in\mathbb R^{L\times512}
\xrightarrow{Q}z\in\mathbb R^{K\times d_z}
\xrightarrow{R}\tilde h\xrightarrow{D_{\rm T5}}\hat x.
\tag{4}
\]

Start at \(K=32,d_z=128\): a four-fold reduction from a 32-by-512 T5 state,
rather than the failed eight-fold reduction.  Train the codec on OWT with

\[
\mathcal L_{\rm codec}=
\mathcal L_{\rm token}
+\lambda_h\|R(Q(h))-h\|_2^2
+\lambda_r\|Q(E_{\rm T5}(D_{\rm T5}(R(Q(h)))))-Q(h)\|_2^2.
\tag{5}
\]

The final term is decode--re-encode consistency.  It directly tests the part
of latent space that the atlas will traverse; it is not a token-simplex loss.
Use masked-context prediction with an EMA target slot encoder as the JEPA
auxiliary task:

\[
\mathcal L_{\rm JEPA}=
\|G_\phi(Q(h_{\rm context}),c,m)-
\operatorname{sg}[\bar Q(h_{\rm target})]\|_2^2.
\tag{6}
\]

I-JEPA motivates predicting latent targets from context rather than merely
reconstructing observations; V-JEPA 2 motivates keeping this predictive head
separate from the state representation.  Here it regularizes codec geometry,
not the deployment map.  [I-JEPA](https://arxiv.org/abs/2301.08243),
[V-JEPA 2](https://arxiv.org/abs/2506.09985)

**Codec gates, all measured on held-out OWT:**

1. codec teacher-forced token CE must be no more than 0.5 nats above the
   native-state decoder CE;
2. autoregressive reconstructions must have mean normalized token edit
   distance at most 0.20;
3. decode--re-encode error must be comparable for interpolated slots and
   native encoded slots; and
4. semantic retrieval of a decoded reconstruction must recover its source
   sentence above a fixed frozen-teacher baseline.

Failure means increase representation capacity or data, not add a text flow,
a target-aware retrieval path, or a corrector.

### Stage 2 — labelled static SST-2 atlas

**Hardware:** Metal MPS.  **Data:** frozen Stage-1 codec; 512 SST-2 examples
per sentiment for the K=1 support; 128 disjoint examples per sentiment for
evaluation.  **Goal:** isolate the atlas from codec failure.

- Allocate 4 of 32 slots to \(s\); retain 28 slots as \(r\).
- Train the orthogonal chart and conditional affine canonicalizers using
  Equation (3) and a two-support barycenter.
- Fit neither a direct pairwise endpoint regressor nor a continuous-time model.
- Decode only after calculating all latent-space metrics.

**Atlas gates:** both directions must reduce held-out semantic OT cost,
target-factor probe must exceed 85%, residual/content similarity must not fall
below the identity baseline, and composition/cycle must stay within numerical
precision.  Failure here means the factorization or empirical pairing is
wrong; it is not evidence for rectified flow.

### Stage 3 — SSL ontology discovery

**Hardware:** Metal MPS for prototypes and assignments.  **Data:** the same
OWT representation plus a stable frozen sentence teacher.  **Goal:** replace
human labels with discovered, repeatable directions while retaining an
explicit target choice.

Learn balanced prototype assignments on \(s\), retain content neighborhoods
on \(r\), and cluster only displacement candidates that recur across such
neighborhoods:

\[
\Delta_{ij}=s_j-s_i,
\quad (i,j)\text{ content-neighbor pair}.
\tag{7}
\]

The selected target is then an anchor sentence or discovered prototype,
not an implicit token-level operation.  Report occupancy, view agreement,
direction reproducibility across disjoint subsets, content preservation, and
post-hoc SST alignment.  Labels are evaluation only in this stage.

### Stage 4 — decoded dynamic-path regularization (CUDA only)

**Hardware:** CUDA GPU, preferably at least 24 GB VRAM.  This phase is not
viable as the default Metal experiment: each update needs several decoded
intermediates and a sparse linear solve at every time slice.

Feng and Strohmer regularize an autoencoder by applying Benamou--Brenier path
energy to decoded latent chords

\[
\rho_t=D((1-t)z_i+t z_j),\qquad
\mathcal L=\mathcal L_{\rm recon}+\alpha J(\rho_{0:T})+
\beta\mathcal L_{\rm mass}.
\tag{8}
\]

For non-negative image densities, \(J\) has a discrete closed form after a
KKT solve per time slice.  The appendix establishes two practically important
facts: the number of time bins should track physical pixel displacement, and
auxiliary training images markedly improve learned interpolation even when
only a small set of paths is regularized.  [Feng & Strohmer (2024), including
Appendix §§6--7.5](https://arxiv.org/pdf/2404.08900)

**Image implementation:** unfreeze a decoder-compatible image codec; sample a
small set of training-support OT pairs; decode 4--8 chord points; downsample
positive image densities to a 32-by-32 or 64-by-64 grid; and backpropagate the
path energy through a batched conjugate-gradient or sparse Cholesky solve.
This is decoder pretraining only.  The deployed class atlas remains Equation
(1), so its static inverse/composition theorem is unchanged.

**Text restriction:** do not apply pixel-mass Benamou--Brenier energy to token
probabilities, T5 coordinates, or an arbitrary PCA grid.  A genuine text
analogue requires a learned continuous semantic-cell graph with non-negative
slot occupancy and a declared graph ground metric.  Until that geometry is
defined and validated, use only the Stage-1 decode--re-encode path diagnostic
and call it *latent path feasibility*, not dynamic OT.

### Stage 5 — optional non-exact quality model

Only after a pure atlas passes Stage 2 may a ChordEdit-style one-shot residual
be evaluated:

\[
z_{\rm out}=T_{a\to b}(z)+R_\psi(z,T_{a\to b}(z),a,b).
\tag{9}
\]

It may improve rendering, but it is not part of the pure theorem: it need not
be invertible or compositional.  Its results must be shown in a separate table
from the static atlas.  [ChordEdit](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_ChordEdit_One-Step_Low-Energy_Transport_for_Image_Editing_CVPR_2026_paper.html)

## Claims and falsifiers

The defensible paper claim is not “OT replaces diffusion.”  It is: *a shared,
factorized family of static empirical Monge maps gives direct compositional
distribution transport in a decoder-compatible continuous representation, and
we measure precisely when this geometry survives decoding.*

The project fails on a setting if any of the following occurs:

- held-out endpoint OT does not beat identity;
- target-factor movement is explained by loss of residual content;
- the codec cannot decode off its training support;
- SSL prototypes collapse or directions are not reproducible; or
- a non-invertible corrector is needed but is reported as pure static OT.

## Literature anchors

- Xue Feng and Thomas Strohmer (2024), [Improving Autoencoder Image
  Interpolation via Dynamic Optimal Transport](https://arxiv.org/abs/2404.08900).
  The correct decoder-geometry reference; its dynamic path energy is a
  GPU-only image pretraining component, not the static endpoint method.
- Mahmoud Assran et al. (2023), [Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243).
- Mido Assran et al. (2025), [V-JEPA 2: Self-Supervised Video Models Enable
  Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985).
- Liangsi Lu et al. (2026), [ChordEdit: One-Step Low-Energy Transport for
  Image Editing](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_ChordEdit_One-Step_Low-Energy_Transport_for_Image_Editing_CVPR_2026_paper.html).
