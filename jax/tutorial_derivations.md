<!-- gaussian -->
### Why a gradient becomes a moment message

The Gaussian is an exponential family with sufficient statistics
$T(u)=(u,uu^\top)$, natural parameters
$\eta=(C^{-1}m,-\tfrac12 C^{-1})$, and expectation parameters
$\mu=E[T]=(m,M)$. Its log partition function satisfies
$\nabla_\eta A=\mu$ and $\nabla_\eta^2 A=\operatorname{Cov}_q(T)=F$.
Consequently the chain rule gives $\nabla_\eta\Phi=F\nabla_\mu\Phi$.
This is a Euclidean derivative in natural coordinates, equivalently the
natural-gradient direction in expectation coordinates. It is not an ordinary
derivative with respect to the mean alone.

To compute it from familiar derivatives, write
$dC=dM-dm\,m^\top-m\,dm^\top$. Substitution into
$d\Phi=g_m^\top dm+\operatorname{tr}(G_C^\top dC)$ gives the
expectation-coordinate covector $(g_m-2G_Cm,G_C)$. For centered Gaussian
$\epsilon=u-m$, odd centered moments vanish and
$E[\epsilon_i\epsilon_j\epsilon_k\epsilon_l]
=C_{ij}C_{kl}+C_{ik}C_{jl}+C_{il}C_{jk}$.
Multiplication by $F=\operatorname{Cov}(T)$ therefore gives

$$d_m=Cg_m,\qquad d_M=md_m^\top+d_mm^\top+2CG_CC.$$

The inverse is $g_m=C^{-1}d_m$ and
$G_C=\tfrac12C^{-1}(d_M-md_m^\top-d_mm^\top)C^{-1}$.
The implementation uses solves, not matrix inverses. The full formula includes
off-diagonal covariances; its restriction to independent coordinates gives
the elementwise diagonal formulas used by the layers.

An input image or token index is observed, not a Gaussian latent variable.
Its zero variance does not admit this inverse. The code carries its ordinary
adjoint without assigning artificial input uncertainty.

### The common reverse operation

For a cached moment map $\mu_y=F_\ell(\mu_x,\mu_w)$:

1. Decode the child signal into ordinary derivatives of its moments.
2. Apply the local vector–Jacobian product of the declared forward map.
3. Sum derivatives arriving at a shared parent from **all** children.
4. Encode the parent derivative using that parent's cached Gaussian belief.
5. Update persistent beliefs only after the reverse sweep is complete.

`literal=True` supplies the handwritten local derivative; `literal=False`
uses `jax.vjp`. Both implement this same contract. A global `jax.grad`
of the scalar objective is an independent coordinate check, not the explicit
message engine. No custom autograd rule is needed to disguise one coordinate
system as another.

<!-- ep -->
### EP: a finite tilted-moment identity

Let $q_\eta(u)=\exp\{\eta^\top T(u)-A(\eta)\}$ be a cavity and let
$f(u)$ be fixed while differentiating. Define
$Z(\eta)=E_{q_\eta}f(u)$ and $\widetilde p(u)=q_\eta(u)f(u)/Z$.
Differentiating under the integral, with the usual integrability condition,

$$\nabla_\eta\log Z
=\frac{E_q[f(T-E_qT)]}{E_q f}
=E_{\widetilde p}T-E_qT
=F\nabla_\mu\log Z.$$

Thus the encoded signal is **exactly a finite correction** to sufficient
statistics when the normalizer and its derivatives are exact. At unit step
$m^+=m+d_m$, $M^+=M+d_M$. Reconstructing the covariance gives
$C^+=C+2CG_CC-d_md_m^\top$. The last term follows from expanding
$(m+d_m)(m+d_m)^\top$; dropping it changes the tilted covariance.

For multiple factors, the cavity is obtained by removing the old site:
$\eta_c=\eta_q-\eta_n$. After moment projection of the tilt, replace the
site by $\eta_n^{\rm target}=\eta_{\rm proj}-\eta_c$ and damp in canonical
coordinates if needed. Updating a site without removing its old contribution
counts the same data again. Damping a site is a different operation from
interpolating raw moments. An approximate nonlinear normalizer gives an
**EP-style projected update**, not a proof of exact nonlinear inference.

### MFVI: a different objective and a different finite update

For the likelihood term $\Psi=E_q[\log p(D\mid W)]$, the ELBO is
$\mathcal L=\Psi-\mathrm{KL}(q\|p_0)$.
Since $\mathrm{KL}=(\eta-\eta_0)^\top\mu-A(\eta)+A(\eta_0)$,
its expectation-coordinate derivative is $\eta-\eta_0$; the other terms
cancel because $dA=\mu^\top d\eta$. A canonical natural-gradient step is

$$\eta^+=(1-\rho)\eta+\rho(\eta_0+\nabla_\mu\Psi).$$

For diagonal Gaussians $\eta=(h,-\tau/2)$ with $\tau=1/v$ and $h=\tau m$,
$\nabla_\mu\Psi=(g_m-2m\odot g_v,g_v)$ gives the displayed update for
$\tau$ and $h$. This is why adding the MFVI raw-moment direction with unit
step is generally wrong.

For a Gaussian target with precision $\Lambda$ and information vector $h_*$,
$g_{v_i}^{\rm like}=-(\Lambda_{ii}-\tau_{0i})/2$.
A single-coordinate canonical step therefore yields
$v_i^+=1/\Lambda_{ii}$ and
$m_i^+=(h_{*i}-\sum_{j\ne i}\Lambda_{ij}m_j)/\Lambda_{ii}$:
exact Gaussian coordinate-ascent VI. At convergence $m=\Lambda^{-1}h_*$,
but $v_i=1/\Lambda_{ii}$ generally differs from $(\Lambda^{-1})_{ii}$.
Parallel unit updates need damping; coordinate-wise ascent does not prove
parallel ascent.

For example, a target covariance with unit diagonal and correlation $0.8$
has diagonal-MFVI variance $1-0.8^2=0.36$. A single global EP moment projection
retains variance $1$. Both can have the same mean. Neither discrepancy is a
failure of the Gaussian coordinate conversion.

For MFVI the activation signals are likelihood natural-gradient directions,
not newly conditioned activation posteriors. For EP they are finite tilted
moment corrections only when the declared normalizer is exact. The common
signal representation does not erase that difference in interpretation.

<!-- covariance -->
### Shared factors on a directed acyclic computation graph

Write a boundary as $h\approx m+U\epsilon/\sqrt R$, $\epsilon\sim N(0,I_R)$.
The same columns represent the same latent perturbations at every use of that
boundary. The covariance is $UU^\top/R$; it is never necessary to allocate
this full matrix for a convolutional feature map or a token sequence.
For a local map $y=f(h,w)$, first-order propagation gives
$U_y=J_hU_h+J_wU_w$ and $m_y=f(m_h,m_w)$.
The JAX carrier computes these actions locally, not an enormous global Jacobian. In code, factor columns are stored on the leading axis; transpose them to match the column-oriented equations here.

**Residual proposition.** For $y=x+G(x)$, the represented covariance is
$(U_x+U_G)(U_x+U_G)^\top/R$. Expanding gives the two marginal covariances
and both cross terms. If $G(x)=x$, it is $4C_x$; replacing the branches by
independent copies gives $2C_x$. For DenseNet concatenation, stack the factors
to obtain a block covariance with off-diagonal block $U_xU_G^\top/R$.
Both statements follow directly by multiplying the stacked/summed factors. ∎

This is a computation DAG with repeated use, not a claim that all neural
posterior factors form an exact junction tree. Reverse traversal sums
sensitivities; cavity removal concerns likelihood sites. These are related
bookkeeping issues, but not interchangeable operations.

### A diagonal weight posterior can induce strongly correlated activations

For fixed $x$, $y=W^\top x+b$ with independent weight columns has diagonal
output covariance. But for uncertain shared input $X$, independent of the
current weight block, total covariance yields
\[
E[y]=M_W^\top m_X+m_b,\qquad
\operatorname{Cov}(y)=M_W^\top C_XM_W+
\operatorname{Diag}\{V_W^\top(\operatorname{diag}C_X+m_X^2)+v_b\}.
\]
**Proof.** Conditional on $X$, the mean is $M_W^\top X+m_b$ and covariance
is diagonal with entries $\sum_iV_{ij}X_i^2+v_{b_j}$. Average the latter and
add the covariance of the conditional mean. ∎

After one mixing layer, $M_W^\top C_XM_W$ is generally dense. In convolution,
the same filter is reused at every position. Independent weight draws per
pixel would be a different model. The global column identifiers in the
implementation preserve reuse across pixels and examples.

### Nonlinear closure: Gaussian ReLU means and the omitted Hermite residual

For $X\sim N(m,s^2)$, put $a=m/s$, $P=\Phi(a)$ and $p=\phi(a)$.
Integrating over $X>0$ gives $E[\max(X,0)]=mP+sp$ and
$E[\max(X,0)^2]=(m^2+s^2)P+msp$. These follow by substituting
$X=m+sZ$, using $\int_{-a}^\infty z\phi(z)dz=\phi(a)$ and
$\int_{-a}^\infty z^2\phi(z)dz=\Phi(a)-a\phi(a)$.
At $s=0$, use the deterministic value—no invented input noise.

Gaussian integration by parts gives $\operatorname{Cov}(X,\operatorname{ReLU}X)
=s^2P$. Hence the first linear/Hermite projection has slope $P$.
For jointly Gaussian inputs its retained covariance is
$\operatorname{Diag}(P)C\operatorname{Diag}(P)$.
The remaining diagonal variance is nonnegative by orthogonal projection:
$v_{\mathrm{ReLU}}-P^2s^2\ge0$. A diagonal-plus-factor closure may retain
this residual separately. The minimal shared-factor implementation here
retains the first component and **does not claim exact ReLU variance**.
Its omission is an explicit ablation against diagonal marginal matching and
complete-network sampling.

### What a finite covariance sketch guarantees

For diagonal weight variance $V$, choose fixed sign columns $e_r$ and
$U_r=J V^{1/2}e_r$. For independent Rademacher signs,
$E[U_rU_r^\top]=JVJ^\top$. For a scalar projection put
$b=V^{1/2}J^\top a$. Expanding fourth moments gives
$\operatorname{Var}((b^\top e)^2)=2(\|b\|^4-\sum_i b_i^4)$,
so the relative root-mean-square error of the averaged directional variance
is at most $\sqrt{2/R}$. The fourth-moment expansion retains only indices
paired an even number of times; its coefficients are 1 for $b_i^4$ and 6 for
$b_i^2b_j^2$. Subtracting $\|b\|^4$ proves the formula.

These are fixed-belief guarantees. Fitting a model to a fixed sketch can adapt
to its directions, so an independent sketch and sampled posterior prediction
are required after fitting. A low-rank covariance sketch of diagonal weights
is also **not** a learned low-rank weight posterior.

Here is a concrete reason for that qualification. Take weight covariance
$I_P$ and one fixed sign vector $e\in\{-1,+1\}^P$. If a fitted scalar map
has sensitivity $a=e/\sqrt P$, its actual variance under fresh Gaussian
weight perturbations is $a^\top I_Pa=1$, whereas the one-direction estimate
is $(a^\top e)^2=P$. Both equalities follow from $e^\top e=P$.
The usual unbiasedness argument holds when $a$ is fixed independently of
the random probe; it cannot be reapplied after choosing $a$ from that probe.
This linear counterexample explains why retaining covariance and satisfying
the local derivative identities do not alone guarantee trained-belief fidelity.
It is a possible mechanism to investigate, not proof that it caused a particular
experimental discrepancy. Independent-direction and shared-network checks distinguish that question from ordinary sampling variability; the small fixture below is not a learned-network diagnosis.

### Why a small number of directions can give a noisy covariance

For a fixed linearization, let $z_r\sim N(0,C)$ independently and
$\widehat C=R^{-1}\sum_r z_rz_r^\top$, with the mean known to be zero.
Then
\[
E\|\widehat C-C\|_F^2
=\frac{(\operatorname{tr}C)^2+\|C\|_F^2}{R}.
\]
**Proof.** Independence makes the variance of the average $1/R$ times that of
one outer product. For entries $i,j$, the Gaussian fourth-moment identity is
$E[z_i^2z_j^2]=C_{ii}C_{jj}+2C_{ij}^2$. Subtract $C_{ij}^2$ to obtain
$\operatorname{Var}(z_iz_j)=C_{ii}C_{jj}+C_{ij}^2$; sum over $i,j$.
For sign sketches $z=Ae$, the fourth moment has the same Gaussian terms minus
$2\sum_k A_{ik}^2A_{jk}^2$, so the displayed expression is an upper bound. ∎

Thus relative root-mean-square covariance error depends on
$\sqrt{[1+(\operatorname{tr}C)^2/\|C\|_F^2]/R}$, not just parameter count.
For $d$ equal-variance independent outputs this is $\sqrt{(d+1)/R}$.
A sketch can preserve shared-variable semantics and still have substantial
finite-direction error. Keeping a diagonal residual, shrinking noisy
off-diagonal entries, or allocating more directions are distinct remedies;
none removes nonlinear closure bias. This explains why the plots compare
rank and sampling uncertainty rather than using “correlated” as a certificate
of accurate covariance.

### Structured weight beliefs: $D+LL^\top$

This is a mathematical extension. The reported reproduction configurations
use diagonal weight beliefs; their correlated **activation** factors do not
change that weight family. A learned structured weight posterior would need
its own objective, updates and full-training evaluation.

A correlated generator is $w=m+D^{1/2}\epsilon+Lz$, with independent standard
Gaussians $\epsilon,z$. Its covariance follows by expanding the outer product
and observing that the two cross expectations vanish. Store $P$ diagonal
entries and $Pr$ factor entries, not a $P\times P$ matrix.
The matrix determinant lemma gives
$\log|D+LL^\top|=\log|D|+\log|I+L^\top D^{-1}L|$;
Woodbury gives its inverse using an $r\times r$ solve. For an isotropic prior
$N(m_0,\sigma_0^2I)$ the Gaussian KL uses
$\mathrm{tr}(D)+\|L\|_F^2$, $\|m-m_0\|^2$, and that determinant.
These identities make structured VI feasible without a dense weight covariance.

Explicitly, with $K=I+L^\top D^{-1}L$,
$C^{-1}=D^{-1}-D^{-1}LK^{-1}L^\top D^{-1}$; multiplying by $D+LL^\top$
verifies the inverse. The determinant identity follows by evaluating the
same block determinant using either Schur complement. Hence, for
$p_0=N(m_0,\sigma_0^2I_P)$,
$\mathrm{KL}(q\Vert p_0)=\tfrac12[(\operatorname{tr}D+\|L\|_F^2+
\|m-m_0\|^2)/\sigma_0^2-P+P\log\sigma_0^2-\log|C|]$.
This follows by taking the expectation of the Gaussian log-density ratio.
It assumes strictly positive diagonal $D$; a singular low-rank-only law
requires separate support analysis.

One column of $L$ can jointly change LoRA factors in several attention blocks.
Block-diagonal $L$ forbids those cross-block correlations; a global $L$ retains
them. If upstream and current-layer weights share $z$, then the current weight
and its incoming activation are correlated. Reusing an independent-layer
moment formula would drop this dependence. Common latent columns handle its
first-order contribution; second-order products need additional moments.

### Error accounting rather than automatic equality claims

Suppose $\|D^2 f\|\le M$ on the relevant neighborhood and
$\delta\sim N(0,C)$. Taylor's remainder satisfies
$\|r(\delta)\|\le M\|\delta\|^2/2$, hence
$\|E f(m+\delta)-f(m)\|\le M\mathrm{tr}(C)/2$.
Let $A=J\delta$. Since
$\operatorname{Cov}(A+r)-\operatorname{Cov}(A)$ is two cross terms plus
$\operatorname{Cov}(r)$, Cauchy–Schwarz bounds its norm by
$2\sqrt{E\|A\|^2 E\|r\|^2}+E\|r\|^2$, with
$E\|r\|^2\le M^2[(\mathrm{tr}C)^2+2\mathrm{tr}(C^2)]/4$.
The Gaussian fourth-moment identity gives the last expression. A local bound
needs a tail term if the Hessian bound holds only locally. ReLU requires
piecewise/crossing analysis rather than pretending it has a bounded classical
Hessian everywhere. Gradient-approximation bounds require bounds on derivatives
of the *loss gradient*, not just good forward moment estimates.

Thus Monte Carlo converges to the learned belief's actual neural distribution;
it need not converge to a fixed Gaussian closure. The plots should reveal the
approximation gap, its depth/rank dependence and its sampling uncertainty.

<!-- training -->
### What the optimizer adds—and what it does not

The mathematical EP/MFVI messages above are independent of Adam or Vadam.
For large-scale experiments we must additionally specify a finite update.
Vadam retains moving averages of the likelihood gradient and its squared
value, uses the second statistic to normalize mean steps, and ties posterior
precision to its optimizer state. $E[g^2]$ is not $(E[g])^2$ and is not the
VI derivative with respect to variance. The moment implementation approximates
it with a linearized gradient: if $g(W)\simeq g(m)+H(W-m)$, then
$E[g(W)^{\odot2}]\simeq g(m)^{\odot2}+\operatorname{diag}(HC H^\top)$.
A shared random direction estimates the last diagonal; the optimizer's time
average remains a separate operation. This is an optimizer approximation,
not the exact Gaussian message-passing theorem.

For MFVI, an unbiased minibatch likelihood **sum** is
$(N/B)\sum_{n\in\mathcal B}E_q\log p(y_n\mid x_n,W)$; the KL is counted
once. An equivalent average objective divides both likelihood and KL by $N$.
Temperature changes that objective and must be reported.

### Tied power EP and the external field

The scalable EP-style training path has a tied Gaussian cavity, rather than
storing one full weight-space site per example. Put $k=\alpha/N$ and
$q_c=q^{1-k}p_0^k/A$. The declared energy is

$$\mathcal F=-\frac{\tau}{\alpha}
\left[\frac1N\sum_n\log E_{q_c}
p(y_n\mid x_n,W)^{\alpha/\tau}+\log A\right].$$

For one diagonal coordinate, let $r=v/v_0$ and $d=1+k(r-1)$.
Completing the Gaussian square gives
$v_c=v/d$, $m_c=m_0+(1-k)(m-m_0)/d$, and
$-\log A=\tfrac12[\log d-k\log r+k(1-k)(m-m_0)^2/(v_0d)]$.
Sum this last expression over coordinates. The external field is part of
the scalar objective every minibatch; it is not diluted by the number of
epochs. Differentiating must include the cavity and field dependence on $q$.

Expanding $\log E_q\exp[(\alpha/\tau)\log p]$ to first order in $\alpha$
and using $\log A=-(\alpha/N)\mathrm{KL}(q\|p_0)+O(\alpha^2)$ gives
$\mathcal F\to-\frac1N\sum_n E_q\log p_n
+\frac\tau N\mathrm{KL}(q\|p_0)$.
This connects the tied energy to VI in its small-power limit. Adam on the
mean and log variance optimizes this energy; it is **not** a literal EP site
replacement. The exact site algorithm and this scalable surrogate are named
separately throughout.

<!-- metrics -->
### Metric definitions and uncertainty decomposition

For posterior draws $w_s$, use $\bar p(y\mid x)=S^{-1}\sum_s p(y\mid x,w_s)$
and predictive NLL $-N^{-1}\sum_n\log\bar p(y_n\mid x_n)$.
The expectation of sampled cross-entropy is a training quantity, not this
predictive NLL. The moment method instead integrates its declared logit
Gaussian; it is a one-moment-pass prediction, not 100 sampled networks charged
to its serving time. Cubature at the small classification head still has cost.

The Brier score is $N^{-1}\sum_n\sum_c(p_{nc}-1\{y_n=c\})^2$.
For bin $B_b$, ECE is $\sum_b |B_b|\,|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)|/N$.
The reliability plot includes counts because a low-occupancy bin is noisy.
ECE depends on binning; it is not a proper scoring rule or a replacement for NLL.

Sort confidence descending; if $e_{(i)}$ is the corresponding 0–1 error,
the selective risk at coverage $k/N$ is $r_k=k^{-1}\sum_{i\le k}e_{(i)}$.
The step-integral AURC is $N^{-1}\sum_k r_k$. Do **not** divide by $N$ again.
Excess AURC subtracts the same error count's oracle ranking, separating ranking
quality from the base error rate. OOD scores use entropy or negative maximum
probability; declare **OOD as the positive class**, report prevalence, AUROC,
AUPRC and FPR at 95% OOD recall. CIFAR-10-C measures corruption robustness;
its severity curves are not automatically an OOD-detection benchmark.

Predictive entropy $H(\bar p)$ combines uncertainties. With actual network
samples, $H(\bar p)-S^{-1}\sum_sH(p_s)$ estimates mutual information with the
sampled weight variable. A moment-head cubature analogue is a quantity for
the represented logit law, not a proof of equality to the full weight mutual
information. Keep those labels distinct.
### Ambiguity is not the same as unfamiliarity

DirtyMNIST combines clean digits with ambiguous images bearing multiple labels.
For one image let $r_c$ be its empirical label frequencies. Any fixed class
prediction gets at most $\max_c r_c$ of those labels right. Its average log
loss is $-\sum_c r_c\log p_c=H(r)+\mathrm{KL}(r\Vert p)\ge H(r)$.
These are finite-label diagnostics, not population Bayes limits. Counting
each label pair is appropriate for classification loss; OOD detection should
count unique images, not duplicate an image for each of its labels.
The native archive here evaluates Fashion-MNIST OOD; it does not contain a
native DirtyMNIST run. Mutual information needs per-network
probabilities, not only their saved average; it is not inferred from the
averaged-probability archives shown below.

<!-- transformer -->
### Conditional Gaussian carriers and probability averaging

The BLoB-inspired construction learns **one diagonal Gaussian adapter
distribution**, not a mixture or an ensemble: $A=M_A+\rho^{\odot2}\odot\epsilon$
with $\epsilon\sim N(0,I)$; $B$ is a point parameter and the pretrained backbone
is frozen. Thus $\operatorname{Var}(A)=\rho^{\odot4}$. The objective is expected
log likelihood plus the source Gaussian KL, with prior standard deviation 0.2
and the source cyclic KL schedule. Likelihood AdamW and the separate
prior/variance SGD step follow that source update. This is **mean-field VI
with a conditional-moment approximation**, not EP or a canonical unit MFVI step.

**Why activation correlations survive diagonal weights.** For one adapter,
write $z=(W+sBA)x$ and $\overline W=W+sBM_A$. With independent $A$ and $x$,
mean $m_x$ and covariance $C_x$, conditional expectation gives
$E[z]=\overline Wm_x$. The law of total covariance gives the exact affine
identity $\operatorname{Cov}(z)=\overline W C_x\overline W^\top
+s^2B\operatorname{Diag}\{V_A(\operatorname{diag}C_x+m_x^{\odot2})\}B^\top$,
where $V_A=\operatorname{Var}(A)$. To derive the second term, condition on $x$:
independent rows of $A$ have conditional output variance
$\sum_j(V_A)_{ij}x_j^2$; then average $x_j^2$.
Multiplication by $B$ induces correlated outputs even though $V_A$ is diagonal.
For different inputs, the same random adapter is reused; their injected
cross-covariance contains $\sum_j(V_A)_{ij}E[x_jx'_j]$, not separate weight draws.

The training approximation retains these dependencies through shared local
directions. For paired nonlinear block outputs $Y_k^\pm$, define their mean
$m=(2K)^{-1}\sum_k(Y_k^++Y_k^-)$, odd deviations
$O_k=(Y_k^+-Y_k^-)/2$ and even deviations
$E_k=(Y_k^++Y_k^-)/2-m$. A moment carrier
$\widetilde Y=m+K^{-1/2}\sum_k(\epsilon_kO_k+\zeta_kE_k)$,
with independent coefficient vectors uniformly distributed on the sphere of
radius $\sqrt K$, has conditional mean $m$
and covariance $K^{-1}\sum_k(O_kO_k^\top+E_kE_k^\top)$.
Indeed, $\epsilon=\sqrt K g/\|g\|$ for $g\sim N(0,I_K)$ satisfies
$E\epsilon=0$ and $E[\epsilon\epsilon^\top]=I_K$ by rotational symmetry and
its fixed squared norm $K$; the independently generated $\zeta$ does too.
Expanding the paired centered outer products proves that this equals the
paired empirical covariance. Two antithetic pairs per block were used in the
training fit. This spherical carrier is stochastic and need not be Gaussian
conditional on its integration nodes; the finite-node nonlinear calculation
remains an approximation. The posterior weights are still a single Gaussian.
The Gaussian coordinate
bridge remains the same; this extension changes the moment closure and update.

The source training code also uses Flipout signs to reduce gradient variance.
Here the signs vary across **tokens**, not only across examples. They preserve
the perturbation marginal of one affine token row, but do not generally preserve
the joint perturbation of a complete sequence. For a scalar shared readout
$z_t=x_t^\top w$, independent Gaussian weight variances give
$\operatorname{Cov}(z_t,z_s)=\sum_i x_{ti}x_{si}v_i$. Independent zero-mean
output signs at distinct tokens instead make this cross-token covariance zero:
the cross product contains the factor $E[s_ts_s]=0$. Attention can depend on
these correlations. Thus increasing the number of source-Flipout predictions
need not reproduce the global shared-weight predictor. The serving table labels
these two probability averages separately. Cross-input covariance diagnostics
likewise use actual shared sampled weights.

The original Gaussian-A+B path offers a separate deterministic predictor:
for SST-2 it integrates $E[\sigma(D)]$ over a Gaussian logit contrast;
for multiclass CLINC it averages softmax probabilities at symmetric output
nodes. The conditional Gaussian-A/point-B experiments use the source-coupled
stochastic carrier instead. Their serving comparison records how many carrier
passes or complete-network draws are averaged. These are different
approximations; neither applies softmax only to averaged logits.


**Training and serving carriers are different.** The spherical law above is
stochastic and bounded conditional on its nodes. A Gaussian carrier with the
same columns has the same first two moments but a different higher-moment law.
`blob.coupled_nodes` couples these nodes to the source sampler's random stream.
If $a$ is a unit vector, $\xi_*$ and $Z_i$ are independent
standard Gaussians, set $\Xi_i=a_i\xi_*+Z_i-a_i\sum_j a_jZ_j$.
Conditional on $a$, the joint covariance is $aa^\top+(I-aa^\top)^2=I$ and
$\sum_i a_i\Xi_i=\xi_*$. Thus the nodes have the intended independent Gaussian
law while preserving the source draw under recombination. Since this law does
not depend on $a$, the nodes are independent of $a$ under the ideal random
construction. This coupling and its stop-gradient/randomness choices matter
when comparing updates, not only predicted moments.

<!-- gaussian_smoothing -->
### Why smoothing and backward moment updates coincide

The following proof applies to any frozen affine layer with independent parent
activation and weight blocks. The plotted fixture freezes the local Jacobians
of a 64 → 12 → 8 → 10 tanh MLP at one checkpoint and uses fixed synthetic
inputs with Gaussian observations. The test suite also checks rectangular
synthetic affine chains. Neither is a nonlinear classification-posterior oracle.

**Forward pass.** For $h_{\ell-1}\sim\mathcal N(m,P)$ and independent
$w_\ell\sim\mathcal N(a,V)$, compute $\mu=c+Am+Ba$ and
$S=APA^\top+BVB^\top$. Retain full cross-example activation covariance.

**Literal backward messages.** If the child has conditioned moments
$(\mu^+,S^+)$, write $\delta=\mu^+-\mu$ and $\Delta=S^+-S$.
Gaussian conditioning gives the two gains $K_h=PA^\top S^{-1}$ and
$K_w=VB^\top S^{-1}$. The parent activation has
$m^+=m+K_h\delta$, $P^+=P+K_h\Delta K_h^\top$; the weight block has
$a^+=a+K_w\delta$, $V^+=V+K_w\Delta K_w^\top$.
Apply these equations from the observed output back through all three layers.

**Why the pseudo-gradient route gives the same weight message.** Decode the
child raw-moment change to $g_\mu=S^{-1}\delta$ and
$G_S=\tfrac12S^{-1}(\Delta+\delta\delta^\top)S^{-1}$.
The local VJP gives $g_a=B^\top g_\mu$ and $G_V=B^\top G_SB$.
Encoding gives $d_a=Vg_a=K_w\delta$ and
$d_{M_w}=ad_a^\top+d_aa^\top+K_w(\Delta+\delta\delta^\top)K_w^\top$.
Installing the raw moments subtracts $d_ad_a^\top$ when recovering covariance;
this cancels the $K_w\delta\delta^\top K_w^\top$ term and leaves exactly
$V^+=V+K_w\Delta K_w^\top$. Replacing $(B,V,a)$ by $(A,P,m)$ proves the
activation identity. Induction from the exact Gaussian observation head proves
agreement at every layer.


These are exact marginal posterior updates. Keeping only those marginals does
not store the complete cross-layer posterior covariance. A deterministic
observed input stays clamped; it does not require a Fisher inverse.

<!-- literal_dense -->
### The handwritten dense-layer reverse calculation

For row-batched inputs use $M=mA+m_b$ and
$V=vA^{\odot2}+(m^{\odot2}+v)S+v_b$, with $A$ shaped input by output.
For child Euclidean adjoints $(a,b)$, direct differentiation gives
$g_m=aA^\top+2m\odot(bS^\top)$,
$g_v=b(A^{\odot2}+S)^\top$,
$g_A=m^\top a+2A\odot(v^\top b)$ and
$g_S=(m^{\odot2}+v)^\top b$.
Bias adjoints sum $a,b$ over the batch. The JAX module uses the displayed [input, output] layout, unlike PyTorch which stores the transpose
of $A,S$, following `nn.Linear`; the equations are identical after transposing
those parameter adjoints. Encoding these pairs gives the literal messages.
EP and MFVI differ in the incoming head signal, not in this layer derivative.

<!-- gaussian_heads -->
### The Gaussian heads already differ before propagation

For Gaussian observation noise $r>0$ and residual $e=y-m$, EP uses
$\Phi_{\rm EP}=\log\mathcal N(y;m,r+v)$. Differentiating gives
$g_m=e/(r+v)$ and $g_v=\tfrac12[e^2/(r+v)^2-1/(r+v)]$.
MFVI instead uses
$\Psi=-\tfrac12[\log(2\pi r)+(e^2+v)/r]$, whose derivatives are
$g_m=e/r$ and $g_v=-1/(2r)$. Sum over observations when appropriate.
The former is a normalizer; the latter is an expected log likelihood.
A likelihood head at the output is not an activation entropy penalty.

<!-- residual_integration -->
### Sampling-assisted moment propagation: integrate the nonlinear remainder

Use the Gaussian moment calculation as a control variate, and sample the part
that its nonlinear approximation omits. This retains a single Gaussian weight
belief while providing a measurable accuracy refinement for its messages.

Write $W=m+\sqrt v\odot\epsilon$, with standard-normal density $\phi$.
Freeze the linear logit approximation
$z_0(W)=f(m_*)+J_*(W-m_*)$ at the current snapshot. Let $h=\log p$ for
MFVI and $h=p$ for EP, and let $h_0$ apply the same likelihood to $z_0$.
For an integration proposal $r$ and its known weight $a=\phi/r$,
$E_\phi h=E_\phi h_0+E_r[a(h-h_0)]$.
The proof is the cancellation
$\int\phi h_0+\int r(\phi/r)(h-h_0)=\int\phi h$.
Differentiating under the integral, for integrable derivatives and fixed
surrogate/proposal coefficients, gives the identical decomposition for
$\partial_m E h$ and $\partial_v E h$. Pairing both paths with the same noise gives
$\operatorname{Var}(h-h_0)=\operatorname{Var}(h)+\operatorname{Var}(h_0)
-2\operatorname{Cov}(h,h_0)$. It reduces variance relative to direct sampling
when $2\operatorname{Cov}(h,h_0)>\operatorname{Var}(h_0)$; the control-variate
identity alone does not guarantee this condition. Importance integration also
depends on the proposal's weighted second moments.

For EP, integrate both $Z=E p$ and $\partial Z$ before dividing by $Z$;
the finite-sample ratio is consistent, not generally unbiased. For MFVI,
integrate $\partial E\log p$ directly. The Gaussian encoding remains
$d_m=v\odot g_m$, $c=2v^2\odot g_v$ and $d_s=2m\odot d_m+c$.
Finite EP and canonical MFVI updates remain distinct, as in the earlier
derivation; the code retains that distinction.

To resolve small derivatives, one may use the following antithetic
importance rule: $r(\epsilon)=\tfrac12\phi(\epsilon)+\tfrac12\phi(\epsilon-\delta)$,
where $\delta=\sqrt v\odot(J_{\rm competitor}-J_y)$ and
$a=2/[1+\exp(\delta^\top\epsilon-\|\delta\|^2/2)]\le2$.
The four nodes $\epsilon,-\epsilon,\delta+\epsilon,\delta-\epsilon$
form one independent integration group. This is an integration proposal;
the weight belief itself stays a single Gaussian.
The frozen surrogate is a function of $W$ anchored at $(m_*,J_*)$: freeze its
coefficients, not the dependence of $W=m+\sqrt v\odot\epsilon$ on the current
belief. Likewise freeze a proposal chosen at that snapshot, and retain all
ratio terms needed for the declared coordinate derivative. A convenient
reparameterized implementation samples $\epsilon$ and uses fixed proposal
coefficients in epsilon space. Stopping the entire surrogate graph would
remove the derivative one is trying to integrate.

<!-- classifier -->
### Full readout covariance and a diagonal-plus-factor hidden state

The image benchmarks retain full Gaussian logit covariance. `layers.readout`
implements the affine total-covariance rule below; `layers.correlated_readout`
adds the rank-four hidden correction used in the moment-EP MLP. The general
categorical head also accepts diagonal logits for the simpler teaching examples.

For ten classes the full **10×10 logit covariance** is
$C_z=M\operatorname{Diag}(v)M^\top+\operatorname{Diag}(V(m^{\odot2}+v)+v_b)$.
This follows directly from total covariance, conditioning first on the hidden
activation. Independent readout weights contribute the diagonal term;
uncertain hidden activations contribute correlations between classes.

In the second hidden layer, define
$D=\operatorname{Diag}(\Phi(m_{z_2}/\sqrt{v_{z_2}}))$ and $Q^\top Q=I_4$,
let $L=DM_2\operatorname{Diag}(\sqrt{v_1})Q$. Use
$C_{h_2}=\operatorname{Diag}(v_2-\operatorname{diag}(LL^\top))$
$+LL^\top$.
The retained part is positive semidefinite and the diagonal is unchanged.
Under the Gaussian ReLU closure, the residual is nonnegative: the first
Hermite component contributes at most the total activation variance, and an
orthogonal projection cannot increase its squared row norm. Explicitly,
Gaussian integration gives $\operatorname{Cov}(Z,Z_+)=v\Phi(m/\sqrt v)$;
Cauchy–Schwarz then implies
$\operatorname{Var}(Z_+)\ge v\Phi(m/\sqrt v)^2$.
Together with $QQ^\top\preceq I$ and the nonnegative weight-variance
contribution to $v_{z_2}$, this bounds each retained row norm by $v_2$.
Hold the basis fixed within each reverse pass. For the once-per-epoch refresh
used by the MLP training loop, include the refresh in training time. Backpropagation
includes the covariance correction's dependence on both layer means and
activation variances; it does not differentiate through the basis refresh.



### Contrast-space probability integration

The reported probability is $\bar p(y\mid x)=E_q[p(y\mid x,W)]$.
NLL is $-\log\bar p(y\mid x)$, not an average of sampled losses. Moment
serving factors the covariance of nine logit contrasts and averages softmax
probabilities over 18 symmetric Gaussian cubature nodes. Those are small
head calculations, **not 18 sampled networks**. Explicitly, let
$D=[I_9,-\mathbf 1]$, $a=Dm_z$ and $LL^\top=DC_zD^\top$.
The nodes $a\pm3Le_j$, each with weight $1/18$, have mean $a$ and covariance
$LL^\top$: the signed terms cancel and
$\sum_{j,\pm}(\pm3Le_j)(\pm3Le_j)^\top/18=LL^\top$.
Append a zero tenth logit before softmax; removing a common logit offset leaves
probabilities unchanged. Matching these first two Gaussian moments does not
make integration of nonlinear softmax exact. Sampled serving instead averages
$S$ complete-network probability vectors; each benchmark records $S$.
Both use the same ten classes.

One moment pass performs three affine products in an ordinary hidden layer:
one for the mean, and two for its variance. It also evaluates the nonlinear
moment formulas and the small covariance-aware head. This explains the
additional arithmetic relative to a point forward pass; the reported clocks
measure its actual implementation rather than inferring a speedup from the
number of network passes.

<!-- optimizer_complete -->
### Complete finite-update recurrences and their state

For the reference Vadam recurrence write $N_{\rm eff}=N/\tau$, prior precision
$p_i=1/v_{0i}$ and $\lambda_i=p_i/N_{\rm eff}$. Here $g$ is the gradient
of the minibatch-average **negative** log likelihood. With moments
$a_i=E[g_i]$ and $b_i=E[g_i^2]$, its temporal state updates are
\[
r_i^+=\beta_1r_i+(1-\beta_1)[a_i+\lambda_i(m_i-m_{0i})],\qquad
s_i^+=\beta_2s_i+(1-\beta_2)b_i.
\]
Then $m_i^+=m_i-\eta\widehat r_i/(\sqrt{\widehat s_i}+\lambda_i)$ and
$v_i^+=1/(N_{\rm eff}s_i^++p_i)$, where hats denote the reference's bias
correction. Its initial $s_i=(v_i^{-1}-p_i)/N_{\rm eff}$ preserves the declared
initial belief variance. This initialization is part of the reproduced
algorithm, not a zero-state Adam replacement.

For sampled MFVI, average $g_i$ and $g_i^2$ separately across complete network
draws. For moment MFVI, the reference uses
$b_i\approx a_i^2+(H\sqrt{\operatorname{Diag}v}\,e)_i^2$ with one fixed
unit-covariance direction and the point-loss Hessian $H$. The executed image
protocol uses Gaussian directions, except for LeNet's Rademacher direction.
To derive the contraction,
linearize $g(w)\approx g(m)+H(w-m)$; independence and $E[ee^\top]=I$ give
$E_e[(H\sqrt{\operatorname{Diag}v}\,e)_i^2]=(H\operatorname{Diag}(v)H^\top)_{ii}$.
A single fixed direction is an approximation to that expectation, not an exact
second moment. This statistic differs from the VI variance derivative and needs
its own validation. The EP optimizer instead consumes the differentiated
cavity likelihood and external field in $(m,\log v)$; it does not replace
posterior variance by Adam's squared-gradient accumulator.

The optional Gaussian-text Vadam path also retains a separate, zero-initialized
likelihood normalization state $u_i^+=\beta_2u_i+(1-\beta_2)b_i$.
When that recorded option is active, the mean denominator uses
$\sqrt{\widehat u_i}+\lambda_i$, while the posterior variance still uses
$1/(N_{\rm eff}s_i+p_i)$. The initial prior-compatible $s_i$ and the temporal
normalizer $u_i$ have different roles. The code records this choice in
`vadam_init(..., likelihood_normalization=True)`; silently identifying these
two states changes the training trajectory even when the sampled gradients
are identical. `vadam_step` handles both states explicitly. The image
experiments use the single-$s$ recurrence. The conditional-Gaussian text
tables use the BLoB split optimizer, not either Vadam variant.
The image curvature refresh frequency is a recorded implementation
choice: one fixed direction, refreshed every four MLP or two LeNet updates,
while mean gradients and EMAs update every minibatch. It is not a theorem that
such stale curvature equals a fresh sampled gradient second moment.

For EP-Adam let $a=\log v$. The ordinary chain rule gives
$g_a=v\odot g_v$. Adam maintains separate first and second temporal
accumulators for $g_m$ and $g_a$ and subtracts their bias-corrected, normalized
steps. These accumulators neither replace the belief variance nor perform EP
site removal. The cavity/field objective supplies EP semantics before Adam.

<!-- power_notation -->
**Match the notation to the executable controls.** The code calls the likelihood
power $\alpha/\tau$ `power`, and calls $\tau$ `temperature` in the training
protocol; `objective_and_grad` passes `temperature * power` as the `alpha`
argument of `power_cavity`. The reproduced EP configurations use
`power = 1`, so their head averages ordinary class probabilities while the
cavity removal fraction is $\tau/N$. Varying `temperature` with `power` fixed is therefore
not the $\alpha\downarrow0$ limit at fixed $\tau$ just proved. This distinction
prevents a regularization sweep from being mislabelled an EP-to-VI limit.

<!-- transformer_calculus -->
### LayerNorm, attention and LoRA: the actual directional equations

For LayerNorm, $a=x-\bar x$, $s=(d^{-1}\|a\|^2+\varepsilon)^{-1/2}$.
Its directional derivative is
$\delta y=\gamma s[\delta x-\overline{\delta x}-a s^2\overline{a\delta x}]
+\delta\gamma\,sa+\delta\beta$.
Differentiate the centering, variance and scale successively to obtain this
expression. Merely scaling each marginal variance by $\gamma^2s^2$ omits both
projection terms. The local JVP in the notebook includes them.

For attention, $S=QK^\top/\sqrt d$, $P=\mathrm{softmax}(S)$, $O=PV$.
The three directional equations are
$\delta S=(\delta QK^\top+Q\delta K^\top)/\sqrt d$,
$\delta P=P\odot[\delta S-\sum_jP_j\delta S_j]$, and
$\delta O=\delta P V+P\delta V$.
Shared columns preserve $Q$–$K$–$V$ cross terms that independent marginal
propagation discards. Causal and padding masks remain deterministic.

For LoRA, $Y=XW_0+cXAB$ and
$\delta Y=\delta X(W_0+cAB)+cX\delta A B+cXA\delta B$.
The first-order rule omits $cX\delta A\delta B$ and related products involving
$\delta X$. If $B=0$ at initialization, the first-order derivative with respect
to $A$ vanishes, while the product uncertainty need not vanish. Therefore the
initial and learned-belief tests must both be run; checking only the mean
network or only late checkpoints misses this failure mode.

<!-- mc_definitions -->
### What the approximation errors measure

For $S$ complete-network draws shared across diagnostic inputs, compute
$\widehat m=S^{-1}\sum_s f_{W_s}(X)$ and
$\widehat v=S^{-1}\sum_s(f_{W_s}(X)-\widehat m)^2$.
This is the empirical central second moment used in the saved arrays. Multiplying
by $S/(S-1)$ gives the usual unbiased variance for independent draws; antithetic
nodes instead require the independent-group uncertainty calculation below.
A relative discrepancy is $\|\widehat t-t_{\rm ref}\|_2/\|t_{\rm ref}\|_2$.
When the reference norm is zero, report the absolute discrepancy instead;
an arbitrary denominator floor is not a relative-error guarantee.
For independent batches at budget $B$, root-mean-square error is
$[K_B^{-1}\sum_b\|\widehat t_b-t_{\rm ref}\|_2^2]^{1/2}/\|t_{\rm ref}\|_2$.
Pointwise $\pm1.96$ MC standard-error bars, when displayed, concern the finite
reference estimator, not the learned posterior or the approximation's bias.
An independent repeat checks stability; it is not itself a confidence interval.
All comparisons must use the same objective, sum/average normalization,
coordinate system, weight belief and set of reported coordinates.
