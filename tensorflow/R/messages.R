# %% Diagonal Gaussian messages and MFVI canonical update
# MFVI in Gaussian mean coordinates. Read this file before either layer file.
# Dependency: R package reticulate and Python TensorFlow (CPU is sufficient).
# All objectives below are MAXIMIZED. Means/variances are stored as (m,v),
# while the exponential-family mean parameter is (m,s), s=m^2+v.
library(reticulate)
tf <- import("tensorflow", convert=FALSE)

mfvi_state <- function(m,v) list(m=m,v=v)
mfvi_number <- function(a,x) tf$cast(a,x$dtype)
mfvi_numpy <- function(x) py_to_r(x$numpy())

# Theorem (coordinate conversion).
# If a=dPhi/dm|v, b=dPhi/dv, then dPhi/dm|s=a-2mb, dPhi/ds=b.
# Multiplying these by the Gaussian Fisher matrix
#   F = [[v,2mv],[2mv,4m^2v+2v^2]]
# gives dm=va and ds=2m*dm+2v^2b. This proves the two formulas below.
# These are natural directions, NOT automatically EP tilted-posterior changes.
# We retain c=ds-2m*dm separately to avoid subtraction at tiny variances.
# Zero-variance coordinates are deterministic: keep their ordinary adjoints
# a0,b0 explicitly; never invert a zero Fisher block or invent uncertainty.
mfvi_encode <- function(x,a,b) {
  tf$debugging$assert_non_negative(x$v)
  dm<-x$v*a; c<-2*x$v*x$v*b; zero<-x$v==0
  list(dm=dm,ds=2*x$m*dm+c,c=c,
       a0=tf$where(zero,a,tf$zeros_like(a)),
       b0=tf$where(zero,b,tf$zeros_like(b)))
}
# Inverse conversion: a=dm/v, b=(ds-2m*dm)/(2v^2)=c/(2v^2).
# Proof: solve the two encoding equations. The saved a0,b0 apply only at v=0,
# where an inverse Fisher map does not exist; they carry ordinary derivatives.
mfvi_decode <- function(x,d) list(
  a=tf$math$divide_no_nan(d$dm,x$v)+d$a0,
  b=tf$math$divide_no_nan(d$c,2*x$v*x$v)+d$b0)

# At ONE forward snapshot these maps are linear. Therefore branch messages
# add componentwise. Absolute posterior means would double-count the parent.
mfvi_sum_messages <- function(a,b) {
  if(is.null(a)) return(b)
  Map(`+`,a,b)
}

# Theorem (MFVI canonical update, independent diagonal Gaussian prior).
# Write tau=1/v, h=m/v and eta=(h,-tau/2). For likelihood adjoints (a,b),
# the raw-moment gradient is (a-2mb,b). KL contributes eta_prior-eta.
# Hence eta+ = (1-rho)eta + rho[eta_prior+(a-2mb,b)], giving:
#   tau+ = (1-rho)/v + rho(1/v0-2b)
#   h+   = (1-rho)m/v + rho(m0/v0+a-2mb).
# Proof: substitute eta into the natural-gradient step, then solve m+=h+/tau+.
# The prior is counted ONCE, not once per example or once per backward layer.
# Unlike EP, a unit additive update to (m,s) is generally WRONG for MFVI.
mfvi_canonical <- function(x,d,prior,rho) {
  g<-mfvi_decode(x,d)
  tau<-(1-rho)/x$v+rho*(1/prior$v-2*g$b)
  h<-(1-rho)*x$m/x$v+rho*(prior$m/prior$v+g$a-2*x$m*g$b)
  tf$debugging$assert_positive(tau,message="Reduce rho: invalid MFVI precision")
  mfvi_state(h/tau,1/tau)
}
# Gaussian KL: E_q[log q-log p0] = 1/2 sum[(v+(m-m0)^2)/v0-1+log(v0/v)].
# Derivation: use E_q[(w-m)^2]=v and E_q[(w-m0)^2]=v+(m-m0)^2 in the two
# Gaussian log densities. This is a global weight penalty, not an activation loss.
mfvi_kl <- function(x,prior) tf$reduce_sum(
  (x$v+(x$m-prior$m)^2)/prior$v-1+tf$math$log(prior$v/x$v))/2

# Exact expected Gaussian log likelihood, with no assumption that activations
# themselves are Gaussian: E[(y-z)^2]=(y-m)^2+v uses only two moments.
# scale=N/B for a uniformly sampled minibatch; use 1 for the entire dataset.
# Sum, rather than silently average, the observation/output contributions.
# Thus Phi=-scale/2 sum[log(2*pi*r)+((y-m)^2+v)/r],
# dPhi/dm=scale*(y-m)/r and dPhi/dv=-scale/(2r). Encode this derivative pair.
mfvi_gaussian_head <- function(z,y,noise_variance=1,scale=1) {
  e<-y-z$m; r<-mfvi_number(noise_variance,z$m)
  phi<- -scale*tf$reduce_sum((e^2+z$v)/r+tf$math$log(2*pi*r))/2
  list(value=phi,message=mfvi_encode(z,scale*e/r,-scale*tf$ones_like(z$v)/(2*r)))
}

# Classification: a declared second-order approximation to E[log p(y|z)].
# Phi = sum_n[y'm - logsumexp(m) - 1/2 sum_i v_i p_i(1-p_i)].
# It assumes diagonal logit covariance; it is NOT log E[p], EP, or an exact
# nonlinear-network ELBO. Let t_i=v_i(1-2p_i). Softmax Jacobian multiplication
# is J t = p*(t-sum(p*t)); this differentiates the variance correction too.
# Derivation: Taylor-expand log p_y(z) at m. The linear term averages to zero;
# its Hessian is -(Diag(p)-p p^T), whose contraction with Diag(v) gives the
# displayed correction. Differentiating it gives a=y-p-(J t)/2 and
# b=-p*(1-p)/2, before multiplying by the minibatch scale.
mfvi_categorical_head <- function(z,labels,scale=1) {
  p<-tf$nn$softmax(z$m); k<-tf$gather(tf$shape(z$m),1L)
  y<-tf$one_hot(labels,k,dtype=z$m$dtype)
  t<-z$v*(1-2*p)
  a<-scale*(y-p-.5*p*(t-tf$reduce_sum(p*t,axis=-1L,keepdims=TRUE)))
  b<- -.5*scale*p*(1-p)
  phi<-scale*(tf$reduce_sum(y*z$m)-tf$reduce_sum(tf$reduce_logsumexp(z$m,axis=-1L))
              -.5*tf$reduce_sum(z$v*p*(1-p)))
  list(value=phi,message=mfvi_encode(z,a,b))
}

# %% Handwritten local message equations
# Literal MFVI layer equations. Source mfvi_gaussian_math.R first.
# No GradientTape, no projected-backward call, no EP cavity or tilted target.
# Each backward takes a CHILD MOMENT SIGNAL and returns PARENT MOMENT SIGNALS.
# Decode -> the handwritten chain rule below -> encode. This is literal
# variational-message transport for the DECLARED forward moment approximation.
# These are explicit R backward functions, not tf$custom_gradient callbacks.
# mfvi_backward calls them during its reverse graph traversal. A custom-gradient
# wrapper could dispatch the same equations, but is not needed by this engine.
# Throughout: * denotes elementwise multiplication, juxtaposition denotes matrix
# multiplication, a=dPhi/dM and b=dPhi/dV are DECODED child adjoints.

# Dense: independent weight coordinates and an input independent of this layer's
# weights. Input need not be Gaussian. Diagonal input covariance is retained.
# M=m A+b; V=v A^2+(m^2+v) S+t. Matrix products sum shared-weight contributions
# across ALL examples; minibatch scaling belongs to the likelihood head only.
mfvi_dense_forward <- function(x,w,bias) mfvi_state(
  tf$matmul(x$m,w$m)+bias$m,
  tf$matmul(x$v,w$m^2)+tf$matmul(x$m^2+x$v,w$v)+bias$v)
# Backward derivation. Differentiate M=m A+mb and V=v A^2+(m^2+v) S+vb:
#   g_m=a A^T+2m*(b S^T),       g_v=b (A^2+S)^T,
#   g_A=m^T a+2A*(v^T b),      g_S=(m^2+v)^T b,
#   g_mb=sum_batch(a),         g_vb=sum_batch(b).
# These follow by collecting coefficients in dPhi=<a,dM>+<b,dV>.
# Encode EACH pair at its own forward belief; do not add it to the variance.
mfvi_dense_backward <- function(x,w,bias,d) {
  y<-mfvi_dense_forward(x,w,bias);g<-mfvi_decode(y,d)
  a<-g$a;b<-g$b
  # dM/dm=A; dV/dm=2m S. BOTH terms matter in the mean message.
  xm<-tf$matmul(a,w$m,transpose_b=TRUE)+2*x$m*tf$matmul(b,w$v,transpose_b=TRUE)
  xv<-tf$matmul(b,w$m^2+w$v,transpose_b=TRUE)
  wm<-tf$matmul(x$m,a,transpose_a=TRUE)+2*w$m*tf$matmul(x$v,b,transpose_a=TRUE)
  wv<-tf$matmul(x$m^2+x$v,b,transpose_a=TRUE)
  list(mfvi_encode(x,xm,xv),mfvi_encode(w,wm,wv),
       mfvi_encode(bias,tf$reduce_sum(a,axis=0L),tf$reduce_sum(b,axis=0L)))
}

# Convolution is the same polynomial with products replaced by convolution.
# Conv2DBackpropInput/Filter are transpose linear operators, not inference.
mfvi_conv <- function(x,w,stride=1L) tf$nn$conv2d(x,w,
  strides=as.integer(c(1,stride,stride,1)),padding="SAME")
mfvi_conv_input <- function(x,w,g,stride) tf$raw_ops$Conv2DBackpropInput(
  input_sizes=tf$shape(x),filter=w,out_backprop=g,
  strides=as.integer(c(1,stride,stride,1)),padding="SAME")
mfvi_conv_weight <- function(x,w,g,stride) tf$raw_ops$Conv2DBackpropFilter(
  input=x,filter_sizes=tf$shape(w),out_backprop=g,
  strides=as.integer(c(1,stride,stride,1)),padding="SAME")
# Let C(m,A) denote convolution. Independence gives
# M=C(m,A)+mb, V=C(v,A^2)+C(m^2+v,S)+vb by the same variance expansion as Dense.
# This retains pointwise variances; shared filters can induce spatial covariance
# that this diagonal representation does not store.
mfvi_conv2d_forward <- function(x,w,bias,stride=1L) mfvi_state(
  mfvi_conv(x$m,w$m,stride)+bias$m,
  mfvi_conv(x$v,w$m^2,stride)+mfvi_conv(x$m^2+x$v,w$v,stride)+bias$v)
# Backward: let I(A,g) be the input adjoint of C(.,A), and K(x,g) its
# filter adjoint. The chain rule gives
# g_m=I(A,a)+2m*I(S,b); g_v=I(A^2+S,b);
# g_A=K(m,a)+2A*K(v,b); g_S=K(m^2+v,b).
# Bias adjoints sum batch AND spatial positions because one bias is shared.
mfvi_conv2d_backward <- function(x,w,bias,d,stride=1L) {
  y<-mfvi_conv2d_forward(x,w,bias,stride);g<-mfvi_decode(y,d);a<-g$a;b<-g$b
  xm<-mfvi_conv_input(x$m,w$m,a,stride)+2*x$m*mfvi_conv_input(x$m,w$v,b,stride)
  xv<-mfvi_conv_input(x$v,w$m^2+w$v,b,stride)
  wm<-mfvi_conv_weight(x$m,w$m,a,stride)+2*w$m*mfvi_conv_weight(x$v,w$m,b,stride)
  wv<-mfvi_conv_weight(x$m^2+x$v,w$v,b,stride)
  list(mfvi_encode(x,xm,xv),mfvi_encode(w,wm,wv),mfvi_encode(bias,
    tf$reduce_sum(a,axis=c(0L,1L,2L)),tf$reduce_sum(b,axis=c(0L,1L,2L))))
}

# ReLU: Gaussian closure is used HERE, not for the observed input distribution.
# s=sqrt(v), alpha=m/s, P=Phi(alpha), p=phi(alpha).
# M=mP+sp; S=(m^2+v)P+msp; V=S-M^2.
# Derivatives: M_m=P, M_v=p/(2s), S_m=2M, S_v=P. Therefore
# V_m=2M(1-P), V_v=P-Mp/s. These differentiate the full moment map,
# unlike a stopped-gate rule which loses the mean/variance coupling.
mfvi_relu_terms <- function(x) {
  positive<-x$v>0;s<-tf$sqrt(tf$where(positive,x$v,tf$ones_like(x$v)))
  alpha<-x$m/s;P<-.5*tf$math$erfc(-alpha/sqrt(2));p<-tf$exp(-alpha^2/2)/sqrt(2*pi)
  M<-x$m*P+s*p;S<-(x$m^2+x$v)*P+x$m*s*p
  gate<-tf$cast(x$m>0,x$m$dtype)
  # max(.,0) removes subtraction roundoff; it adds no variance floor.
  V<-tf$maximum(S-M^2,tf$zeros_like(S))
  list(M=tf$where(positive,M,tf$nn$relu(x$m)),V=tf$where(positive,V,tf$zeros_like(V)),
       P=P,p=p,s=s,positive=positive,gate=gate,active=S-M^2>0)
}
# Forward: retain the Gaussian integrals M and V computed above. At v=0 the
# declared deterministic branch returns (relu(m),0), without adding a floor.
mfvi_relu_forward <- function(x) {r<-mfvi_relu_terms(x);mfvi_state(r$M,r$V)}
# Backward: g_m=a*M_m+b*V_m, g_v=a*M_v+b*V_v; encode this pair at (m,v).
# Proof: substitute dM=M_m dm+M_v dv and dV=V_m dm+V_v dv into dPhi.
# At v=0 differentiate the explicitly chosen deterministic branch, rather than
# claiming its variance derivative equals the one-sided limit from v>0.
mfvi_relu_backward <- function(x,d) {
  r<-mfvi_relu_terms(x);g<-mfvi_decode(mfvi_state(r$M,r$V),d)
  # Differentiate the explicit zero-variance branch as written in forward.
  a<-tf$where(r$positive,r$P,r$gate)
  b<-tf$where(r$positive,r$p/(2*r$s),tf$zeros_like(x$v))
  c<-tf$where(r$positive & r$active,2*r$M*(1-r$P),tf$zeros_like(x$v))
  e<-tf$where(r$positive & r$active,r$P-r$M*r$p/r$s,tf$zeros_like(x$v))
  list(mfvi_encode(x,g$a*a+g$b*c,g$a*b+g$b*e))
}

# Average pool: M=A m, V=(A elementwise squared)v. For k*k pixels,
# tf.avg_pool(v)/k^2 implements A^2 v, provided retained pixels are independent.
mfvi_pool_forward <- function(x,k=2L) mfvi_state(
  tf$nn$avg_pool2d(x$m,as.integer(k),as.integer(k),"VALID"),
  tf$nn$avg_pool2d(x$v,as.integer(k),as.integer(k),"VALID")/(k*k))
# Backward: g_m=A^T a and g_v=(A elementwise squared)^T b.
# Proof: transpose each of the two linear forward maps. For uniform windows,
# the second transpose is AvgPoolGrad(b)/k^2. Encode at the unpooled belief.
mfvi_pool_backward <- function(x,d,k=2L) {
  g<-mfvi_decode(mfvi_pool_forward(x,k),d)
  back<-function(a) tf$raw_ops$AvgPoolGrad(orig_input_shape=tf$shape(x$m),grad=a,
    ksize=as.integer(c(1,k,k,1)),strides=as.integer(c(1,k,k,1)),padding="VALID")
  list(mfvi_encode(x,back(g$a),back(g$b)/(k*k)))
}
# Flatten: let P be the coordinate rearrangement, leaving the batch axis intact.
# M=P m, V=P v. Every scalar coordinate is retained, so this introduces no
# additional moment approximation; the represented covariance remains diagonal.
mfvi_flatten_forward <- function(x) {
  dims<-tf$stack(list(tf$gather(tf$shape(x$m),0L),tf$constant(-1L)))
  mfvi_state(tf$reshape(x$m,dims),tf$reshape(x$v,dims))
}
# Backward: g_m=P^T a, g_v=P^T b. Proof: P is a permutation/reshape, hence
# P^T is its inverse reshape. Decode at the flattened belief, reshape both
# adjoints to the parent shape, then encode at the parent belief.
mfvi_flatten_backward <- function(x,d) {
  g<-mfvi_decode(mfvi_flatten_forward(x),d)
  list(mfvi_encode(x,tf$reshape(g$a,tf$shape(x$m)),tf$reshape(g$b,tf$shape(x$v))))
}

# Diagonal branch projection: a+b has V=Va+Vb AFTER discarding cross covariance.
# This is deliberately a baseline, NOT exact for correlated residual branches.
# For the SAME random variable a+a, V=4Va; handle that alias explicitly.
# This forward covariance issue is distinct from accumulating backward messages.
mfvi_add_forward <- function(x,y,same=FALSE) {
  if(same) return(mfvi_state(2*x$m,4*x$v))
  mfvi_state(x$m+y$m,x$v+y$v)
}
# Backward for distinct parents: both receive adjoints (a,b), because both
# forward Jacobians are identity. Encode separately at their respective beliefs.
# For the identical parent x+x, differentiate (2m,4v) to obtain (2a,4b) ONCE.
# Other shared ancestors accumulate their messages in the graph reverse pass.
mfvi_add_backward <- function(x,y,d,same=FALSE) {
  g<-mfvi_decode(mfvi_add_forward(x,y,same),d)
  if(same) return(list(mfvi_encode(x,2*g$a,4*g$b)))
  list(mfvi_encode(x,g$a,g$b),mfvi_encode(y,g$a,g$b))
}
# Concatenate: M=[mx,my], V=[vx,vy] along the channel axis. This stacks marginal
# moments; a diagonal belief does not retain the off-diagonal covariance blocks
# between the two inputs. No new independent weight or likelihood factor is added.
mfvi_concat_forward <- function(x,y) mfvi_state(
  tf$concat(list(x$m,y$m),axis=-1L),tf$concat(list(x$v,y$v),axis=-1L))
# Backward: if Sx,Sy select the respective channel ranges, parent adjoints are
# (Sx*a,Sx*b) and (Sy*a,Sy*b). Proof: the transpose of concatenation is slicing.
# Encode each slice at its parent's belief. Repeated parents are merged by the
# graph's message sum, rather than by adding absolute posterior target means.
mfvi_concat_backward <- function(x,y,d) {
  g<-mfvi_decode(mfvi_concat_forward(x,y),d)
  sizes<-tf$stack(list(tf$gather(tf$shape(x$m),tf$rank(x$m)-1L),
                       tf$gather(tf$shape(y$m),tf$rank(y$m)-1L)))
  a<-py_to_r(tf$split(g$a,sizes,axis=-1L));b<-py_to_r(tf$split(g$b,sizes,axis=-1L))
  list(mfvi_encode(x,a[[1]],b[[1]]),mfvi_encode(y,a[[2]],b[[2]]))
}

# %% The same maps through local autodiff
# Shared candidate layers; independent handwritten references are in
# mfvi_literal_layers.R. EP and MFVI choose heads/updates, not duplicate maps.
# Projected MFVI backward messages. Source math and literal layer files first.
# The forward moment maps are deliberately shared; the backward calculations
# are independent: TensorFlow differentiates each LOCAL moment map here.
# Nothing in this file calls a literal *_backward function to compute its VJP.
#
# Theorem: for y=F(x), encoding E_x and decoding D_y,
#   message_x = E_x [ J_F(x)^T D_y(message_y) ].
# Proof: D_y recovers the Euclidean derivative of the declared scalar Phi.
# The chain rule supplies J_F^T times that derivative. E_x then applies the
# Gaussian Fisher map proved in mfvi_gaussian_math.R. Thus literal derivatives
# and this projected construction agree whenever they use the SAME F and Phi.
# This does not prove a nonlinear Gaussian closure equals the true BNN ELBO.
#
# TensorFlow autodiff supplies ONLY the Euclidean local VJP. Encoding, decoding,
# graph accumulation and the canonical MFVI update remain explicit operations.
# Neither this path nor the literal path registers tf$custom_gradient callbacks;
# the explicit reverse traversal already provides their message-routing role.
# Every layer has call/backward operations; weights update only AFTER the reverse
# sweep. A custom-gradient API wrapper is optional, not a different inference rule.
mfvi_projected <- function(forward,parents) {
  parents<-lapply(parents,function(x)mfvi_state(tf$identity(x$m),tf$identity(x$v)))
  inputs<-unlist(lapply(parents,moment_values),recursive=FALSE)
  evaluate<-function(a)do.call(forward,lapply(seq_along(parents),function(i)
    list(m=a[[2L*i-1L]],v=a[[2L*i]])))
  cache<-moment_cache(inputs,evaluate,persistent=TRUE)
  backward<-function(child) {
    g<-mfvi_decode(cache$state,child);d<-moment_pullback(cache,list(g$a,g$b))
    lapply(seq_along(parents),function(i)mfvi_encode(parents[[i]],d[[2L*i-1L]],d[[2L*i]]))
  }
  list(state=cache$state,backward=backward)
}

# Named layer counterparts. Literal equations remain independently readable
# in the previous file; these wrappers spell out the corresponding local map.
# Dense projection. Write x=(m,v), w=(A,S), bias=(mb,vb).
# M=m A+mb; V=v A^2+(m^2+v) S+vb. With child adjoints (a,b), the VJP is
# (a A^T+2m*(b S^T), b (A^2+S)^T) for x,
# (m^T a+2A*(v^T b), (m^2+v)^T b) for w, and (sum a,sum b) for bias.
# Proof: differentiate the two polynomials and collect coefficients. The shared
# engine obtains these derivatives by autodiff and encodes each parent pair.
mfvi_projected_dense <- function(x,w,bias)
  mfvi_projected(moment_dense,list(x,w,bias))
# Conv2D projection. Replace dense products by C(x,w), the convolution operator:
# M=C(m,A)+mb; V=C(v,A^2)+C(m^2+v,S)+vb.
# Its input/filter VJPs are the transpose convolution operators. Bilinearity of
# C gives the same four terms as Dense; the shared VJP includes all of them.
# Spatial correlations induced by shared filters are outside this diagonal map.
mfvi_projected_conv2d <- function(x,w,bias,stride=1L)
  mfvi_projected(function(x,w,bias)moment_conv2d(x,w,bias,c(1L,stride,stride,1L)),list(x,w,bias))
# ReLU projection under Gaussian closure: t=m/sqrt(v), P=Phi(t), p=phi(t),
# M=mP+sqrt(v)p; V=(m^2+v)P+m*sqrt(v)p-M^2.
# For v>0 and unclipped V, derivatives are M_m=P, M_v=p/(2sqrt(v)),
# V_m=2M(1-P), V_v=P-Mp/sqrt(v). Thus the VJP is
# (a*M_m+b*V_m, a*M_v+b*V_v), including both mean/variance cross terms.
# Autodiff follows the written deterministic/clipping branches at their boundaries.
mfvi_projected_relu <- function(x) mfvi_projected(moment_relu,list(x))
# Average-pool projection: M=A m, V=A^{elementwise 2}v for independent pixels.
# Therefore the VJP is (A^T a, (A^{elementwise 2})^T b). For k*k pixels,
# A^{elementwise 2}=A/k^2. This follows directly by transposing the linear maps.
mfvi_projected_pool <- function(x,k=2L)
  mfvi_projected(function(x)moment_average_pool(x,k),list(x))
# Flatten projection: M=P m, V=P v for a coordinate permutation/reshape P.
# The local VJP is (P^T a,P^T b), i.e. inverse reshaping, followed by encoding.
# No distributional assumption beyond the incoming representation is introduced.
mfvi_projected_flatten <- function(x) mfvi_projected(moment_flatten,list(x))
# Residual projection: M=mx+my, V=vx+vy after dropping cross covariance.
# Each distinct parent receives adjoints (a,b). For the SAME parent, M=2m,V=4v;
# its VJP is (2a,4b). The one-parent closure below enforces this identity and
# prevents mistaking two occurrences of x for two independent random variables.
mfvi_projected_add <- function(x,y,same=FALSE) {
  if(same) return(mfvi_projected(function(x)moment_add(x,x,TRUE),list(x)))
  mfvi_projected(moment_add,list(x,y))
}
# Concatenation projection: M=[mx,my], V=[vx,vy]. The transpose Jacobian slices
# (a,b) into the matching channel ranges. Encode those slices at each parent.
# Proof: concatenation is a linear injection for each parent; its adjoint is
# channel selection. The graph sums contributions when a parent is reused.
mfvi_projected_concat <- function(x,y) mfvi_projected(moment_concat,list(x,y))

# A graph is needed only to represent residual/concatenation reuse. Each row
# is one mathematical map, with integer parent IDs and weight IDs. No callbacks
# for logging, device management, schedulers, checkpointing or configuration.
mfvi_model <- function() {
  model<-new.env(parent=emptyenv());model$nodes<-list();model$weights<-list();model
}
mfvi_parameter <- function(model,shape,prior_variance) {
  shape<-as.list(as.integer(shape))
  p<-mfvi_state(tf$Variable(tf$random$normal(shape,stddev=sqrt(prior_variance),dtype=tf$float64)),
                tf$Variable(tf$fill(shape,tf$constant(.01*prior_variance,tf$float64))))
  p$prior<-mfvi_state(tf$zeros(shape,tf$float64),tf$fill(shape,tf$constant(prior_variance,tf$float64)))
  model$weights[[length(model$weights)+1L]]<-p
  length(model$weights)
}
mfvi_node <- function(model,parents,forward,literal,weights=integer()) {
  model$nodes[[length(model$nodes)+1L]]<-list(parents=parents,weights=weights,
                                           forward=forward,literal=literal)
  length(model$nodes)+1L # State 1 is the observed input, with exactly zero variance.
}
mfvi_run <- function(model,x,path="projected") {
  stopifnot(path %in% c("literal","projected"))
  states<-list(mfvi_state(x,tf$zeros_like(x)));cache<-list()
  weights<-lapply(model$weights,function(w)mfvi_state(tf$identity(w$m),tf$identity(w$v)))
  if(path=="projected") {
    layer<-function(op,ps,wm,wv,pids)do.call(op$forward,c(ps,Map(mfvi_state,wm,wv)))
    f<-moment_graph_forward(model$nodes,states[[1]],lapply(weights,`[[`,"m"),
      lapply(weights,`[[`,"v"),layer,record=TRUE)
    f$weights<-weights;f$path<-path;return(f)
  }
  for(op in model$nodes) {
    parents<-c(states[op$parents],weights[op$weights])
    if(path=="projected") c<-mfvi_projected(op$forward,parents) else {
      c<-list(state=do.call(op$forward,parents),parents=parents,literal=op$literal)
    }
    cache[[length(cache)+1L]]<-c;states[[length(states)+1L]]<-c$state
  }
  list(states=states,weights=weights,cache=cache,path=path,output=tail(states,1)[[1]])
}
mfvi_backward <- function(model,pass,child) {
  if(pass$path=="projected") {
    d<-mfvi_decode(pass$output,child)
    a<-moment_graph_reverse(pass,list(d$a,d$b),lapply(pass$weights,`[[`,"m"))
    signals<-lapply(seq_along(pass$states),function(i) {
      v<-a$boundary_adjoints[[i]]
      if(is.null(v))v<-list(tf$zeros_like(pass$states[[i]]$m),tf$zeros_like(pass$states[[i]]$v))
      mfvi_encode(pass$states[[i]],v[[1]],v[[2]])
    })
    return(list(activations=signals,weights=Map(mfvi_encode,pass$weights,a$gm,a$gv)))
  }
  messages<-vector("list",length(pass$states));messages[[length(messages)]]<-child
  wd<-vector("list",length(model$weights))
  for(j in rev(seq_along(model$nodes))) {
    op<-model$nodes[[j]];c<-pass$cache[[j]];d<-messages[[j+1L]]
    if(pass$path=="projected") parents<-c$backward(d) else
      parents<-do.call(c$literal,c(c$parents,list(d=d)))
    nx<-length(op$parents)
    for(k in seq_len(nx)) {
      i<-op$parents[[k]];messages[[i]]<-mfvi_sum_messages(messages[[i]],parents[[k]])
    }
    for(k in seq_along(op$weights)) {
      i<-op$weights[[k]];wd[[i]]<-mfvi_sum_messages(wd[[i]],parents[[nx+k]])
    }
  }
  list(activations=messages,weights=wd)
}
mfvi_apply <- function(model,pass,messages,rho) {
  # Compute and validate EVERY new belief before assigning any of them.
  updated<-lapply(seq_along(model$weights),function(i)
    mfvi_canonical(pass$weights[[i]],messages$weights[[i]],model$weights[[i]]$prior,rho))
  for(i in seq_along(updated)) {
    model$weights[[i]]$m$assign(updated[[i]]$m)
    model$weights[[i]]$v$assign(updated[[i]]$v)
  }
  invisible(updated)
}

# %% The nonlinear MLP used for the equality check
# Readable architectures made ONLY from the literal/projected MFVI primitives.
# Source gaussian_math, literal_layers, projected_layers, then this file.
# Every builder can run either reverse path; the forward equations are identical.
# The layer maps and candidate traversal are shared with the benchmarks.
# These small reference architectures differ in topology and normalization.

mfvi_dense <- function(g,x,input,output) {
  force(x)
  w<-mfvi_parameter(g,c(input,output),2/input)
  b<-mfvi_parameter(g,output,1/input)
  mfvi_node(g,x,moment_dense,mfvi_dense_backward,c(w,b))
}
mfvi_conv2d <- function(g,x,input,output,k=3L,stride=1L) {
  force(x);force(stride)
  w<-mfvi_parameter(g,c(k,k,input,output),2/(k*k*input))
  b<-mfvi_parameter(g,output,1/(k*k*input))
  mfvi_node(g,x,function(x,w,bias)moment_conv2d(x,w,bias,c(1L,stride,stride,1L)),
    function(x,w,bias,d)mfvi_conv2d_backward(x,w,bias,d,stride),c(w,b))
}
mfvi_relu <- function(g,x) {force(x);mfvi_node(g,x,moment_relu,mfvi_relu_backward)}
mfvi_pool <- function(g,x,k=2L) {
  force(x);force(k)
  mfvi_node(g,x,function(x)moment_average_pool(x,k),function(x,d)mfvi_pool_backward(x,d,k))
}
mfvi_flatten <- function(g,x) {force(x);mfvi_node(g,x,moment_flatten,mfvi_flatten_backward)}
mfvi_add <- function(g,x,y) {
  force(x);force(y)
  if(x==y) return(mfvi_node(g,x,function(x)moment_add(x,x,TRUE),
                           function(x,d)mfvi_add_backward(x,x,d,TRUE)))
  mfvi_node(g,c(x,y),moment_add,mfvi_add_backward)
}
mfvi_concat <- function(g,x,y) {
  force(x);force(y);mfvi_node(g,c(x,y),moment_concat,mfvi_concat_backward)
}

# An MLP example: change only these widths to study depth/width effects.
mfvi_mlp <- function(widths=c(16L,8L,8L,10L)) {
  g<-mfvi_model();x<-1L
  for(i in seq_len(length(widths)-1L)) {
    x<-mfvi_dense(g,x,widths[[i]],widths[[i+1L]])
    if(i<length(widths)-1L)x<-mfvi_relu(g,x)
  }
  g
}

# DenseNet counterpart of the original EP catalogue: CIFAR input 32x32x3,
# three bottleneck/concatenation stages, transitions and a 10-class head.
# Concatenation retains marginal variances but discards cross-channel covariance;

# %% EP and MFVI select different objectives and finite updates
# Two objectives on ONE set of forward layers and ONE reverse traversal.
# Source mfvi_gaussian_math.R, mfvi_literal_layers.R, mfvi_projected_layers.R
# and mfvi_architectures.R first. The literal derivatives are check references;
# the actual candidate uses moment_layers.R and moment_backprop.R.

# EP: Phi=log N(y;m,r+v), so a=(y-m)/(r+v) and
# b=[(y-m)^2/(r+v)^2-1/(r+v)]/2. These are normalizer derivatives.
# MFVI: Phi=E[log N(y;z,r)], a=(y-m)/r, b=-1/(2r).
# For a nonlinear moment network these are two DECLARED Gaussian closures;
# agreement of literal derivatives and backprop is not an exact-posterior claim.
gaussian_objective_message <- function(z,y,method="mfvi",r=.8) {
  if(method=="mfvi")return(mfvi_gaussian_head(z,y,r))
  stopifnot(method=="ep");S<-z$v+tf$constant(r,dtype=z$m$dtype);e<-y-z$m
  list(value=-tf$reduce_sum(tf$math$log(2*pi*S)+e*e/S)/2,
       message=mfvi_encode(z,e/S,(e*e/(S*S)-1/S)/2))
}

# Finite tilted update is in RAW moments. Substituting s=m^2+v gives
# m+=m+rho*dm; v+=v+rho*c-rho^2*dm^2. The last term cannot be dropped.
# This installs an exact tilt only when the supplied normalizer is exact.
ep_install_message <- function(q,d,rho=1) {
  m<-q$m+rho*d$dm;v<-q$v+rho*d$c-rho*rho*d$dm*d$dm
  tf$debugging$assert_positive(v,message="Invalid projected EP tilt; reduce damping")
  list(m=m,v=v)
}

# Literal and backprop paths use identical beliefs and objectives. Only the
# backward implementation differs. No optimizer or posterior oracle is called.
ep_mfvi_messages <- function(model,x,y,method="mfvi",path="projected") {
  f<-mfvi_run(model,x,path)
  head<-gaussian_objective_message(f$output,y,method)
  list(forward=f,objective=head$value,messages=mfvi_backward(model,f,head$message))
}

