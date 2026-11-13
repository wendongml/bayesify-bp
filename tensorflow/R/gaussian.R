# %% Gaussian Fisher coordinates and finite updates
# %% Gaussian geometry: exactly the displayed equations
encode_diagonal <- function(m, v, gm, gv) {
  dm <- v * gm
  centered<-2*v*v*gv
  list(dm = dm, ds = 2*m*dm + centered, centered= centered)
}
decode_diagonal <- function(m, v, signal) {
  tf$debugging$assert_positive(v)
  # Retain the centered increment to avoid catastrophic subtraction when v is
  # tiny. It is the same quantity ds-2*m*dm, not a different update rule.
  centered<-if(is.null(signal$centered))signal$ds-2*m*signal$dm else signal$centered
  list(gm = signal$dm/v, gv = centered/(2*v*v))
}
encode_full <- function(m, C, gm, GC) {
  GC <- (GC + tf$transpose(GC))/2
  dm <- tf$linalg$matvec(C, gm)
  outer <- function(a,b) tf$einsum("i,j->ij", a,b)
  list(dm=dm, dM=outer(m,dm)+outer(dm,m)+2*tf$matmul(tf$matmul(C,GC),C))
}
decode_full <- function(m, C, signal) {
  L <- tf$linalg$cholesky(C)
  solveC <- function(A) tf$linalg$cholesky_solve(L,A)
  dm <- signal$dm
  Q <- signal$dM-tf$einsum("i,j->ij",m,dm)-tf$einsum("i,j->ij",dm,m)
  list(gm=tf$squeeze(solveC(tf$expand_dims(dm,-1L)),-1L),
       GC=tf$transpose(solveC(tf$transpose(solveC(Q))))/2)
}
tilted_update <- function(m,v,signal,rho=1) {
  mp <- m+rho*signal$dm
  vp <- v+m*m+rho*signal$ds-mp*mp
  tf$debugging$assert_positive(vp, message="Invalid finite tilted update: do not silently clip.")
  list(m=mp,v=vp)
}
canonical_mfvi <- function(m,v,gm_like,gv_like,m0,v0,rho) {
  precision <- (1-rho)/v + rho*(1/v0-2*gv_like)
  information <- (1-rho)*m/v + rho*(m0/v0+gm_like-2*m*gv_like)
  tf$debugging$assert_positive(precision)
  list(m=information/precision,v=1/precision)
}

# %% Gaussian MLP: forward and backward messages
# Full Gaussian belief q=N(m,C), with expectation coordinates (m,M=C+mm').
# Encode ordinary adjoints of a MAXIMIZED scalar objective as moment messages.
gaussian_encode <- function(q,g) {
  dm <- q$C %*% g$m
  dM <- tcrossprod(q$m,dm)+tcrossprod(dm,q$m)+2*q$C%*%g$C%*%q$C
  list(dm=as.numeric(dm),dM=dM)
}

# Solve C x=b by Cholesky; no variance floor or model-changing jitter.
chol_solve <- function(C,b) {
  U <- chol(C)
  backsolve(U,forwardsolve(t(U),b))
}

# Decode d_m=C g_m and d_M=m d_m'+d_m m'+2 C G_C C.
gaussian_decode <- function(q,d) {
  V <- d$dM-tcrossprod(q$m,d$dm)-tcrossprod(d$dm,q$m)
  G <- .5*t(chol_solve(q$C,t(chol_solve(q$C,V))))
  list(m=as.numeric(chol_solve(q$C,d$dm)),C=(G+t(G))/2)
}

# Frozen affine layer h = c + A h_parent + B w. One shared weight block
# acts on ALL examples; A/B therefore include cross-example dependence.
# Each tape contains ONLY this local map, for its VJP in the reverse pass.
gaussian_mlp_forward <- function(A,B,c,weights,input) {
  T64<-function(x)tf$constant(x,tf$float64)
  host<-function(x)reticulate::py_to_r(x$numpy())
  ms<-lapply(weights,function(w)T64(matrix(w$m,ncol=1)))
  vs<-lapply(weights,function(w)T64(w$C))
  ops<-lapply(seq_along(weights),function(l)list(parents=l,weights=l,
    A=T64(A[[l]]),B=T64(B[[l]]),offset=T64(matrix(c[[l]],ncol=1))))
  layer<-function(op,ps,wm,wv,pids)moment_affine_gaussian(ps[[1]],
    list(m=wm[[1]],v=wv[[1]]),op$A,op$B,op$offset)
  f<-moment_graph_forward(ops,list(m=T64(matrix(input$m,ncol=1)),v=T64(input$C)),
                          ms,vs,layer,record=TRUE)
  h<-lapply(f$states,function(a)list(m=as.numeric(host(a$m)),C=host(a$v)))
  list(h=h,weights=weights,graph=f,means=ms,variances=vs)
}

# The exact Gaussian normalizer seeds the SAME reverse traversal used by
# nonlinear benchmarks. Encoding after transport is identical to encode /
# decode at each nonsingular boundary, with fewer roundoff-prone solves.
gaussian_mlp_backward <- function(forward,y,R) {
  h<-forward$h;L<-length(forward$weights);q<-h[[L+1]];S<-q$C+R
  gm<-as.numeric(chol_solve(S,y-q$m))
  G<-.5*(tcrossprod(gm)-chol_solve(S,diag(length(gm))))
  a<-moment_graph_reverse(forward$graph,list(
    tf$constant(matrix(gm,ncol=1),tf$float64),tf$constant(G,tf$float64)),
    forward$means,variances=forward$variances)
  host<-function(x)reticulate::py_to_r(x$numpy())
  activation<-vector("list",L+1);weight<-vector("list",L)
  for(l in 1:L) {
    z<-a$boundary_adjoints[[l+1]]
    activation[[l+1]]<-gaussian_encode(h[[l+1]],list(m=host(z[[1]]),C=host(z[[2]])))
    weight[[l]]<-gaussian_encode(forward$weights[[l]],list(m=host(a$gm[[l]]),C=host(a$gv[[l]])))
  }
  list(activation=activation,weight=weight)
}

# Install an exact tilted raw-moment message; the quadratic term is essential.
gaussian_install <- function(q,d) {
  list(m=q$m+d$dm,
    C=q$C+d$dM-tcrossprod(q$m,d$dm)-tcrossprod(d$dm,q$m)-tcrossprod(d$dm))
}

# %% Independent Gaussian conditioning and smoothing
# Independent base-R references. No pseudo-gradient or encode/decode calls.
# First compose one global affine predictor and condition the shared weights.
gaussian_mlp_condition <- function(A,B,c,weights,input,y,R) {
  sizes<-vapply(weights,function(w)length(w$m),integer(1))
  ends<-cumsum(sizes);starts<-c(1L,head(ends,-1)+1L)
  blocks<-Map(seq.int,starts,ends);p<-sum(sizes)
  m<-unlist(lapply(weights,`[[`,'m'));C<-matrix(0,p,p)
  for(l in seq_along(blocks))C[blocks[[l]],blocks[[l]]]<-weights[[l]]$C
  J<-list(matrix(0,length(input$m),p));offset<-list(input$m)
  for(l in seq_along(blocks)) {
    J[[l+1]]<-A[[l]]%*%J[[l]]
    J[[l+1]][,blocks[[l]]]<-J[[l+1]][,blocks[[l]]]+B[[l]]
    offset[[l+1]]<-as.numeric(A[[l]]%*%offset[[l]]+c[[l]])
  }
  H<-tail(J,1)[[1]];b<-tail(offset,1)[[1]];S<-H%*%C%*%t(H)+R
  gain<-t(solve(S,H%*%C))
  mp<-as.numeric(m+gain%*%(y-b-H%*%m));Cp<-C-gain%*%H%*%C
  activation<-lapply(seq_along(J),function(i)
    list(m=as.numeric(offset[[i]]+J[[i]]%*%mp),C=J[[i]]%*%Cp%*%t(J[[i]])))
  weight<-lapply(blocks,function(i)list(m=mp[i],C=Cp[i,i,drop=FALSE]))
  list(activation=activation,weight=weight,mean=mp,covariance=Cp,
       J=J,offset=offset,blocks=blocks,prior_mean=m,prior_covariance=C)
}

# Second reference: forward Gaussian predictions, terminal conditioning,
# then local conditional gains for both the parent activation AND layer weights.
gaussian_mlp_smooth <- function(A,B,c,weights,input,y,R) {
  L<-length(weights);h<-list(input)
  for(l in 1:L)h[[l+1]]<-list(
    m=as.numeric(A[[l]]%*%h[[l]]$m+B[[l]]%*%weights[[l]]$m+c[[l]]),
    C=A[[l]]%*%h[[l]]$C%*%t(A[[l]])+B[[l]]%*%weights[[l]]$C%*%t(B[[l]]))
  activation<-vector('list',L+1);weight<-vector('list',L)
  last<-h[[L+1]];K<-t(solve(last$C+R,last$C))
  activation[[L+1]]<-list(m=as.numeric(last$m+K%*%(y-last$m)),C=last$C-K%*%last$C)
  for(l in L:1) {
    child<-h[[l+1]];post<-activation[[l+1]]
    Kw<-t(solve(child$C,B[[l]]%*%weights[[l]]$C))
    weight[[l]]<-list(m=as.numeric(weights[[l]]$m+Kw%*%(post$m-child$m)),
      C=weights[[l]]$C+Kw%*%(post$C-child$C)%*%t(Kw))
    if(l>1) {
      Kh<-t(solve(child$C,A[[l]]%*%h[[l]]$C))
      activation[[l]]<-list(m=as.numeric(h[[l]]$m+Kh%*%(post$m-child$m)),
        C=h[[l]]$C+Kh%*%(post$C-child$C)%*%t(Kh))
    }
  }
  list(forward=h,activation=activation,weight=weight)
}

