# %% Affine, convolution, activation and merge moments
# Shared numerical layers for the tutorial AND the training examples.
# TensorFlow tensors are supplied by the caller; sourcing this file has no
# installation, download, model construction, or training side effects.
# Inference objectives (EP versus MFVI) do not change these forward equations.

# Exact frozen affine Gaussian map, used by the linear-network proof example.
# Here v stores the full covariance. x and w are independent before observing y.
moment_affine_gaussian <- function(x,w,A,B,offset)list(
  m=tf$matmul(A,x$m)+tf$matmul(B,w$m)+offset,
  v=tf$matmul(tf$matmul(A,x$v),A,transpose_b=TRUE)+
    tf$matmul(tf$matmul(B,w$v),B,transpose_b=TRUE))

# Independent affine factors: E[XW]=m A and
# Var(XW)=v A^2+(m^2+v) S. Inputs need not be Gaussian for this identity.
# Bias has its own mean/variance; an absent bias is exactly deterministic zero.
moment_dense <- function(x,w,bias=NULL,full=FALSE) {
  m<-tf$matmul(x$m,w$m)
  noise<-tf$matmul(x$m*x$m+x$v,w$v)
  if(!is.null(bias)){m<-m+bias$m;noise<-noise+bias$v}
  if(full)return(list(m=m,v=tf$einsum("bi,ic,id->bcd",x$v,w$m,w$m)+tf$linalg$diag(noise)))
  list(m=m,v=tf$matmul(x$v,w$m*w$m)+noise)
}

# Convolution is the same independent-product identity with convolution in
# place of matrix multiplication. Spatial cross covariance is projected out.
moment_conv2d <- function(x,w,bias=NULL,strides=c(1L,1L,1L,1L),padding="SAME") {
  conv<-function(a,b)tf$nn$conv2d(a,b,strides=as.integer(strides),padding=padding)
  m<-conv(x$m,w$m);v<-conv(x$v,w$m*w$m)+conv(x$m*x$m+x$v,w$v)
  if(!is.null(bias)){m<-m+bias$m;v<-v+bias$v}
  list(m=m,v=v)
}

# Gaussian ReLU integrals. P=Phi(m/s), p=phi(m/s), s=sqrt(v):
# M=mP+sp; S=(m^2+v)P+msp; V=S-M^2. Zero variance follows the point rule.
# The optional tail cutoff is the existing shared-direction approximation's
# numerical branch, not a change to the diagonal Gaussian moment formula.
moment_relu_terms <- function(m,v,tail=Inf) {
  positive<-v>0;s<-tf$sqrt(tf$where(positive,v,tf$ones_like(v)))
  central<-positive & (tf$abs(m)<=tail*s)
  a<-tf$where(central,m,tf$zeros_like(m))
  b<-tf$where(central,s,tf$ones_like(s));t<-a/b
  P<-.5*tf$math$erfc(-t/sqrt(2));p<-tf$exp(-t*t/2)/sqrt(2*pi)
  M<-a*P+b*p;V<-tf$maximum((a*a+v)*P+a*b*p-M*M,tf$zeros_like(v))
  list(m=tf$where(central,M,tf$nn$relu(m)),
       v=tf$where(central,V,tf$zeros_like(v)),
       gate=tf$where(central,P,tf$cast(m>0,m$dtype)))
}
moment_relu <- function(x) {
  a<-moment_relu_terms(x$m,x$v);list(m=a$m,v=a$v)
}

# Average pooling is linear: m'=A m, v'=(A elementwise squared) v.
moment_average_pool <- function(x,k=2L)list(
  m=tf$nn$avg_pool2d(x$m,as.integer(k),as.integer(k),"VALID"),
  v=tf$nn$avg_pool2d(x$v,as.integer(k),as.integer(k),"VALID")/(k*k))
moment_flatten <- function(x) {
  dims<-tf$stack(list(tf$gather(tf$shape(x$m),0L),tf$constant(-1L)))
  list(m=tf$reshape(x$m,dims),v=tf$reshape(x$v,dims))
}
# Distinct parents: discard their cross covariance only in the diagonal option.
# A repeated parent remains ONE random variable, so Var(X+X)=4 Var(X).
moment_add <- function(x,y,same=FALSE) {
  if(same)return(list(m=2*x$m,v=4*x$v))
  list(m=x$m+y$m,v=x$v+y$v)
}
moment_concat <- function(x,y)list(m=tf$concat(list(x$m,y$m),-1L),
                                  v=tf$concat(list(x$v,y$v),-1L))

# Correlated option: columns of U/sqrt(K) are shared across branches.
# ReLU uses its Gaussian mean and the stated first-order covariance closure.
# This is deliberately NOT called the exact Gaussian ReLU covariance.
moment_relu_shared <- function(m,U) {
  v<-Reduce(`+`,lapply(U,tf$square))/length(U)
  double<-identical(reticulate::py_to_r(m$dtype$name),"float64")
  tiny<-if(double)2.2250738585072014e-308 else 1.1754943508222875e-38
  limit<-if(double)40 else 12
  positive<-v>tf$constant(tiny,dtype=v$dtype)
  s<-tf$sqrt(tf$where(positive,v,tf$ones_like(v)))
  central<-positive & (tf$abs(m)<=limit*s)
  a<-tf$where(central,m,tf$zeros_like(m));b<-tf$where(central,s,tf$ones_like(s))
  t<-a/b;P<-.5*tf$math$erfc(-t/sqrt(2));p<-tf$exp(-.5*t*t)/sqrt(2*pi)
  mean<-tf$where(central,a*P+b*p,tf$nn$relu(m))
  gate<-tf$where(central,P,tf$cast(m>0,m$dtype))
  list(m=mean,U=lapply(U,function(u)gate*u))
}

# %% Cached forward maps and explicit reverse traversal
# ONE cached forward / reverse traversal for diagonal and correlated states.
# A state is (m,v), or (m,U[,C]); U columns remain shared across graph branches.
# The numerical layer map is independent of the likelihood and optimizer.

moment_values <- function(x) {
  if(!is.null(x$v))return(list(x$m,x$v))
  c(list(x$m),x$U,if(!is.null(x$C))list(x$C))
}
moment_restore <- function(a,template) {
  if(!is.null(template$v))return(list(m=a[[1]],v=a[[2]]))
  k<-length(template$U);z<-list(m=a[[1]],U=a[1L+seq_len(k)])
  if(!is.null(template$C))z$C<-a[[k+2L]]
  z
}

# Freeze each local forward snapshot. Recompute optionally trades memory for
# repeating the SAME local equations, using the cached inputs and no updates.
moment_cache <- function(inputs,evaluate,recompute=FALSE,persistent=FALSE) {
  if(recompute)return(list(inputs=inputs,evaluate=evaluate,
                           state=evaluate(inputs),recompute=TRUE))
  with(tf$GradientTape(persistent=persistent,watch_accessed_variables=FALSE) %as% tape,{
    tape$watch(inputs);z<-evaluate(inputs)
  })
  list(inputs=inputs,state=z,tape=tape,recompute=FALSE)
}
moment_pullback <- function(cache,adjoint) {
  if(cache$recompute) {
    with(tf$control_dependencies(adjoint),{
      with(tf$GradientTape(watch_accessed_variables=FALSE) %as% tape,{
        tape$watch(cache$inputs);z<-cache$evaluate(cache$inputs)
      })
      d<-tape$gradient(moment_values(z),cache$inputs,output_gradients=adjoint,
                      unconnected_gradients=tf$UnconnectedGradients$ZERO)
    })
  } else d<-cache$tape$gradient(moment_values(cache$state),cache$inputs,
             output_gradients=adjoint,unconnected_gradients=tf$UnconnectedGradients$ZERO)
  lapply(reticulate::py_to_r(d),tf$convert_to_tensor)
}

# 'layer' supplies only the selected local moment formula. Parameter variance
# can be held fixed for a mean-only update without silently changing the map.
moment_graph_forward <- function(ops,input,ms,vs,layer,record=FALSE,
                                 need_variance=TRUE,recompute=FALSE) {
  states<-list(input);cache<-list()
  for(op in ops) {
    ids<-unique(op$parents);ix<-match(op$parents,ids)
    pids<-unique(op$weights);wi<-match(op$weights,pids)
    templates<-states[ids];sizes<-vapply(templates,function(s)length(moment_values(s)),integer(1))
    starts<-c(0L,cumsum(sizes));nx<-sum(sizes);nw<-length(pids)
    inputs<-c(unlist(lapply(templates,moment_values),recursive=FALSE),ms[pids],
              if(need_variance)vs[pids])
    evaluate<-local({op<-op;ix<-ix;wi<-wi;pids<-pids;templates<-templates
      starts<-starts;sizes<-sizes;nx<-nx;nw<-nw;fixed_v<-vs[pids]
      function(a) {
        ps<-lapply(seq_along(templates),function(i)
          moment_restore(a[starts[i]+seq_len(sizes[i])],templates[[i]]))
        wm<-a[nx+seq_len(nw)]
        wv<-if(need_variance)a[nx+nw+seq_len(nw)] else fixed_v
        layer(op,ps[ix],wm[wi],wv[wi],pids)
      }
    })
    if(record) {
      cc<-moment_cache(inputs,evaluate,recompute)
      cc$parents<-ids;cc$parameters<-pids;cc$starts<-starts;cc$sizes<-sizes
      cc$nx<-nx;cc$need_variance<-need_variance
      cache[[length(cache)+1L]]<-cc;z<-cc$state
    } else z<-evaluate(inputs)
    states[[length(states)+1L]]<-z
  }
  list(output=tail(states,1L)[[1]],states=states,cache=cache,recompute=recompute)
}

# Decode the objective's signal before this traversal and encode the returned
# adjoints at the persistent Gaussian beliefs afterwards. Their linearity at
# one snapshot means shared-parent contributions SUM; beliefs never multiply.
moment_graph_reverse <- function(forward,child,ms,retain=TRUE,variances=NULL) {
  adj<-vector("list",length(forward$states));adj[[length(adj)]]<-child
  gm<-lapply(ms,tf$zeros_like)
  gv<-lapply(if(is.null(variances))ms else variances,tf$zeros_like)
  for(j in rev(seq_along(forward$cache))) {
    cc<-forward$cache[[j]];a<-adj[[j+1L]];if(is.null(a))next
    d<-moment_pullback(cc,a)
    if(!retain)adj[j+1L]<-list(NULL)
    for(k in seq_along(cc$parents)) {
      p<-cc$parents[[k]];v<-d[cc$starts[k]+seq_len(cc$sizes[k])]
      adj[[p]]<-if(is.null(adj[[p]]))v else Map(`+`,adj[[p]],v)
    }
    n<-length(cc$parameters)
    for(k in seq_len(n)) {
      p<-cc$parameters[[k]];gm[[p]]<-gm[[p]]+d[[cc$nx+k]]
      if(cc$need_variance)gv[[p]]<-gv[[p]]+d[[cc$nx+n+k]]
    }
  }
  list(gm=gm,gv=gv,boundary_adjoints=adj)
}

