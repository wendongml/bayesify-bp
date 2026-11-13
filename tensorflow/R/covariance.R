# %% Shared covariance directions on the common graph
# %% A visible graph, not a framework: nodes, parents and Gaussian parameters
# State = list(m, U): m is a tensor and U is a list of shared covariance columns.
# The SAME column index is used across branches, layers, tokens and examples.
new_graph <- function(rank=4L, seed=41L) {
  g <- new.env(parent=emptyenv())
  g$nodes <- list(); g$weights <- list(); g$rank <- rank; g$seed <- seed
  g
}
weight <- function(g, name, initial, variance=1e-5, prior_variance=1e-3) {
  i <- length(g$weights)+1L
  g$weights[[i]] <- list(name=name, m=tf$Variable(initial, name=paste0(name,"_mean")),
    lv=tf$Variable(tf$ones_like(initial)*log(variance), name=paste0(name,"_log_variance")),
    m0=if(identical(g$prior_center,"zero"))zero(initial) else tf$identity(initial), v0=prior_variance)
  i
}
node <- function(g, name, parents, point, weights=integer(), kind="delta") {
  force(parents);force(weights) # R promises must not mutate the graph mid-registration.
  g$nodes[[length(g$nodes)+1L]] <- list(name=name, parents=parents,
    point=point, weights=weights, kind=kind)
  length(g$nodes)+1L # state 1 is the observed input
}
belief <- function(g) list(m=lapply(g$weights,function(w) tf$identity(w$m)),
                          v=lapply(g$weights,function(w) tf$exp(w$lv)))
directions <- function(g,i,v) {
  if(!is.null(g$fixed_directions))return(lapply(g$fixed_directions[[i]],function(e)tf$sqrt(v)*e))
  lapply(seq_len(g$rank),function(r) {
    e <- 2*tf$cast(tf$random$stateless_uniform(tf$shape(v),
      seed=as.integer(c(g$seed+1009L*i,7919L*r)),minval=0L,maxval=2L,dtype=tf$int32),v$dtype)-1
    tf$sqrt(v)*e
  })
}
covariance <- function(U) {
  # Outputs [batch,classes]; no input-by-input covariance is materialized.
  Reduce(`+`,lapply(U,function(u) tf$einsum("bi,bj->bij",u,u)))/length(U)
}
gaussian_relu <- function(m,U) moment_relu_shared(m,U)
local_moments <- function(op,states,ws,ds,rank,vs=list()) {
  # Identity views make local argument slots distinct for ForwardAccumulator;
  # their outer derivatives still accumulate at the one shared graph parent.
  ms <- lapply(states,function(s)tf$identity(s$m));ws<-lapply(ws,tf$identity)
  if(op$kind=="relu") return(gaussian_relu(ms[[1]],states[[1]]$U))
  if(op$kind=="gaussian_readout") {
    # For z=h W+b with independent Gaussian readout weights:
    # Cov(z)=W_mean' Cov(h) W_mean+diag(E[h^2]' Var(W)+Var(b)).
    # Integrating this small head is exact for the supplied hidden moments;
    # a first-order JVP omits Var(h)*Var(W) and sketches readout uncertainty.
    U<-lapply(states[[1]]$U,function(u)tf$matmul(u,ws[[1]]))
    h2<-ms[[1]]^2+Reduce(`+`,lapply(states[[1]]$U,tf$square))/rank
    C<-covariance(U)+tf$linalg$diag(tf$matmul(h2,vs[[1]])+vs[[2]])
    return(list(m=op$point(ms,ws),U=U,C=C))
  }
  mean <- op$point(ms,ws)
  U <- lapply(seq_len(rank),function(r) {
    # This is a local JVP, not a second sampled nonlinear network.
    primals <- c(ms,ws)
    tangents <- c(lapply(states,function(s) s$U[[r]]),lapply(ds,function(d) d[[r]]))
    with(tf$autodiff$ForwardAccumulator(primals,tangents) %as% acc, {
      y <- op$point(ms,ws)
    })
    # Token indices and masks are observed, non-differentiable inputs. Their
    # unconnected tangent is exactly zero, not a Python None carried onward.
    acc$jvp(y,unconnected_gradients=tf$UnconnectedGradients$ZERO)
  })
  list(m=mean,U=U)
}
point_forward <- function(g,x,ws=NULL,record=FALSE) {
  if(is.null(ws)) ws <- lapply(g$weights,`[[`,"m")
  states <- list(x)
  for(op in g$nodes) states[[length(states)+1L]] <- op$point(states[op$parents],ws[op$weights])
  if(record) states else states[[length(states)]]
}
moment_forward <- function(g,x,ms=NULL,vs=NULL,record=FALSE,need_variance=TRUE,
                           point_linearization=FALSE,recompute=FALSE) {
  rank<-if(point_linearization)1L else g$rank
  if(is.null(ms)){b<-belief(g);ms<-b$m;vs<-b$v}
  layer<-function(op,ps,wm,wv,pids) {
    if(point_linearization)op$kind<-"delta"
    if(any(vapply(ps,function(a)!is.null(a$C),logical(1))))
      stop("Full class covariance is a terminal boundary.")
    # op weights may repeat; directions belong to the one underlying parameter.
    ds<-lapply(seq_along(op$weights),function(k)directions(g,op$weights[k],wv[[k]]))
    local_moments(op,ps,wm,ds,rank,wv)
  }
  f<-moment_graph_forward(g$nodes,list(m=x,U=replicate(rank,zero(x),simplify=FALSE)),
    ms,vs,layer,record,need_variance,recompute)
  f$rank<-rank;f
}
moment_reverse <- function(g,forward,child,retain_boundary_adjoints=TRUE) {
  moment_graph_reverse(forward,child,lapply(g$weights,`[[`,"m"),retain_boundary_adjoints)
}

# Hessian-vector multiplication by the same local forward/reverse interface.
point_hvp <- function(g,x,ms,vs,y) {
  f<-moment_forward(g,x,ms,vs,record=TRUE,need_variance=FALSE,
                    point_linearization=TRUE,recompute=TRUE)
  z<-f$output$m;dz<-f$output$U[[1]];p<-tf$nn$softmax(z)
  batch<-tf$cast(tf$gather(tf$shape(z),0L),z$dtype)
  gz<-(p-tf$one_hot(y,tf$gather(tf$shape(z),1L),dtype=z$dtype))/batch
  hz<-p*(dz-tf$reduce_sum(p*dz,axis=1L,keepdims=TRUE))/batch
  moment_reverse(g,f,list(hz,gz),retain_boundary_adjoints=FALSE)$gm
}

# %% Diagonal image-layer covariance approximation
# %% A true diagonal activation baseline, evaluated at the SAME learned weights
diagonal_image_layer<-function(op,parents,wm,wv,pids=NULL) {
  m<-lapply(parents,`[[`,"m");v<-lapply(parents,`[[`,"v")
    mean<-op$point(m,wm)
    if(op$kind=="conv") {
      # Var(sum_i X_i W_i), with independent inputs/weights in this closure.
      variance<-op$point(v,lapply(wm,tf$square))+
        op$point(list(v[[1]]+m[[1]]^2),wv)
    } else if(op$kind=="bn") {
      s2<-op$normalization$v+1e-5;a<-(m[[1]]-op$normalization$m)/tf$sqrt(s2)
      t<-v[[1]]/s2
      variance<-wm[[1]]^2*t+wv[[1]]*(t+a^2)+wv[[2]]
    } else if(op$kind=="relu") {
      z<-moment_relu(list(m=m[[1]],v=v[[1]]));mean<-z$m;variance<-z$v
    } else if(op$kind=="gaussian_readout") {
      z<-moment_dense(parents[[1]],list(m=wm[[1]],v=wv[[1]]),list(m=wm[[2]],v=wv[[2]]))
      mean<-z$m;variance<-z$v
    } else if(grepl("_sum$",op$name)) {
      # Distinct residual branches are approximated as independent here.
      # A literally repeated parent is still ONE variable: Var(2X)=4 Var(X).
      ids<-unique(op$parents);count<-tabulate(match(op$parents,ids),nbins=length(ids))
      variance<-Reduce(`+`,lapply(seq_along(ids),function(i)count[i]^2*parents[[match(ids[i],op$parents)]]$v))
    } else if(grepl("_concat$|_shortcut$",op$name)) {
      variance<-op$point(v,list())
    } else if(op$name=="global_pool") {
      variance<-op$point(v,list())/prod(shape(m[[1]])[2:3])
    } else if(grepl("_pool$",op$name)) {
      variance<-op$point(v,list())/4 # nonoverlapping2x2 average pooling
    } else stop("Declare a diagonal moment rule before using layer: ",op$name)
  list(m=mean,v=variance)
}

diagonal_image_forward<-function(g,x,ms=NULL,vs=NULL) {
  stopifnot(!isTRUE(g$training))
  if(is.null(ms)){b<-belief(g);ms<-b$m;vs<-b$v}
  moment_graph_forward(g$nodes,list(m=x,v=zero(x)),ms,vs,
    diagonal_image_layer,record=FALSE)$states
}

diagonal_fidelity<-function(g,x,reference,layers) {
  states<-diagonal_image_forward(g,x)
  do.call(rbind,lapply(seq_along(layers),function(k) {
    n<-ncol(reference$samples[[k]]);z<-states[[layers[k]]]
    m<-as.numeric(value(tf$reshape(z$m,tuple(-1L))))[seq_len(n)]
    v<-as.numeric(value(tf$reshape(z$v,tuple(-1L))))[seq_len(n)]
    C<-diag(v,nrow=n);ref<-reference$moments[[k]]
    data.frame(node=layers[k],name=reference$table$name[k],draws=nrow(reference$samples[[k]]),
      mean_relative=sqrt(sum((m-ref$mc_mean)^2))/max(sqrt(sum(ref$mc_mean^2)),1e-30),
      covariance_relative=norm(C-ref$mc_covariance,"F")/max(norm(ref$mc_covariance,"F"),1e-30),
      scope="Diagonal activations throughout; independent distinct branches/spatial positions")
  }))
}

# %% Image-gradient covariance calculation
# %% Exact execution rearrangement of the same image moment approximation
# No new covariance closure, optimizer, weight belief, or quadrature rule.
# This serving path packs the mean and activation directions into one conv.
image_moment_forward_packed<-function(g,x,ms=NULL,vs=NULL) {
  stopifnot(g$architecture %in% c("resnet20","densenet40"),!isTRUE(g$training))
  if(is.null(ms)){b<-belief(g);ms<-b$m;vs<-b$v}
  states<-list(list(m=x,U=replicate(g$rank,zero(x),simplify=FALSE)))
  for(op in g$nodes) {
    ps<-states[op$parents];ww<-ms[op$weights];vv<-vs[op$weights]
    ds<-lapply(seq_along(op$weights),function(k)directions(g,op$weights[[k]],vv[[k]]))
    if(op$kind=="conv") {
      # conv is bilinear: d conv(X,W)=conv(dX,W)+conv(X,dW).
      # Pack only the first two terms' shared-W operation; FLOPs are unchanged.
      packed<-tf$concat(c(list(ps[[1]]$m),ps[[1]]$U),axis=0L)
      pieces<-as_r(tf$split(op$point(list(packed),ww),g$rank+1L,axis=0L))
      z<-list(m=pieces[[1]],U=lapply(seq_len(g$rank),function(r)
        pieces[[r+1L]]+op$point(list(ps[[1]]$m),list(ds[[1]][[r]]))))
    } else if(op$kind=="bn") {
      # Frozen running statistics; uncertain gamma/beta. This is the SAME
      # first-order product rule, not the independent-product variance formula.
      s<-tf$math$rsqrt(op$normalization$v+1e-5)
      centered<-ps[[1]]$m-op$normalization$m
      z<-list(m=op$point(list(ps[[1]]$m),ww),U=lapply(seq_len(g$rank),function(r)
        s*(ww[[1]]*ps[[1]]$U[[r]]+centered*ds[[1]][[r]])+ds[[2]][[r]]))
    } else if(op$kind=="relu") {
      z<-gaussian_relu(ps[[1]]$m,ps[[1]]$U)
    } else if(op$kind=="gaussian_readout") {
      z<-local_moments(op,ps,ww,ds,g$rank,vv)
    } else {
      # CIFAR's remaining nodes are add, concatenate, subsampling, padding,
      # average pooling and global averaging: deterministic linear maps.
      stopifnot(length(op$weights)==0L)
      z<-list(m=op$point(lapply(ps,`[[`,"m"),list()),
        U=lapply(seq_len(g$rank),function(r)op$point(lapply(ps,function(p)p$U[[r]]),list())))
    }
    states[[length(states)+1L]]<-z
  }
  list(output=states[[length(states)]],states=states)
}
image_probabilities_packed<-function(g,x) {
  z<-image_moment_forward_packed(g,x)$output
  tf$reduce_mean(tf$nn$softmax(head_nodes(z,normals=g$head_normals)),axis=0L)
}

# Freeze the existing direction VALUES before XLA: changing its random-number
# implementation would change the approximation rather than just its execution.
# Compile once per learned belief/architecture; measure compilation separately.
compiled_image_predictor<-function(g,mode="moment",xla=TRUE) {
  stopifnot(mode %in% c("point","moment"),!isTRUE(g$training))
  if(mode=="moment") {
    b<-belief(g)
    g$fixed_directions<-lapply(seq_along(b$v),function(i)
      lapply(directions(g,i,b$v[[i]]),tf$sign))
  }
  f<-if(mode=="moment")function(x)image_probabilities_packed(g,x) else
    function(x)predict_probabilities(g,x,"point")
  tf$`function`(f,jit_compile=xla,reduce_retracing=TRUE)
}

# %% Scalar readout moment and sensitivity approximation
# %% Scalar-output covariance: retain every first-order weight direction
# For the binary contrast z=f_positive-f_negative, Var(z) is sum(J_l^2 v_l).
# A reverse sensitivity contracts this sum without an input-by-input matrix.
# Below, each little operation has its own VJP. Reversing this calculation
# differentiates the variance as well as the mean; no whole-model gradient is
# substituted for the candidate's explicit message traversal.
scalar_moment_forward<-function(g,x,ms,vs,record=TRUE,retain_jacobian=FALSE) {
  n<-length(ms);values<-c(ms,vs,list(x));tapes<-list()
  add<-function(parents,fn) {
    ids<-unique(parents);ix<-match(parents,ids);inputs<-values[ids]
    if(record) {
      with(tf$GradientTape(watch_accessed_variables=FALSE) %as% tape, {
        tape$watch(inputs);out<-fn(inputs[ix])
      })
    } else out<-fn(inputs[ix])
    new<-length(values)+seq_along(out)
    values<<-c(values,out)
    if(record)tapes[[length(tapes)+1L]]<<-list(tape=tape,inputs=inputs,outputs=out,parents=ids,children=new)
    new
  }
  # Forward means at the current weight means.
  boundary<-c(2L*n+1L)
  for(op in g$nodes) {
    np<-length(op$parents);nw<-length(op$weights)
    boundary<-c(boundary,add(c(boundary[op$parents],op$weights),function(a)
      list(op$point(a[seq_len(np)],if(nw)a[np+seq_len(nw)] else list()))))
  }
  mean_id<-add(tail(boundary,1),function(a)list(tf$gather(a[[1]],1L,axis=1L)-tf$gather(a[[1]],0L,axis=1L)))
  signal<-rep(NA_integer_,length(boundary))
  signal[[length(boundary)]]<-add(tail(boundary,1),function(a)
    list(tf$ones_like(a[[1]])*tf$constant(c(-1,1),dtype=a[[1]]$dtype)))
  contributions<-integer();jacobian_ids<-rep(NA_integer_,n)
  # Backward sensitivities, adding all contributions at a shared parent.
  for(j in rev(seq_along(g$nodes))) {
    op<-g$nodes[[j]];sid<-signal[[j+1L]]
    if(is.na(sid))next
    if(op$kind=="lora") {
      stopifnot(length(op$parents)==1L,length(op$weights)==2L)
      A<-op$weights[[1]];B<-op$weights[[2]];scale<-op$lora_scale
      answer<-add(c(boundary[op$parents],A,B,sid,n+A,n+B),function(a) {
        H<-a[[1]];W_A<-a[[2]];W_B<-a[[3]];S<-a[[4]]
        SB<-tf$matmul(S,W_B,transpose_b=TRUE)
        J_A<-scale*tf$einsum("bti,btj->bij",H,SB)
        J_B<-scale*tf$einsum("bti,btj->bij",tf$matmul(H,W_A),S)
        variance<-tf$reduce_sum(J_A^2*a[[5]],axis=c(1L,2L))+
                  tf$reduce_sum(J_B^2*a[[6]],axis=c(1L,2L))
        parent<-tf$matmul(S,op$frozen_weight,transpose_b=TRUE)+
                scale*tf$matmul(SB,W_A,transpose_b=TRUE)
        if(retain_jacobian)list(parent,variance,J_A,J_B) else list(parent,variance)
      })
      parent_signals<-answer[[1]];contributions<-c(contributions,answer[[2]])
      if(retain_jacobian) {jacobian_ids[[A]]<-answer[[3]];jacobian_ids[[B]]<-answer[[4]]}
    } else {
      stopifnot(length(op$weights)==0L)
      # Transformers have no cross-example operations here. The VJP of the
      # batched map therefore supplies one sensitivity per input example.
      ids<-unique(op$parents);ix<-match(op$parents,ids);k<-length(ids)
      parent_signals<-add(c(boundary[ids],sid),function(a) {
        parents<-a[seq_len(k)]
        with(tf$GradientTape(watch_accessed_variables=FALSE) %as% pull, {
          pull$watch(parents);out<-op$point(parents[ix],list())
        })
        as_r(pull$gradient(out,parents,output_gradients=a[[k+1L]],
          unconnected_gradients=tf$UnconnectedGradients$ZERO))
      })
      # Repeated parents were differentiated together; do not count twice.
      op$parents<-ids
    }
    for(k in seq_along(op$parents)) {
      p<-op$parents[[k]];s<-parent_signals[[k]]
      signal[[p]]<-if(is.na(signal[[p]]))s else
        add(c(signal[[p]],s),function(a)list(a[[1]]+a[[2]]))
    }
  }
  stopifnot(length(contributions)>0L)
  variance_id<-add(contributions,function(a)list(Reduce(`+`,a)))
  list(mean=values[[mean_id]],variance=values[[variance_id]],mean_id=mean_id,
       variance_id=variance_id,values=values,tapes=tapes,n=n,
       jacobians=if(retain_jacobian)values[jacobian_ids] else NULL)
}
scalar_moment_reverse<-function(f,gm,gv) {
  adj<-vector("list",length(f$values));adj[[f$mean_id]]<-gm;adj[[f$variance_id]]<-gv
  for(a in rev(f$tapes)) {
    child<-lapply(a$children,function(i)if(is.null(adj[[i]]))zero(f$values[[i]]) else adj[[i]])
    d<-as_r(a$tape$gradient(a$outputs,a$inputs,output_gradients=child,
      unconnected_gradients=tf$UnconnectedGradients$ZERO))
    for(k in seq_along(a$parents)) {
      i<-a$parents[[k]];adj[[i]]<-if(is.null(adj[[i]]))d[[k]] else adj[[i]]+d[[k]]
    }
  }
  get<-function(i)if(is.null(adj[[i]]))zero(f$values[[i]]) else adj[[i]]
  list(gm=lapply(seq_len(f$n),get),gv=lapply(f$n+seq_len(f$n),get))
}

# %% Binary Gaussian head: integrate probabilities, not logits
# Gauss-Hermite nodes integrate the declared scalar Gaussian contrast. They
# are quadrature points, not sampled whole networks or independent token logits.
binary_gaussian_head<-function(m,v,y,expected=FALSE,order=20L) {
  rule<-as_r(np$polynomial$hermite$hermgauss(as.integer(order)))
  nodes<-tf$constant(as.numeric(rule[[1]])*sqrt(2),dtype=m$dtype)
  weights<-tf$constant(as.numeric(rule[[2]])/sqrt(pi),dtype=m$dtype)
  z<-tf$expand_dims(m,1L)+tf$expand_dims(tf$sqrt(v),1L)*nodes
  signed<-tf$expand_dims(2*tf$cast(y,m$dtype)-1,1L)*z
  logp<--tf$math$softplus(-signed)
  loss<-if(expected)-tf$reduce_mean(tf$reduce_sum(logp*weights,axis=1L)) else
    -tf$reduce_mean(tf$reduce_logsumexp(logp+tf$math$log(weights),axis=1L))
  positive<-tf$reduce_sum(tf$math$sigmoid(z)*weights,axis=1L)
  list(loss=loss,probabilities=tf$stack(list(1-positive,positive),axis=1L))
}

# %% Frozen-Jacobian MFVI: correlated minibatch gradient moments, not E[g]^2
gaussian_residual_moments<-function(m,C,y,order=20L) {
  # r_i=sigmoid(z_i)-y_i. The SAME uncertain weights generate every z_i.
  # Hence E[g_j^2]=J_j' E[rr'] J_j/B^2, including cross-example terms.
  rule<-as_r(np$polynomial$hermite$hermgauss(as.integer(order)))
  t<-tf$constant(as.numeric(rule[[1]])*sqrt(2),dtype=m$dtype)
  w<-tf$constant(as.numeric(rule[[2]])/sqrt(pi),dtype=m$dtype)
  sd<-tf$sqrt(tf$maximum(tf$linalg$diag_part(C),zero(m)))
  outer<-tf$einsum("i,j->ij",sd,sd)
  correlation<-tf$where(outer>0,C/tf$maximum(outer,scalar(1e-30,outer)),zero(outer))
  tf$debugging$assert_less_equal(tf$reduce_max(tf$abs(correlation)),scalar(1+1e-5,m))
  correlation<-tf$clip_by_value(correlation,scalar(-1,m),scalar(1,m))
  residual<-tf$math$sigmoid(tf$expand_dims(m,1L)+tf$expand_dims(sd,1L)*t)-
              tf$expand_dims(tf$cast(y,m$dtype),1L)
  first<-tf$reduce_sum(residual*w,axis=1L)
  # Shape [i,j,t,u]; integrate the bivariate Gaussian for each example pair.
  rho<-tf$expand_dims(tf$expand_dims(correlation,2L),3L)
  tj<-tf$reshape(t,c(1L,1L,order,1L));uj<-tf$reshape(t,c(1L,1L,1L,order))
  zj<-tf$reshape(m,c(1L,-1L,1L,1L))+tf$reshape(sd,c(1L,-1L,1L,1L))*
       (rho*tj+tf$sqrt(tf$maximum(1-rho^2,zero(rho)))*uj)
  rj<-tf$math$sigmoid(zj)-tf$reshape(tf$cast(y,m$dtype),c(1L,-1L,1L,1L))
  second<-tf$reduce_sum(tf$expand_dims(tf$expand_dims(residual,1L),3L)*rj*
    tf$reshape(w,c(1L,1L,order,1L))*tf$reshape(w,c(1L,1L,1L,order)),axis=c(2L,3L))
  second<-(second+tf$transpose(second))/2
  second<-tf$linalg$set_diag(second,tf$reduce_sum(residual^2*w,axis=1L))
  list(first=first,second=second)
}
frozen_gradient_moments<-function(m,J,vs,y,order=20L) {
  # J is recomputed at EACH current mean, then held fixed for this update.
  # Only the small minibatch covariance and contractions use float64.
  B<-tf$gather(tf$shape(m),0L);Bd<-tf$cast(B,tf$float64)
  flat<-lapply(J,function(j)tf$cast(tf$reshape(j,tf$stack(list(B,-1L))),tf$float64))
  vv<-lapply(vs,function(v)tf$cast(tf$reshape(v,tuple(-1L)),tf$float64))
  C<-Reduce(`+`,Map(function(j,v)tf$matmul(j*v,j,transpose_b=TRUE),flat,vv))
  r<-gaussian_residual_moments(tf$cast(m,tf$float64),C,y,order)
  gm<-Map(function(j,v)tf$reshape(tf$linalg$matvec(j,r$first,transpose_a=TRUE)/Bd,tf$shape(v)),flat,vs)
  g2<-Map(function(j,v)tf$reshape(tf$reduce_sum(j*tf$matmul(r$second,j),axis=0L)/(Bd^2),tf$shape(v)),flat,vs)
  maximum<-tf$reduce_max(tf$stack(lapply(g2,function(a)tf$reduce_max(tf$abs(a)))))
  minimum<-tf$reduce_min(tf$stack(lapply(g2,tf$reduce_min)))
  tf$debugging$assert_greater_equal(minimum,-tf$maximum(scalar(1e-24,maximum),scalar(1e-10,maximum)*maximum),
    message="Gaussian quadrature produced a materially negative gradient second moment.")
  list(gm=Map(function(a,v)tf$cast(a,v$dtype),gm,vs),
       g2=Map(function(a,v)tf$cast(tf$maximum(a,zero(a)),v$dtype),g2,vs),C=C)
}

# %% Frozen sensitivity second moments
# %% Frozen local Jacobians: propagate several output sensitivities on one DAG
frozen_graph_reverse<-function(g,states,ms,seeds) {
  # Each seed is an output adjoint. The first gives E[g]; the others give
  # J' H_loss J delta_w. No derivative of J is included in this closure.
  ns<-length(seeds);adj<-lapply(seeds,function(a) {
    v<-vector("list",length(states));v[[length(v)]]<-a;v
  })
  gradients<-lapply(seeds,function(a)lapply(ms,zero))
  for(j in rev(seq_along(g$nodes))) {
    op<-g$nodes[[j]];parents<-unique(op$parents);ix<-match(op$parents,parents)
    parameters<-unique(op$weights);wi<-match(op$weights,parameters)
    inputs<-c(lapply(states[parents],`[[`,"m"),ms[parameters]);np<-length(parents);nw<-length(parameters)
    with(tf$GradientTape(persistent=TRUE,watch_accessed_variables=FALSE) %as% tape, {
      tape$watch(inputs)
      output<-op$point(inputs[ix],if(nw)inputs[np+wi] else list())
    })
    for(s in seq_len(ns)) {
      if(is.null(adj[[s]][[j+1L]]))next
      d<-dense_gradients(tape$gradient(output,inputs,output_gradients=adj[[s]][[j+1L]],
                            unconnected_gradients=tf$UnconnectedGradients$ZERO))
      for(k in seq_along(parents)) {
        p<-parents[[k]]
        adj[[s]][[p]]<-if(is.null(adj[[s]][[p]]))d[[k]] else adj[[s]][[p]]+d[[k]]
      }
      for(k in seq_along(parameters)) {
        p<-parameters[[k]];gradients[[s]][[p]]<-gradients[[s]][[p]]+d[[np+k]]
      }
    }
  }
  gradients
}
frozen_shared_statistics<-function(g,x,ms,vs,y) {
  f<-moment_forward(g,x,ms,vs,record=FALSE)
  h<-head_backward(f$output,y,"mfvi",1,normals=g$head_normals)
  p<-tf$nn$softmax(f$output$m);B<-tf$cast(tf$gather(tf$shape(p),0L),p$dtype)
  # Hessian of categorical cross-entropy: diag(p)-p p'.
  variation<-lapply(f$output$U,function(u) {
    u<-u/sqrt(g$rank)
    p*(u-tf$reduce_sum(p*u,axis=1L,keepdims=TRUE))/B
  })
  gradients<-frozen_graph_reverse(g,f$states,ms,c(list(h$adj[[1]]),variation))
  first<-gradients[[1]]
  second<-lapply(seq_along(ms),function(i)first[[i]]^2+
    Reduce(`+`,lapply(gradients[-1],function(a)a[[i]]^2)))
  list(gm=first,g2=second,loss=h$loss,forward=f)
}

# %% Gaussian integral plus a paired nonlinear residual
# A Gaussian expectation plus a paired nonlinear residual.
#
# Let q = N(m,diag(v)), h = log p(y|W) for VI, or h = p(y|W) for EP.
# Write W=m+sqrt(v)*epsilon. Freeze a linear surrogate h0 and an integration
# proposal r(epsilon) at this snapshot. With phi the standard-normal density,
# E_phi[h] = E_phi[h0] + E_r[(phi/r)(h-h0)], also for pathwise m,v derivatives.
# No fitted multiplier is used. Sampling approximates only the residual term.
# This is a stochastic correction to moment propagation, not a deterministic
# Gaussian closure and not a new mixture distribution over network weights.
#
# Inputs use rows = integration draws, columns = parameter coordinates.
# gaussian$gm and gaussian$gv integrate derivatives of E_q[h0], before EP log.
# nonlinear / surrogate store log likelihood, pathwise gm and gv of LOG p.
# weight is the known Gaussian target/proposal density ratio (one for q draws).
# The same Gaussian noise must be used in both paths; adapt the surrogate only
# between complete forward/backward passes. EP corrects Z and dZ, then divides.
residual_backward <- function(m, v, gaussian, nonlinear, surrogate,
                              weight=rep(1,nrow(nonlinear$gm)), method="mfvi") {
  stopifnot(method %in% c("mfvi","ep"), all(v>0), all(weight>=0))
  n <- length(weight)
  stopifnot(nrow(nonlinear$gm)==n, identical(dim(nonlinear$gm),dim(surrogate$gm)))
  a <- if(method=="ep") exp(nonlinear$log_likelihood) else rep(1,n)
  b <- if(method=="ep") exp(surrogate$log_likelihood) else rep(1,n)
  # R recycles a row-length vector down each matrix column, exactly as needed.
  gm <- gaussian$gm+colMeans(weight*(a*nonlinear$gm-b*surrogate$gm))
  gv <- gaussian$gv+colMeans(weight*(a*nonlinear$gv-b*surrogate$gv))
  Z <- if(method=="ep") gaussian$Z+mean(weight*(
    expm1(nonlinear$log_likelihood)-expm1(surrogate$log_likelihood))) else 1
  stopifnot(is.finite(Z),Z>0,all(is.finite(gm)),all(is.finite(gv)))
  gm <- gm/Z; gv <- gv/Z
  dm <- v*gm; centered <- 2*v*v*gv
  list(gm=gm,gv=gv,dm=dm,ds=2*m*dm+centered,centered=centered,Z=Z)
}

# Gaussian information geometry is unchanged by the integration rule.
# EP updates raw moments: v+ = v + rho*(ds-2*m*dm) - rho^2*dm^2.
# VI updates canonical parameters, with the prior counted once.
residual_ep_update <- function(m,v,s,rho=1) {
  out <- list(m=m+rho*s$dm,v=v+rho*s$centered-rho^2*s$dm^2)
  stopifnot(all(out$v>0));out
}
residual_mfvi_update <- function(m,v,s,m0,v0,rho) {
  tau <- (1-rho)/v+rho*(1/v0-2*s$gv)
  h <- (1-rho)*m/v+rho*(m0/v0+s$gm-2*m*s$gv)
  stopifnot(all(tau>0));list(m=h/tau,v=1/tau)
}

# A variance diagnostic that does not borrow the mean term in ds.
# v_i(a_b)=exp(a_b)*v_i within block b gives
# dPhi/da_b=sum_i v_i*dPhi/dv_i. Its Fisher information is n_b/2,
# so its natural signal is (2/n_b)*sum_i v_i*dPhi/dv_i.
# This tests a block scale; it does NOT certify every individual dv_i.
variance_scale_signal <- function(v,gv,block) {
  sums <- tapply(v*gv,block,sum);sizes <- table(block)
  2*sums/as.numeric(sizes[names(sums)])
}

# %% Apply the residual identity to one network
# Fixed-belief nonlinear residual integration for the existing tutorial graph.
# Requires geometry_and_graph.R and backward_residual.R. No optimizer or sites
# are changed here. The linear surrogate is frozen for the complete pass.

# z0(W)=f(m)+J(W-m) is Gaussian: E[z0]=f(m), Cov(z0)=sum J_i diag(v_i) J_i'.
linear_gaussian_surrogate <- function(g,x,b) {
  with(tf$GradientTape(persistent=TRUE,watch_accessed_variables=FALSE) %as% tape, {
    tape$watch(b$m);f<-point_forward(g,x,b$m)
    classes<-as_r(tf$unstack(tf$squeeze(f,0L)))
  })
  gradients<-lapply(classes,function(z)as_r(tape$gradient(z,b$m,
    unconnected_gradients=tf$UnconnectedGradients$ZERO)))
  K<-length(classes)
  J<-lapply(seq_along(b$m),function(i)tf$stack(lapply(gradients,function(d)
    tf$reshape(d[[i]],c(-1L)))))
  C<-Reduce(`+`,Map(function(A,v)tf$einsum("ki,li,i->kl",A,A,
    tf$reshape(v,c(-1L))),J,b$v))
  list(m=tf$cast(tf$squeeze(f,0L),tf$float64),C=tf$cast(C,tf$float64),
       J=J,K=K)
}

# Gaussian integration is only K-1 dimensional. The proposal in standard-normal
# coordinates is r(e)=(phi(e)+phi(e-delta))/2. Its exact ratio phi/r is <=2.
# delta points toward the strongest competing logit; this resolves a very small
# likelihood correction without changing q. E[grad_z h] and E[Hessian_z h]/2
# give the ordinary mean and covariance adjoints (Gaussian derivative identity).
surrogate_head <- function(s,y,method,power=18L,seed=173L) {
  K<-s$K;other<-setdiff(0:(K-1L),y);D<-diag(K)[other+1L,,drop=FALSE]-
    matrix(diag(K)[y+1L,],K-1L,K,byrow=TRUE)
  D<-tf$constant(D,tf$float64);a<-tf$linalg$matvec(D,s$m)
  L<-tf$linalg$cholesky(tf$matmul(tf$matmul(D,s$C),D,transpose_b=TRUE))
  competitor<-which.max(as.numeric(value(a)))-1L
  delta<-tf$gather(L,as.integer(competitor))
  scipy<-import("scipy",convert=FALSE)
  u<-scipy$stats$qmc$Sobol(as.integer(K-1L),scramble=TRUE,seed=as.integer(seed))$random_base2(as.integer(power-1L))
  e<-tf$constant(scipy$special$ndtri(u),tf$float64)
  e<-tf$concat(list(e,e+delta),0L)
  r<-2*tf$math$sigmoid(tf$reduce_sum(delta*delta)/2-tf$linalg$matvec(e,delta))
  contrasts<-a+tf$matmul(e,L,transpose_b=TRUE)
  z<-tf$matmul(contrasts,tf$one_hot(as.integer(other),K,dtype=tf$float64))
  lp<-tf$nn$log_softmax(z);p<-tf$exp(lp)
  score<-tf$one_hot(as.integer(y),K,dtype=tf$float64)-p
  H<-tf$einsum("ni,nj->nij",p,p)-tf$linalg$diag(p)
  py<-tf$gather(p,as.integer(y),axis=1L)
  if(method=="ep") {
    Z<-1-tf$reduce_mean(r*(-tf$math$expm1(tf$gather(lp,as.integer(y),axis=1L))))
    H<-tf$expand_dims(tf$expand_dims(py,-1L),-1L)*(H+tf$einsum("ni,nj->nij",score,score))
    score<-tf$expand_dims(py,-1L)*score
  } else Z<-tf$constant(1,tf$float64)
  gm<-tf$reduce_mean(tf$expand_dims(r,-1L)*score,0L)
  GC<-tf$reduce_mean(tf$expand_dims(tf$expand_dims(r,-1L),-1L)*H,0L)/2
  # The natural signal is encoded only after correcting the expectation.
  list(Z=Z,gm=lapply(s$J,function(J)tf$linalg$matvec(tf$cast(J,tf$float64),gm,transpose_a=TRUE)),
       gv=lapply(s$J,function(J)tf$einsum("ik,ij,jk->k",tf$cast(J,tf$float64),GC,tf$cast(J,tf$float64))),
       competitor=other[competitor+1L])
}

# Each iteration pairs nonlinear and linear responses to the SAME perturbation.
# EP uses p*grad(log p) in the numerator, then divides by its corrected Z.
# MFVI uses grad(log p) directly. These two heads must never be interchanged.
network_residual_backward <- function(g,x,y,method="mfvi",pairs=512L,seed=1700001L,
                                     head_power=18L,head_seed=173L) {
  b<-belief(g);s<-linear_gaussian_surrogate(g,x,b)
  h<-surrogate_head(s,y,method,head_power,head_seed)
  shift<-Map(function(J,v)tf$reshape(tf$sqrt(tf$reshape(v,c(-1L)))*
    (tf$gather(J,as.integer(h$competitor))-tf$gather(J,as.integer(y))),tf$shape(v)),s$J,b$v)
  shift2<-Reduce(`+`,lapply(shift,function(d)tf$reduce_sum(tf$cast(d,tf$float64)^2)))
  dm<-dv<-lapply(h$gm,tf$zeros_like);dZ<-tf$constant(0,tf$float64)
  for(k in 0:(pairs-1L)) for(component in 0:1) for(sign in c(1,-1)) {
    e<-lapply(seq_along(b$m),function(i)sign*tf$random$stateless_normal(tf$shape(b$m[[i]]),
      seed=as.integer(c(seed+k,1009L*i+6053L+41L)))+component*shift[[i]])
    dot<-Reduce(`+`,Map(function(a,d)tf$reduce_sum(tf$cast(a,tf$float64)*tf$cast(d,tf$float64)),e,shift))
    importance<-2*tf$math$sigmoid(shift2/2-dot)
    ws<-Map(function(m,v,e)m+tf$sqrt(v)*e,b$m,b$v,e)
    with(tf$GradientTape(watch_accessed_variables=FALSE) %as% tape, {
      tape$watch(ws);z<-tf$cast(point_forward(g,x,ws),tf$float64)
      ell<-tf$gather(tf$squeeze(tf$nn$log_softmax(z),0L),as.integer(y))
    })
    gradient<-as_r(tape$gradient(ell,ws,unconnected_gradients=tf$UnconnectedGradients$ZERO))
    dz<-Reduce(`+`,Map(function(J,v,e)tf$linalg$matvec(J,tf$reshape(tf$sqrt(v)*e,c(-1L))),s$J,b$v,e))
    lp0<-tf$nn$log_softmax(s$m+tf$cast(dz,tf$float64))
    ell0<-tf$gather(lp0,as.integer(y));gz<-tf$one_hot(as.integer(y),s$K,dtype=tf$float64)-tf$exp(lp0)
    a<-if(method=="ep")tf$exp(ell) else tf$constant(1,tf$float64)
    a0<-if(method=="ep")tf$exp(ell0) else tf$constant(1,tf$float64)
    for(i in seq_along(dm)) {
      gm<-tf$cast(tf$reshape(gradient[[i]],c(-1L)),tf$float64)
      gm0<-tf$linalg$matvec(tf$cast(s$J[[i]],tf$float64),gz,transpose_a=TRUE)
      residual<-importance*(a*gm-a0*gm0)
      dm[[i]]<-dm[[i]]+residual/(4*pairs)
      dv[[i]]<-dv[[i]]+residual*tf$cast(tf$reshape(e[[i]],c(-1L)),tf$float64)/
        (2*tf$sqrt(tf$cast(tf$reshape(b$v[[i]],c(-1L)),tf$float64)))/(4*pairs)
    }
    if(method=="ep")dZ<-dZ+importance*(tf$math$expm1(ell)-tf$math$expm1(ell0))/(4*pairs)
  }
  Z<-h$Z+dZ;tf$debugging$assert_positive(Z)
  gm<-Map(function(a,b)(a+b)/Z,h$gm,dm);gv<-Map(function(a,b)(a+b)/Z,h$gv,dv)
  signals<-lapply(seq_along(gm),function(i)encode_diagonal(
    tf$cast(tf$reshape(b$m[[i]],c(-1L)),tf$float64),
    tf$cast(tf$reshape(b$v[[i]],c(-1L)),tf$float64),gm[[i]],gv[[i]]))
  list(gm=gm,gv=gv,signals=signals,Z=Z)
}

