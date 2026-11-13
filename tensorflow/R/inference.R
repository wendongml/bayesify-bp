# %% Probability integration, objective, cavity and KL
# %% Likelihood heads: expected log probability is not log expected probability
head_nodes <- function(state,normals=NULL) {
  if(!is.null(normals))return(tf$expand_dims(state$m,0L)+
    tf$einsum("qr,rbc->qbc",tf$cast(normals,state$m$dtype),tf$stack(state$U)/sqrt(length(state$U))))
  if(!is.null(state$C)) {
    # Softmax is invariant to a shared logit offset. Integrate K-1 contrasts,
    # retaining their FULL covariance, not K independent marginal variances.
    K<-shape(state$m)[2L];d<-K-1L
    D<-cbind(diag(d),-rep(1,d))
    D<-tf$constant(D,dtype=state$m$dtype)
    a<-tf$matmul(state$m,D,transpose_b=TRUE)
    V<-tf$einsum("ij,bjk,lk->bil",D,state$C,D)
    L<-tf$linalg$cholesky(V) # Readout bias variance makes V positive definite.
    nodes<-lapply(seq_len(d),function(j)sqrt(d)*tf$gather(L,as.integer(j-1L),axis=2L))
    logits<-c(lapply(nodes,function(u)a+u),lapply(nodes,function(u)a-u))
    return(tf$stack(lapply(logits,function(z)tf$concat(list(z,
      tf$expand_dims(zero(tf$gather(z,0L,axis=1L)),1L)),axis=1L))))
  }
  # Integrate the represented low-rank Gaussian in its latent coordinates.
  # +/- sqrt(R) axes, equally weighted: exact Gaussian first/second moments.
  R<-length(state$U)
  tf$stack(c(lapply(state$U,function(u)state$m+u),
             lapply(state$U,function(u)state$m-u)),axis=0L)
  # U stores unnormalised columns; U/sqrt(R) times +/-sqrt(R) is +/-U.
}
head_loss <- function(z,y,method="mfvi",power=1) {
  logp<-tf$nn$log_softmax(z,axis=-1L)
  yhot<-tf$one_hot(y,depth=tf$gather(tf$shape(z),2L),dtype=z$dtype)
  ll<-tf$reduce_sum(logp*tf$expand_dims(yhot,0L),axis=-1L)
  if(method=="mfvi") return(-tf$reduce_mean(ll))
  -tf$reduce_mean(tf$reduce_logsumexp(power*ll,axis=0L)-
       tf$math$log(tf$cast(tf$gather(tf$shape(ll),0L),ll$dtype)))/power
}
head_backward <- function(state,y,method,power,normals=NULL) {
  xs<-c(list(state$m),state$U,if(!is.null(state$C))list(state$C))
  with(tf$GradientTape(watch_accessed_variables=FALSE) %as% tape, {
    tape$watch(xs);loss<-head_loss(head_nodes(state,normals),y,method,power)
  })
  list(loss=loss,adj=as_r(tape$gradient(loss,xs,unconnected_gradients=tf$UnconnectedGradients$ZERO)))
}
gaussian_kl <- function(ms,vs,g) {
  Reduce(`+`,lapply(seq_along(ms),function(i) {
    w<-g$weights[[i]]
    tf$reduce_sum((vs[[i]]+tf$square(ms[[i]]-w$m0))/w$v0-1+log(w$v0)-tf$math$log(vs[[i]]))/2
  }))
}
power_cavity <- function(ms,vs,g,N,alpha) {
  k<-alpha/N
  m<-v<-list();logA<-scalar(0,ms[[1]])
  original_dtype<-ms[[1]]$dtype
  use_double<-isTRUE(g$field_float64)
  if(use_double) {
    ms<-lapply(ms,function(a)tf$cast(a,tf$float64));vs<-lapply(vs,function(a)tf$cast(a,tf$float64))
    logA<-tf$cast(logA,tf$float64)
  }
  for(i in seq_along(ms)) {
    w<-g$weights[[i]];e<-k*(vs[[i]]/w$v0-1);d<-1+e
    m0<-tf$cast(w$m0,ms[[i]]$dtype)
    tf$debugging$assert_positive(d,message="Invalid power cavity precision.")
    v[[i]]<-vs[[i]]/d
    m[[i]]<-m0+(1-k)*(ms[[i]]-m0)/d
    logA<-logA-tf$reduce_sum(tf$math$log1p(e)-k*tf$math$log(vs[[i]]/w$v0)+
           k*(1-k)*tf$square(ms[[i]]-m0)/(w$v0*d))/2
  }
  if(use_double) {
    m<-lapply(m,function(a)tf$cast(a,original_dtype));v<-lapply(v,function(a)tf$cast(a,original_dtype))
    logA<-tf$cast(logA,original_dtype)
  }
  list(m=m,v=v,logA=logA)
}

# %% The optimizer recurrences used in the experiments
# %% Optimizer recurrences: posterior variance is not Adam's second accumulator
gaussian_optimizer <- function(g,kind,N,tau,lr,variance_lr=lr,decay=0) {
  # Write the short recurrence explicitly: Keras Adam places epsilon differently
  # from the reference. Vadam uses likelihood E[g^2], not (E[g])^2 or dL/dv.
  initial<-list(m=lapply(g$weights,`[[`,"m"),
    v=if(kind=="vadam")lapply(g$weights,function(w)tf$exp(w$lv)) else NULL)
  n<-length(g$weights);N_eff<-N/tau
  variable<-function(x)tf$Variable(x,trainable=FALSE)
  rate<-variable(scalar(lr,initial$m[[1]]));vrate<-variable(scalar(variance_lr,initial$m[[1]]))
  t<-variable(scalar(0,initial$m[[1]]))
  r<-lapply(initial$m,function(m)variable(zero(m)))
  s<-lapply(seq_len(n),function(i)variable(if(kind=="vadam")
    (1/initial$v[[i]]-1/g$weights[[i]]$v0)/N_eff else zero(initial$m[[i]])))
  likelihood_normalization<-kind=="vadam" && identical(g$vadam_normalization,"likelihood_ema_initial_zero")
  u<-if(likelihood_normalization)lapply(initial$m,function(m)variable(zero(m))) else list()
  rv<-sv<-list()
  if(kind=="ep_adam") {
    rv<-lapply(initial$m,function(m)variable(zero(m)))
    sv<-lapply(initial$m,function(m)variable(zero(m)))
  }
  b1<-.9;b2<-if(kind=="vadam").99 else .999
  if(!is.null(g$optimizer_betas)){b1<-g$optimizer_betas[[1]];b2<-g$optimizer_betas[[2]]}
  variance_epsilon<-if(is.null(g$variance_epsilon))1e-8 else g$variance_epsilon
  step<-function(gm,gv=NULL,g2=NULL) {
    t$assign_add(scalar(1,t))
    for(i in seq_len(n)) {
      w<-g$weights[[i]];gradient<-gm[[i]]
      if(kind=="vadam") {
        lambda<-1/(w$v0*N_eff)
        r[[i]]$assign(b1*r[[i]]+(1-b1)*(gradient+lambda*(w$m-w$m0)))
        s[[i]]$assign(b2*s[[i]]+(1-b2)*g2[[i]])
        if(likelihood_normalization)u[[i]]$assign(b2*u[[i]]+(1-b2)*g2[[i]])
        normalization<-if(likelihood_normalization)u[[i]] else s[[i]]
        w$m$assign_sub(rate*(r[[i]]/(1-b1^t))/(tf$sqrt(normalization/(1-b2^t))+lambda))
        w$lv$assign(-tf$math$log(N_eff*s[[i]]+1/w$v0))
      } else {
        r[[i]]$assign(b1*r[[i]]+(1-b1)*gradient)
        s[[i]]$assign(b2*s[[i]]+(1-b2)*gradient^2)
        w$m$assign((1-rate*decay)*w$m-rate*(r[[i]]/(1-b1^t))/(tf$sqrt(s[[i]]/(1-b2^t))+1e-8))
        if(kind=="ep_adam") {
          gradient<-tf$exp(w$lv)*gv[[i]]
          rv[[i]]$assign(b1*rv[[i]]+(1-b1)*gradient)
          sv[[i]]$assign(b2*sv[[i]]+(1-b2)*gradient^2)
          w$lv$assign_sub(vrate*(rv[[i]]/(1-b1^t))/(tf$sqrt(sv[[i]]/(1-b2^t))+variance_epsilon))
        }
      }
    }
  }
  named<-function(x,prefix)if(length(x))setNames(x,paste0(prefix,seq_along(x))) else list()
  state<-c(list(t=t,lr=rate,variance_lr=vrate),setNames(r,paste0("r",seq_len(n))),
    setNames(s,paste0("s",seq_len(n))),named(rv,"rv"),
    named(sv,"sv"),named(u,"u"))
  list(step=step,learning_rate=rate,variance_learning_rate=vrate,state=state,
       checkpoint=do.call(tf$train$Checkpoint,state),kind=kind)
}

# %% One complete sampled or moment minibatch update
# %% Belief update: local moment reverse traversal, then the reference optimizer
make_update <- function(g,method="mfvi",mode="moment",N,tau=.01,power=1,
                        lr=1e-3,variance_lr=lr,samples=1L,decay=0,capture=FALSE) {
  kind<-if(mode=="point")"adamw" else if(method=="mfvi")"vadam" else "ep_adam"
  optimizer<-gaussian_optimizer(g,kind,N,tau,lr,variance_lr,decay)
  diagnostic<-new.env(parent=emptyenv())
  draw_noise<-function(i,s,like)tf$random$stateless_normal(tf$shape(like),dtype=like$dtype,
    seed=tf$stack(list(tf$constant(as.integer(g$seed+1009L*i)),
      tf$cast(optimizer$state$t,tf$int32)*as.integer(samples)+as.integer(s))))
  mean_statistics<-function(draws)lapply(seq_along(draws[[1]]),function(i)list(
    m=Reduce(`+`,lapply(draws,function(d)d[[i]]$m))/length(draws),
    v=Reduce(`+`,lapply(draws,function(d)d[[i]]$v))/length(draws)))
  step<-function(x,y,noise=NULL) {
    if(mode=="point") {ms<-lapply(g$weights,function(w)tf$identity(w$m));vs<-NULL}
    else {b<-belief(g);ms<-b$m;vs<-b$v}
    n<-length(ms);g2<-NULL;gv<-NULL
    if(mode=="point") {
      with(tf$GradientTape(watch_accessed_variables=FALSE) %as% tape, {
        tape$watch(ms);states<-point_forward(g,x,ms,record=TRUE)
        loss<-tf$reduce_mean(tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,logits=states[[length(states)]]))
      })
      gm<-dense_gradients(tape$gradient(loss,ms));statistics<-norm_statistics(g,states)
    } else if(mode=="sampled" && method=="mfvi") {
      gradients<-losses<-statistics_draws<-list()
      for(s in seq_len(samples)) {
        ws<-lapply(seq_len(n),function(i)ms[[i]]+tf$sqrt(vs[[i]])*
          if(is.null(noise))draw_noise(i,s,ms[[i]]) else noise[[s]][[i]])
        with(tf$GradientTape(watch_accessed_variables=FALSE) %as% tape, {
          tape$watch(ws);states<-point_forward(g,x,ws,record=TRUE)
          loss<-tf$reduce_mean(tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,logits=states[[length(states)]]))
        })
        gradients[[s]]<-dense_gradients(tape$gradient(loss,ws));losses[[s]]<-loss
        statistics_draws[[s]]<-norm_statistics(g,states)
      }
      gm<-lapply(seq_len(n),function(i)Reduce(`+`,lapply(gradients,function(d)d[[i]]))/samples)
      g2<-lapply(seq_len(n),function(i)Reduce(`+`,lapply(gradients,function(d)d[[i]]^2))/samples)
      loss<-Reduce(`+`,losses)/samples;statistics<-mean_statistics(statistics_draws)
    } else {
      with(tf$GradientTape(persistent=TRUE,watch_accessed_variables=FALSE) %as% field_tape, {
        field_tape$watch(c(ms,vs))
        if(method=="ep") {
          cavity<-power_cavity(ms,vs,g,N,alpha=tau*power)
          cm<-cavity$m;cv<-cavity$v;field<- -cavity$logA/power
        } else {cm<-ms;cv<-vs;field<-scalar(0,ms[[1]])}
      })
      if(mode=="moment") {
        if(identical(g$moment_backend,"frozen_shared")) {
          stopifnot(method=="mfvi")
          moments<-frozen_shared_statistics(g,x,cm,cv,y);g2<-moments$g2
          a<-list(gm=moments$gm,gv=lapply(cv,zero));loss<-moments$loss
          statistics<-norm_statistics(g,moments$forward$states,moment=TRUE)
        } else if(identical(g$moment_backend,"frozen_scalar")) {
          stopifnot(method=="mfvi")
          f<-scalar_moment_forward(g,x,cm,cv,record=FALSE,retain_jacobian=TRUE)
          moments<-frozen_gradient_moments(f$mean,f$jacobians,cv,y)
          g2<-moments$g2
          a<-list(gm=moments$gm,gv=lapply(cv,zero))
          loss<-binary_gaussian_head(f$mean,f$variance,y,expected=TRUE)$loss
          statistics<-list()
        } else if(identical(g$moment_backend,"scalar_adjoint")) {
          stopifnot(method=="ep") # The recovered MFVI winner has a different curvature contract.
          f<-scalar_moment_forward(g,x,cm,cv,record=TRUE)
          with(tf$GradientTape(watch_accessed_variables=FALSE) %as% head_tape, {
            head_tape$watch(list(f$mean,f$variance))
            h<-binary_gaussian_head(f$mean,f$variance,y,expected=FALSE)
          })
          hg<-as_r(head_tape$gradient(h$loss,list(f$mean,f$variance)))
          a<-scalar_moment_reverse(f,hg[[1]],hg[[2]]);loss<-h$loss;statistics<-list()
        } else {
          f<-moment_forward(g,x,cm,cv,record=TRUE,need_variance=method=="ep",
            recompute=isTRUE(g$recompute_moments))
          h<-head_backward(f$output,y,method,power,normals=g$head_normals)
          a<-moment_reverse(g,f,h$adj,retain_boundary_adjoints=FALSE);loss<-h$loss
          statistics<-norm_statistics(g,f$states,moment=TRUE)
          a<-list(gm=a$gm,gv=a$gv);f<-NULL;h<-NULL
        }
        # Encode/decode at the persistent Gaussian boundary. Intermediate
        # singular low-rank factors use additive Euclidean adjoints.
        decoded<-lapply(seq_len(n),function(i)decode_diagonal(cm[[i]],cv[[i]],
          encode_diagonal(cm[[i]],cv[[i]],-a$gm[[i]],-a$gv[[i]])))
        gm<-lapply(decoded,function(d)-d$gm);gv<-lapply(decoded,function(d)-d$gv)
        if(method=="mfvi" && is.null(g2)) {
          # Reference moment-Vadam statistic: E[g]^2+(H sqrt(v) e)^2.
          # H is the point categorical-loss Hessian. No posterior oracle.
          if(!isTRUE(as_r(tf$inside_function()))) {gc();import("gc")$collect()}
          with(tf$control_dependencies(c(gm,unlist(statistics,recursive=FALSE))), {
          if(isTRUE(g$recompute_moments)) {
            hd<-point_hvp(g,x,ms,vs,y)
          } else {
          d<-lapply(seq_len(n),function(i)directions(g,i,vs[[i]])[[1]])
          with(tf$autodiff$ForwardAccumulator(ms,d) %as% accumulator, {
            with(tf$GradientTape(watch_accessed_variables=FALSE) %as% tape, {
              tape$watch(ms);z<-point_forward(g,x,ms)
              ell<-tf$reduce_mean(tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,logits=z))
            })
            gradient<-dense_gradients(tape$gradient(ell,ms))
          })
          hd<-dense_gradients(accumulator$jvp(gradient))
          }
          })
          g2<-Map(function(a,h)a^2+h^2,gm,hd)
        }
      } else {
        statistics_draws<-list()
        with(tf$GradientTape(watch_accessed_variables=FALSE) %as% tape, {
          tape$watch(c(cm,cv))
          logits<-lapply(seq_len(samples),function(s) {
            ws<-lapply(seq_len(n),function(i)cm[[i]]+tf$sqrt(cv[[i]])*
              if(is.null(noise))draw_noise(i,s,cm[[i]]) else noise[[s]][[i]])
            states<-point_forward(g,x,ws,record=TRUE)
            statistics_draws[[s]]<<-norm_statistics(g,states)
            states[[length(states)]]
          })
          loss<-head_loss(tf$stack(logits),y,method,power)
        })
        a<-as_r(tape$gradient(loss,c(cm,cv)));gm<-a[seq_len(n)];gv<-a[n+seq_len(n)]
        statistics<-mean_statistics(statistics_draws)
      }
      if(method=="ep") {
        a<-as_r(field_tape$gradient(c(cm,cv),c(ms,vs),output_gradients=c(gm,gv)))
        p<-as_r(field_tape$gradient(field,c(ms,vs)))
        gm<-Map(`+`,a[seq_len(n)],p[seq_len(n)]);gv<-Map(`+`,a[n+seq_len(n)],p[n+seq_len(n)])
        loss<-loss+field
      }
    }
    commit_statistics(g,statistics)
    if(capture) {diagnostic$gm<-gm;diagnostic$gv<-gv;diagnostic$g2<-g2}
    optimizer$step(gm,gv,g2)
    loss
  }
  # The image size is fixed; the minibatch size is not. A declared dynamic
  # batch avoids a second expensive DenseNet trace for the last short batch.
  compiled<-if(identical(g$architecture,"densenet40"))tf$`function`(step,
    input_signature=list(tf$TensorSpec(tuple(NULL,32L,32L,3L),g$weights[[1]]$m$dtype),
                         tf$TensorSpec(tuple(NULL),tf$int32))) else
    tf$`function`(step,reduce_retracing=TRUE)
  list(step=compiled,eager_step=step,
    mean_optimizer=optimizer,
    variance_optimizer=if(kind=="ep_adam")list(learning_rate=optimizer$variance_learning_rate) else NULL,
    checkpoint=optimizer$checkpoint,kind=kind,diagnostic=diagnostic)
}

# %% Predictive probability averaging
# %% Native prediction and shared-network Monte Carlo are separate protocols
predict_probabilities <- function(g,x,mode="moment",samples=32L) {
  if(mode=="point") return(tf$nn$softmax(point_forward(g,x)))
  b<-belief(g)
  if(mode=="moment") {
    if(isTRUE(g$moment_backend %in% c("scalar_adjoint","frozen_scalar"))) {
      f<-scalar_moment_forward(g,x,b$m,b$v,record=FALSE)
      return(binary_gaussian_head(f$mean,f$variance,tf$zeros_like(f$mean),expected=FALSE)$probabilities)
    }
    f<-moment_forward(g,x,b$m,b$v)$output
    return(tf$reduce_mean(tf$nn$softmax(head_nodes(f,normals=g$head_normals)),axis=0L))
  }
  probs<-lapply(seq_len(samples),function(s) {
    # The same draw is a complete network across all evaluation minibatches.
    ws<-lapply(seq_along(b$m),function(i)b$m[[i]]+tf$sqrt(b$v[[i]])*
      tf$random$stateless_normal(tf$shape(b$m[[i]]),dtype=b$m[[i]]$dtype,
        seed=as.integer(c(g$seed+1009L*i,32749L*s))))
    tf$nn$softmax(point_forward(g,x,ws))
  })
  Reduce(`+`,probs)/samples
}

# %% MNIST updates and predictive averaging
# %% Same objectives and optimizer recurrences as the successful MNIST runs
# MFVI minimizes E_q[-log p(y|W)] plus tau/N KL(q||p0). The optimizer's
# gradient-square accumulator is separate from the learned weight variance.
# Power EP uses q_c proportional to q^(1-alpha/N) p0^(alpha/N), alpha=tau=.01,
# -log E_qc[p(y|W)] - log A. The global field -log A is applied once,
# not diluted by the number of minibatches. This is the tied-energy benchmark,
# distinct from the literal finite-site EP checks earlier in the tutorial.
mnist_moment_gradient <- function(g,x,y,ms,vs,method) {
  f<-mnist_forward(g,x,ms,vs,record=TRUE)
  state<-list(m=f$output$m,C=f$output$v,U=list())
  h<-head_backward(state,y,method,1)
  a<-mnist_reverse(g,f,h$adj)
  list(loss=h$loss,gm=a$gm,gv=a$gv)
}
mnist_hvp <- function(g,x,y,ms,vs,probes) {
  d<-Map(function(v,e)tf$sqrt(v)*e,vs,probes)
  with(tf$autodiff$ForwardAccumulator(ms,d) %as% acc,{
    with(tf$GradientTape(watch_accessed_variables=FALSE) %as% tape,{
      tape$watch(ms);z<-mnist_point(g,x,ms)
      ell<-tf$reduce_mean(tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,logits=z))
    })
    gradient<-dense_gradients(tape$gradient(ell,ms))
  })
  dense_gradients(acc$jvp(gradient))
}
mnist_update <- function(g,method,mode,N,lr,variance_lr=lr,refresh=1L) {
  kind<-if(mode=="point")"adamw" else if(method=="mfvi")"vadam" else "ep_adam"
  g$optimizer_betas<-if(mode=="point")c(.9,.99) else NULL
  opt<-gaussian_optimizer(g,kind,N,tau=.01,lr=lr,variance_lr=variance_lr)
  ms<-lapply(g$weights,`[[`,"m");n<-length(ms)
  probes<-lapply(seq_len(n),function(i) {
    if(g$architecture=="lenet")return(2*tf$cast(tf$random$stateless_uniform(tf$shape(ms[[i]]),
      seed=as.integer(c(750L+g$seed,i)),minval=0L,maxval=2L,dtype=tf$int32),ms[[i]]$dtype)-1)
    tf$random$stateless_normal(tf$shape(ms[[i]]),seed=as.integer(c(703L+g$seed,i)),dtype=ms[[i]]$dtype)
  })
  cache<-lapply(ms,function(m)tf$Variable(zero(m),trainable=FALSE))
  noise<-function(i,s,m)tf$random$stateless_normal(tf$shape(m),dtype=m$dtype,
    seed=tf$stack(list(tf$constant(as.integer(g$seed+1700L+19L*i)),
      tf$cast(opt$state$t,tf$int32)*4L+as.integer(s))))
  step<-function(x,y,update_curvature=TRUE) {
    b<-belief(g);ms<-b$m;vs<-b$v;gv<-g2<-NULL
    if(mode=="point") {
      with(tf$GradientTape(watch_accessed_variables=FALSE) %as% tape,{
        tape$watch(ms);z<-mnist_point(g,x,ms)
        loss<-tf$reduce_mean(tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,logits=z))
      });gm<-dense_gradients(tape$gradient(loss,ms))
    } else if(method=="mfvi") {
      if(mode=="sampled") {
        ws<-lapply(seq_len(n),function(i)ms[[i]]+tf$sqrt(vs[[i]])*noise(i,0L,ms[[i]]))
        with(tf$GradientTape(watch_accessed_variables=FALSE) %as% tape,{
          tape$watch(ws);z<-mnist_point(g,x,ws)
          loss<-tf$reduce_mean(tf$nn$sparse_softmax_cross_entropy_with_logits(labels=y,logits=z))
        });gm<-dense_gradients(tape$gradient(loss,ws));g2<-lapply(gm,tf$square)
      } else {
        a<-mnist_moment_gradient(g,x,y,ms,vs,"mfvi");gm<-a$gm;loss<-a$loss
        if(update_curvature) {
          h<-mnist_hvp(g,x,y,ms,vs,probes)
          for(i in seq_len(n))cache[[i]]$assign(h[[i]]^2)
        }
        g2<-Map(function(g,c)g*g+c,gm,cache)
      }
    } else {
      with(tf$GradientTape(persistent=TRUE,watch_accessed_variables=FALSE) %as% field_tape,{
        field_tape$watch(c(ms,vs));cavity<-power_cavity(ms,vs,g,N,alpha=.01)
        cm<-cavity$m;cv<-cavity$v;field<- -cavity$logA
      })
      if(mode=="moment")a<-mnist_moment_gradient(g,x,y,cm,cv,"ep")
      else {
        with(tf$GradientTape(watch_accessed_variables=FALSE) %as% tape,{
          tape$watch(c(cm,cv))
          logits<-lapply(0:3,function(s)mnist_point(g,x,lapply(seq_len(n),
            function(i)cm[[i]]+tf$sqrt(cv[[i]])*noise(i,s,cm[[i]]))))
          loss<-head_loss(tf$stack(logits),y,"ep",1)
        })
        a0<-as_r(tape$gradient(loss,c(cm,cv)))
        a<-list(loss=loss,gm=a0[seq_len(n)],gv=a0[n+seq_len(n)])
      }
      d<-as_r(field_tape$gradient(c(cm,cv),c(ms,vs),output_gradients=c(a$gm,a$gv)))
      p<-as_r(field_tape$gradient(field,c(ms,vs)))
      gm<-Map(`+`,d[seq_len(n)],p[seq_len(n)]);gv<-Map(`+`,d[n+seq_len(n)],p[n+seq_len(n)])
      loss<-a$loss+field
    }
    opt$step(gm,gv,g2);loss
  }
  list(eager=step,step=tf$`function`(function(x,y)step(x,y,TRUE),reduce_retracing=TRUE),
       cached=tf$`function`(function(x,y)step(x,y,FALSE),reduce_retracing=TRUE),optimizer=opt,
       refresh=refresh,cache=cache,probes=probes)
}

# %% Predictive averaging: probabilities, not average logits or average losses
mnist_predict <- function(g,x,mode="moment",draws=16L,seed=192L,components=FALSE) {
  b<-belief(g)
  if(mode=="point")z<-tf$expand_dims(mnist_point(g,x,b$m),0L)
  else if(mode=="moment") {
    a<-mnist_forward(g,x,b$m,b$v)$output
    z<-head_nodes(list(m=a$m,C=a$v,U=list()))
  } else z<-tf$stack(lapply(seq_len(draws),function(s)mnist_point(g,x,
    lapply(seq_along(b$m),function(i)b$m[[i]]+tf$sqrt(b$v[[i]])*
      tf$random$stateless_normal(tf$shape(b$m[[i]]),seed=as.integer(c(seed+1009L*i,s)),dtype=b$m[[i]]$dtype)))))
  p<-tf$nn$softmax(z);average<-tf$reduce_mean(p,axis=0L)
  if(!components)return(average)
  entropy<-function(p)-tf$reduce_sum(p*tf$math$log(tf$maximum(p,scalar(1e-30,p))),axis=-1L)
  list(p=average,entropy=entropy(average),mi=entropy(average)-tf$reduce_mean(entropy(p),axis=0L))
}

# Gaussian readout derivatives without a Gaussian assumption on hidden h.
# For z=h W+b, condition on h and apply Price's identity:
# d_m E[f]=E[h df/dz], d_v E[f]=E[h^2 d^2f/dz^2]/2.
# For log p_y: r=e_y-p, diag(H)=-p(1-p). For p_y replace H by
# p_y*(H+r^2) and r by p_y*r. Average BEFORE EP normalizer division.
# Shapes: features [draw,input], logits [draw,class], derivatives [draw,input,class].
sampled_readout_adjoint <- function(features,logits,label,method="mfvi") {
  lp<-tf$nn$log_softmax(logits);p<-tf$exp(lp)
  r<-tf$one_hot(as.integer(label),shape(logits)[2L],dtype=p$dtype)-p
  H<- -p*(1-p)
  if(method=="mfvi") {value<-tf$gather(lp,as.integer(label),axis=1L);a<-tf$ones_like(value)}
  else if(method=="ep") {value<-tf$gather(p,as.integer(label),axis=1L);a<-value;H<-H+r*r}
  else stop("use mfvi or ep")
  h<-tf$expand_dims(features,-1L);a<-tf$reshape(a,c(-1L,1L,1L))
  list(value=value,gm=a*h*tf$expand_dims(r,1L),gv=.5*a*h*h*tf$expand_dims(H,1L))
}
