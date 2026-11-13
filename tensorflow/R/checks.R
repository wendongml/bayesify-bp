# %% Independent numerical check: message_equivalence
check_gaussian <- function(out=file.path(OUT,"message_equivalence")) {
# Exact linearized MNIST network: three shared uncertain weight blocks.
# Run from the supplement root: Rscript R/gaussian_mlp_check.R [output_dir]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
library(reticulate)
tf<-import('tensorflow',convert=FALSE)
set.seed(20261106L)
data<-read.csv('advanced_results/message_equivalence/mnist_first16_8x8.csv')[1:8,]
X<-as.matrix(data[,-1]);N<-nrow(X);widths<-c(64L,12L,8L,10L)
target<-diag(10)[data$label+1L,,drop=FALSE];y<-as.numeric(t(target))
block_diag<-function(blocks) {
  nr<-vapply(blocks,nrow,integer(1));nc<-vapply(blocks,ncol,integer(1))
  z<-matrix(0,sum(nr),sum(nc));r<-c(0,cumsum(nr));s<-c(0,cumsum(nc))
  for(i in seq_along(blocks))z[(r[i]+1):r[i+1],(s[i]+1):s[i+1]]<-blocks[[i]]
  z
}

# Analytic frozen Jacobians of tanh(W_l [h;1]); the last layer is linear.
# Vectorize each weight matrix row by row. One B_l uses the SAME block for
# every input. Offsets ensure f_lin(x,W0)=f(x,W0), including nonzero biases.
A<-B<-offset<-weights<-checkpoint<-list();H<-X
for(l in 1:3) {
  di<-widths[l];do<-widths[l+1];W<-matrix(rnorm(do*(di+1),sd=.55/sqrt(di)),do)
  pre<-cbind(H,1)%*%t(W);nextH<-if(l<3)tanh(pre) else pre
  D<-if(l<3)1-nextH^2 else matrix(1,N,do)
  A[[l]]<-block_diag(lapply(1:N,function(i)diag(D[i,],do)%*%W[,1:di,drop=FALSE]))
  B[[l]]<-do.call(rbind,lapply(1:N,function(i)
    diag(D[i,],do)%*%kronecker(diag(do),matrix(c(H[i,],1),nrow=1))))
  w0<-as.numeric(t(W));p<-length(w0)
  offset[[l]]<-as.numeric(t(nextH))-as.numeric(A[[l]]%*%as.numeric(t(H))+B[[l]]%*%w0)
  U<-matrix(rnorm(p*2)/sqrt(p),p,2)
  weights[[l]]<-list(m=w0+rnorm(p,sd=.025),C=.025*diag(p)+.01*tcrossprod(U))
  checkpoint[[l]]<-w0;H<-nextH
}
input<-list(m=as.numeric(t(X)),C=matrix(0,length(X),length(X)))
R<-.8*diag(length(y))
saveRDS(list(A=A,B=B,c=offset,weights=weights,input=input,y=y,R=R,
             checkpoint=checkpoint,base_output=as.numeric(t(H))),file.path(out,'gaussian_mlp_exact_fixture.rds'))

# Candidate: layerwise forward moments, explicit reverse VJPs, moment encoding.
f<-gaussian_mlp_forward(A,B,offset,weights,input)
d<-gaussian_mlp_backward(f,y,R)
# Independent references: global shared-weight conditioning and local smoothing.
dense<-gaussian_mlp_condition(A,B,offset,weights,input,y,R)
smooth<-gaussian_mlp_smooth(A,B,offset,weights,input,y,R)
checks<-pairs<-list()
check<-function(name,a,b,atol=1e-10,rtol=1e-8) {
  err<-max(abs(a-b));tol<-atol+rtol*max(abs(b))
  checks[[length(checks)+1L]]<<-data.frame(quantity=name,max_abs=err,tolerance=tol,status=if(err<=tol)'PASS' else 'FAIL')
}
record<-function(kind,block,q,post,message) {
  dm<-post$m-q$m
  dM<-post$C+tcrossprod(post$m)-q$C-tcrossprod(q$m)
  # The figure retains EVERY mean and MARGINAL second moment, parallel to
  # diagonal EP/MFVI. Full within-block second matrices are checked below.
  for(k in 1:2)pairs[[length(pairs)+1L]]<<-data.frame(method='gaussian_mlp_exact',
    quantity=paste(kind,c('mean message','second-moment message')[k]),block=block,
    coordinate=seq_along(dm),message=if(k==1)dm else diag(dM),
    backprop=if(k==1)message$dm else diag(message$dM))
  check(paste(kind,block,'full mean message'),message$dm,dm)
  check(paste(kind,block,'full second message'),message$dM,dM)
  installed<-gaussian_install(q,message)
  check(paste(kind,block,'conditioned mean'),installed$m,post$m)
  check(paste(kind,block,'conditioned covariance'),installed$C,post$C)
}
for(l in 1:3) {
  check(paste('forward mean',l),f$h[[l+1]]$m,smooth$forward[[l+1]]$m)
  check(paste('forward covariance',l),f$h[[l+1]]$C,smooth$forward[[l+1]]$C)
  for(k in c('m','C')) {
    check(paste('activation dense/smoother',l,k),smooth$activation[[l+1]][[k]],dense$activation[[l+1]][[k]])
    check(paste('weight dense/smoother',l,k),smooth$weight[[l]][[k]],dense$weight[[l]][[k]])
  }
  record('Activation',paste0('h',l),f$h[[l+1]],smooth$activation[[l+1]],d$activation[[l+1]])
  record('Weight',paste0('W',l,' + bias'),weights[[l]],smooth$weight[[l]],d$weight[[l]])
}

# Independent Euclidean autodiff through the entire moment graph.
T64<-function(x)tf$constant(x,tf$float64)
z<-unlist(lapply(weights,function(w)list(T64(matrix(w$m,ncol=1)),T64(w$C))),recursive=FALSE)
with(tf$GradientTape(persistent=TRUE) %as% tape, {
  tape$watch(z);m<-T64(matrix(input$m,ncol=1));C<-T64(input$C)
  for(l in 1:3) {
    a<-T64(A[[l]]);b<-T64(B[[l]])
    m<-tf$matmul(a,m)+tf$matmul(b,z[[2*l-1]])+T64(matrix(offset[[l]],ncol=1))
    C<-tf$matmul(tf$matmul(a,C),a,transpose_b=TRUE)+tf$matmul(tf$matmul(b,z[[2*l]]),b,transpose_b=TRUE)
  }
  e<-T64(matrix(y,ncol=1))-m;L<-tf$linalg$cholesky(C+T64(R))
  objective<--.5*(tf$reduce_sum(e*tf$linalg$cholesky_solve(L,e))+2*tf$reduce_sum(tf$math$log(tf$linalg$diag_part(L))))
})
g<-lapply(py_to_r(tape$gradient(objective,z)),function(x)py_to_r(x$numpy()))
for(l in 1:3) {
  decoded<-gaussian_decode(weights[[l]],d$weight[[l]])
  check(paste('global ordinary autodiff mean',l),decoded$m,g[[2*l-1]])
  check(paste('global ordinary autodiff covariance',l),decoded$C,g[[2*l]])
}

# Structural controls: checkpoint offset, cross-example and cross-layer terms.
base<-input$m
for(l in 1:3)base<-as.numeric(A[[l]]%*%base+B[[l]]%*%checkpoint[[l]]+offset[[l]])
check('frozen predictor equals nonlinear checkpoint',base,as.numeric(t(H)))
cross_example<-max(abs(f$h[[4]]$C[1:10,11:20]))
cross_layer<-max(abs(dense$covariance[dense$blocks[[1]],dense$blocks[[2]]]))
stopifnot(cross_example>1e-8,cross_layer>1e-8)
q<-weights[[1]];step<-d$weight[[1]]
wrong<-q$C+step$dM-tcrossprod(q$m,step$dm)-tcrossprod(step$dm,q$m)
negative_error<-max(abs(wrong-dense$weight[[1]]$C))
stopifnot(negative_error>1e-8) # Missing -dm dm' must be detected.
condition<-vapply(f$h[-1],function(q)kappa(q$C,exact=TRUE),numeric(1))
allchecks<-do.call(rbind,checks);points<-do.call(rbind,pairs)
write.csv(allchecks,file.path(out,'gaussian_mlp_exact_message_checks.csv'),row.names=FALSE)
points$message<-sprintf('%.17g',points$message);points$backprop<-sprintf('%.17g',points$backprop)
write.csv(points,file.path(out,'gaussian_mlp_exact_messages.csv'),row.names=FALSE)
write.csv(data.frame(layer=1:3,activation_condition_number=condition),file.path(out,'gaussian_mlp_exact_conditioning.csv'),row.names=FALSE)
write.csv(data.frame(cross_example_covariance_max=cross_example,cross_layer_weight_covariance_max=cross_layer,
  missing_raw_second_quadratic_term_error=negative_error),file.path(out,'gaussian_mlp_exact_controls.csv'),row.names=FALSE)
print(allchecks);print(condition)
stopifnot(all(allchecks$status=='PASS'))
cat('Exact Gaussian MLP:',nrow(allchecks),'checks PASS;',nrow(points),'plotted coordinate pairs.\n')
}

# %% Independent numerical check: shared_layers
check_messages <- function(out=file.path(OUT,"shared_layers")) {
# Teaching experiment using the SAME candidate layers as the benchmarks.
# Run: Rscript R/shared_messages_check.R [output_directory]
library(reticulate)
tf<-import("tensorflow",convert=FALSE)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
data<-read.csv("advanced_results/message_equivalence/mnist_first16_8x8.csv")
x<-tf$constant(as.matrix(data[,-1]),tf$float64)
y<-tf$one_hot(as.integer(data$label),10L,dtype=tf$float64)
rows<-coordinates<-list()
check<-function(method,kind,block,a,b) {
  a<-as.numeric(py_to_r(a$numpy()));b<-as.numeric(py_to_r(b$numpy()))
  error<-max(abs(a-b));tol<-1e-10+1e-8*max(abs(b))
  rows[[length(rows)+1L]]<<-data.frame(method=method,quantity=kind,block=block,
    max_abs=error,tolerance=tol,status=if(error<=tol)"PASS" else "FAIL")
  if(grepl("message$",kind))coordinates[[length(coordinates)+1L]]<<-data.frame(
    method=method,quantity=kind,block=block,coordinate=seq_along(a),
    message=sprintf("%.17g",a),backprop=sprintf("%.17g",b))
}
for(method in c("ep","mfvi")) {
  tf$random$set_seed(20261106L)
  g<-mfvi_mlp(c(64L,12L,8L,10L))
  literal<-ep_mfvi_messages(g,x,y,method,"literal")
  candidate<-ep_mfvi_messages(g,x,y,method,"projected")
  check(method,"objective",0,candidate$objective,literal$objective)
  for(j in 2:5) {
    a<-literal$messages$activations[[j]];b<-candidate$messages$activations[[j]]
    check(method,"Activation mean message",j,a$dm,b$dm)
    check(method,"Activation second-moment message",j,a$ds,b$ds)
    check(method,"Activation centered-second",j,a$c,b$c)
  }
  for(j in seq_along(g$weights)) {
    a<-literal$messages$weights[[j]];b<-candidate$messages$weights[[j]]
    check(method,"Weight mean message",j,a$dm,b$dm)
    check(method,"Weight second-moment message",j,a$ds,b$ds)
    check(method,"Weight centered-second",j,a$c,b$c)
    q<-candidate$forward$weights[[j]]
    if(method=="mfvi") {
      aa<-mfvi_canonical(q,a,g$weights[[j]]$prior,1e-4)
      bb<-mfvi_canonical(q,b,g$weights[[j]]$prior,1e-4)
    } else {aa<-ep_install_message(q,a,1e-4);bb<-ep_install_message(q,b,1e-4)}
    check(method,"Updated weight mean",j,aa$m,bb$m)
    check(method,"Updated weight variance",j,aa$v,bb$v)
  }
  # Ordinary differentiation of the declared scalar objective is the third path.
  variables<-unlist(lapply(g$weights,function(w)list(w$m,w$v)),recursive=FALSE)
  with(tf$GradientTape() %as% tape,{
    tape$watch(variables);f<-mfvi_run(g,x,"literal")
    objective<-gaussian_objective_message(f$output,y,method)$value
  })
  ad<-py_to_r(tape$gradient(objective,variables))
  for(j in seq_along(g$weights)) {
    d<-mfvi_decode(candidate$forward$weights[[j]],candidate$messages$weights[[j]])
    check(method,"Global AD mean",j,d$a,ad[[2*j-1]])
    check(method,"Global AD variance",j,d$b,ad[[2*j]])
  }
}
rows<-do.call(rbind,rows);points<-do.call(rbind,coordinates)
write.csv(rows,file.path(out,"message_checks.csv"),row.names=FALSE)
for(method in c("ep","mfvi"))write.csv(points[points$method==method,],
    file.path(out,paste0(method,"_messages.csv")),row.names=FALSE)
stopifnot(all(rows$status=="PASS"))
cat("SHARED_LAYERS_MESSAGES",nrow(rows),"PASS; maximum discrepancy",max(rows$max_abs),"\n")
}

