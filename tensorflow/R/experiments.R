# %% Public MNIST data, complete training and evaluation
# Public data only. The network code below is R/TensorFlow; PyTorch is used here
# solely for the archived split generator and the authors' .pt dataset format.
mnist_data <- function(root="data") {
  dir.create(root,recursive=TRUE,showWarnings=FALSE)
  torch<-import("torch",convert=FALSE);vision<-import("torchvision.datasets",convert=FALSE)
  dirty<-import("ddu_dirty_mnist",convert=FALSE)
  a<-vision$MNIST(root,train=TRUE,download=TRUE)
  b<-vision$MNIST(root,train=FALSE,download=TRUE)
  c<-vision$FashionMNIST(root,train=FALSE,download=TRUE)
  save<-function(name,array)np$save(file.path(root,paste0(name,".npy")),array)
  save("clean_train_x",a$data$numpy()$astype("float32")/255)
  save("clean_train_y",a$targets$numpy()$astype("int32"))
  save("clean_test_x",b$data$numpy()$astype("float32")/255)
  save("clean_test_y",b$targets$numpy()$astype("int32"))
  save("fashion_x",c$data$numpy()$astype("float32")/255)
  # Exactly the archived 55k/5k stratified partition, reconstructed from its seed.
  generator<-torch$Generator()$manual_seed(20260928L);fit<-validation<-list()
  for(label in 0:9) {
    ids<-torch$nonzero(a$targets==as.integer(label))$flatten()
    ids<-ids$index_select(0L,torch$randperm(ids$numel(),generator=generator))
    validation[[label+1L]]<-ids$narrow(0L,0L,500L)
    fit[[label+1L]]<-ids$narrow(0L,500L,as.integer(as_r(ids$numel())-500L))
  }
  fit<-torch$cat(fit);validation<-torch$cat(validation)
  fit<-fit$index_select(0L,torch$randperm(fit$numel(),generator=generator))
  validation<-validation$index_select(0L,torch$randperm(validation$numel(),generator=generator))
  save("fit",fit$numpy()$astype("int32"));save("validation",validation$numpy()$astype("int32"))
  # Keep the original image scale and disable added noise for this extension.
  dirty$AmbiguousMNIST$mirrors<-list("https://github.com/BlackHC/ddu_dirty_mnist/releases/download/data-v1.0.0/")
  d<-dirty$AmbiguousMNIST(root,train=TRUE,download=TRUE,normalize=FALSE,noise_stddev=0)
  e<-dirty$AmbiguousMNIST(root,train=FALSE,download=TRUE,normalize=FALSE,noise_stddev=0)
  save("ambiguous_train_x",d$data$squeeze(1L)$numpy())
  save("ambiguous_train_y",d$targets$numpy()$astype("int32"))
  # Test predictions are evaluated once per unique image, then scored against
  # its ten labels; identical images do not multiply OOD ROC sample counts.
  ids<-torch$arange(0L,60000L,10L)
  save("ambiguous_test_x",e$data$index_select(0L,ids)$squeeze(1L)$numpy())
  save("ambiguous_test_y",e$targets$reshape(tuple(6000L,10L))$numpy()$astype("int32"))
  groups<-as.integer(as_r(np$random$default_rng(20261107L)$permutation(6000L)))
  expand<-function(ids)as.integer(as.vector(t(outer(10L*ids,0:9,`+`))))
  save("ambiguous_fit",np$array(expand(groups[1:5500]),dtype="int32"))
  save("ambiguous_validation",np$array(expand(groups[5501:6000]),dtype="int32"))
  invisible(root)
}

# Full schedule; paired means and separate training/evaluation clocks.
mnist_fit <- function(arch="mlp",dataset="clean",label="moment_mfvi",seed=41L,
                      epochs=if(arch=="mlp")40L else 30L) {
  mode<-if(label=="ordinary")"point" else strsplit(label,"_")[[1]][1]
  method<-if(grepl("ep",label))"ep" else "mfvi"
  tag<-paste(arch,dataset,label,seed,sep="_");folder<-file.path("results",tag)
  dir.create(folder,recursive=TRUE,showWarnings=FALSE)
  load<-function(name)tf$convert_to_tensor(np$load(file.path("data",paste0(name,".npy"))))
  reshape<-function(x)if(arch=="mlp")tf$reshape(x,tuple(-1L,784L)) else
    tf$pad(tf$expand_dims(x,-1L),rbind(c(0L,0L),c(2L,2L),c(2L,2L),c(0L,0L)))
  allx<-load("clean_train_x");ally<-load("clean_train_y")
  fit<-load("fit");val<-load("validation")
  x<-tf$gather(allx,fit);y<-tf$gather(ally,fit);xv<-tf$gather(allx,val);yv<-tf$gather(ally,val)
  if(dataset=="dirty") {
    ax<-load("ambiguous_train_x");ay<-load("ambiguous_train_y")
    af<-load("ambiguous_fit");av<-load("ambiguous_validation")
    x<-tf$concat(list(x,tf$gather(ax,af)),0L);y<-tf$concat(list(y,tf$gather(ay,af)),0L)
    xv<-tf$concat(list(xv,tf$gather(ax,av)),0L);yv<-tf$concat(list(yv,tf$gather(ay,av)),0L)
  }
  x<-reshape(x);xv<-reshape(xv);N<-shape(y)[1L]
  g<-mnist_network(arch,seed);g$correlated<-arch=="mlp" && mode=="moment" && method=="ep"
  lr<-if(arch=="mlp" && (mode=="point" || (method=="ep" && mode=="sampled")))3e-4 else .001
  vlr<-if(arch=="mlp" && method=="ep" && mode=="sampled") .003 else lr
  refresh<-if(arch=="mlp")4L else 2L
  u<-mnist_update(g,method,mode,N,lr,vlr,refresh)
  all_variables<-c(setNames(lapply(g$weights,`[[`,"m"),paste0("m",seq_along(g$weights))),
    setNames(lapply(g$weights,`[[`,"lv"),paste0("lv",seq_along(g$weights))),u$optimizer$state,
    setNames(u$cache,paste0("cache",seq_along(u$cache))),if(g$correlated)list(basis=g$basis))
  checkpoint<-do.call(tf$train$Checkpoint,all_variables)
  snapshot<-function()lapply(all_variables,function(v)tf$identity(v))
  restore<-function(values)for(i in seq_along(all_variables))all_variables[[i]]$assign(values[[i]])
  clock<-function(){invisible(value(u$optimizer$state$t));unname(proc.time()["elapsed"])}
  refresh_basis<-function()if(g$correlated) {
    w<-tf$slice(g$weights[[2]]$m,c(0L,0L),c(-1L,400L))
    svd<-as_r(tf$linalg$svd(w,full_matrices=FALSE));g$basis$assign(tf$gather(svd[[3]],0:3,axis=1L))
  }
  epoch_body<-function(indices) {
    body<-function(i,loss) {
      ids<-tf$slice(indices,tf$expand_dims(i,0L),tf$expand_dims(tf$minimum(128L,N-i),0L))
      xx<-tf$gather(x,ids);yy<-tf$gather(y,ids)
      if(mode=="moment" && method=="mfvi")a<-tf$cond(tf$math$floormod(u$optimizer$state$t,scalar(refresh,u$optimizer$state$t))==0,
        function()u$step(xx,yy),function()u$cached(xx,yy)) else a<-u$step(xx,yy)
      list(i+tf$size(ids),loss+a*tf$cast(tf$size(ids),tf$float32))
    }
    out<-as_r(tf$while_loop(function(i,loss)i<N,body,list(tf$constant(0L),tf$constant(0,tf$float32)),parallel_iterations=1L))
    out[[2]]/N
  }
  epoch_run<-tf$`function`(epoch_body,autograph=FALSE)
  pred<-tf$`function`(function(xx)mnist_predict(g,xx,mode,20L),reduce_retracing=TRUE)
  predict_chunks<-function(xx) {
    count<-shape(xx)[1L]
    tf$concat(lapply(seq.int(0L,count-1L,by=256L),function(i)pred(tf$gather(xx,as.integer(i:min(i+255L,count-1L))))),0L)
  }
  score<-function(p,y)list(accuracy=value(tf$reduce_mean(tf$cast(tf$argmax(p,axis=1L,output_type=tf$int32)==y,tf$float32))),
   nll=value(-tf$reduce_mean(tf$math$log(tf$maximum(tf$reduce_sum(p*tf$one_hot(y,10L),1L),scalar(1e-30,p))))))
  initial<-snapshot();start<-clock()
  refresh_basis();invisible(value(epoch_run(tf$range(N,dtype=tf$int32))));restore(initial)
  invisible(value(pred(tf$gather(xv,0:255))));compile_seconds<-clock()-start
  cat("COMPILED",tag,compile_seconds,"seconds\n");flush.console()
  curves<-data.frame();train_seconds<-evaluation_seconds<-checkpoint_seconds<-0;best<-Inf;best_epoch<-0L
  for(epoch in seq_len(epochs)) {
    cool<-.03+.97*.5*(1+cos(pi*(epoch-1)/max(1,epochs-1)))
    u$optimizer$learning_rate$assign(scalar(lr*cool,u$optimizer$learning_rate))
    u$optimizer$variance_learning_rate$assign(scalar(vlr*cool,u$optimizer$variance_learning_rate))
    t<-clock();refresh_basis()
    indices<-tf$constant(np$random$default_rng(as.integer(seed*1000L+epoch))$permutation(as.integer(N))$astype("int32"))
    loss<-value(epoch_run(indices));train_seconds<-train_seconds+clock()-t
    t<-clock();p<-predict_chunks(xv);s<-score(p,yv);evaluation_seconds<-evaluation_seconds+clock()-t
    stopifnot(is.finite(loss),is.finite(s$nll))
    curves<-rbind(curves,data.frame(architecture=arch,dataset=dataset,method=label,seed=seed,epoch=epoch,
      train_seconds=train_seconds,evaluation_seconds=evaluation_seconds,training_objective=loss,
      validation_nll=s$nll,validation_accuracy=s$accuracy,learning_rate=lr*cool))
    t<-clock()
    if(s$nll<best) {
      best<-s$nll;best_epoch<-epoch;b<-belief(g)
      arrays<-c(setNames(lapply(b$m,function(a)a$numpy()),paste0("m",seq_along(b$m))),
                setNames(lapply(b$v,function(a)a$numpy()),paste0("v",seq_along(b$v))),
                if(g$correlated)list(basis=g$basis$numpy()))
      do.call(np$savez,c(list(file.path(folder,"best.npz")),arrays))
    }
    invisible(checkpoint$write(file.path(folder,"latest")))
    write.csv(curves,file.path(folder,"curves.csv"),row.names=FALSE)
    checkpoint_seconds<-checkpoint_seconds+clock()-t
    jsonlite::write_json(list(architecture=arch,dataset=dataset,method=label,seed=seed,
      epoch=epoch,epochs=epochs,best_epoch=best_epoch,best_validation_nll=best,train_seconds=train_seconds,
      evaluation_seconds=evaluation_seconds,checkpoint_seconds=checkpoint_seconds,compile_seconds=compile_seconds,
      learning_rate=lr,variance_learning_rate=vlr,temperature=.01,prior_precision=1000,
      batch_size=128,curvature_refresh=if(mode=="moment"&&method=="mfvi")refresh else NA_integer_,
      covariance_rank=if(g$correlated)4 else 0,tf=as_r(tf$`__version__`),R=R.version.string,
      status=if(epoch==epochs)"complete_fixed_budget" else "running"),file.path(folder,"run.json"),auto_unbox=TRUE,pretty=TRUE)
    cat("EPOCH",tag,epoch,"validation",s$accuracy,s$nll,"train",train_seconds,"\n");flush.console()
  }
  cat("TRAINING_COMPLETE",tag,"\n")
  invisible(folder)
}

# Saved-belief evaluation. Validation selects the checkpoint; test does not.
mnist_evaluate <- function(arch="mlp",dataset="clean",label="moment_mfvi",seed=41L) {
  tag<-paste(arch,dataset,label,seed,sep="_");folder<-file.path("results",tag)
  run<-jsonlite::read_json(file.path(folder,"run.json"),simplifyVector=TRUE)
  stopifnot(run$status=="complete_fixed_budget")
  mode<-if(label=="ordinary")"point" else strsplit(label,"_")[[1]][1]
  g<-mnist_network(arch,seed);g$correlated<-arch=="mlp" && label=="moment_ep"
  b<-np$load(file.path(folder,"best.npz"))
  for(i in seq_along(g$weights)) {
    g$weights[[i]]$m$assign(b$get(paste0("m",i)))
    g$weights[[i]]$lv$assign(tf$math$log(tf$convert_to_tensor(b$get(paste0("v",i)))))
  }
  if(g$correlated)g$basis$assign(b$get("basis"))
  load<-function(name)tf$convert_to_tensor(np$load(file.path("data",paste0(name,".npy"))))
  reshape<-function(x)if(arch=="mlp")tf$reshape(x,tuple(-1L,784L)) else
    tf$pad(tf$expand_dims(x,-1L),rbind(c(0L,0L),c(2L,2L),c(2L,2L),c(0L,0L)))
  x<-reshape(load("clean_test_x"));y<-as.integer(value(load("clean_test_y")))
  fashion<-reshape(load("fashion_x"));ambiguous<-reshape(load("ambiguous_test_x"))
  ay<-value(load("ambiguous_test_y"))
  # One shared random network per draw across ALL inputs, fixed stateless seeds.
  # Main sampled prediction uses 16 networks; moment uses 18 logit nodes, 0 nets.
  draws<-16L
  batch_predict<-tf$`function`(function(xx) {
    a<-mnist_predict(g,xx,mode,draws,seed=192L,components=TRUE)
    tf$concat(list(a$p,tf$expand_dims(a$entropy,-1L),tf$expand_dims(a$mi,-1L)),axis=1L)
  })
  all_predict<-tf$`function`(function(xx) {
    count<-tf$gather(tf$shape(xx),0L);padding<-tf$math$floormod(-count,256L)
    pad<-tf$concat(list(tf$reshape(tf$stack(list(0L,padding)),c(1L,2L)),
      tf$zeros(c(length(shape(xx))-1L,2L),tf$int32)),axis=0L)
    xp<-tf$pad(xx,pad)
    chunks<-tf$reshape(xp,tf$concat(list(tf$constant(c(-1L,256L)),tf$slice(tf$shape(xx),list(1L),list(-1L))),0L))
    out<-tf$map_fn(function(a)batch_predict(a),chunks,
      fn_output_signature=tf$TensorSpec(c(256L,12L),tf$float32),parallel_iterations=1L)
    tf$slice(tf$reshape(out,c(-1L,12L)),c(0L,0L),tf$stack(list(count,12L)))
  },autograph=FALSE)
  clock<-function(){invisible(value(tf$reduce_sum(g$weights[[1]]$m)));unname(proc.time()["elapsed"])}
  t<-clock();result<-all_predict(x);invisible(value(result));compile_seconds<-clock()-t
  times<-numeric(7)
  for(i in seq_along(times)){t<-clock();result<-all_predict(x);invisible(value(tf$reduce_sum(result)));times[i]<-clock()-t}
  result<-value(result);ood<-value(all_predict(fashion));amb<-value(all_predict(ambiguous))
  p<-result[,1:10];po<-ood[,1:10];pa<-amb[,1:10]
  metrics<-uncertainty_metrics(p,y);om<-ood_metrics(p,po)
  # Standard Dirty test mixture has 10k clean examples + 60k ambiguous label pairs.
  ai<-rep(seq_len(nrow(pa)),each=10L);alabel<-as.integer(t(ay))
  dirty_metrics<-uncertainty_metrics(rbind(p,pa[ai,]),c(y,alabel))
  dirty_ood<-ood_metrics(rbind(p,pa),po) # unique images, not 10 duplicate scores
  metrics<-c(list(architecture=arch,dataset=dataset,method=label,seed=seed),metrics,as.list(om),
   list(inference_seconds=median(times),inference_seconds_sd=sd(times),inference_repeats=times,
        inference_compile_seconds=compile_seconds,prediction_network_draws=if(mode=="sampled")draws else 0,
        logit_nodes=if(mode=="moment")18 else 0,dirty_test=dirty_metrics,dirty_ood_unique=as.list(dirty_ood),
        entropy_clean=mean(result[,11]),entropy_ambiguous=mean(amb[,11]),entropy_fashion=mean(ood[,11]),
        mi_clean=mean(result[,12]),mi_ambiguous=mean(amb[,12]),mi_fashion=mean(ood[,12])))
  jsonlite::write_json(metrics,file.path(folder,"metrics.json"),auto_unbox=TRUE,pretty=TRUE)
  np$savez_compressed(file.path(folder,"predictions.npz"),p=p,y=y,p_ood=po,p_ambiguous=pa,
    ambiguous_labels=ay,entropy=result[,11],mi=result[,12],ood_entropy=ood[,11],ood_mi=ood[,12],
    ambiguous_entropy=amb[,11],ambiguous_mi=amb[,12])
  saveRDS(list(p=p,y=y,ood=po,ambiguous=pa,ambiguous_labels=ay,
    uncertainty=list(clean=result[,11:12],ambiguous=amb[,11:12],fashion=ood[,11:12]),
    run=run,metrics=metrics,history=read.csv(file.path(folder,"curves.csv"))),file.path(folder,"evaluation.rds"))
  cat("EVALUATED",tag,"accuracy",metrics$accuracy,"NLL",metrics$nll,"OOD AUROC",metrics$msp_auroc,
      "seconds",metrics$inference_seconds,"\n")
  invisible(readRDS(file.path(folder,"evaluation.rds")))
}

# %% Fixed benchmark settings
# Protocol controls used by the original full recipes; sourcing does not train.
RUN_LARGE_SEMANTIC_CHECKS <- FALSE
RUN_PIPELINE_CHECKS <- FALSE
RUN_OOD_EVALUATION <- TRUE
RUN_OFFICIAL_CLINC <- TRUE
RUN_TEXT_TRANSFER <- FALSE
RUN_MATCHED_MFVI <- FALSE
RUN_SERVING_ABLATION <- FALSE
BENCHMARKS <- c("resnet20_cifar10","densenet40_cifar10","gpt2_lora_sst2","roberta_clinc")
METHODS <- c("ordinary","sampled_mfvi","moment_mfvi","sampled_ep","moment_ep")
SEEDS <- NULL
SOURCE_SEEDS <- c(resnet20_cifar10=41L,densenet40_cifar10=41L,gpt2_lora_sst2=41L,roberta_clinc=42L)
# # Fixed archived output quadrature; no network samples.
# NPY SHA256 c2615a1c7482e806477101ed0ba1954e7555cbcefe0a66e4f94827d06e94f893
clinc_quadrature<-function() {
  z<-c(0.58437865972518921,-2.4569301605224609,-0.38708853721618652,1.1663821935653687,0.94783222675323486,-0.18053328990936279,-0.81607609987258911,0.12470864504575729,
    0.24076876044273376,-1.115301251411438,-0.065834425389766693,0.68678587675094604,1.6848517656326294,-0.51654201745986938,-1.3877550363540649,0.45175164937973022,
    0.32348242402076721,-1.2781552076339722,-0.65743839740753174,1.9131494760513306,0.83192139863967896,-0.10165075957775116,-0.93112123012542725,0.20457404851913452,
    0.0031001938041299582,-0.75133353471755981,-0.30529060959815979,1.0256928205490112,1.4231822490692139,-0.4284483790397644,-1.6306099891662598,0.54007053375244141,
    0.4961603581905365,-1.8059725761413574,-0.46996745467185974,1.3193442821502686,1.0913810729980469,-0.2630256712436676,-0.7014230489730835,0.043339859694242477,
    0.16235882043838501,-0.99014359712600708,-0.14125244319438934,0.77848261594772339,2.2505931854248047,-0.61012941598892212,-1.1893608570098877,0.36296957731246948,
    0.40891841053962708,-1.5002168416976929,-0.55907678604125977,1.5347484350204468,0.73423540592193604,-0.024702426046133041,-1.0452913045883179,0.28180071711540222,
    0.083702780306339264,-0.8699791431427002,-0.2213091105222702,0.88965833187103271,1.2492804527282715,-0.34630981087684631,-1.9899431467056274,0.62973719835281372,
    0.65092355012893677,-1.8817839622497559,-0.32846486568450928,1.2883970737457275,0.9232751727104187,-0.19912946224212646,-0.83839738368988037,0.10632205009460449,
    0.30034613609313965,-1.0181200504302979,-0.0083584301173686981,0.75822228193283081,1.6135696172714233,-0.53470051288604736,-1.4373865127563477,0.43430539965629578,
    0.38220667839050293,-1.1573286056518555,-0.59040266275405884,2.5833127498626709,0.80955809354782104,-0.12022062391042709,-0.95604431629180908,0.18576301634311676,
    0.060757730156183243,-0.68009626865386963,-0.24543546140193939,1.1239255666732788,1.3742049932479858,-0.44606035947799683,-1.7045509815216064,0.52164244651794434,
    0.56514525413513184,-1.552048921585083,-0.40403810143470764,1.4858232736587524,1.054552435874939,-0.28711867332458496,-0.72816193103790283,0.020129876211285591,
    0.22591812908649445,-0.89669930934906006,-0.078502252697944641,0.86232393980026245,2.0247752666473389,-0.63531887531280518,-1.2377240657806396,0.34089398384094238,
    0.47511893510818481,-1.3302427530288696,-0.49017959833145142,1.7806761264801025,0.70713406801223755,-0.048031795769929886,-1.0820941925048828,0.25762349367141724,
    0.14660662412643433,-0.78536367416381836,-0.15768909454345703,0.98276925086975098,1.1998598575592041,-0.36854994297027588,-2.1977658271789551,0.60449683666229248)
  tf$constant(matrix(c(z,-z),ncol=1L),dtype=tf$float32)
}

# Historical transformer contracts: not the generic CIFAR defaults
text_protocol<-function(benchmark,method) {
  cfg<-protocols[protocols$benchmark==benchmark,];cfg$method<-method
  cfg$batch<-32L;cfg$rank<-1L;cfg$training_draws<-if(grepl("_ep$",method))4L else 1L
  cfg$training_draws<-if(grepl("moment",method))0L else cfg$training_draws
  cfg$optimizer<-if(method=="ordinary")"AdamW" else if(grepl("mfvi",method))"archived Vadam recurrence" else "Adam on mean/log variance"
  if(benchmark=="gpt2_lora_sst2") {
    cfg$epochs<-if(method=="moment_mfvi")1L else 3L;cfg$schedule_epochs<-3L
    cfg$schedule<-"warmup_cosine_floor_0.1";cfg$warmup_fraction<-.05
    cfg$mean_lr<-.0003;cfg$variance_lr<-.01
    cfg$tau<-if(grepl("_ep$",method)).01 else if(method=="sampled_mfvi").001 else .0001
    cfg$stopping<-"historical public validation selection; NOT independent final test"
    cfg$validation_draws<-16L;cfg$prediction_draws<-16L
  } else {
    stopifnot(benchmark=="roberta_clinc")
    cfg$epochs<-6L;cfg$schedule_epochs<-6L;cfg$schedule<-"warmup_linear"
    cfg$warmup_fraction<-.06;cfg$mean_lr<-5e-5;cfg$variance_lr<-.005
    cfg$tau<-if(method=="moment_mfvi")3e-6 else 1e-6
    cfg$stopping<-"duplicate-grouped 1500 training rows; public development untouched during selection"
    cfg$validation_draws<-8L;cfg$prediction_draws<-8L
  }
  cfg$status<-"faithful configuration; full-model implementation gate required before training"
  cfg
}
configure_text_belief<-function(g,benchmark,method) {
  if(benchmark=="gpt2_lora_sst2") {
    g$moment_backend<-if(method=="moment_mfvi")"frozen_scalar" else "scalar_adjoint"
    if(method=="sampled_mfvi")g$vadam_normalization<-"likelihood_ema_initial_zero"
  } else if(benchmark=="roberta_clinc") {
    g$moment_backend<-if(method=="moment_mfvi")"frozen_shared" else "shared_curved"
    g$vadam_normalization<-"likelihood_ema_initial_zero"
    g$optimizer_betas<-c(.9,if(method=="ordinary").999 else .99)
    g$variance_epsilon<-1e-12;g$field_float64<-TRUE
    g$head_normals<-clinc_quadrature()
  }
  invisible(g)
}
learning_rate_multiplier<-function(cfg,step,N) {
  # step is one-based, just as in the reference loop.
  if(is.null(cfg$schedule))return(.5*(1+cos(pi*(step-1)/(cfg$epochs*ceiling(N/cfg$batch)))))
  total<-cfg$schedule_epochs*ceiling(N/cfg$batch)
  warm<-max(1L,round(cfg$warmup_fraction*total))
  if(step<=warm)return(step/warm)
  fraction<-(step-warm)/(total-warm)
  if(cfg$schedule=="warmup_linear")max(0,1-fraction) else .1+.9*.5*(1+cos(pi*fraction))
}

# %% Public data and complete benchmark training
# %% Fixed full-data protocols: these are not tiny substitutes for the benchmarks
benchmark_seeds<-function(task) if(is.null(SEEDS)) SOURCE_SEEDS[[task]] else SEEDS
protocols<-data.frame(
  benchmark=c("resnet20_cifar10","densenet40_cifar10","gpt2_lora_sst2","roberta_clinc"),
  epochs=c(100L,100L,3L,6L),batch=c(125L,125L,32L,32L),
  mean_lr=c(.003,.003,.0003,5e-5),variance_lr=c(.003,.003,.01,.005),
  tau=c(.001,.001,.01,1e-6),rank=c(1L,1L,1L,1L),
  training_draws=1L,validation_draws=16L,prediction_draws=16L,
  weight_decay=c(5e-4,5e-4,.01,.01),
  stopping=c("separate stratified subset; minimum stopping predictive NLL",
    "separate stratified subset; minimum stopping predictive NLL",
    "historical public validation selection; NOT independent final test",
    "duplicate-grouped training-only stopping set; public development separate"),
  status=c("source-matched CIFAR reproduction","source-matched CIFAR reproduction",
           "source-matched transformer contract; full training reproduction pending",
           "source-matched transformer contract; full training reproduction pending"))
# These are cohort defaults; benchmark_protocol/text_protocol supply each
# method's actual optimizer and regularization. Do not present generic defaults
# or a historical descriptive status string as a completed-run result.
print(protocols[,c("benchmark","epochs","batch","stopping")])

benchmark_protocol<-function(benchmark,method) {
  cfg<-protocols[protocols$benchmark==benchmark,]
  if(!grepl("cifar10",benchmark))return(text_protocol(benchmark,method))
  cfg$method<-method;cfg$training_draws<-if(method=="sampled_ep")4L else 1L
  # Both archived selected ordinary runs use0.005. The training function's
  # generic default0.0005 is not their selected configuration.
  if(method=="ordinary")cfg$weight_decay<-.005
  cfg$tau<-if(method=="moment_ep").01 else .001
  cfg$optimizer<-if(method=="ordinary")"AdamW" else if(grepl("mfvi",method))"reference Vadam" else "Adam on mean/log variance"
  cfg
}

if(RUN_IMPLEMENTATION_CHECKS) {
  # Values recovered from the selected runs' immutable config.json files,
  # rather than the generic defaults in their training function.
  methods<-c("ordinary","sampled_mfvi","moment_mfvi","sampled_ep","moment_ep")
  checks<-do.call(rbind,lapply(c("resnet20_cifar10","densenet40_cifar10"),function(task)
    do.call(rbind,lapply(methods,function(method) {
      a<-benchmark_protocol(task,method)
      stopifnot(a$mean_lr==.003,a$epochs==100L,a$batch==125L,a$rank==1L)
      if(method=="ordinary")stopifnot(a$weight_decay==.005)
      else stopifnot(a$tau==if(method=="moment_ep").01 else .001)
      data.frame(benchmark=task,method=method,status="PASS")
    }))))
  write.csv(checks,file.path(OUT,"selected_cifar_configuration_checks.csv"),row.names=FALSE)
}

stratified_holdout<-function(y,per_class,seed=19041L) {
  set.seed(seed)
  unlist(lapply(sort(unique(y)),function(k)sample(which(y==k),per_class)))
}
load_cifar<-function() {
  # The public mirror and NumPy split are the ones used by the successful
  # reference. Cache real image bytes, not predictions or trained weights.
  py_run_string(r"---(
from pathlib import Path
import hashlib,json,time
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO
from huggingface_hub import hf_hub_download
def tutorial_cifar():
    cache=Path('/content/bnn_data_cache/cifar10');cache.mkdir(parents=True,exist_ok=True)
    result={};receipts=[]
    for split,n in [('train',50000),('test',10000)]:
        begin=time.time();saved=cache/(split+'.npz');receipt=cache/(split+'.json')
        if not saved.exists():
            path=Path(hf_hub_download('uoft-cs/cifar10',f'plain_text/{split}-00000-of-00001.parquet',
                revision='0b2714987fa478483af9968de7c934580d0bb9a2',repo_type='dataset',token=False))
            table=pq.read_table(path).to_pydict()
            x=np.stack([np.asarray(Image.open(BytesIO(row['bytes'])).convert('RGB')) for row in table['img']])
            y=np.asarray(table['label'],dtype=np.int32)
            assert x.shape==(n,32,32,3) and np.array_equal(np.bincount(y),np.full(10,n//10))
            np.savez(saved,x=x,y=y)
            receipt.write_text(json.dumps(dict(repo='uoft-cs/cifar10',resolved_asset=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
                rows=n,download_decode_seconds=time.time()-begin)))
        with np.load(saved) as a:result[split+'_x']=a['x'];result[split+'_y']=a['y']
        receipts.append(json.loads(receipt.read_text()))
    rng=np.random.default_rng(20261021);fit=[];stop=[]
    for c in range(10):
        ids=rng.permutation(np.flatnonzero(result['train_y']==c));stop.extend(ids[:500]);fit.extend(ids[500:])
    fit=np.asarray(fit);stop=np.asarray(stop);rng.shuffle(fit);rng.shuffle(stop)
    result.update(fit=fit,stop=stop)
    Path('advanced_results').mkdir(exist_ok=True)
    Path('advanced_results/data_assets.json').write_text(json.dumps(receipts,indent=2))
    return result
)---")
  a<-import_main(convert=FALSE)$tutorial_cifar()
  y<-as.integer(as_r(a$get("train_y")));yt<-as.integer(as_r(a$get("test_y")))
  fit<-as.integer(as_r(a$get("fit")))+1L;stop<-as.integer(as_r(a$get("stop")))+1L
  # Observed pixels have zero probabilistic variance; dataset variability is
  # not represented as a Gaussian input belief.
  with(tf$device("/CPU:0"), {
    x<-tf$cast(tf$convert_to_tensor(a$get("train_x")),tf$float32)/255
    xt<-tf$cast(tf$convert_to_tensor(a$get("test_x")),tf$float32)/255
    center<-tf$constant(c(.4914,.4822,.4465));scale<-tf$constant(c(.2470,.2435,.2616))
    x<-(x-center)/scale;xt<-(xt-center)/scale
  })
  list(x=x,y=y,fit=fit,stop=stop,test_x=xt,test_y=yt,kind="image")
}
load_svhn_ood<-function() {
  # Optional diagnostic data, not training or checkpoint-selection data.
  # Cache 100 predetermined examples per digit on the VM, not old predictions.
  py_run_string(r"---(
def tutorial_svhn_ood():
    from pathlib import Path
    import json,hashlib
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    from io import BytesIO
    from huggingface_hub import hf_hub_download
    cache=Path('/content/bnn_data_cache/svhn');cache.mkdir(parents=True,exist_ok=True)
    file=cache/'fixed_1000.npz';receipt=cache/'fixed_1000.json'
    if not file.exists():
        revision='f9e1717d73324ebbc1ece84267ccd80d0aca0690'
        path=Path(hf_hub_download('ufldl-stanford/svhn','cropped_digits/test-00000-of-00001.parquet',
            revision=revision,repo_type='dataset',token=False))
        data=pq.read_table(path).to_pydict()
        y=np.asarray(data['label']);rng=np.random.default_rng(20261030)
        ids=np.concatenate([rng.permutation(np.flatnonzero(y==k))[:100] for k in range(10)])
        x=np.stack([np.asarray(Image.open(BytesIO(data['image'][int(i)]['bytes'])).convert('RGB')) for i in ids])
        np.savez(file,x=x,ids=ids,labels=y[ids])
        receipt.write_text(json.dumps(dict(dataset='ufldl-stanford/svhn',revision=revision,
            split='test',rows=len(ids),selection='100 per digit; RNG20261030; no predictions used',
            sha256=hashlib.sha256(file.read_bytes()).hexdigest())))
    Path('advanced_results').mkdir(exist_ok=True)
    Path('advanced_results/svhn_data_receipt.json').write_text(receipt.read_text())
    with np.load(file) as a:return a['x'].copy()
)---")
  with(tf$device('/CPU:0'), {
    x<-tf$cast(tf$convert_to_tensor(import_main(convert=FALSE)$tutorial_svhn_ood()),tf$float32)/255
    x<-(x-tf$constant(c(.4914,.4822,.4465)))/tf$constant(c(.2470,.2435,.2616))
  });x
}
load_text<-function(g,task) {
  ds<-import("datasets",convert=FALSE)
  # Recent datasets versions return a Column object rather than a Python list.
  # Materialize the text/label column explicitly before converting it to R.
  builtins<-import_builtins(convert=FALSE)
  column<-function(d,name)as_r(builtins$list(d$`__getitem__`(name)))
  if(task=="sst2") {
    revision<-"8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
    train<-ds$load_dataset("stanfordnlp/sst2",split="train",revision=revision,token=FALSE)
    test<-ds$load_dataset("stanfordnlp/sst2",split="validation",revision=revision,token=FALSE)
    text<-as.character(column(train,"sentence"));y<-as.integer(column(train,"label"))
    test_text<-as.character(column(test,"sentence"));yt<-as.integer(column(test,"label"))
    stop<-integer() # Historical protocol used all 67,349 training examples.
    g$tokenizer$pad_token<-g$tokenizer$eos_token
    g$tokenizer$padding_side<-"right"
    ood<-NULL;names<-c("negative","positive")
    stopifnot(length(y)==67349L,length(yt)==872L,all(y %in% 0:1),all(yt %in% 0:1))
    saveRDS(list(repository="stanfordnlp/sst2",revision=revision,
      training_fingerprint=as_r(train$`_fingerprint`),validation_fingerprint=as_r(test$`_fingerprint`),
      training_rows=length(y),validation_rows=length(yt),
      selection="public validation is also reporting; not independent test"),
      file.path(OUT,"sst2_dataset_receipt.rds"))
  } else {
    train<-ds$load_dataset("clinc/clinc_oos","plus",split="train",revision="155b9c710419136e17307b80d0a13e68cd46b4ec",token=FALSE)
    test<-ds$load_dataset("clinc/clinc_oos","plus",split="validation",revision="155b9c710419136e17307b80d0a13e68cd46b4ec",token=FALSE)
    names<-as.character(as_r(train$features$get("intent")$names))
    oos<-match("oos",names)-1L;inlabels<-setdiff(seq_along(names)-1L,oos)
    yy<-as.integer(column(train,"intent"));tt<-as.character(column(train,"text"))
    keep<-yy!=oos;y<-match(yy[keep],inlabels)-1L;text<-tt[keep]
    yy<-as.integer(column(test,"intent"));tt<-as.character(column(test,"text"))
    yt<-match(yy[yy!=oos],inlabels)-1L;test_text<-tt[yy!=oos];ood<-tt[yy==oos]
    names<-names[inlabels+1L]
    encoded<-g$tokenizer(as.list(text),padding=FALSE,truncation=TRUE,max_length=128L)
    stop<-clinc_split(encoded$get("input_ids"),y)
    stopifnot(length(y)==15000L,length(stop)==1500L,length(yt)==3000L)
  }
  tokenize<-function(text) {
    if(task=="sst2")return(sst2_inputs(g$tokenizer,text))
    a<-g$tokenizer(as.list(text),padding="max_length",truncation=TRUE,max_length=128L,return_tensors="np")
    with(tf$device("/CPU:0"), {
      x<-tf$stack(list(tf$cast(tf$convert_to_tensor(a$get("input_ids")),tf$float32),
                      tf$cast(tf$convert_to_tensor(a$get("attention_mask")),tf$float32)),axis=2L)
    });x
  }
  x<-tokenize(text);xt<-tokenize(test_text)
  list(x=x,y=y,fit=setdiff(seq_along(y),stop),stop=stop,
       stop_x=if(task=="sst2")xt else tf$gather(x,as.list(as.integer(stop-1L))),
       stop_y=if(task=="sst2")yt else y[stop],test_x=xt,test_y=yt,
       reporting_cohort=if(task=="sst2")"public validation (also checkpoint selection)" else "public development (separate training-only selection)",
       ood_x=if(is.null(ood))NULL else tokenize(ood),ood_cohort="validation OOS, not official test OOS",
       labels=names,kind="text")
}
sst2_inputs<-function(tokenizer,sentences,max_length=128L) {
  # Faithful historical prompt: preserve BOTH demonstrations and the decision
  # suffix. Truncate only the new review body, never the demonstrations.
  encode<-function(s)as.integer(as_r(tokenizer$encode(s,add_special_tokens=FALSE)))
  prefix<-encode(paste0("Review: great acting and direction\nSentiment: positive.\n\n",
                       "Review: boring and predictable\nSentiment: negative.\n\n"))
  suffix<-encode("\nSentiment:");budget<-max_length-length(prefix)-length(suffix)
  stopifnot(budget>0L)
  ids<-matrix(as.integer(as_r(tokenizer$eos_token_id)),nrow=length(sentences),ncol=max_length)
  mask<-matrix(0L,nrow=length(sentences),ncol=max_length)
  for(i in seq_along(sentences)) {
    body<-head(encode(paste0("Review: ",sentences[[i]])),budget)
    tokens<-c(prefix,body,suffix);ids[i,seq_along(tokens)]<-tokens;mask[i,seq_along(tokens)]<-1L
  }
  with(tf$device("/CPU:0"), {
    x<-tf$stack(list(tf$constant(ids,dtype=tf$float32),tf$constant(mask,dtype=tf$float32)),axis=2L)
  });x
}
augment_images<-function(x,seed) {
    center<-tf$constant(c(.4914,.4822,.4465));scale<-tf$constant(c(.2470,.2435,.2616))
    x<-tf$pad(x*scale+center,list(c(0L,0L),c(4L,4L),c(4L,4L),c(0L,0L)))
    # Stateless per-update augmentation makes checkpoint resumption independent
    # of compilation or validation calls consuming a global RNG stream.
    seeds<-tf$random$experimental$stateless_split(seed,tf$gather(tf$shape(x),0L))
    x<-tf$map_fn(function(pair) {
      pair<-as_r(pair) # Python tuples are zero-indexed; R lists are one-indexed.
      a<-tf$image$stateless_random_crop(pair[[1]],c(32L,32L,3L),pair[[2]])
      tf$image$stateless_random_flip_left_right(a,pair[[2]]+c(17L,29L))
    },tuple(x,seeds),fn_output_signature=tf$TensorSpec(c(32L,32L,3L),tf$float32))
    (x-center)/scale
}
# Reporting saved predictions needs no TensorFlow; training initializes it first.
if (exists("tf")) compiled_augmentation<-tf$`function`(augment_images,reduce_retracing=TRUE)
minibatch<-function(data,idx,training=FALSE,seed=c(41L,0L)) {
  x<-tf$gather(data$x,as.list(as.integer(idx-1L)))
  y<-tf$constant(as.list(as.integer(data$y[idx])),dtype=tf$int32)
  if(training && data$kind=="image")x<-compiled_augmentation(x,tf$constant(as.integer(seed)))
  if(data$kind=="text")x<-trim_text_batch(x)
  list(x=x,y=y)
}
trim_text_batch<-function(x) {
  length<-tf$cast(tf$reduce_max(tf$reduce_sum(tf$gather(x,1L,axis=2L),axis=1L)),tf$int32)
  tf$slice(x,c(0L,0L,0L),tf$stack(list(tf$constant(-1L),length,tf$constant(-1L))))
}

# %% Evaluation: compile once, average probabilities, synchronize timing
evaluate<-function(g,x,y,mode,batch=128L,draws=32L) {
  g$training<-FALSE
  # Reuse the compiled serving graph throughout training; variables remain live.
  if(is.null(g$serving))g$serving<-list()
  key<-paste(mode,draws,sep="_")
  if(is.null(g$serving[[key]]))g$serving[[key]]<-tf$`function`(
    function(a)predict_probabilities(g,if(grepl("gpt2|roberta",g$architecture))trim_text_batch(a) else a,mode,draws),reduce_retracing=TRUE)
  serve<-g$serving[[key]]
  ii<-seq_len(min(batch,length(y)))
  t0<-proc.time()[[3]];invisible(value(serve(tf$gather(x,as.list(as.integer(ii-1L))))))
  # Warm the short last batch as well. Otherwise its first graph trace can be
  # misreported as prediction time when evaluating a freshly restored belief.
  remainder<-length(y)%%batch
  if(length(y)>batch && remainder>0L) {
    tail_ids<-as.integer((length(y)-remainder):(length(y)-1L))
    invisible(value(serve(tf$gather(x,tail_ids))))
  }
  compile_seconds<-proc.time()[[3]]-t0
  t0<-proc.time()[[3]];p<-list()
  for(start in seq.int(1L,length(y),by=batch)) {
    ii<-start:min(start+batch-1L,length(y))
    p[[length(p)+1L]]<-value(serve(tf$gather(x,as.list(as.integer(ii-1L)))))
  }
  seconds<-proc.time()[[3]]-t0
  p<-do.call(rbind,p)
  list(p=p,metrics=uncertainty_metrics(p,y),seconds=seconds,compile_seconds=compile_seconds)
}

# One compiled sampled network, reused across the whole reporting cohort.
# This diagnostic does not charge 64 sampled networks to moment serving.
prediction_draw_sweep<-function(g,x,y,batch=128L,counts=c(1L,4L,16L,64L)) {
  g$training<-FALSE
  # One complete weight draw is shared across ALL input minibatches. Cache it
  # once: regenerating millions of identical random entries for every batch
  # wastes work and is not part of the probability-average estimator.
  ws<-lapply(g$weights,function(w)tf$Variable(zero(w$m),trainable=FALSE))
  draw<-tf$`function`(function(s) {
    assigned<-lapply(seq_along(g$weights),function(i) {
      w<-g$weights[[i]]
      key<-tf$stack(list(tf$constant(as.integer(g$seed+7919L*i)),s))
      ws[[i]]$assign(w$m+tf$exp(w$lv/2)*tf$random$stateless_normal(tf$shape(w$m),seed=key,dtype=w$m$dtype))
    })
    with(tf$control_dependencies(assigned),tf$constant(0L))
  })
  serve<-tf$`function`(function(a) {
    if(grepl("gpt2|roberta",g$architecture))a<-trim_text_batch(a)
    tf$nn$softmax(point_forward(g,a,ws))
  },reduce_retracing=TRUE)
  t0<-proc.time()[[3]]
  invisible(value(draw(tf$constant(0L))))
  invisible(value(serve(tf$gather(x,as.integer(seq_len(min(batch,length(y)))-1L)))))
  remainder<-length(y)%%batch
  if(length(y)>batch && remainder>0L)
    invisible(value(serve(tf$gather(x,as.integer((length(y)-remainder):(length(y)-1L))))))
  compile_seconds<-proc.time()[[3]]-t0
  total<-NULL;seconds<-0;rows<-snapshots<-list()
  for(s in seq_len(max(counts))) {
    p<-list();t0<-proc.time()[[3]]
    invisible(value(draw(tf$constant(as.integer(s)))))
    for(start in seq.int(1L,length(y),by=batch)) {
      ids<-start:min(start+batch-1L,length(y))
      p[[length(p)+1L]]<-value(serve(tf$gather(x,as.integer(ids-1L))))
    }
    seconds<-seconds+proc.time()[[3]]-t0;p<-do.call(rbind,p)
    total<-if(is.null(total))p else total+p
    if(s %in% counts) {
      average<-total/s;snapshots[[as.character(s)]]<-average
      rows[[length(rows)+1L]]<-cbind(data.frame(draws=s,seconds=seconds),as.data.frame(uncertainty_metrics(average,y)))
    }
  }
  list(table=do.call(rbind,rows),probabilities=snapshots,labels=y,compile_seconds=compile_seconds,
    semantics="one shared network per draw across all examples; probability averaging; diagnostic only")
}
checkpoint_object<-function(g,update) {
  values<-c(lapply(g$weights,`[[`,"m"),lapply(g$weights,`[[`,"lv"))
  values<-Filter(Negate(is.null),values)
  names(values)<-paste0("parameter_",seq_along(values))
  for(i in seq_along(g$nodes)) if(!is.null(g$nodes[[i]]$normalization)) {
    values[[paste0("bn_mean_",i)]]<-g$nodes[[i]]$normalization$m
    values[[paste0("bn_var_",i)]]<-g$nodes[[i]]$normalization$v
  }
  values$optimizer<-update$checkpoint
  do.call(tf$train$Checkpoint,values)
}
compare_compiled_update<-function(g,update,a,ckpt,prefix,atol,rtol) {
  initial<-as_r(ckpt$save(prefix))
  tracked<-c(lapply(g$weights,`[[`,"m"),Filter(Negate(is.null),lapply(g$weights,`[[`,"lv")),
             update$mean_optimizer$state)
  for(op in g$nodes)if(!is.null(op$normalization))tracked<-c(tracked,list(op$normalization$m,op$normalization$v))
  eager_loss<-as.numeric(value(update$eager_step(a$x,a$y)));expected<-lapply(tracked,value)
  ckpt$restore(initial)$assert_existing_objects_matched()
  t0<-proc.time()[[3]];compiled_loss<-as.numeric(value(update$step(a$x,a$y)))
  seconds<-proc.time()[[3]]-t0
  errors<-do.call(rbind,lapply(seq_along(tracked),function(i) {
    error<-max(abs(value(tracked[[i]])-expected[[i]]));scale<-max(abs(expected[[i]]))
    data.frame(state=i,max_abs=error,reference_scale=scale,atol=atol,rtol=rtol,
      status=if(is.finite(error)&&error<=atol+rtol*scale)"PASS" else "FAIL")
  }))
  ckpt$restore(initial)$assert_existing_objects_matched()
  write.csv(errors,paste0(prefix,"_agreement.csv"),row.names=FALSE)
  list(errors=errors,seconds=seconds,loss_difference=abs(eager_loss-compiled_loss))
}
double_precision_check<-function(g,update,a,cfg,method,inference,mode,N,run) {
  # A separate copy checks the SAME equations/update in float64. It is never
  # used to supply training gradients or overwrite the float32 learned model.
  text_model<-grepl("gpt2|roberta",g$architecture)
  cpu_strict<-text_model || identical(g$architecture,"densenet40")
  # The full transformer state is too large to duplicate in float64 on an
  # already occupied GPU. Its strict gate uses every parameter on two actual
  # examples on CPU; the separately retained float32 check uses the full batch.
  # DenseNet's full-batch training graph also stays resident. Use every weight
  # on two real examples for its separate strict arithmetic check, preserving
  # the full-batch float32 discrepancy instead of weakening its tolerance.
  with(tf$device(if(cpu_strict)"/CPU:0" else "/GPU:0"), {
  d<-if(grepl("gpt2",g$architecture))build_gpt2_lora(g$rank,g$seed,dtype=tf$float64)
    else if(grepl("roberta",g$architecture))build_roberta(g$rank,g$seed,dtype=tf$float64)
    else build_cifar(g$architecture,g$rank,g$seed)
  if(text_model)configure_text_belief(d,cfg$benchmark,method)
  if(!is.null(cfg$vadam_normalization))d$vadam_normalization<-cfg$vadam_normalization
  d$training<-TRUE
  for(i in seq_along(g$weights)) {
    w<-g$weights[[i]]
    d$weights[[i]]<-w
    d$weights[[i]]$m<-tf$Variable(tf$cast(w$m,tf$float64))
    d$weights[[i]]$lv<-if(is.null(w$lv))NULL else tf$Variable(tf$cast(w$lv,tf$float64))
    d$weights[[i]]$m0<-if(is.null(w$m0))NULL else tf$cast(w$m0,tf$float64)
  }
  for(i in seq_along(g$nodes))if(!is.null(g$nodes[[i]]$normalization)) {
    d$nodes[[i]]$normalization$m<-tf$Variable(tf$cast(g$nodes[[i]]$normalization$m,tf$float64))
    d$nodes[[i]]$normalization$v<-tf$Variable(tf$cast(g$nodes[[i]]$normalization$v,tf$float64))
  }
  du<-make_update(d,inference,mode,N,tau=cfg$tau,power=1,lr=cfg$mean_lr,
    variance_lr=cfg$variance_lr,samples=cfg$training_draws,decay=if(mode=="point")cfg$weight_decay else 0)
  for(k in names(update$mean_optimizer$state))
    du$mean_optimizer$state[[k]]$assign(tf$cast(update$mean_optimizer$state[[k]],tf$float64))
  aa<-if(cpu_strict)list(x=tf$gather(a$x,list(0L,1L)),y=tf$gather(a$y,list(0L,1L))) else a
  result<-compare_compiled_update(d,du,list(x=tf$cast(aa$x,tf$float64),y=aa$y),checkpoint_object(d,du),
    file.path(run,"compile_float64"),1e-10,1e-8)
  result$scope<-if(cpu_strict)"all parameters; two real examples; CPU float64" else "all parameters; full minibatch; float64"
  });result
}

# %% End-to-end training: same initial means, data and minibatch order
train_one<-function(benchmark,method="moment_mfvi",seed=41L,cfg=NULL,data=NULL,diagnostics=TRUE) {
  # A notebook rerun must not start another model beyond its allocation limit.
  if(as.numeric(Sys.time())>=as.numeric(Sys.getenv("BNN_STOP_UTC",unset="Inf")))
    stop("Allocation boundary reached before starting ",benchmark," / ",method)
  if(is.null(cfg))cfg<-benchmark_protocol(benchmark,method)
  stopifnot(nrow(cfg)==1L,method %in% c("ordinary","sampled_mfvi","moment_mfvi","sampled_ep","moment_ep"))
  g<-switch(benchmark,resnet20_cifar10=build_cifar("resnet20",cfg$rank,seed),
    densenet40_cifar10=build_cifar("densenet40",cfg$rank,seed),
    gpt2_lora_sst2=build_gpt2_lora(cfg$rank,seed),roberta_clinc=build_roberta(cfg$rank,seed))
  if(!grepl("cifar10",benchmark)) {
    # Independent full-model reference audits are development evidence, not
    # a private CSV that a reader must obtain to execute this notebook.
    # This run still executes the actual compiled/eager update gate below.
    configure_text_belief(g,benchmark,method)
  }
  if(!is.null(cfg$vadam_normalization))g$vadam_normalization<-cfg$vadam_normalization
  if(is.null(data))data<-if(grepl("cifar10",benchmark))load_cifar() else load_text(g,if(grepl("gpt2",benchmark))"sst2" else "clinc")
  if(benchmark=="roberta_clinc" && exists("RUN_OFFICIAL_CLINC") && RUN_OFFICIAL_CLINC) {
    data$official<-load_official_clinc(g$tokenizer)
    stopifnot(identical(data$labels,data$official$labels))
  }
  if(grepl("cifar10",benchmark) && exists("RUN_OOD_EVALUATION") && RUN_OOD_EVALUATION) {
    data$ood_x<-load_svhn_ood();data$ood_cohort<-"SVHN official test; fixed balanced 1000, never used in training/selection"
  }
  if(exists("RUN_OOD_EVALUATION") && !RUN_OOD_EVALUATION)data$ood_x<-NULL
  mode<-if(method=="ordinary")"point" else if(grepl("sampled",method))"sampled" else "moment"
  if(mode=="point")g$weights<-lapply(g$weights,function(w) {
    w$lv<-NULL;w$m0<-NULL;w$v0<-NULL;w
  })
  inference<-if(grepl("_ep$",method))"ep" else "mfvi"
  run<-file.path(OUT,paste(benchmark,method,seed,sep="_"));dir.create(run,recursive=TRUE,showWarnings=FALSE)
  saveRDS(list(fit=data$fit,stop=data$stop,labels=data$labels,reporting_cohort=data$reporting_cohort),file.path(run,"split.rds"))
  stop_x<-if(is.null(data$stop_x))tf$gather(data$x,as.list(as.integer(data$stop-1L))) else data$stop_x
  stop_y<-if(is.null(data$stop_y))data$y[data$stop] else data$stop_y
  write.csv(cfg,file.path(run,"protocol.csv"),row.names=FALSE)
  g$training<-TRUE
  update<-make_update(g,inference,mode,length(data$fit),tau=cfg$tau,power=1,
    lr=cfg$mean_lr,variance_lr=cfg$variance_lr,samples=cfg$training_draws,decay=if(mode=="point")cfg$weight_decay else 0)
  ckpt<-checkpoint_object(g,update)
  best<-tf$train$CheckpointManager(ckpt,file.path(run,"best"),max_to_keep=1L)
  last<-tf$train$CheckpointManager(ckpt,file.path(run,"latest"),max_to_keep=2L)
  history<-list();best_nll<-Inf;training_seconds<-0;selected_seconds<-NA_real_;selected_epoch<-NA_integer_
  completed_epoch<-0L;recovery_file<-file.path(run,"recovery.rds")
  if(!file.exists(recovery_file) && !file.exists(file.path(run,"evaluation.rds")) &&
     !file.exists(file.path(run,"initial.index")))ckpt$write(file.path(run,"initial"))
  if(file.exists(file.path(run,"evaluation.rds"))) {
    previous<-readRDS(file.path(run,"evaluation.rds"))
    stopifnot(identical(previous$protocol,cfg))
    ckpt$restore(previous$checkpoint)$assert_existing_objects_matched()
    # Prediction requires the learned belief, not Adam/Vadam state or its
    # compiled training graph. Free those before the diagnostic forward passes.
    rm(update,ckpt,best,last);gc();import("gc")$collect()
    g$training<-FALSE
    if(!is.null(data$official) && !file.exists(file.path(run,"official_test.rds"))) {
      evaluate_official_clinc(g,data,mode,cfg,previous$checkpoint,run)
    }
    # Complete a newly requested diagnostic at the exact saved belief. Never
    # replace the completed run's predictions or claim this is new training.
    if(!is.null(data$ood_x) && !file.exists(file.path(run,"ood_predictions.rds"))) {
      ood<-evaluate(g,data$ood_x,rep(0L,shape(data$ood_x)[1]),mode,cfg$batch,cfg$prediction_draws)
      saveRDS(list(p=ood$p,cohort=data$ood_cohort,seconds=ood$seconds,
        checkpoint=previous$checkpoint),file.path(run,"ood_predictions.rds"))
    }
    if(diagnostics && mode!="point" && !file.exists(file.path(run,"prediction_draw_sweep.rds"))) {
      sweep<-prediction_draw_sweep(g,data$test_x,data$test_y,cfg$batch)
      saveRDS(sweep,file.path(run,"prediction_draw_sweep.rds"))
    }
    if(diagnostics && mode!="point" && !file.exists(file.path(run,"same_belief_diagnostic.rds"))) {
      probe<-tf$gather(data$test_x,as.integer(0:3))
      layers<-unique(as.integer(round(seq(2,length(g$nodes)+1L,length.out=6))))
      diagnostic<-if(grepl("gpt2",g$architecture))scalar_belief_diagnostic(g,probe,draws=128L)
        else same_belief_diagnostic(g,probe,layers,draws=128L)
      saveRDS(diagnostic,file.path(run,"same_belief_diagnostic.rds"))
    }
    cat("REUSE completed notebook run",run,"\n");return(invisible(run))
  }
  if(file.exists(recovery_file)) {
    recovered<-readRDS(recovery_file);stopifnot(identical(recovered$protocol,cfg))
    ckpt$restore(recovered$checkpoint)$assert_existing_objects_matched()
    history<-recovered$history;completed_epoch<-recovered$epoch
    best_nll<-recovered$best_nll;training_seconds<-recovered$training_seconds
    selected_seconds<-recovered$selected_seconds;selected_epoch<-recovered$selected_epoch
    cat("RESTORED",run,"epoch",completed_epoch,"\n")
  }
  pending_file<-file.path(run,"pending_evaluation.rds");pending<-NULL
  if(file.exists(pending_file)) {
    a<-readRDS(pending_file);stopifnot(identical(a$protocol,cfg))
    if(a$epoch>completed_epoch) {
      pending<-a;ckpt$restore(a$checkpoint)$assert_existing_objects_matched()
      history<-a$history;completed_epoch<-a$epoch-1L;best_nll<-a$best_nll
      training_seconds<-a$training_seconds;selected_seconds<-a$selected_seconds;selected_epoch<-a$selected_epoch
      cat("RESTORED completed updates; only evaluation pending for epoch",a$epoch,"\n")
    }
  }
  elapsed_start<-proc.time()[[3]]
  # Compile with a real minibatch, preserving every model/optimizer state.
  # Compare an actual eager update with the actual compiled update, not just
  # their losses. Both start at the very same checkpoint and stateless RNG key.
  a<-minibatch(data,data$fit[seq_len(cfg$batch)],TRUE,c(seed,0L))
  memory_bound_reference<-identical(g$architecture,"densenet40") ||
    (identical(g$architecture,"roberta_base_full") && inference=="ep")
  if(memory_bound_reference) {
    # The eager full-batch reference exceeds40GB on these graphs. Keep that
    # resource failure in the audit; do not weaken a numerical tolerance.
    # Compare EVERY parameter/state in float64 on two real examples, then
    # execute the actual compiled full batch and restore its initial state.
    validation_start<-proc.time()[[3]]
    strict<-double_precision_check(g,update,a,cfg,method,inference,mode,length(data$fit),run)
    stopifnot(all(strict$errors$status=="PASS"))
    validation_seconds<-proc.time()[[3]]-validation_start
    # The strict CPU replica and its compiled graph have finished their job.
    # Release both R and Python references before compiling the real GPU batch.
    gc();import("gc")$collect();gc()
    saved<-as_r(ckpt$save(file.path(run,"compile_full_batch")))
    start<-proc.time()[[3]];loss<-as.numeric(value(update$step(a$x,a$y)))
    check<-list(errors=strict$errors,seconds=proc.time()[[3]]-start)
    stopifnot(is.finite(loss));ckpt$restore(saved)$assert_existing_objects_matched()
    write.csv(data.frame(math_gate="PASS_FLOAT64",float32_equality="NOT_TESTED_FULL_BATCH_EAGER_MEMORY",
      full_batch_compiled="PASS_FINITE_UPDATE",validation_seconds=validation_seconds,
      explanation=strict$scope),file.path(run,"compilation_gate.csv"),row.names=FALSE)
  } else check<-compare_compiled_update(g,update,a,ckpt,file.path(run,"compile_float32"),3e-5,3e-4)
  compile_seconds<-check$seconds
  write.csv(check$errors,file.path(run,"compiled_update_agreement.csv"),row.names=FALSE)
  if(any(check$errors$status=="FAIL")) {
    strict<-double_precision_check(g,update,a,cfg,method,inference,mode,length(data$fit),run)
    stopifnot(all(strict$errors$status=="PASS"))
    cat("FLOAT64_UPDATE_PASS; float32 equality failures retained (ReLU/round-off sensitivity).\n")
    write.csv(data.frame(math_gate="PASS_FLOAT64",float32_equality="FAIL_RETAINED",
      explanation=paste("Strict float64 whole-update comparison; no tolerance relaxation or float64 training.",strict$scope)),
      file.path(run,"compilation_gate.csv"),row.names=FALSE)
    gc()
  } else if(!memory_bound_reference)write.csv(data.frame(math_gate="PASS_FLOAT32",float32_equality="PASS"),
      file.path(run,"compilation_gate.csv"),row.names=FALSE)
  cat("COMPILED_UPDATE_GATE",benchmark,method,"seconds",compile_seconds,"\n");flush.console()
  if(completed_epoch<cfg$epochs)for(epoch in seq.int(completed_epoch+1L,cfg$epochs)) {
    if(is.null(pending)||pending$epoch!=epoch) {
    set.seed(seed*1000L+epoch);order<-sample(data$fit)
    # Augmentation, weight draws and covariance probes are stateless below.
    # Do not reset TensorFlow's global execution context at each epoch.
    g$training<-TRUE;loss<-0;count<-0
    for(start in seq.int(1L,length(order),by=cfg$batch)) {
      ids<-order[start:min(start+cfg$batch-1L,length(order))]
      # Reference cosine schedule changes at every minibatch, without a new
      # 5% floor. Paired methods receive the same order and augmentation keys.
      rate<-learning_rate_multiplier(cfg,(epoch-1L)*ceiling(length(order)/cfg$batch)+(start-1L)%/%cfg$batch+1L,length(order))
      update$mean_optimizer$learning_rate$assign(cfg$mean_lr*rate)
      if(!is.null(update$variance_optimizer))update$variance_optimizer$learning_rate$assign(cfg$variance_lr*rate)
      t0<-proc.time()[[3]];a<-minibatch(data,ids,TRUE,c(seed*1000L+epoch,start))
      ell<-as.numeric(value(update$step(a$x,a$y)))
      training_seconds<-training_seconds+proc.time()[[3]]-t0
      if(!is.finite(ell)) stop("Nonfinite objective; failed run retained: ",run)
      loss<-loss+ell*length(ids);count<-count+length(ids)
      if((start-1L)%/%cfg$batch%%60L==0L) {
        saveRDS(list(benchmark=benchmark,method=method,epoch=epoch,
          examples_this_epoch=count,updates=as.numeric(value(update$mean_optimizer$state$t)),
          training_seconds=training_seconds,time=as.character(Sys.time())),file.path(run,"progress.rds"))
        write.csv(data.frame(benchmark=benchmark,method=method,epoch=epoch,examples_this_epoch=count,
          updates=as.numeric(value(update$mean_optimizer$state$t)),training_seconds=training_seconds,
          time=as.character(Sys.time())),file.path(run,"progress.csv"),row.names=FALSE)
        cat("PROGRESS",benchmark,method,"epoch",epoch,"examples",count,"\n");flush.console()
      }
    }
    stopifnot(count==length(data$fit))
    saved<-as_r(last$save(checkpoint_number=as.integer(epoch)))
    saveRDS(list(protocol=cfg,epoch=epoch,checkpoint=saved,history=history,best_nll=best_nll,
      training_seconds=training_seconds,selected_seconds=selected_seconds,selected_epoch=selected_epoch,
      loss=loss,count=count,rate=rate),pending_file)
    } else {
      loss<-pending$loss;count<-pending$count;rate<-pending$rate;saved<-pending$checkpoint
    }
    stopping<-evaluate(g,stop_x,stop_y,mode,cfg$batch,cfg$validation_draws)
    fit_ids<-data$fit[seq_len(min(2048L,length(data$fit)))]
    fitted<-evaluate(g,tf$gather(data$x,as.integer(fit_ids-1L)),data$y[fit_ids],mode,cfg$batch,cfg$validation_draws)
    history[[epoch]]<-data.frame(epoch=epoch,updates=as.numeric(value(update$mean_optimizer$state$t)),
      examples_this_epoch=count,objective=loss/count,
      training_subset_nll=fitted$metrics$nll,training_subset_accuracy=fitted$metrics$accuracy,
      stopping_nll=stopping$metrics$nll,stopping_accuracy=stopping$metrics$accuracy,
      training_seconds=training_seconds,evaluation_seconds=stopping$seconds+fitted$seconds,
      compilation_seconds=compile_seconds,mean_lr=cfg$mean_lr*rate,variance_lr=cfg$variance_lr*rate)
    print(history[[epoch]])
    write.csv(do.call(rbind,history),file.path(run,"learning_curves.csv"),row.names=FALSE)
    if(stopping$metrics$nll<best_nll) {
      best_nll<-stopping$metrics$nll;best$save(checkpoint_number=as.integer(epoch))
      selected_epoch<-epoch;selected_seconds<-training_seconds
    }
    saveRDS(list(protocol=cfg,epoch=epoch,checkpoint=saved,history=history,best_nll=best_nll,
      training_seconds=training_seconds,selected_seconds=selected_seconds,
      selected_epoch=selected_epoch,seed=seed,
      rng="stateless TensorFlow keys + seed/epoch R permutation; optimizer t saved"),recovery_file)
    # A real allocation deadline checkpoints and yields; it never invents a
    # completed result or starts another run after the budget expires.
    deadline<-as.numeric(Sys.getenv("BNN_STOP_UTC",unset="Inf"))
    if(as.numeric(Sys.time())>deadline && epoch<cfg$epochs)
      stop("Checkpoint-preserving allocation boundary: resume from ",recovery_file)
  }
  completed_updates<-as.numeric(value(update$mean_optimizer$state$t))
  ckpt$restore(best$latest_checkpoint)$assert_existing_objects_matched()
  final<-evaluate(g,data$test_x,data$test_y,mode,cfg$batch,draws=cfg$prediction_draws)
  selected_validation<-evaluate(g,stop_x,stop_y,mode,cfg$batch,draws=cfg$prediction_draws)
  saveRDS(list(probabilities=final$p,labels=data$test_y,metrics=final$metrics,
    benchmark=benchmark,method=method,mode=mode,prediction_draws=if(mode=="sampled")cfg$prediction_draws else if(mode=="point")1L else 0L,
    prediction_seconds=final$seconds,training_seconds=training_seconds,history=history,
    selected_epoch=selected_epoch,selected_training_seconds=selected_seconds,
    compilation_seconds=compile_seconds,total_elapsed_seconds=proc.time()[[3]]-elapsed_start,protocol=cfg,
    optimizer=if(mode=="point")"AdamW" else if(inference=="mfvi")"reference Vadam" else "Adam on mean/log variance",
    validation_probabilities=selected_validation$p,validation_labels=stop_y,
    reporting_cohort=if(is.null(data$reporting_cohort))"official test" else data$reporting_cohort,
    validation_metrics=selected_validation$metrics,
    examples_per_epoch=length(data$fit),updates=completed_updates,
    selected_updates=as.numeric(value(update$mean_optimizer$state$t)),
    source="New training executed by this notebook; no campaign checkpoint or prediction import",
    checkpoint=as_r(best$latest_checkpoint),architecture=g$architecture,seed=seed,
    target="approximate neural weight belief; categorical predictive; not exact posterior"),file.path(run,"evaluation.rds"))
  selected_checkpoint<-as_r(best$latest_checkpoint)
  rm(update,ckpt,best,last);gc();import("gc")$collect()
  cat("POST_TRAINING_EVALUATION",benchmark,method,"checkpoint",selected_checkpoint,"\n");flush.console()
  if(!is.null(data$ood_x)) {
    ood<-evaluate(g,data$ood_x,rep(0L,shape(data$ood_x)[1]),mode,cfg$batch,draws=cfg$prediction_draws)
    saveRDS(list(p=ood$p,cohort=data$ood_cohort,seconds=ood$seconds,
      checkpoint=selected_checkpoint),file.path(run,"ood_predictions.rds"))
  }
  # The expensive learned-belief audit is deliberately separate from serving.
  if(!is.null(data$official))evaluate_official_clinc(g,data,mode,cfg,selected_checkpoint,run)
  if(diagnostics && mode!="point") {
    cat("SAME_BELIEF_DIAGNOSTIC",benchmark,method,"\n");flush.console()
    probe<-tf$gather(data$test_x,as.integer(0:3))
    layers<-unique(as.integer(round(seq(2,length(g$nodes)+1L,length.out=6))))
    diagnostic<-if(grepl("gpt2",g$architecture))scalar_belief_diagnostic(g,probe,draws=128L)
      else same_belief_diagnostic(g,probe,layers,draws=128L)
    saveRDS(diagnostic,file.path(run,"same_belief_diagnostic.rds"))
    rm(diagnostic);gc();import("gc")$collect()
    cat("PREDICTION_DRAW_SWEEP",benchmark,method,"\n");flush.console()
    sweep<-prediction_draw_sweep(g,data$test_x,data$test_y,cfg$batch)
    saveRDS(sweep,file.path(run,"prediction_draw_sweep.rds"))
  }
  invisible(run)
}
train_benchmarks<-function() {
  roster<-if(exists("TRAINING_ROSTER") && !is.null(TRAINING_ROSTER))TRAINING_ROSTER else
    do.call(rbind,lapply(BENCHMARKS,function(task)
      expand.grid(method=METHODS,seed=benchmark_seeds(task),benchmark=task,stringsAsFactors=FALSE)))
  for(i in seq_len(nrow(roster))) {
    train_one(roster$benchmark[i],roster$method[i],as.integer(roster$seed[i]))
    tf$keras$backend$clear_session();gc();import("gc")$collect();gc()
  }
  summarize_new_runs()
}

# %% Official CLINC evaluation
# %% A final CLINC test, separate from training, stopping and development
load_official_clinc<-function(tokenizer) {
  ds<-import("datasets",convert=FALSE);builtins<-import_builtins(convert=FALSE)
  revision<-"155b9c710419136e17307b80d0a13e68cd46b4ec"
  test<-ds$load_dataset("clinc/clinc_oos","plus",split="test",revision=revision,token=FALSE)
  names<-as.character(as_r(test$features$get("intent")$names))
  oos<-match("oos",names)-1L;classes<-setdiff(seq_along(names)-1L,oos)
  yy<-as.integer(as_r(builtins$list(test$`__getitem__`("intent"))))
  text<-as.character(as_r(builtins$list(test$`__getitem__`("text"))))
  inside<-which(yy!=oos);outside<-which(yy==oos)
  y<-match(yy[inside],classes)-1L
  stopifnot(length(inside)==4500L,length(outside)==1000L,all(table(y)==30L))
  tokens<-tokenizer(as.list(text),padding="max_length",truncation=TRUE,max_length=128L,return_tensors="np")
  with(tf$device("/CPU:0"), {
    x<-tf$stack(list(tf$cast(tf$convert_to_tensor(tokens$get("input_ids")),tf$float32),
                    tf$cast(tf$convert_to_tensor(tokens$get("attention_mask")),tf$float32)),axis=2L)
  })
  receipt<-list(repository="clinc/clinc_oos",configuration="plus",revision=revision,split="test",
    fingerprint=as_r(test$`_fingerprint`),in_domain_rows=inside,out_of_scope_rows=outside,
    labels=names[classes+1L],selection="No test data in training, checkpoint selection or hyperparameter selection")
  saveRDS(receipt,file.path(OUT,"clinc_official_test_receipt.rds"))
  list(x=tf$gather(x,as.integer(inside-1L)),y=y,ood_x=tf$gather(x,as.integer(outside-1L)),
       labels=names[classes+1L])
}
evaluate_official_clinc<-function(g,data,mode,cfg,checkpoint,run) {
  output<-file.path(run,"official_test.rds")
  if(is.null(data$official)||file.exists(output))return(invisible(NULL))
  a<-evaluate(g,data$official$x,data$official$y,mode,cfg$batch,cfg$prediction_draws)
  b<-evaluate(g,data$official$ood_x,rep(0L,1000L),mode,cfg$batch,cfg$prediction_draws)
  saveRDS(list(p=a$p,y=data$official$y,ood=b$p,checkpoint=checkpoint,
    prediction_seconds=a$seconds,ood_seconds=b$seconds,
    cohort="official CLINC test:4500 in-domain,1000 out-of-scope; never selected on this set"),output)
}
report_official_clinc<-function(records) {
  rows<-lapply(records,function(r) {
    file<-file.path(dirname(r$file),"official_test.rds")
    if(!file.exists(file))return(NULL)
    a<-readRDS(file)
    list(p=a$p,y=a$y,ood=a$ood,label=paste(r$result$method,"seed",r$result$seed),
      method=r$result$method,seed=r$result$seed,family="Official CLINC test",seconds=a$prediction_seconds,
      training_seconds=r$result$training_seconds,selected_training_seconds=r$result$selected_training_seconds)
  })
  rows<-Filter(Negate(is.null),rows);if(!length(rows))return(invisible(NULL))
  quality<-do.call(rbind,lapply(rows,function(r)cbind(data.frame(method=r$method,seed=r$seed,
    training_seconds=r$training_seconds,selected_training_seconds=r$selected_training_seconds,
    prediction_seconds=r$seconds),as.data.frame(uncertainty_metrics(r$p,r$y)))))
  ood<-do.call(rbind,lapply(rows,function(r)cbind(data.frame(method=r$method,seed=r$seed),ood_metrics(r$p,r$ood))))
  write.csv(quality,file.path(OUT,"clinc_official_test_metrics.csv"),row.names=FALSE)
  write.csv(ood,file.path(OUT,"clinc_official_ood_metrics.csv"),row.names=FALSE)
  show_table(quality,"Official CLINC test: accuracy/AUROC higher; NLL/Brier/ECE/AURC/time lower")
  show_table(ood,"Official CLINC out-of-scope: AUROC/AUPRC higher; FPR95 lower")
  for(seed in unique(quality$seed)) {
    same<-Filter(function(r)r$seed==seed,rows)
    tag<-paste0("clinc_official_test_seed",seed)
    diagnostic_plots(same,paste0("new_",tag))
    a<-quality[quality$seed==seed,];a$family<-tag;a$label<-a$method
    a$inference_seconds<-a$prediction_seconds
    quality_comparison(a,paste("— held-out test, seed",seed))
    cost_plots(a)
    base<-Filter(function(r)r$method=="ordinary",same)
    if(length(base)==1L && length(same)>1L) {
      changes<-do.call(rbind,lapply(Filter(function(r)r$method!="ordinary",same),function(r) {
        stopifnot(identical(r$y,base[[1]]$y))
        cbind(data.frame(method=r$method,seed=seed),paired_prediction_change(r$p,base[[1]]$p,r$y))
      }))
      write.csv(changes,file.path(OUT,paste0(tag,"_paired_changes.csv")),row.names=FALSE)
      show_table(changes,"Held-out paired-example changes; positive favours BNN; not across-seed uncertainty")
    }
  }
}

# %% Same-belief Monte Carlo checks
# Fixed learned beliefs, fixed inputs, and the SAME 512 network draws.
# This file only reads saved diagnostics: no fitting or neural inference.

# Diagonal propagation stored its full covariance error, not individual v_i.
# Frobenius orthogonality recovers its variance-vector error exactly:
# ||Diag(v)-C||_F^2 = ||v-diag(C)||_2^2 + ||C-Diag(diag(C))||_F^2.
variance_rank_evidence <- function() {
  points <- summaries <- list()
  for (method in c("mfvi", "ep")) {
    run <- file.path(OUT, paste0("resnet20_cifar10_moment_", method, "_41"))
    a <- readRDS(file.path(run, "learned_rank_fidelity.rds"))
    diagonal <- read.csv(file.path(run, "learned_diagonal_fidelity.csv"))
    settings <- unique(a$table[,c("probe_set", "rank")])
    stopifnot(nrow(settings)==length(a$moments))
    for (k in seq_along(a$reference$moments)) {
      C <- a$reference$moments[[k]]$mc_covariance
      S <- a$reference$samples[[k]]
      # Delta-method uncertainty for the unbiased sample variance.
      n <- nrow(S); centered2 <- sweep(S,2,colMeans(S),"-")^2
      se <- apply(centered2,2,sd)/sqrt(n)*n/(n-1)
      denominator <- sqrt(sum(diag(C)^2))
      d2 <- diagonal$covariance_relative[k]^2*sum(C^2) -
            sum((C-diag(diag(C)))^2)
      stopifnot(d2 >= -1e-12*sum(C^2), denominator>0)
      summaries[[length(summaries)+1L]] <- data.frame(method=method,
        node=a$table$node[k], boundary=a$table$name[k], representation="Layerwise diagonal",
        probe_set="not applicable", rank=0L, variance_relative=sqrt(max(0,d2))/denominator,
        covariance_relative=diagonal$covariance_relative[k],
        mean_relative=diagonal$mean_relative[k], variance_reference_mcse=sqrt(sum(se^2))/denominator)
      for (j in seq_along(a$moments)) {
        z <- a$moments[[j]][[k]]
        stopifnot(max(abs(z$mc_covariance-C))<1e-12)
        v <- diag(z$covariance)
        points[[length(points)+1L]] <- data.frame(method=method,
          node=a$table$node[k], boundary=a$table$name[k],
          probe_set=settings$probe_set[j], rank=settings$rank[j],
          coordinate=seq_along(v), mc_variance=diag(C), moment_variance=v, mcse=se)
        summaries[[length(summaries)+1L]] <- data.frame(method=method,
          node=a$table$node[k], boundary=a$table$name[k], representation="Shared activation factors",
          probe_set=settings$probe_set[j], rank=settings$rank[j],
          variance_relative=sqrt(sum((v-diag(C))^2))/denominator,
          covariance_relative=sqrt(sum((z$covariance-C)^2)/sum(C^2)),
          mean_relative=sqrt(sum((z$mean-z$mc_mean)^2)/sum(z$mc_mean^2)),
          variance_reference_mcse=sqrt(sum(se^2))/denominator)
      }
    }
  }
  points <- do.call(rbind,points); summaries <- do.call(rbind,summaries)
  write.csv(points,file.path(OUT,"activation_variance_coordinates.csv"),row.names=FALSE)
  write.csv(summaries,file.path(OUT,"activation_variance_rank_summary.csv"),row.names=FALSE)
  list(points=points,summary=summaries)
}

# Predetermined boundaries: the third of six recorded states, and the readout.
# Show rank 1 versus 64 with identical axes, including zero, at each boundary.
variance_rank_plots <- function(a) {
  for (method in c("mfvi","ep")) {
    p <- a$points[a$points$method==method & a$points$probe_set=="training keys",]
    nodes <- sort(unique(p$node))[c(3L,6L)]
    figure_pair(paste0("tutorial_",method,"_variance_rank_scatter"),function() {
      par(mfrow=c(2,2),mar=c(4.4,4.7,3.0,.8),oma=c(0,0,2.2,0),las=1,pty="s")
      for (node in nodes) {
        q <- p[p$node==node & p$rank %in% c(1L,64L),]
        upper <- 1.45*max(q$moment_variance,q$mc_variance+2*q$mcse)
        for (rank in c(1L,64L)) {
          z <- q[q$rank==rank,]
          plot(z$mc_variance,z$moment_variance,xlim=c(0,upper),ylim=c(0,upper),asp=1,
            xlab="Monte Carlo variance",ylab="Propagated moment variance",pch=16,xaxs="i",yaxs="i",
            col=if(rank==1L)"#D55E00" else "#0072B2",
            main=paste(if(node==nodes[1])"Middle-layer" else "Readout", "variance: rank",rank))
          segments(pmax(0,z$mc_variance-2*z$mcse),z$moment_variance,
                   z$mc_variance+2*z$mcse,z$moment_variance,col="#888888")
          abline(0,1,col="#444444",lty=2)
          error <- sqrt(sum((z$moment_variance-z$mc_variance)^2)/sum(z$mc_variance^2))
          legend("topleft",c(sprintf("Relative variance error: %.1f%%",100*error),
            "Identity", "Horizontal bars: +/-2 MCSE"),lty=c(NA,2,1),
            col=c("black","#444444","#888888"),cex=.77,bg="white",box.col="grey80")
        }
      }
      mtext(paste("ResNet-20",toupper(method),"- fixed belief and 512 shared-network draws"),
            outer=TRUE,line=.55,font=2,cex=1.03)
    },width=10,height=8.4)
  }
}

# These are LIKELIHOOD MOMENT SIGNALS, before an optimizer or prior update.
# EP uses the lower-noise derivative estimator of the same log normalizer.
backward_agreement_evidence <- function() {
  rows <- noise <- list()
  for (method in c("mfvi","ep")) {
    run <- file.path(OUT,paste0("resnet20_cifar10_moment_",method,"_41"))
    a <- read.csv(file.path(run,"backward_rank_fidelity.csv"))
    for (rank in sort(unique(a$rank))) for (signal in c("mean","raw_second")) {
      z <- a[a$rank==rank,]
      ref <- z[[paste0(if(method=="ep")"derivative_delta_" else "mc_delta_",signal)]]
      se <- z[[paste0(if(method=="ep")"derivative_mcse_" else "mcse_delta_",signal)]]
      pred <- z[[paste0("moment_delta_",signal)]]
      # Center before measuring spread: a nearly constant vector has tiny SD,
      # whereas an attenuated copy can have high correlation but slope < 1.
      xc <- ref-mean(ref); yc <- pred-mean(pred)
      slope <- sum(xc*yc)/sum(xc^2)
      rows[[length(rows)+1L]] <- data.frame(method=method,rank=rank,signal=signal,
        relative_error=sqrt(sum((pred-ref)^2)/sum(ref^2)),
        reference_relative_mcse=sqrt(sum(se^2)/sum(ref^2)),
        norm_ratio=sqrt(sum(pred^2)/sum(ref^2)),
        cosine=sum(pred*ref)/sqrt(sum(pred^2)*sum(ref^2)),
        within_two_mcse=mean(abs(pred-ref)<=2*se),coordinates=nrow(z),
        correlation=cor(ref,pred),spread_ratio=sd(pred)/sd(ref),
        slope=slope,intercept=mean(pred)-slope*mean(ref),
        linear_R2=cor(ref,pred)^2,distinct_moment_values=length(unique(pred)))
    }
    if(method=="ep") {
      z <- read.csv(file.path(run,"learned_backward_fidelity.csv"))
      for(signal in c("mean","raw_second")) noise[[length(noise)+1L]] <- data.frame(
        signal=signal,raw_mc_signal_norm=sqrt(sum(z[[paste0("mc_delta_",signal)]]^2)),
        raw_mcse_norm=sqrt(sum(z[[paste0("mcse_delta_",signal)]]^2)),
        derivative_signal_norm=sqrt(sum(z[[paste0("derivative_delta_",signal)]]^2)),
        derivative_mcse_norm=sqrt(sum(z[[paste0("derivative_mcse_",signal)]]^2)))
    }
  }
  answer <- do.call(rbind,rows); noise <- do.call(rbind,noise)
  write.csv(answer,file.path(OUT,"backward_agreement_summary.csv"),row.names=FALSE)
  write.csv(noise,file.path(OUT,"ep_backward_reference_noise.csv"),row.names=FALSE)
  answer
}

backward_refinement_plots <- function() {
  for (method in c("mfvi","ep")) {
    run <- file.path(OUT,paste0("resnet20_cifar10_moment_",method,"_41"))
    a <- read.csv(file.path(run,"backward_rank_fidelity.csv"))
    figure_pair(paste0("tutorial_",method,"_backward_refinement"),function() {
      par(mfrow=c(2,2),mar=c(4.5,4.9,3,.8),oma=c(0,0,2.3,0),las=1,pty="s")
      for(rank in c(1L,64L)) for(signal in c("mean","raw_second")) {
        z <- a[a$rank==rank,]
        ref <- z[[paste0(if(method=="ep")"derivative_delta_" else "mc_delta_",signal)]]
        se <- z[[paste0(if(method=="ep")"derivative_mcse_" else "mcse_delta_",signal)]]
        pred <- z[[paste0("moment_delta_",signal)]]
        # SAME units on x and y. Rank 64 gets its own expanded window: rank-1
        # outliers must not compress the refined signals. No coordinate omitted.
        scale <- sqrt(sum(ref^2))
        limits <- range(0,ref-2*se,ref+2*se,pred)/scale
        limits <- limits+c(-.08,.36)*diff(limits)
        plot(ref/scale,pred/scale,xlim=limits,ylim=limits,asp=1,pch=16,cex=.45,col="#0072B2",
          xlab="MC signal / reference norm",ylab="Moment signal / reference norm",
          main=paste(if(signal=="mean")"Mean signal" else "Raw-second signal",
                     if(rank==1L)"- rank 1" else "- rank 64 (expanded)"))
        segments((ref-2*se)/scale,pred/scale,(ref+2*se)/scale,pred/scale,col=adjustcolor("#888888",.4))
        abline(0,1,lty=2,col="#444444")
        error <- sqrt(sum((pred-ref)^2)/sum(ref^2))
        xc <- ref-mean(ref); yc <- pred-mean(pred)
        slope <- sum(xc*yc)/sum(xc^2)
        intercept <- mean(pred)-slope*mean(ref)
        abline(intercept/scale,slope,col="#D55E00",lwd=1.6)
        legend(if(rank==1L)"bottomright" else "topleft",c(sprintf("Slope %.3f; correlation %.3f",slope,cor(ref,pred)),
          sprintf("Spread: %.3f x MC; error %.1f%%",sd(pred)/sd(ref),100*error),
          "Identity", "Fitted line (descriptive)", "Horizontal bars: +/-2 MCSE"),
          lty=c(NA,NA,2,1,1),col=c("black","black","#444444","#D55E00","#888888"),
          cex=.72,bg="white",box.col="grey80")
      }
      mtext(paste("ResNet-20",toupper(method),"- same factor and sampled reference across ranks"),
            outer=TRUE,line=.6,font=2,cex=1.01)
    },width=10,height=8.4)
  }
}

