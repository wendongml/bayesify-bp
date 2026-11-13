# %% Compute forward approximation errors
# One learned diagonal-Gaussian weight belief per method; identical MC draws
# across all covariance ranks. Error = ||approximation-reference|| / ||reference||.
forward_accuracy_cost <- function() {
  if(!file.exists(file.path(OUT,"resnet20_cifar10_moment_mfvi_41/learned_rank_fidelity.rds")))
    return(list(summary=read.csv(file.path(OUT,"approximation_cost/forward_summary.csv")),
                coordinates=read.csv(file.path(OUT,"approximation_cost/forward_coordinates.csv"))))
  rows <- coordinates <- list()
  for (method in c("mfvi", "ep")) {
    path <- file.path(OUT, paste0("resnet20_cifar10_moment_",method,"_41"))
    a <- readRDS(file.path(path,"learned_rank_fidelity.rds"))
    settings <- a$table[a$table$name=="classifier",]
    k <- length(a$reference$moments)
    S <- a$reference$samples[[k]]; n <- nrow(S)
    m <- colMeans(S); v <- diag(cov(S))
    mean_se <- sqrt(v/n)
    variance_se <- apply(sweep(S,2,m,"-")^2,2,sd)/sqrt(n)*n/(n-1)
    for (j in seq_len(nrow(settings))) {
      z <- a$moments[[j]][[k]]
      rows[[length(rows)+1L]] <- data.frame(method=method,
        rank=settings$rank[j], sketch=settings$probe_set[j],
        seconds=settings$diagnostic_seconds[j],
        mean_error=100*sqrt(sum((z$mean-m)^2)/sum(m^2)),
        variance_error=100*sqrt(sum((diag(z$covariance)-v)^2)/sum(v^2)),
        mean_mcse=100*sqrt(sum(mean_se^2)/sum(m^2)),
        variance_mcse=100*sqrt(sum(variance_se^2)/sum(v^2)))
      coordinates[[length(coordinates)+1L]] <- data.frame(method=method,
        rank=settings$rank[j],sketch=settings$probe_set[j],coordinate=seq_along(v),
        reference=v,approximation=diag(z$covariance),reference_mcse=variance_se)
    }
  }
  a <- do.call(rbind,rows)
  # Curves and scatters use the original configured sketch. The independent
  # sketch is preserved in the full CSV; neither is chosen by its accuracy.
  summary <- a[a$sketch=="training keys",]
  coordinates <- do.call(rbind,coordinates)
  write.csv(a,file.path(OUT,"approximation_cost/forward_sketches.csv"),row.names=FALSE)
  write.csv(summary,file.path(OUT,"approximation_cost/forward_summary.csv"),row.names=FALSE)
  write.csv(coordinates,file.path(OUT,"approximation_cost/forward_coordinates.csv"),row.names=FALSE)
  list(summary=summary,coordinates=coordinates)
}

# %% Compute backward approximation errors
# Each row stores the average of four dependent antithetic/shifted nodes.
# Different rows are independent groups. For EP, numerator/denominator is
# the derivative of its normalizer divided by that normalizer; for VI the
# denominator is one. Both sampling estimators use exactly the same draws.
backward_accuracy_cost <- function() {
  if(!file.exists(file.path(OUT,"approximation_cost/backward_groups.rds")))
    return(read.csv(file.path(OUT,"approximation_cost/backward_summary.csv")))
  saved <- readRDS(file.path(OUT,"approximation_cost/backward_groups.rds"))
  rows <- list()
  for (a in saved) for (budget in c(32L,64L,128L,256L,512L,1024L,2048L)) {
    size <- budget/4L
    groups <- split(seq_len(nrow(a$sampling$numerator)),
                    rep(seq_len(nrow(a$sampling$numerator)/size),each=size))
    ref <- a$reference[,1]; ref_se <- a$reference[,2]
    for (estimator in c("sampling","corrected")) for (b in seq_along(groups)) {
      ids <- groups[[b]]; z <- a[[estimator]]
      denominator <- sum(z$denominator[ids])
      stopifnot(denominator>0)
      estimate <- colSums(z$numerator[ids,,drop=FALSE])/denominator
      rows[[length(rows)+1L]] <- data.frame(method=a$method,input=a$input,
        estimator=estimator,budget=budget,batch=b,
        squared_relative_error=sum((estimate-ref)^2)/sum(ref^2),
        squared_reference_mcse=sum(ref_se^2)/sum(ref^2))
    }
  }
  a <- do.call(rbind,rows)
  groups <- split(a,list(a$method,a$estimator,a$budget),drop=TRUE)
  summary <- do.call(rbind,lapply(groups,function(z) data.frame(
    method=z$method[1],estimator=z$estimator[1],budget=z$budget[1],batches=nrow(z),
    error=100*sqrt(mean(z$squared_relative_error)),
    reference_mcse=100*sqrt(mean(z$squared_reference_mcse)))))
  write.csv(a,file.path(OUT,"approximation_cost/backward_batches.csv"),row.names=FALSE)
  write.csv(summary,file.path(OUT,"approximation_cost/backward_summary.csv"),row.names=FALSE)
  summary
}

# %% Read per-example benchmark predictions
# Saved per-example predictions, not a substituted performance table.
files <- list.files(OUT, pattern="^evaluation[.]rds$", recursive=TRUE, full.names=TRUE)
files <- files[grepl("/(resnet20|densenet40)_cifar10_",files)]
tutorial_records <- lapply(files, function(file) {
  a <- readRDS(file)
  if(is.null(a$benchmark))return(NULL)
  o <- file.path(dirname(file), "ood_predictions.rds")
  list(p=a$probabilities, y=a$labels, label=a$method,
       family=a$benchmark, result=a, file=file,
       ood=if(file.exists(o)) readRDS(o)$p else NULL)
})
tutorial_records <- Filter(Negate(is.null),tutorial_records)
curves <- do.call(rbind, lapply(tutorial_records, function(r) {
  h <- do.call(rbind, r$result$history)
  data.frame(family=r$family, cohort=paste0("new_R_training_seed",r$result$seed),
    label=r$result$method, seed=r$result$seed, epoch=h$epoch,
    training_seconds=h$training_seconds, validation_nll=h$stopping_nll,
    validation_accuracy=h$stopping_accuracy)
}))
write.csv(curves, file.path(OUT,"new_learning_curves.csv"), row.names=FALSE)

# Every row uses the probabilities of the learned checkpoint on the same cohort.
# Training time covers its complete recorded schedule; prediction is native serving.
benchmark_summary <- function(records) {
  a <- do.call(rbind, lapply(records, function(r) {
    m <- uncertainty_metrics(r$p, r$y)
    data.frame(Method=r$label, Accuracy=100*m$accuracy, NLL=m$nll,
      ECE=100*m$ece, Brier=m$brier, AURC=m$aurc,
      Train_s=r$result$training_seconds,
      Predict_s=if(is.null(r$seconds)) r$result$prediction_seconds else r$seconds)
  }))
  base <- a[a$Method=="ordinary", ]
  a$Train_x <- a$Train_s/base$Train_s
  a$Predict_x <- a$Predict_s/base$Predict_s
  a$Method <- vapply(a$Method, function(x) method_style(x)$name, character(1))
  names(a) <- c("Method", "Accuracy % (up)", "NLL (down)", "ECE % (down)",
    "Brier (down)", "AURC (down)", "Training s (down)", "Prediction s (down)",
    "Training / NN (down)", "Prediction / NN (down)")
  a
}

# %% Read selected GPT-2 predictions and settings
# Recompute the pinned GPT-2 campaign from saved per-example probabilities.
# R4 averages METRICS over three covariance-probe seeds, not probabilities.
# These PyTorch runs and the R/TensorFlow reproduction have separate provenance.
gpt2_selected_evidence <- function() {
  path <- file.path(OUT,"gpt2_selected_campaign")
  p <- read.csv(file.path(path,"predictions.csv"))
  cost <- read.csv(file.path(path,"recorded_costs.csv"))
  reference <- read.csv(file.path(path,"reference_metrics.csv"))
  metrics <- do.call(rbind,lapply(cost$key,function(key) {
    z <- p[p$key==key,]
    seeds <- sort(unique(z$probe_seed))
    do.call(rbind,lapply(seeds,function(seed) {
      d <- z[z$probe_seed==seed,];d <- d[order(d$row),]
      stopifnot(identical(d$row,1:872))
      data.frame(key=key,probe_seed=seed,
        sentiment_auroc=ranking_curve(d$p1,d$label==1L)$auc,
        as.data.frame(uncertainty_metrics(cbind(1-d$p1,d$p1),d$label)))
    }))
  }))
  metric_names <- setdiff(names(metrics),c("key","probe_seed"))
  average <- aggregate(metrics[metric_names],list(key=metrics$key),mean)
  average <- average[match(cost$key,average$key),]
  reference <- reference[match(cost$key,reference$key),]
  stopifnot(max(abs(average$accuracy-reference$accuracy))<1e-7,
    max(abs(average$nll-reference$nll))<1e-6,
    max(abs(average$ece-reference$ece))<1e-6)
  a <- cbind(cost,average[metric_names])
  a$training_multiple <- a$training_seconds/a$training_seconds[1]
  a$inference_multiple <- a$inference_seconds/a$inference_seconds[1]
  write.csv(a,file.path(path,"recomputed_metrics.csv"),row.names=FALSE)
  write.csv(metrics,file.path(path,"recomputed_metrics_by_probe.csv"),row.names=FALSE)
  list(summary=a,predictions=p,
    config=read.csv(file.path(path,"configurations.csv")),
    curves=read.csv(file.path(path,"learning_curves.csv")))
}

# %% Read the original transformer prediction records
# Saved single-Gaussian LoRA experiments, not a new fit or an R training port.
# All scores below are recomputed from full per-example probability vectors.
compact_transformer_evidence <- function() {
  path <- file.path(OUT,"compact_transformers")
  recorded <- read.csv(file.path(path,"prediction.csv"))
  steps <- read.csv(file.path(path,"training_steps.csv"))
  fits <- read.csv(file.path(path,"historical_full_fit_cost.csv"))
  rows <- list(); predictions <- list()
  for(task in c("sst2","clinc")) for(method in c("ordinary","hybrid")) {
    key <- paste(task,method,sep="_")
    d <- readRDS(file.path(path,paste0(key,".rds")))
    u <- uncertainty_metrics(d$p,d$y)
    r <- recorded[recorded$task==task & recorded$method==method,]
    stopifnot(abs(u$accuracy-r$accuracy)<1e-10,abs(u$nll-r$nll)<1e-7,
              abs(u$ece-r$ece15)<1e-7)
    task_name <- if(task=="sst2")"GPT-2 / SST-2" else "RoBERTa / CLINC"
    fit <- fits[fits$task==task_name & fits$method==if(method=="ordinary")"Ordinary" else "Revised K8",]
    mode <- if(method=="hybrid")"antithetic_shared" else
            if(task=="sst2")"ordinary_compiled" else "ordinary_sdpa"
    step <- steps[steps$task==task & steps$mode==mode,]
    stopifnot(nrow(step)==3L,nrow(fit)==1L)
    rows[[length(rows)+1L]] <- data.frame(task=task,method=method,
      predictor=if(method=="ordinary")"Ordinary AdamW" else
        if(task=="sst2")"Revised moment VI, K1" else "Revised moment VI, K2",
      as.data.frame(u),sentiment_auroc=if(task=="sst2")ranking_curve(d$p[,2],d$y==1L)$auc else NA_real_,
      fit_seconds=fit$training_seconds,fit_gpu=fit$hardware,
      step_ms=1000*mean(step$seconds),inference_seconds=r$seconds,setup_seconds=r$setup_seconds)
    predictions[[key]] <- d
  }
  a <- do.call(rbind,rows)
  for(task in c("sst2","clinc")) {
    i<-which(a$task==task)
    a$step_multiple[i] <- a$step_ms[i]/a$step_ms[i[1]]
    a$inference_multiple[i] <- a$inference_seconds[i]/a$inference_seconds[i[1]]
  }
  write.csv(a,file.path(path,"recomputed_metrics.csv"),row.names=FALSE)
  list(summary=a,predictions=predictions,intervals=read.csv(file.path(path,"paired_intervals.csv")))
}

# %% Common transformer tables and metrics
# Use the same uncertainty table and four-panel plot as ResNet and DenseNet.
# Different training durations keep their own ordinary comparison and cost.
transformer_records <- function(task) {
  compact<-compact_transformer_evidence(); records<-list()
  if(task=="sst2") {
    saved<-gpt2_selected_evidence()
    for(key in c("adamw","sampled_mfvi","moment_ep")) {
      p<-saved$predictions[saved$predictions$key==key,]
      p<-p[order(p$row),];a<-saved$summary[saved$summary$key==key,]
      label<-c(adamw="Ordinary AdamW, 1 epoch",sampled_mfvi="Sampled MFVI, 1 epoch",
               moment_ep="Moment power-EP, 1 epoch")[[key]]
      records[[length(records)+1L]]<-list(p=cbind(1-p$p1,p$p1),y=p$label,label=label,
        epochs=1L,train=a$training_seconds,infer=a$inference_seconds,fit_gpu="A100",control=1L)
    }
  }
  for(method in c("ordinary","hybrid")) {
    d<-compact$predictions[[paste(task,method,sep="_")]]
    a<-compact$summary[compact$summary$task==task & compact$summary$method==method,]
    label<-if(method=="ordinary")"Ordinary AdamW" else "Moment MFVI"
    if(task=="sst2")label<-paste0(label,", 3 epochs")
    records[[length(records)+1L]]<-list(p=d$p,y=d$y,label=label,
      epochs=if(task=="sst2")3L else 6L,train=a$fit_seconds,infer=a$inference_seconds,
      fit_gpu=a$fit_gpu,control=if(task=="sst2")4L else 1L)
  }
  records
}

transformer_table <- function(records,task) {
  a<-do.call(rbind,lapply(records,function(r) {
    u<-uncertainty_metrics(r$p,r$y);base<-records[[r$control]]
    data.frame(Method=method_style(r$label)$name,Epoch=r$epochs,
      Accuracy=100*u$accuracy,NLL=u$nll,ECE=100*u$ece,
      Brier=u$brier,AURC=u$aurc,Error_AUROC=u$error_auroc,
      Error_AUPRC=u$error_auprc,Risk80=u$risk80,
      Train_s=r$train,Infer_s=r$infer,Train_x=r$train/base$train,Infer_x=r$infer/base$infer,
      Fit_GPU=r$fit_gpu)
  }))
  write.csv(a,file.path(OUT,paste0("tutorial_",task,"_summary.csv")),row.names=FALSE)
  show_table(a[,c("Method","Epoch","Accuracy","NLL","ECE","Train_s","Infer_s","Train_x","Infer_x")] |>
    setNames(c("Method","Epoch","Accuracy % (up)","NLL (down)","ECE % (down)",
      "Training s (down)","Prediction s (down)","Training / NN (down)","Prediction / NN (down)")),
    "Predictive quality and recorded cost; ratios use the corresponding ordinary protocol")
  show_table(a[,c("Method","Brier","AURC","Error_AUROC","Error_AUPRC","Risk80")] |>
    setNames(c("Method","Brier (down)","AURC (down)","Error AUROC (up)",
      "Error AUPRC (up)","Risk at 80% (down)")),
    "Uncertainty from the same saved predictions")
  invisible(a)
}

# %% Native Keras summaries and expanded diagrams
# Native model inspection. This file never draws architecture boxes or arrows.
# Saved output is from tf.keras.utils.plot_model() and Model.summary().
native_architecture <- function(name, rebuild=FALSE) {
  path<-file.path(OUT,"architectures",name)
  if(rebuild) {
    g<-switch(name,resnet20=build_cifar("resnet20",1L,41L),
      densenet40=build_cifar("densenet40",1L,41L),
      gpt2=build_gpt2_lora(1L,41L),roberta=build_roberta(1L,41L),
      roberta_lora=build_roberta_lora(1L,41L))
    stopifnot(!is.null(g))
    x<-if(name %in% c("resnet20","densenet40"))
      tf$random$stateless_normal(c(2L,32L,32L,3L),c(1L,2L)) else
      tf$constant(array(c(0,2,30,40,50,60,70,80,rep(1,8)),c(1,8,2)),dtype=tf$float32)
    view<-keras_inspection(g,x)
    view$module$write_inspection(view$model,path)
    view$model$summary(expand_nested=FALSE,line_length=78L)
  } else {
    # Verbatim native summary, not a separately maintained table of layers.
    cat(readLines(file.path(path,"inline_summary.txt")),sep="\n")
  }
  invisible(path)
}

# %% Build the native Keras inspection view
# The Keras leaves reuse node$point and the existing weight variables.
# The grouping below names scopes only. Edges come from node$parents.
keras_scope <- function(name, architecture) {
  if(architecture=="resnet20") {
    if(grepl("^stage[1-3]_block[1-3]",name)) {
      block<-sub("^(stage[1-3]_block[1-3]).*","\\1",name)
      return(c(sub("_block.*","",block),block))
    }
    return(if(grepl("^stem",name))"stem" else "readout")
  }
  if(architecture=="densenet40") {
    if(grepl("^dense[1-3]_",name)) {
      block<-sub("^(dense[1-3]).*","\\1",name)
      unit<-sub("^(dense[1-3]_[0-9]+).*","\\1",name)
      return(c(block,unit))
    }
    if(grepl("^transition[12]_",name))return(sub("^(transition[12]).*","\\1",name))
    return(if(name=="stem")"stem" else "readout")
  }
  if(grepl("^h\\.[0-9]+\\.",name))
    return(c("decoder_12_blocks",sub("^(h\\.[0-9]+).*","\\1",name)))
  if(grepl("^roberta.encoder.layer\\.[0-9]+\\.",name))
    return(c("encoder_12_blocks",sub("^roberta.encoder.layer\\.([0-9]+).*","encoder_\\1",name)))
  if(grepl("embedding",name))return("embeddings")
  "readout"
}

keras_inspection <- function(g, example, helper="python/keras_graph.py") {
  # Execute the same graph once to obtain actual tensor shapes, not guessed ones.
  states<-point_forward(g,example,record=TRUE)
  nodes<-lapply(g$nodes,function(op) {
    buffers<-if(is.null(op$normalization))list() else list(op$normalization$m,op$normalization$v)
    # Frozen constants are captured by the existing callback. Give them tracked
    # read-only Variable storage so Keras includes them in its native summary.
    # The callback itself is unchanged and now reads that same storage.
    for(name in codetools::findGlobals(op$point,merge=FALSE)$variables) {
      en<-environment(op$point)
      if(!exists(name,en,inherits=FALSE))next
      a<-get(name,en)
      if(!inherits(a,"python.builtin.object") || !py_has_attr(a,"dtype") ||
         !py_has_attr(a,"shape") || !isTRUE(as_r(tf$is_tensor(a))))next
      if(!py_has_attr(a,"assign")) {
        a<-tf$Variable(a,trainable=FALSE,name=paste0(gsub("\\.","_",op$name),"_",name))
        assign(name,a,en)
      }
      buffers[[length(buffers)+1L]]<-a
    }
    callback<-function(xs,ws)op$point(as_r(xs),as_r(ws))
    list(name=op$name,parents=as.list(as.integer(op$parents)),callback=callback,
      parameters=lapply(g$weights[op$weights],`[[`,"m"),buffers=buffers,
      group=as.list(gsub("\\.","_",keras_scope(op$name,g$architecture))))
  })
  module<-import_from_path("keras_graph",path=dirname(helper),convert=FALSE)
  model<-module$from_graph(nodes,lapply(states,function(x)as.list(shape(x))),g$architecture)
  actual<-model(list(example),training=FALSE)
  if(inherits(actual,"python.builtin.list"))actual<-actual[[0L]]
  discrepancy<-max(abs(value(actual)-value(tail(states,1)[[1]])))
  stopifnot(is.finite(discrepancy),discrepancy<1e-5)
  parameters<-unname(c(lapply(g$weights,`[[`,"m"),g$point_weights))
  errors<-as_r(module$verify_nodes(model,nodes,states,parameters))
  list(model=model,module=module,max_absolute_difference=discrepancy,
       node_errors=data.frame(node=vapply(g$nodes,`[[`,character(1),"name"),max_abs=unlist(errors)))
}

# %% Read new original-PyTorch training outputs
# The displayed historical transformer fits were trained in Python/PyTorch.
# These functions read NEW outputs from those original training implementations.
# They never substitute historical scores when a new training run fails.
fresh_transformer_records <- function(task) {
  records<-list()
  if(task=="sst2")for(key in c("adamw","sampled_mfvi","moment_ep")) {
    cfg<-jsonlite::read_json(file.path("python/gpt2",paste0(key,".json")),simplifyVector=TRUE)
    folder<-file.path("python/gpt2/runs",cfg$name)
    a<-np$load(file.path(folder,"epoch1_predictions.npz"));h<-read.csv(file.path(folder,"learning_curves.csv"))
    records[[length(records)+1L]]<-list(p=as_r(a$get("probabilities")),y=as.integer(as_r(a$get("labels"))),
      label=c(adamw="Ordinary AdamW, 1 epoch",sampled_mfvi="Sampled MFVI, 1 epoch",moment_ep="Moment power-EP, 1 epoch")[[key]],
      epochs=1L,train=h$training_seconds[1],infer=h$inference_seconds[1],fit_gpu="Current runtime",control=1L)
  }
  for(method in c("ordinary","hybrid")) {
    folder<-file.path("python/blob_mirror/results",task,paste0("v3_",method,"_seed41"))
    p<-np$load(file.path(folder,"native_predictions.npz"))
    a<-jsonlite::read_json(file.path(folder,"native_prediction.json"),simplifyVector=TRUE)
    label<-if(method=="ordinary")"Ordinary AdamW" else "Moment MFVI"
    if(task=="sst2")label<-paste0(label,", 3 epochs")
    records[[length(records)+1L]]<-list(p=as_r(p$get("probabilities")),y=as.integer(as_r(p$get("labels"))),
      label=label,epochs=a$epoch,train=a$training_seconds,infer=a$inference_seconds,fit_gpu=a$hardware,
      control=if(task=="sst2")4L else 1L)
  }
  records
}

