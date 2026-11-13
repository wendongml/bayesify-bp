# %% Table display
# Presentation only: no model, inference, result selection or scoring here.
show_table <- function(a,caption) {
  if(nzchar(Sys.getenv("BAYESIFY_PYTHON_NOTEBOOK")) || (requireNamespace("IRdisplay",quietly=TRUE) && nzchar(Sys.getenv("JPY_PARENT_PID")))) {
    escape<-function(s)gsub(">","&gt;",gsub("<","&lt;",gsub("&","&amp;",s,fixed=TRUE),fixed=TRUE),fixed=TRUE)
    a[]<-lapply(a,function(x)if(is.numeric(x))format(x,digits=5,trim=TRUE) else as.character(x))
    header<-paste0("<th>",escape(names(a)),"</th>",collapse="")
    rows<-apply(a,1,function(x)paste0("<tr>",paste0("<td>",escape(x),"</td>",collapse=""),"</tr>"))
    html<-paste0("<h4>",escape(caption),"</h4><table><thead><tr>",
      header,"</tr></thead><tbody>",paste(rows,collapse=""),"</tbody></table>")
    if(nzchar(Sys.getenv("BAYESIFY_PYTHON_NOTEBOOK"))) {
      display<-import("IPython.display",convert=FALSE);display$display(display$HTML(html))
    } else IRdisplay::display_html(html)
  } else {cat(caption,"\n");print(a,digits=5,row.names=FALSE)}
}

# %% Shared calibration, selection and ROC plotting
# Same learned beliefs and fixed directions as the other native implementations.
# CSV values are moments/messages, never weights; the common MC reference and
# paired readout correction are explicitly attributed to their recorded run.
matched_approximation_plots <- function(path=file.path(OUT,"matched")) {
  a<-read.csv(file.path(path,"coordinates.csv"))
  a<-a[a$backend=="tensorflow",]
  errors<-read.csv(file.path(path,"approximation_errors.csv"))
  errors<-errors[errors$backend=="tensorflow",]
  colours<-c("#0072B2","#D55E00")
  scatter<-function(method,quantity,schemes,labels) {
    d<-a[a$method==method & a$quantity==quantity & a$scheme %in% schemes,]
    bounds<-range(c(d$reference,d$estimate));pad<-diff(bounds)*.05
    bounds<-bounds+c(-pad,pad)
    plot(NA,xlim=bounds,ylim=bounds,asp=1,xlab="MC reference",ylab="Computed value",
         main=paste(toupper(method),switch(quantity,covariance="logit variance",
                     mean_signal="mean signal",variance_signal="variance signal")))
    abline(0,1,lty=2,col="grey40")
    for(i in seq_along(schemes)) {
      z<-d[d$scheme==schemes[i],]
      points(z$reference,z$estimate,pch=c(16,17)[i],cex=.55,
             col=adjustcolor(colours[i],alpha.f=.65))
    }
    legend("topleft",labels,col=colours,pch=c(16,17),cex=1,bg="white",bty="n")
  }
  figure_pair("matched_backward",function() {
    par(mfrow=c(2,2),mar=c(4,4,3,1))
    for(quantity in c("mean_signal","variance_signal"))for(method in c("mfvi","ep"))
      scatter(method,quantity,c("corrected_s4096","corrected_s16384"),
              c("Paired correction: 4,096","Paired correction: 16,384"))
  },width=9,height=9)
  figure_pair("matched_forward",function() {
    par(mfrow=c(1,2),mar=c(4,4,3,1))
    for(method in c("mfvi","ep"))scatter(method,"covariance",c("r1","r64"),
                     c("Ablation: 1 direction","Ablation: 64 directions"))
  },width=9,height=4.5)
  figure_pair("matched_rank",function() {
    par(mfrow=c(1,2),mar=c(4,4,3,1))
    kinds<-c("covariance","mean_signal","variance_signal")
    for(method in c("mfvi","ep")) {
      d<-errors[errors$method==method & errors$quantity %in% kinds,]
      plot(NA,xlim=c(1,64),ylim=range(100*d$relative_error),log="xy",xaxt="n",
           xlab="Retained directions",ylab="Relative error (%)",main=paste(toupper(method),"closure ablation"))
      axis(1,c(1,4,16,64),c(1,4,16,64))
      for(i in seq_along(kinds)) {
        z<-d[d$quantity==kinds[i],];lines(z$rank,100*z$relative_error,
             type="b",col=c(colours,"#009E73")[i],pch=i+14)
      }
      legend("topleft",c("Output covariance","Readout mean","Readout variance"),
             col=c(colours,"#009E73"),pch=15:17,lty=1,cex=.7,bg="white",bty="n")
    }
  },width=9,height=4.5)
}

# %% Figures are derived from predictions, never from manually entered scores
figure_pair<-function(name,draw,width=11,height=8.8) {
  # The tutorial selects a small reading sequence. An unset/NULL selection
  # regenerates the complete companion gallery with the same plotting code.
  if(exists("TUTORIAL_FIGURES") && !is.null(TUTORIAL_FIGURES) &&
     !name %in% TUTORIAL_FIGURES)return(invisible(NULL))
  for(format in c("pdf","png")) {
    path<-file.path(OUT,"figures",paste0(name,".",format))
    if(format=="pdf") cairo_pdf(path,width=width,height=height) else png(path,width=width*130,height=height*130,res=130,type="cairo-png")
    draw();dev.off()
  }
  if(nzchar(Sys.getenv("BAYESIFY_PYTHON_NOTEBOOK"))) {
    display<-import("IPython.display",convert=FALSE)
    display$display(display$Image(filename=file.path(OUT,"figures",paste0(name,".png"))))
  } else if(requireNamespace("IRdisplay",quietly=TRUE) && nzchar(Sys.getenv("JPY_PARENT_PID")))
    IRdisplay::display_png(file=file.path(OUT,"figures",paste0(name,".png")))
}

# A method has the same colour and line style in every figure. Legends are
# repeated INSIDE each panel, so a reader need not find a detached legend.
method_style<-function(label) {
  for(key in c("sampled_mfvi","moment_mfvi","sampled_ep","moment_ep","ordinary"))
    label<-gsub(key,c(sampled_mfvi="Sampled MFVI",moment_mfvi="Moment MFVI",
      sampled_ep="Sampled power-EP",moment_ep="Moment power-EP",ordinary="Ordinary AdamW")[[key]],label,fixed=TRUE)
  name<-gsub(" — ",", ",label,fixed=TRUE)
  col<-"#202020";lty<-1L;pch<-16L
  if(grepl("Ordinary AdamW.*3 epochs",name)){col<-"#777777";lty<-2L;pch<-8L}
  if(grepl("Ordinary SGD",name)){col<-"#666666";lty<-3L;pch<-8L}
  if(grepl("Moment MFVI",name)){col<-"#0072B2";pch<-17L}
  if(grepl("NLL setting",name)){col<-"#7B3294";pch<-15L}
  if(grepl("Sampled MFVI",name)){col<-"#56B4E9";lty<-2L;pch<-1L}
  if(grepl("Moment power-EP",name)){col<-"#D55E00";pch<-18L}
  if(grepl("frozen",name)){col<-"#E69F00";lty<-3L;pch<-2L}
  if(grepl("Sampled power-EP",name)){col<-"#CC79A7";lty<-2L;pch<-5L}
  name<-gsub("power-EP","EP",name,fixed=TRUE)
  name<-gsub(" — ",", ",name,fixed=TRUE)
  name<-gsub("full curvature","full",name,fixed=TRUE)
  name<-gsub("frozen curvature","frozen",name,fixed=TRUE)
  name<-gsub("accuracy setting","accuracy",name,fixed=TRUE)
  name<-gsub("NLL setting","NLL",name,fixed=TRUE)
  list(name=name,col=col,lty=lty,pch=pch)
}
figure_layout<-function(n=4L) {
  if(n==4L) layout(matrix(1:4,2,2,byrow=TRUE))
  else layout(matrix(1:2,1,2))
  par(mar=c(4.2,4.6,2.8,1.2),oma=c(0,0,2.4,0),cex=.95,las=1)
}
panel_title<-function(name) {
  for(method in c("moment_mfvi","sampled_mfvi","moment_ep","sampled_ep"))
    name<-gsub(paste0(method,"_([0-9]+)"),paste0(method_style(method)$name," (seed \\1)"),name)
  name<-gsub("resnet20_cifar10","resnet20",name,fixed=TRUE)
  name<-gsub("densenet40_cifar10","densenet40",name,fixed=TRUE)
  name<-gsub("gpt2_lora_sst2","sst2",name,fixed=TRUE)
  name<-gsub("roberta_clinc","clinc",name,fixed=TRUE)
  name<-gsub("resnet20","ResNet-20 / CIFAR-10",name,fixed=TRUE)
  name<-gsub("densenet40","DenseNet-40 / CIFAR-10",name,fixed=TRUE)
  name<-gsub("sst2","GPT-2 LoRA / SST-2",name,fixed=TRUE)
  name<-gsub("clinc","RoBERTa-base / CLINC",name,fixed=TRUE)
  mtext(gsub("_"," ",name,fixed=TRUE),side=3,outer=TRUE,line=.7,font=2,cex=1.08)
}
method_legend<-function(labels,where="topright") {
  ss<-lapply(labels,method_style)
  legend(where,vapply(ss,`[[`,character(1),"name"),
    col=vapply(ss,`[[`,character(1),"col"),lty=vapply(ss,`[[`,integer(1),"lty"),
    pch=vapply(ss,`[[`,integer(1),"pch"),lwd=1.8,cex=.78,ncol=1,
    bg=adjustcolor("white",alpha.f=.96),box.col="grey80",inset=.015,x.intersp=.65,xpd=NA)
}
diagnostic_plots<-function(records,family,id_title=family,ood_title=paste(family,"OOD")) {
  labels<-vapply(records,`[[`,character(1),"label")
  styles<-lapply(labels,method_style);cols<-vapply(styles,`[[`,character(1),"col")
  ltys<-vapply(styles,`[[`,integer(1),"lty")
  risks<-lapply(records,function(r) {
    conf<-apply(r$p,1,max);err<-max.col(r$p,ties.method="first")!=r$y+1L
    data.frame(coverage=seq_along(err)/length(err),risk=cumsum(err[order(conf,decreasing=TRUE)])/seq_along(err))
  })
  full_top<-max(.02,1.08*max(vapply(risks,function(r)max(r$risk),numeric(1))))
  detail_top<-max(.01,1.08*max(vapply(risks,function(r)max(r$risk[r$coverage>=.5]),numeric(1))))
  figure_pair(paste0(family,"_calibration_and_selection"),function() {
    figure_layout()
    plot(c(0,1),c(0,1),type="n",xlab="Predicted confidence",ylab="Observed accuracy",
         main="Reliability");abline(0,1,lty=2,col="grey")
    for(i in seq_along(records)) {
      r<-reliability(records[[i]]$p,records[[i]]$y)
      lines(r$confidence,r$accuracy,col=adjustcolor(cols[i],alpha.f=.65),lty=ltys[i],lwd=1.5)
      points(r$confidence,r$accuracy,col=cols[i],pch=16,cex=.25+1.6*sqrt(r$n/max(r$n)))
    }
    mtext("Larger points: more examples in the bin",side=3,line=.2,cex=.8)
    method_legend(labels,"bottomright")
    plot(c(0,1),c(0,full_top),type="n",xlab="Coverage",ylab="Retained error rate",main="Selective prediction")
    for(i in seq_along(records)) {
      r<-records[[i]];confidence<-apply(r$p,1,max);err<-max.col(r$p,ties.method="first")!=r$y+1L
      lines(seq_along(err)/length(err),cumsum(err[order(confidence,decreasing=TRUE)])/seq_along(err),col=cols[i],lty=ltys[i],lwd=1.8)
    }
    method_legend(labels,"topleft")
    # Shared observed range for all methods in this family, not a fixed range
    # that hides a two-percent classification error at the bottom of the plot.
    plot(c(.5,1),c(0,detail_top),type="n",xlab="Coverage",ylab="Retained error rate",main="Selective-risk detail")
    for(i in seq_along(records)) {
      r<-records[[i]];err<-max.col(r$p,ties.method="first")!=r$y+1L;c<-apply(r$p,1,max)
      lines(seq_along(err)/length(err),cumsum(err[order(c,decreasing=TRUE)])/seq_along(err),col=cols[i],lty=ltys[i],lwd=1.8)
    }
    method_legend(labels,"topleft")
    plot(c(0,1),c(0,1),type="n",xlab="False-positive rate",ylab="Error-detection recall",main="Error detection")
    for(i in seq_along(records)) {
      r<-records[[i]];a<-ranking_curve(1-apply(r$p,1,max),max.col(r$p,ties.method="first")!=r$y+1L)
      if(!is.null(a$fpr))lines(a$fpr,a$tpr,col=cols[i],lty=ltys[i],lwd=1.8)
    }
    method_legend(labels,"bottomright")
    panel_title(id_title)
  })
  have<-which(vapply(records,function(r)!is.null(r$ood),logical(1)))
  if(length(have)) figure_pair(paste0(family,"_ood"),function() {
    figure_layout(2L);entropy<-function(p)-rowSums(p*log(pmax(p,1e-30)))
    plot(c(0,1),c(0,1),type="n",xlab="ID false-positive rate",ylab="OOD recall",main="Entropy OOD ROC")
    for(i in have) {
      r<-records[[i]];a<-ranking_curve(c(entropy(r$p),entropy(r$ood)),c(rep(FALSE,nrow(r$p)),rep(TRUE,nrow(r$ood))))
      lines(a$fpr,a$tpr,col=cols[i],lty=ltys[i],lwd=1.8)
    }
    method_legend(labels[have],"bottomright")
    plot(c(0,1),c(0,1),type="n",xlab="OOD recall",ylab="OOD precision",main="Entropy OOD precision–recall")
    for(i in have) {
      r<-records[[i]];a<-ranking_curve(c(entropy(r$p),entropy(r$ood)),c(rep(FALSE,nrow(r$p)),rep(TRUE,nrow(r$ood))))
      lines(a$recall,a$precision,col=cols[i],lty=ltys[i],lwd=1.8)
    }
    method_legend(labels[have],"bottomleft")
    panel_title(ood_title)
  },height=6)
  # Each population is normalized separately; class imbalance must not make
  # the less numerous errors or OOD cases disappear in the histogram.
  density_bins<-function(x,breaks) {
    h<-hist(x,breaks=breaks,plot=FALSE)
    if(length(x))h$counts/sum(h$counts)/diff(breaks) else rep(0,length(h$counts))
  }
  for(kind in c("confidence",if(length(have))"ood_entropy" else character())) {
    ids<-if(kind=="confidence")seq_along(records) else have
    figure_pair(paste0(family,"_",kind,"_histograms"),function() {
      par(mfrow=c(ceiling(length(ids)/2),2),mar=c(4.2,4.4,3,1),oma=c(0,0,2.4,0),las=1)
      for(i in ids) {
        r<-records[[i]]
        if(kind=="confidence") {
          score<-apply(r$p,1,max);correct<-max.col(r$p,ties.method="first")==r$y+1L
          a<-score[correct];b<-score[!correct];breaks<-seq(0,1,length.out=26)
          labels2<-c(paste("Correct, n =",length(a)),paste("Incorrect, n =",length(b)))
        } else {
          entropy<-function(p)-rowSums(p*log(pmax(p,1e-30)))
          a<-entropy(r$p);b<-entropy(r$ood);breaks<-seq(0,log(ncol(r$p))+1e-6,length.out=26)
          labels2<-c(paste("In domain, n =",length(a)),paste("OOD, n =",length(b)))
        }
        da<-density_bins(a,breaks);db<-density_bins(b,breaks);mid<-(head(breaks,-1)+tail(breaks,-1))/2
        plot(mid,da,type="h",lwd=3,col=adjustcolor("#0072B2",.7),ylim=c(0,max(da,db)*1.1),
          xlim=range(breaks),xlab=if(kind=="confidence")"Maximum predictive probability" else "Predictive entropy",
          ylab="Within-group density",main=method_style(r$label)$name)
        lines(mid,db,type="h",lwd=2,col=adjustcolor("#D55E00",.7))
        legend("topleft",labels2,col=c("#0072B2","#D55E00"),lwd=c(3,2),bg="white",cex=.75)
      }
      panel_title(if(kind=="confidence")id_title else ood_title)
    },height=max(4,3.4*ceiling(length(ids)/2)))
  }
}

# %% Learning curves from recorded epochs
learning_plots<-function(path,families) {
  d<-read.csv(path,stringsAsFactors=FALSE);d<-d[d$family %in% families,]
  for(f in unique(d$family)) for(cohort in unique(d$cohort[d$family==f])) {
    a<-d[d$family==f & d$cohort==cohort,];if(!nrow(a))next
    # A line is one recorded configuration/seed in the declared cohort.
    groups<-split(a,paste(a$label,a$seed,sep=" | seed "))
    styles<-lapply(names(groups),method_style)
    figure_pair(paste0(f,"_",cohort,"_learning"),function() {
      figure_layout()
      for(metric in c("validation_nll","validation_accuracy")) for(axis in c("epoch","training_seconds")) {
        yy<-a[[metric]];xx<-a[[axis]];ok<-is.finite(xx)&is.finite(yy)
        if(!any(ok)){plot.new();title("No recorded curve");next}
        plot(range(xx[ok]),range(yy[ok]),type="n",xlab=if(axis=="epoch")"Epoch" else "Recorded cumulative training seconds",
             ylab=if(metric=="validation_nll")"Predictive NLL (lower)" else "Accuracy (higher)",
             main=if(metric=="validation_nll")"Stopping-set probability loss" else "Stopping-set classification accuracy")
        for(i in seq_along(groups)) {
          g<-groups[[i]];g<-g[order(g$epoch),];s<-styles[[i]]
          # Markers are essential: a single recorded checkpoint has no line.
          lines(g[[axis]],g[[metric]],col=s$col,lty=s$lty,lwd=1.7)
          points(g[[axis]],g[[metric]],col=s$col,pch=s$pch,cex=if(nrow(g)==1L)1.15 else .35)
        }
        method_legend(names(groups),if(metric=="validation_nll")"topright" else "bottomright")
      }
      panel_title(paste(f,cohort,"learning"))
    })
  }
}

# %% Equal-axis message plots
# Figures of independently computed coordinates; no training or tensor library.
# All points are retained. Both axes have identical limits and asp = 1.
message_pairs <- function(method, quantity, block, message, backprop) {
  data.frame(method=method, quantity=quantity, block=block,
    coordinate=seq_along(as.numeric(as.array(message))),
    message=as.numeric(as.array(message)), backprop=as.numeric(as.array(backprop)))
}

message_equivalence_plot <- function(method, directory=file.path(OUT,"message_equivalence"), prefix="tutorial") {
  file <- file.path(directory,paste0(method,"_messages.csv"))
  d <- read.csv(file)
  quantities <- c("Activation mean message","Activation second-moment message")
  titles <- c("Activation: mean message","Activation: second-moment message")
  if(method!="gaussian_mlp") {
    quantities <- c(quantities,"Weight mean message","Weight second-moment message")
    titles <- c(titles,"Weight: mean message","Weight: second-moment message")
  }
  figure_pair(paste0(prefix,"_",method,"_message_equivalence"),function() {
    old <- par(mfrow=if(length(quantities)==2)c(1,2) else c(2,2),
               mar=c(4.3,4.6,3.3,.7),mgp=c(2.7,.8,0),pty="s")
    on.exit(par(old))
    for(i in seq_along(quantities)) {
      a <- d[d$quantity==quantities[i],]
      lim <- range(c(a$message,a$backprop));span <- diff(lim)
      if(span==0)span <- max(abs(lim),1)*.1
      lim <- lim+c(-.08,.08)*span
      groups <- sort(unique(a$block))
      colors <- c("#0072B2","#D55E00","#009E73","#CC79A7","#6A3D9A","#333333")
      plot(a$message,a$backprop,type="n",asp=1,xlim=lim,ylim=lim,
        xaxs="i",yaxs="i",xlab="Literal message passing",ylab="Backpropagation",
        main=titles[i],cex.main=.94)
      abline(0,1,col="#444444",lty=2,lwd=1.5)
      for(j in seq_along(groups)) {
        b <- a[a$block==groups[j],]
        points(b$message,b$backprop,pch=16,cex=if(method=="exact").8 else .46,
               col=adjustcolor(colors[j],.65))
      }
      names <- if(method %in% c("gaussian_mlp","gaussian_mlp_exact"))groups else if(method=="exact")"All coordinates" else
        if(grepl("Activation",quantities[i]))c("z1: affine 1","h1: ReLU 1","z2: affine 2","h2: ReLU 2")[as.integer(groups)-1L] else
        if(method=="ep" && length(groups)<=3L)paste("W",(as.integer(groups)+1L)%/%2L,sep="") else
        c("W1","bias 1","W2","bias 2","W3","bias 3")[as.integer(groups)]
      legend("topleft",legend=c("Identity",names),lty=c(2,rep(NA,length(groups))),
        pch=c(NA,rep(16,length(groups))),col=c("#444444",colors[seq_along(groups)]),
        bty="n",cex=.84,inset=.015)
      err <- max(abs(a$backprop-a$message))
      mtext(sprintf("All %s coordinates; max |difference| = %.2e",nrow(a),err),
            side=3,line=.25,cex=.74)
    }
  },width=8.8,height=if(length(quantities)==2)4.5 else 8.1)
  invisible(d)
}

message_equivalence_table <- function(method, directory=file.path(OUT,"message_equivalence")) {
  file <- file.path(directory,paste0(method,"_message_checks.csv"))
  if(!file.exists(file))file<-file.path(directory,"message_checks.csv")
  if(file.exists(file)) {
    checks <- read.csv(file)
    if("method" %in% names(checks))checks<-checks[checks$method==method,]
    stopifnot(all(checks$status=="PASS"))
    cat(toupper(method),":",nrow(checks),"moment-message checks PASS; maximum absolute discrepancy",
        format(max(checks$max_abs),digits=4),"\n")
  }
}

# Decimal round-trip precision keeps plotting error equal to calculation error.
write_message_pairs <- function(pairs,method) {
  pairs$message<-sprintf("%.17g",pairs$message)
  pairs$backprop<-sprintf("%.17g",pairs$backprop)
  write.csv(pairs,file.path(OUT,"message_equivalence",paste0(method,"_messages.csv")),row.names=FALSE)
}

# %% MNIST tables and uncertainty figures
# Reuse the tutorial's probability metrics; no framework or saved-score oracle.
mnist_labels<-c(ordinary="Ordinary Adam",sampled_mfvi="Sampled MFVI",
  moment_mfvi="Moment MFVI",sampled_ep="Sampled power-EP",moment_ep="Moment power-EP")
mnist_colours<-c(ordinary="#202020",sampled_mfvi="#56B4E9",moment_mfvi="#0072B2",
  sampled_ep="#CC79A7",moment_ep="#D55E00")
mnist_lines<-c(ordinary=1,sampled_mfvi=2,moment_mfvi=1,sampled_ep=2,moment_ep=1)
mnist_architectures<-function()figure_pair("mnist_architectures",function() {
  par(mar=c(.2,.2,2,.2));plot.new();plot.window(c(0,1),c(0,1))
  pipelines<-list(c("Observed pixels: 784","Gaussian Dense: 400 + ReLU",
    "Gaussian Dense: 400 + ReLU","10 logits: full covariance","Average class probabilities"),
    c("Observed pixels: pad to 32 × 32","Conv 5 × 5: 6; ReLU + pool 2",
      "Conv 5 × 5: 16; ReLU + pool 2","Flatten 400 to Dense 120 + ReLU",
      "Dense 84 + ReLU to 10 logits","Average class probabilities"))
  for(j in 1:2) {
    x<-c(.25,.75)[j];labels<-pipelines[[j]];yy<-seq(.86,.1,length.out=length(labels))
    text(x,.98,if(j==1)"MLP: 478,410 parameters" else "Modern LeNet: 61,706 parameters",font=2,cex=.95)
    for(i in seq_along(labels)) {
      rect(x-.23,yy[i]-.045,x+.23,yy[i]+.045,col=if(i==1)"#F4F4F4" else "#EAF2F8",border="#24435A")
      text(x,yy[i],labels[i],cex=.85)
      if(i<length(labels))arrows(x,yy[i]-.045,x,yy[i+1]+.045,length=.07)
    }
  }
},width=10,height=5.5)
mnist_records<-function(root)lapply(list.files(root,pattern="^evaluation.rds$",recursive=TRUE,full.names=TRUE),readRDS)
mnist_table<-function(records) {
  raw<-do.call(rbind,lapply(records,function(a) {
    dirty<-a$run$dataset=="dirty"
    if(dirty) {
      ids<-rep(seq_len(nrow(a$ambiguous)),each=10L)
      m<-uncertainty_metrics(rbind(a$p,a$ambiguous[ids,]),c(a$y,as.integer(t(a$ambiguous_labels))))
      o<-ood_metrics(rbind(a$p,a$ambiguous),a$ood)
    } else {m<-uncertainty_metrics(a$p,a$y);o<-ood_metrics(a$p,a$ood)}
    data.frame(architecture=a$run$architecture,dataset=a$run$dataset,method=a$run$method,seed=a$run$seed,
      accuracy=100*m$accuracy,nll=m$nll,ece=100*m$ece,brier=m$brier,aurc=m$aurc,
      error_auroc=m$error_auroc,ood_auroc=o$msp_auroc,ood_fpr95=o$msp_fpr95,
      training_seconds=a$run$train_seconds,inference_seconds=a$metrics$inference_seconds)
  }))
  ordinary<-raw[raw$method=="ordinary",];key<-function(a)paste(a$architecture,a$dataset,a$seed)
  j<-match(key(raw),key(ordinary));stopifnot(!anyNA(j))
  raw$training_ratio<-raw$training_seconds/ordinary$training_seconds[j]
  raw$inference_ratio<-raw$inference_seconds/ordinary$inference_seconds[j]
  raw<-raw[order(match(raw$dataset,c("clean","dirty")),match(raw$architecture,c("mlp","lenet")),
    match(raw$method,names(mnist_labels)),raw$seed),]
  key<-paste(raw$architecture,raw$dataset,raw$method)
  display<-do.call(rbind,lapply(split(raw,factor(key,levels=unique(key))),function(a) {
    stopifnot(nrow(a)==3L,identical(sort(a$seed),41:43))
    row<-data.frame(Model=if(a$architecture[1]=="mlp")"MLP" else "LeNet",
      Dataset=if(a$dataset[1]=="clean")"MNIST" else "Dirty-MNIST",Method=mnist_labels[a$method[1]])
    for(k in names(a)[5:ncol(a)]) {
      digits<-if(k %in% c("accuracy","ece","training_seconds","training_ratio","inference_ratio"))2 else if(k=="aurc")6 else if(k %in% c("nll","brier"))5 else 4
      row[[k]]<-paste0(formatC(mean(a[[k]]),digits=digits,format="f")," ± ",
                       formatC(sd(a[[k]]),digits=digits,format="f"))
    }
    row
  }))
  rownames(display)<-NULL
  names(display)<-c("Model","Dataset","Method","Accuracy % ↑","NLL ↓","ECE % ↓","Brier ↓","AURC ↓",
    "Error AUROC ↑","OOD AUROC ↑","OOD FPR95 ↓","Training s ↓","Prediction 10k s ↓","Training / NN ↓","Inference / NN ↓")
  list(per_seed=raw,display=display)
}
mnist_legend<-function(methods,where="topleft",cex=.95,ncol=1L)legend(where,
  (if(ncol==2L)c(ordinary="Ordinary",sampled_mfvi="MFVI MC",moment_mfvi="MFVI moment",
    sampled_ep="EP MC",moment_ep="EP moment") else c(ordinary="Ordinary",sampled_mfvi="Sampled MFVI",moment_mfvi="Moment MFVI",
    sampled_ep="Sampled EP",moment_ep="Moment EP"))[methods],
  col=mnist_colours[methods],lty=mnist_lines[methods],lwd=1.8,cex=cex,ncol=ncol,
  seg.len=if(ncol==2L).6 else 2,x.intersp=.3,bg="white",box.col="grey80",inset=.02)

# Same four diagnostic questions for every architecture; all five methods.
mnist_diagnostic<-function(records,arch,dataset="clean",seed=41L) {
  a<-Filter(function(a)a$run$architecture==arch && a$run$dataset==dataset && a$run$seed==seed,records)
  a<-a[match(names(mnist_labels),vapply(a,function(a)a$run$method,character(1)))]
  methods<-names(mnist_labels)
  if(dataset=="dirty")a<-lapply(a,function(r) {
    ids<-rep(seq_len(nrow(r$ambiguous)),each=10L)
    r$ood_id<-rbind(r$p,r$ambiguous)
    r$p<-rbind(r$p,r$ambiguous[ids,]);r$y<-c(r$y,as.integer(t(r$ambiguous_labels)));r
  })
  # Match the shared helpers exactly: classification normalizes rows;
  # OOD uses the recorded probability vectors for its common score.
  a<-lapply(a,function(r){if(is.null(r$ood_id))r$ood_id<-r$p;r$p<-r$p/rowSums(r$p);r})
  figure_pair(paste0("mnist_",arch,"_",dataset,"_uncertainty"),function() {
    par(mfrow=c(2,2),mar=c(4.2,4.3,2.8,1),oma=c(0,0,1.8,0),las=1,pty="s",xaxs="i",yaxs="i",cex=1)
    plot(c(0,1),c(0,1),type="n",asp=1,xlab="Confidence",ylab="Observed accuracy",main="Calibration")
    abline(0,1,lty=3,col="grey65")
    for(i in seq_along(a)){r<-reliability(a[[i]]$p,a[[i]]$y);ok<-r$n>=20
      lines(r$confidence[ok],r$accuracy[ok],col=mnist_colours[i],lty=mnist_lines[i])
      points(r$confidence[ok],r$accuracy[ok],pch=16,cex=.3+1.2*sqrt(r$n[ok]/max(r$n)),col=mnist_colours[i])}
    mtext("Larger markers: more examples",side=3,line=.2,cex=.7)
    mnist_legend(methods,"bottomright",cex=.8,ncol=2L)
    risks<-lapply(a,function(r){e<-max.col(r$p,ties.method="first")!=r$y+1L
      cumsum(e[order(apply(r$p,1,max),decreasing=TRUE)])/seq_along(e)})
    plot(c(.5,1),c(0,max(vapply(risks,function(r)max(tail(r,ceiling(length(r)/2))),numeric(1)))*105),type="n",
      xlab="Retained fraction",ylab="Classification error (%)",main="Selective prediction")
    for(i in seq_along(a))lines(seq_along(risks[[i]])/length(risks[[i]]),100*risks[[i]],col=mnist_colours[i],lty=mnist_lines[i])
    mnist_legend(methods)
    for(ood in c(FALSE,TRUE)) {
      plot(c(0,1),c(0,1),type="n",asp=1,xlab="False-positive rate",ylab="True-positive rate",
        main=if(ood)"Fashion-MNIST OOD" else "Error detection")
      abline(0,1,lty=3,col="grey65")
      for(i in seq_along(a)) {
        r<-a[[i]]
        if(ood){id<-if(is.null(r$ood_id))r$p else r$ood_id
          curve<-ranking_curve(c(1-apply(id,1,max),1-apply(r$ood,1,max)),
                               c(rep(FALSE,nrow(id)),rep(TRUE,nrow(r$ood))))}
        else curve<-ranking_curve(1-apply(r$p,1,max),max.col(r$p,ties.method="first")!=r$y+1L)
        lines(curve$fpr,curve$tpr,col=mnist_colours[i],lty=mnist_lines[i],lwd=1.5)
      }
      mnist_legend(methods,"bottomright")
    }
    mtext(paste(if(arch=="mlp")"MLP" else "LeNet",if(dataset=="clean")"MNIST" else "Dirty-MNIST","— seed",seed),outer=TRUE,font=2)
  },width=8.5,height=7.5)
}

# Curves use actual histories. No extrapolation and no carrying stopped runs.
mnist_learning<-function(records,arch,dataset="clean") {
  records<-Filter(function(a)a$run$architecture==arch && a$run$dataset==dataset,records)
  d<-do.call(rbind,lapply(records,`[[`,"history"));methods<-names(mnist_labels)
  d$validation_accuracy<-100*d$validation_accuracy
  figure_pair(paste0("mnist_",arch,"_",dataset,"_learning"),function() {
    par(mfrow=c(2,2),mar=c(4.2,4.3,2.8,1),oma=c(0,0,1.8,0),las=1,cex=1)
    for(metric in c("validation_nll","validation_accuracy"))for(axis in c("epoch","train_seconds")) {
      yy_range<-range(d[[metric]])
      # Leave space above the curves for the inside legend; never cover the
      # ordinary model's late-epoch NLL rise with the legend box.
      if(metric=="validation_nll")yy_range[2]<-3*yy_range[2]
      plot(range(d[[axis]]),yy_range,type="n",log=if(metric=="validation_nll")"y" else "",xlab=if(axis=="epoch")"Epoch" else "Training seconds",
        ylab=if(metric=="validation_nll")"Predictive NLL (log scale)" else "Accuracy (%)",main="Validation set")
      for(method in methods) {
        a<-d[d$method==method,]
        for(seed in unique(a$seed)){b<-a[a$seed==seed,];lines(b[[axis]],b[[metric]],
          col=adjustcolor(mnist_colours[method],.25),lty=mnist_lines[method])}
        # Average at the same epoch; time coordinate is mean actual elapsed time.
        z<-aggregate(a[c(axis,metric)],list(epoch=a$epoch),mean)
        dev<-aggregate(a[metric],list(epoch=a$epoch),sd)[[metric]]
        xx<-if(axis=="epoch")z$epoch else z[[axis]];yy<-z[[metric]]
        polygon(c(xx,rev(xx)),c(yy-dev,rev(yy+dev)),col=adjustcolor(mnist_colours[method],.1),border=NA)
        lines(xx,yy,col=mnist_colours[method],lty=mnist_lines[method],lwd=1.8)
      }
      mnist_legend(methods,if(metric=="validation_nll")"topright" else "bottomright")
    }
    mtext(paste(if(arch=="mlp")"MLP" else "LeNet",if(dataset=="clean")"MNIST" else "Dirty-MNIST",
      "— mean ± one seed SD; thin lines are individual runs"),outer=TRUE,font=2,cex=.9)
  },width=8.5,height=7.5)
}

# One focused ambiguity illustration, not a gallery of every available score.
mnist_ambiguity<-function(records,seed=41L,method="moment_mfvi") {
  # Fixed before inspecting Dirty results: one belief illustrates the two
  # different uncertainty quantities. The table and ROC retain all methods.
  a<-Filter(function(a)a$run$architecture=="lenet" && a$run$dataset=="dirty" &&
              a$run$seed==seed && a$run$method==method,records)[[1L]]
  colours<-c(clean="#009E73",ambiguous="#E69F00",fashion="#7B3294")
  figure_pair("mnist_dirty_ambiguity",function() {
    par(mfrow=c(1,2),mar=c(4.1,4.1,2.8,1),oma=c(0,0,1.8,0),las=1,cex=1)
    for(k in 1:2) {
      u<-a$uncertainty
      limit<-if(k==1)c(0,log(10)) else c(0,max(.01,max(vapply(u,function(x)max(x[,2]),numeric(1)))))
      plot(limit,c(0,1),type="n",xlab=if(k==1)"Predictive entropy (nats)" else "Model disagreement (nats)",
        ylab="Fraction at or below score",main=if(k==1)"Total uncertainty" else "Model disagreement")
      for(name in names(colours))plot(ecdf(u[[name]][,k]),add=TRUE,verticals=TRUE,do.points=FALSE,
        col=colours[name],lwd=1.7)
      legend("bottomright",c("Clean digits","Ambiguous digits","Fashion (OOD)"),
        col=colours,lty=1,lwd=1.7,cex=.95,bg="white",box.col="grey80")
    }
    mtext(paste("Dirty-MNIST:",mnist_labels[method],"— seed",seed),outer=TRUE,font=2)
  },width=8.5,height=4.2)
}

# Companion long table retains every metric computed by the shared helpers.
# The reading table selects complementary metrics rather than repeating them all.
mnist_all_metrics<-function(records)do.call(rbind,lapply(records,function(a) {
  dirty<-a$run$dataset=="dirty"
  ids<-rep(seq_len(nrow(a$ambiguous)),each=10L)
  p<-if(dirty)rbind(a$p,a$ambiguous[ids,]) else a$p
  y<-if(dirty)c(a$y,as.integer(t(a$ambiguous_labels))) else a$y
  id<-if(dirty)rbind(a$p,a$ambiguous) else a$p
  u<-unlist(lapply(names(a$uncertainty),function(population)
    setNames(colMeans(a$uncertainty[[population]]),paste(c("entropy","mi"),population,sep="_"))))
  m<-c(uncertainty_metrics(p,y),as.list(ood_metrics(id,a$ood)),as.list(u))
  data.frame(architecture=a$run$architecture,dataset=a$run$dataset,method=a$run$method,
    seed=a$run$seed,metric=names(m),value=as.numeric(m))
}))

# %% Approximation-versus-cost and MC agreement figures
# Drawing only. Values are recomputed from saved network evaluations by the
# two short mathematical functions shown in the tutorial.
forward_accuracy_cost_plot <- function(a) {
  figure_pair("tutorial_forward_accuracy_cost",function() {
    par(mfrow=c(1,2),mar=c(4.5,4.7,3.1,.9),oma=c(0,0,1.8,0),las=1)
    colors <- c(mfvi="#0072B2",ep="#D55E00")
    for (quantity in c("mean","variance")) {
      error <- a[[paste0(quantity,"_error")]]
      noise <- max(a[[paste0(quantity,"_mcse")]])
      upper <- max(error,noise)*1.30
      plot(range(a$seconds)*c(.8,1.18),c(0,upper),type="n",log="x",
        xaxt="n",xlab="Measured diagnostic time (seconds)",ylab="Relative error (%)",
        main=paste("Output",quantity,"accuracy"))
      axis(1,at=c(1,2,4,8,16,32),labels=c(1,2,4,8,16,32))
      bounds <- par("usr");rect(10^bounds[1],0,10^bounds[2],noise,
        col="#E8E8E8",border=NA)
      for (method in c("mfvi","ep")) {
        z <- a[a$method==method,];z <- z[order(z$rank),]
        lines(z$seconds,z[[paste0(quantity,"_error")]],type="b",
          pch=if(method=="mfvi")16 else 17,lwd=2,col=colors[method])
      }
      for (rank in sort(unique(a$rank))) {
        z <- a[a$rank==rank,]
        text(exp(mean(log(z$seconds))),max(z[[paste0(quantity,"_error")]]),
          labels=paste0("r = ",rank),pos=3,cex=.82,col="#444444",offset=.5)
      }
      legend("topright",c("Moment MFVI","Moment EP"),
        col=colors,pch=c(16,17),lty=1,lwd=2,cex=.90,bg="white",box.col="grey85",inset=.02)
      mtext("r: retained covariance directions",side=3,line=.35,cex=.78)
    }
    mtext("Forward moments: accuracy versus computation",outer=TRUE,font=2,cex=1.02)
  },width=9,height=4.9)
}

backward_accuracy_cost_plot <- function(a) {
  figure_pair("tutorial_backward_accuracy_cost",function() {
    par(mfrow=c(1,2),mar=c(4.5,4.7,3.1,.9),oma=c(0,0,1.8,0),las=1)
    upper <- max(a$error)*1.3
    for (method in c("mfvi","ep")) {
      z <- a[a$method==method,]
      plot(c(26,2500),c(0,upper),type="n",log="x",xaxt="n",
        xlab="Nonlinear network evaluations",ylab="Mean-message error (%)",
        main=toupper(method))
      axis(1,at=c(32,128,512,2048),labels=c(32,128,512,2048))
      bounds <- par("usr");rect(10^bounds[1],0,10^bounds[2],max(z$reference_mcse),
        col="#E8E8E8",border=NA)
      for (estimator in c("sampling","corrected")) {
        q <- z[z$estimator==estimator,];q <- q[order(q$budget),]
        lines(q$budget,q$error,type="b",lwd=2,
          col=if(estimator=="sampling")"#666666" else "#0072B2",
          pch=if(estimator=="sampling")1 else 16,
          lty=if(estimator=="sampling")2 else 1)
      }
      legend("topright",c("Monte Carlo","Moments + sampling"),
        col=c("#666666","#0072B2"),pch=c(1,16),lty=c(2,1),lwd=2,
        cex=.90,bg="white",box.col="grey85",inset=.02)
      mtext("Same draws for both estimators",side=3,line=.35,cex=.78)
    }
    mtext("Backward means: accuracy versus sampling budget",outer=TRUE,font=2,cex=1.02)
  },width=9,height=4.9)
}

# The diagonal is perfect agreement. Keep all predetermined coordinates;
# bars describe the finite MC reference, not a tolerance for the approximation.
forward_mc_agreement_plot <- function(a) {
  figure_pair("tutorial_forward_mc_agreement",function() {
    par(mfrow=c(1,2),mar=c(4.3,4.8,3,.8),oma=c(0,0,1.8,0),las=1,pty="s")
    for (method in c("mfvi","ep")) {
      z <- a[a$method==method & a$sketch=="training keys" & a$rank %in% c(16L,64L),]
      limit <- 1.32*max(z$reference+1.96*z$reference_mcse,z$approximation)
      plot(c(0,limit),c(0,limit),type="n",asp=1,xaxs="i",yaxs="i",
        xlab="MC output variance (512 networks)",ylab="Approximated output variance",
        main=paste(toupper(method),"forward variance"))
      abline(0,1,lty=2,col="#333333")
      for (rank in c(16L,64L)) {
        q <- z[z$rank==rank,]
        segments(pmax(0,q$reference-1.96*q$reference_mcse),q$approximation,
          q$reference+1.96*q$reference_mcse,q$approximation,col="#C3C3C3",lwd=.7)
        points(q$reference,q$approximation,col=if(rank==16L)"#D55E00" else "#0072B2",
          pch=if(rank==16L)1 else 16,cex=.8)
      }
      legend("topleft",c("16 directions","64 directions","Perfect agreement"),
        col=c("#D55E00","#0072B2","#333333"),pch=c(1,16,NA),lty=c(NA,NA,2),
        cex=.86,bg="white",box.col="grey85",inset=.02)
    }
    mtext("Forward approximation against sampled networks",outer=TRUE,font=2,cex=1.02)
  },width=9,height=5.1)
}

backward_mc_agreement_plot <- function() {
  full <- file.path(OUT,"approximation_cost/backward_groups.rds")
  saved <- if(file.exists(full))readRDS(full) else readRDS(file.path(OUT,"approximation_cost/backward_coordinates.rds"))
  figure_pair("tutorial_backward_mc_agreement",function() {
    par(mfrow=c(1,2),mar=c(4.3,4.8,3,.8),oma=c(0,0,1.8,0),las=1,pty="s")
    for (method in c("mfvi","ep")) {
      a <- saved[[paste0(method,"_0")]]
      estimates <- lapply(c(128L,2048L),function(budget) {
        if(!is.null(a$estimates))return(a$estimates[[as.character(budget)]])
        ids <- seq_len(budget/4L)
        colSums(a$corrected$numerator[ids,,drop=FALSE])/sum(a$corrected$denominator[ids])
      })
      scale <- 10^floor(log10(max(abs(a$reference[,1]))))
      x <- a$reference[,1]/scale;se <- a$reference[,2]/scale
      y <- lapply(estimates,`/`,scale)
      limits <- range(x-1.96*se,x+1.96*se,unlist(y))
      limits <- limits+c(-.10,.25)*diff(limits)
      unit <- paste0(" (x ",format(scale,scientific=TRUE),")")
      plot(limits,limits,type="n",asp=1,xaxs="i",yaxs="i",
        xlab=paste0("MC mean message",unit),ylab=paste0("Approximated mean message",unit),
        main=paste(toupper(method),"backward mean"))
      abline(0,1,lty=2,col="#333333")
      segments(x-1.96*se,y[[2]],x+1.96*se,y[[2]],col="#DDDDDD",lwd=.6)
      points(x,y[[1]],col=adjustcolor("#D55E00",.7),pch=1,cex=.55)
      points(x,y[[2]],col=adjustcolor("#0072B2",.8),pch=16,cex=.5)
      legend("topleft",c("Moments + 128 samples","Moments + 2,048 samples","Perfect agreement"),
        col=c("#D55E00","#0072B2","#333333"),pch=c(1,16,NA),lty=c(NA,NA,2),
        cex=.78,bg="white",box.col="grey85",inset=.02)
    }
    mtext("Backward approximation against an independent 8,192-network reference",outer=TRUE,font=2,cex=.94)
  },width=9,height=5.1)
}

