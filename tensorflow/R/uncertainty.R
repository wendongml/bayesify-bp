# %% Uncertainty metrics and their exact helpers
# %% Uncertainty metrics: probabilities and labels, independent of framework
reliability<-function(p,y,bins=15L) {
  confidence<-apply(p,1,max);correct<-max.col(p,ties.method="first")==y+1L
  group<-pmin(floor(confidence*bins)+1L,bins)
  do.call(rbind,lapply(seq_len(bins),function(i) {
    ii<-which(group==i)
    data.frame(bin=i,n=length(ii),confidence=if(length(ii))mean(confidence[ii]) else NA_real_,
      accuracy=if(length(ii))mean(correct[ii]) else NA_real_)
  }))
}
ranking_curve<-function(score,positive) {
  positive<-as.logical(positive);o<-order(score,decreasing=TRUE)
  score<-score[o];positive<-positive[o]
  # Evaluate only after an entire tie group: no favorable arbitrary tie order.
  end<-c(which(diff(score)!=0),length(score))
  tp<-cumsum(positive)[end];fp<-end-tp;P<-sum(positive);N<-sum(!positive)
  if(P==0L || N==0L) return(list(auc=NA_real_,ap=NA_real_,fpr95=NA_real_))
  tpr<-c(0,tp/P);fpr<-c(0,fp/N)
  precision<-tp/(tp+fp);recall<-tp/P
  list(auc=sum(diff(fpr)*(head(tpr,-1)+tail(tpr,-1))/2),
    ap=sum(diff(c(0,recall))*precision),fpr95=min(fpr[tpr>=.95]),
    fpr=fpr,tpr=tpr,precision=precision,recall=recall)
}
uncertainty_metrics<-function(p,y) {
  p<-as.matrix(p);y<-as.integer(y)
  stopifnot(nrow(p)==length(y),all(is.finite(p)),min(p)>=0,
            max(abs(rowSums(p)-1))<1e-4,all(y>=0),all(y<ncol(p)))
  p<-p/rowSums(p);n<-length(y);confidence<-apply(p,1,max)
  errors<-max.col(p,ties.method="first")!=y+1L
  target<-matrix(0,n,ncol(p));target[cbind(seq_len(n),y+1L)]<-1
  r<-reliability(p,y);risk<-cumsum(errors[order(confidence,decreasing=TRUE)])/seq_len(n)
  oracle<-cumsum(sort(errors))/seq_len(n)
  detection<-ranking_curve(1-confidence,errors)
  list(n=n,accuracy=mean(!errors),nll=-mean(log(pmax(p[cbind(seq_len(n),y+1L)],1e-30))),
    brier=mean(rowSums((p-target)^2)),ece=sum(r$n*abs(r$confidence-r$accuracy),na.rm=TRUE)/n,
    aurc=mean(risk),excess_aurc=mean(risk-oracle),error_auroc=detection$auc,error_auprc=detection$ap,
    risk80=risk[ceiling(.8*n)],risk90=risk[ceiling(.9*n)],risk95=risk[ceiling(.95*n)],
    entropy=mean(-rowSums(p*log(pmax(p,1e-30)))))
}
ood_metrics<-function(p_id,p_ood) {
  entropy<-function(p)-rowSums(p*log(pmax(p,1e-30)))
  positive<-c(rep(FALSE,nrow(p_id)),rep(TRUE,nrow(p_ood)))
  a<-ranking_curve(c(entropy(p_id),entropy(p_ood)),positive)
  b<-ranking_curve(c(1-apply(p_id,1,max),1-apply(p_ood,1,max)),positive)
  data.frame(entropy_auroc=a$auc,entropy_auprc=a$ap,entropy_fpr95=a$fpr95,
    msp_auroc=b$auc,msp_auprc=b$ap,msp_fpr95=b$fpr95,
    id_n=nrow(p_id),ood_n=nrow(p_ood),ood_prevalence=mean(positive))
}

