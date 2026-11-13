# %% Complete MLP and LeNet construction and local moments
# %% MNIST layers: the same weights define point, sampled and moment networks
# Observed pixels are deterministic. For independent W and h,
# E[Wh]=M_W m_h and Var(Wh)=V_W(m_h^2+v_h)+M_W^2 v_h.
# Bias is the final column of each weight matrix; its input is exactly one.
mnist_affine <- function(x,w,channels=NULL) {
  n<-shape(w)[2L]-1L
  W<-tf$slice(w,c(0L,0L),c(-1L,n));b<-tf$gather(w,n,axis=1L)
  if(is.null(channels))return(tf$matmul(x,W,transpose_b=TRUE)+b)
  K<-tf$transpose(tf$reshape(W,c(shape(w)[1L],channels,5L,5L)),c(2L,3L,1L,0L))
  tf$nn$conv2d(x,K,strides=1L,padding="VALID")+b
}
mnist_affine_moments <- function(a,w,v,channels=NULL,full=FALSE) {
  n<-shape(w)[2L]-1L
  wm<-tf$slice(w,c(0L,0L),c(-1L,n));wv<-tf$slice(v,c(0L,0L),c(-1L,n))
  bias<-list(m=tf$gather(w,n,axis=1L),v=tf$gather(v,n,axis=1L))
  if(is.null(channels))return(moment_dense(a,
    list(m=tf$transpose(wm),v=tf$transpose(wv)),bias,full))
  kernel<-function(W)tf$transpose(tf$reshape(W,c(shape(w)[1L],channels,5L,5L)),c(2L,3L,1L,0L))
  moment_conv2d(a,list(m=kernel(wm),v=kernel(wv)),bias,padding="VALID")
}

# %% Gaussian ReLU and a small Gaussian maximum closure
# t=m/s, P=Phi(t), phi=phi(t): E[ReLU]=mP+s*phi;
# E[ReLU^2]=(m^2+v)P+m*s*phi. Zero variance gives the point operation.
mnist_relu <- function(m,v) moment_relu(list(m=m,v=v))
# For independent X1,X2: d=m1-m2, s^2=v1+v2, P=Phi(d/s), c=s*phi(d/s).
# E[max]=m2+dP+c; Var(max)=v1 P+v2(1-P)+d^2 P+d c-(dP+c)^2.
# Four inputs use a balanced tree of three such merges, with a Gaussian
# projection after each merge, then Gaussian ReLU. This is a declared closure.
mnist_gaussian_max <- function(a,b) {
  ss<-a$v+b$v;positive<-ss>0;s<-tf$sqrt(tf$where(positive,ss,tf$ones_like(ss)))
  d<-a$m-b$m;t<-d/s;P<-.5*tf$math$erfc(-t/sqrt(2))
  c<-s*tf$exp(-t*t/2)/sqrt(2*pi);u<-d*P+c
  v<-a$v*P+b$v*(1-P)+d*d*P+d*c-u*u
  list(m=tf$where(positive,b$m+u,tf$maximum(a$m,b$m)),
       v=tf$where(positive,tf$maximum(v,zero(v)),zero(v)))
}
mnist_pool <- function(a) {
  patches<-function(x) {
    p<-tf$image$extract_patches(x,sizes=c(1L,2L,2L,1L),strides=c(1L,2L,2L,1L),
                               rates=c(1L,1L,1L,1L),padding="VALID")
    tf$reshape(p,tf$concat(list(tf$gather(tf$shape(p),0:2),tf$constant(c(4L,shape(x)[4L]))),0L))
  }
  m<-patches(a$m);v<-patches(a$v)
  pair<-lapply(0:3,function(i)list(m=tf$gather(m,as.integer(i),axis=3L),v=tf$gather(v,as.integer(i),axis=3L)))
  z<-mnist_gaussian_max(mnist_gaussian_max(pair[[1]],pair[[2]]),
                        mnist_gaussian_max(pair[[3]],pair[[4]]))
  mnist_relu(z$m,z$v)
}
mnist_flatten <- function(x) {
  # NCHW order matches the archived convolutional checkpoint coordinates.
  if(length(shape(x))==4L)x<-tf$transpose(x,c(0L,3L,1L,2L))
  tf$reshape(x,tf$stack(list(tf$gather(tf$shape(x),0L),-1L)))
}

# %% Two architectures; no extra weight distribution or training framework
mnist_network <- function(architecture="mlp",seed=41L,dtype=tf$float32) {
  g<-new_graph(seed=seed);g$prior_center<-"zero";g$architecture<-architecture
  g$field_float64<-TRUE;g$ops<-list()
  add<-function(kind,parents,weights=integer(),channels=NULL) {
    g$ops[[length(g$ops)+1L]]<-list(kind=kind,parents=parents,weights=weights,channels=channels)
    length(g$ops)+1L
  }
  affine<-function(parent,inputs,outputs,channels=NULL,full=FALSE) {
    i<-length(g$weights)+1L
    w<-if(architecture=="mlp")tf$random$stateless_uniform(c(outputs,inputs),
      seed=as.integer(c(seed,i)),minval=-1/sqrt(inputs),maxval=1/sqrt(inputs),dtype=dtype) else
      tf$random$stateless_normal(c(outputs,inputs),seed=as.integer(c(seed,i)),dtype=dtype)*sqrt(2/inputs)
    w<-tf$concat(list(w,tf$zeros(c(outputs,1L),dtype)),1L)
    id<-weight(g,paste0("layer",i),w,variance=1e-4,prior_variance=1e-3)
    add(if(full)"readout" else "affine",parent,id,channels)
  }
  if(architecture=="mlp") {
    a<-affine(1L,784L,400L);a<-add("relu",a)
    z<-affine(a,400L,400L);h<-add("relu",z)
    out<-affine(h,400L,10L,full=TRUE)
    # The optional EP readout also depends on first hidden variance and second
    # preactivation moments. Their reverse messages must accumulate, not replace.
    g$ops[[out-1L]]$parents<-c(h,a,z)
    g$basis<-tf$Variable(tf$gather(tf$eye(400L,dtype=dtype),0:3,axis=1L),trainable=FALSE)
    g$correlated<-FALSE
  } else {
    stopifnot(architecture=="lenet")
    a<-affine(1L,25L,6L,1L);a<-add("pool",a)
    a<-affine(a,150L,16L,6L);a<-add("pool",a);a<-add("flatten",a)
    a<-affine(a,400L,120L);a<-add("relu",a)
    a<-affine(a,120L,84L);a<-add("relu",a);affine(a,84L,10L,full=TRUE)
  }
  g
}
mnist_point <- function(g,x,ms) {
  a<-x;j<-0L
  for(op in g$ops) {
    if(length(op$weights)){j<-j+1L;a<-mnist_affine(a,ms[[j]],op$channels)}
    else if(op$kind=="relu")a<-tf$nn$relu(a)
    else if(op$kind=="pool")a<-tf$nn$relu(tf$nn$max_pool2d(a,ksize=2L,strides=2L,padding="VALID"))
    else a<-mnist_flatten(a)
  }
  a
}

# %% Correlated hidden activations without a 400 by 400 covariance per example
# L=diag(Phi(z2/s2)) M2 diag(sqrt(v1)) Q, Q'Q=I.
# C2=diag(v2-diag(LL'))+LL' preserves marginal variances and PSD.
# The small readout uses O C2 O' + diag(E[h^2]' V_out).
mnist_local <- function(g,op,parents,ws,vs,all_ms) {
  a<-parents[[1]]
  if(op$kind=="relu")return(mnist_relu(a$m,a$v))
  if(op$kind=="pool")return(mnist_pool(a))
  if(op$kind=="flatten")return(list(m=mnist_flatten(a$m),v=mnist_flatten(a$v)))
  z<-mnist_affine_moments(a,ws[[1]],vs[[1]],op$channels,op$kind=="readout")
  if(op$kind=="readout" && isTRUE(g$correlated)) {
    first<-parents[[2]];pre<-parents[[3]]
    W2<-tf$slice(all_ms[[2]],c(0L,0L),c(-1L,400L))
    O<-tf$slice(ws[[1]],c(0L,0L),c(-1L,400L))
    gate<-.5*tf$math$erfc(-pre$m/tf$sqrt(pre$v)/sqrt(2))
    # A saturated ReLU may have exactly zero retained variance. Mask before
    # sqrt as well as after it: otherwise AD evaluates 0 * Inf at that boundary.
    positive<-first$v>0
    sd<-tf$where(positive,tf$sqrt(tf$where(positive,first$v,tf$ones_like(first$v))),zero(first$v))
    L<-tf$expand_dims(gate,-1L)*tf$einsum("hi,bir->bhr",W2,
      tf$expand_dims(sd,-1L)*tf$expand_dims(g$basis,0L))
    U<-tf$einsum("ch,bhr->bcr",O,L)
    diag_shared<-tf$reduce_sum(L*L,axis=2L)
    residual<-tf$maximum(a$v-diag_shared,zero(a$v))
    noise<-mnist_affine(a$m*a$m+a$v,vs[[1]])
    z$v<-tf$einsum("bh,ch,dh->bcd",residual,O,O)+tf$matmul(U,U,transpose_b=TRUE)+tf$linalg$diag(noise)
  }
  z
}
mnist_forward <- function(g,x,ms,vs,record=FALSE) {
  ops<-lapply(g$ops,function(op) {
    if(op$kind=="readout" && isTRUE(g$correlated))op$weights<-c(op$weights,2L)
    op
  })
  layer<-function(op,ps,wm,wv,pids) {
    am<-ms;if(length(pids))am[pids]<-wm
    mnist_local(g,op,ps,wm,wv,am)
  }
  moment_graph_forward(ops,list(m=x,v=zero(x)),ms,vs,layer,record)
}
mnist_reverse <- function(g,f,adj) {
  a<-moment_graph_reverse(f,adj,lapply(g$weights,`[[`,"m"))
  list(gm=a$gm,gv=a$gv,boundaries=a$boundary_adjoints)
}

# %% Complete ResNet-20 and DenseNet-40 construction
# %% Residual and dense image networks: ordinary and moment paths share the graph
normal <- function(dims,scale,dtype=tf$float32) tf$random$normal(as.integer(dims),stddev=scale,dtype=dtype)
conv_node <- function(g,x,cin,cout,k=3L,stride=1L,name) {
  force(x) # Materialize a nested parent before registering this layer's weights.
  stride<-as.integer(stride)
  w<-weight(g,paste0(name,"_W"),normal(c(k,k,cin,cout),sqrt(2/(k*k*cin))))
  # PyTorch's k//2 padding is symmetric, including at stride two. TensorFlow
  # SAME would pad even-size inputs asymmetrically and shift the residual path.
  pad<-as.integer(k%/%2L)
  op<-function(xs,ws)tf$nn$conv2d(
    tf$pad(xs[[1]],list(c(0L,0L),c(pad,pad),c(pad,pad),c(0L,0L))),ws[[1]],
    strides=as.integer(c(1,stride,stride,1)),padding="VALID")
  node(g,name,x,op,w,kind="conv")
}
relu_node <- function(g,x,name) node(g,name,x,function(xs,ws)tf$nn$relu(xs[[1]]),kind="relu")
add_node <- function(g,a,b,name) node(g,name,c(a,b),function(xs,ws)xs[[1]]+xs[[2]])
concat_node <- function(g,a,b,name) node(g,name,c(a,b),function(xs,ws)tf$concat(xs,axis=-1L))
bn_node <- function(g,x,c,name) {
  force(x)
  gamma<-weight(g,paste0(name,"_gamma"),tf$ones(as.integer(c)),variance=1e-5)
  beta<-weight(g,paste0(name,"_beta"),tf$zeros(as.integer(c)),variance=1e-5)
  running<-new.env(parent=emptyenv())
  running$m<-tf$Variable(tf$zeros(as.integer(c)),trainable=FALSE)
  running$v<-tf$Variable(tf$ones(as.integer(c)),trainable=FALSE)
  i<-node(g,name,x,function(xs,ws) {
    a<-xs[[1]]
    if(isTRUE(g$training)) {
      m<-tf$reduce_mean(a,axis=as.integer(c(0,1,2)),keepdims=TRUE)
      v<-tf$reduce_mean(tf$square(a-m),axis=as.integer(c(0,1,2)),keepdims=TRUE)
    } else {m<-running$m;v<-running$v}
    (a-m)*tf$math$rsqrt(v+1e-5)*ws[[1]]+ws[[2]]
  },c(gamma,beta),kind="bn")
  g$nodes[[i-1L]]$normalization<-running
  i
}
norm_statistics <- function(g,states,moment=FALSE) {
  # Training normalizes with biased batch variance, but the stored running
  # variance uses the unbiased estimate, exactly as the reference BatchNorm.
  result<-list()
  for(op in g$nodes) if(!is.null(op$normalization)) {
    a<-states[[op$parents[[1]]]]
    if(moment)a<-a$m
    m<-tf$reduce_mean(a,axis=as.integer(c(0,1,2)))
    v<-tf$reduce_mean(tf$square(a-m),axis=as.integer(c(0,1,2)))
    n<-tf$cast(tf$size(a)/tf$gather(tf$shape(a),3L),a$dtype)
    result[[length(result)+1L]]<-list(m=m,v=v*n/(n-1))
  }
  result
}
commit_statistics <- function(g,statistics) {
  j<-0L
  for(op in g$nodes) if(!is.null(op$normalization)) {
    j<-j+1L
    op$normalization$m$assign(.9*op$normalization$m+.1*statistics[[j]]$m)
    op$normalization$v$assign(.9*op$normalization$v+.1*statistics[[j]]$v)
  }
}
commit_norm <- function(g,states,moment=FALSE) {
  commit_statistics(g,norm_statistics(g,states,moment))
}
dense_node <- function(g,x,cin,cout,name,readout=FALSE,initial_scale=sqrt(2/cin),
                       variance=1e-5,prior_variance=1e-3) {
  force(x)
  w<-weight(g,paste0(name,"_W"),normal(c(cin,cout),initial_scale),variance,prior_variance)
  b<-weight(g,paste0(name,"_b"),tf$zeros(as.integer(cout)),variance,prior_variance)
  node(g,name,x,function(xs,ws)tf$matmul(xs[[1]],ws[[1]])+ws[[2]],c(w,b),
       kind=if(readout)"gaussian_readout" else "delta")
}
build_cifar <- function(architecture="resnet20",rank=4L,seed=41L) {
  tf$random$set_seed(as.integer(seed));g<-new_graph(rank,seed);g$training<-FALSE
  # Recompute local derivatives for the dense graph; same moments and updates.
  g$recompute_moments<-architecture=="densenet40"
  g$prior_center<-"zero"
  if(architecture=="resnet20") {
    x<-relu_node(g,bn_node(g,conv_node(g,1L,3L,16L,name="stem"),16L,"stem_bn"),"stem_relu")
    cin<-16L
    for(stage in 1:3) for(block in 1:3) {
      cout<-c(16L,32L,64L)[[stage]];s<-if(stage>1 && block==1) 2L else 1L
      p<-paste0("stage",stage,"_block",block)
      skip<-x
      if(s==2L) {
        pad<-cin/2L
        skip<-node(g,paste0(p,"_shortcut"),x,local({pad<-pad;function(xs,ws) {
          a<-tf$strided_slice(xs[[1]],begin=c(0L,0L,0L,0L),end=tf$shape(xs[[1]]),
                             strides=c(1L,2L,2L,1L))
          tf$pad(a,list(c(0L,0L),c(0L,0L),c(0L,0L),as.integer(c(pad,pad))))
        }}))
      }
      h<-conv_node(g,x,cin,cout,stride=s,name=paste0(p,"_conv1"))
      h<-relu_node(g,bn_node(g,h,cout,paste0(p,"_bn1")),paste0(p,"_relu1"))
      h<-bn_node(g,conv_node(g,h,cout,cout,name=paste0(p,"_conv2")),cout,paste0(p,"_bn2"))
      x<-relu_node(g,add_node(g,skip,h,paste0(p,"_sum")),paste0(p,"_relu2"));cin<-cout
    }
  } else {
    stopifnot(architecture=="densenet40")
    cin<-16L;x<-conv_node(g,1L,3L,cin,name="stem")
    for(stage in 1:3) {
      for(block in 1:12) {
        p<-paste0("dense",stage,"_",block)
        h<-relu_node(g,bn_node(g,x,cin,paste0(p,"_bn")),paste0(p,"_relu"))
        h<-conv_node(g,h,cin,12L,name=paste0(p,"_conv"))
        x<-concat_node(g,x,h,paste0(p,"_concat"));cin<-cin+12L
      }
      if(stage<3) {
        p<-paste0("transition",stage)
        x<-relu_node(g,bn_node(g,x,cin,paste0(p,"_bn")),paste0(p,"_relu"))
        x<-conv_node(g,x,cin,cin,k=1L,name=paste0(p,"_conv"))
        x<-node(g,paste0(p,"_pool"),x,function(xs,ws)tf$nn$avg_pool2d(xs[[1]],2L,2L,"VALID"))
      }
    }
    x<-relu_node(g,bn_node(g,x,cin,"final_bn"),"final_relu")
  }
  x<-node(g,"global_pool",x,function(xs,ws)tf$reduce_mean(xs[[1]],axis=c(1L,2L)))
  dense_node(g,x,cin,10L,"classifier",readout=TRUE)
  g$architecture<-architecture;g
}

# %% Shared transformer primitives: LayerNorm and attention keep branch correlations

# %% Attention, normalization and GELU
layer_norm <- function(x,gamma,beta,eps) {
  m<-tf$reduce_mean(x,axis=-1L,keepdims=TRUE)
  v<-tf$reduce_mean(tf$square(x-m),axis=-1L,keepdims=TRUE)
  (x-m)*tf$math$rsqrt(v+scalar(eps,x))*gamma+beta
}
split_heads <- function(x) {
  s<-tf$shape(x)
  tf$transpose(tf$reshape(x,tf$stack(list(tf$gather(s,0L),tf$gather(s,1L),12L,64L))),c(0L,2L,1L,3L))
}
merge_heads <- function(x) {
  s<-tf$shape(x)
  tf$reshape(tf$transpose(x,c(0L,2L,1L,3L)),tf$stack(list(tf$gather(s,0L),tf$gather(s,2L),768L)))
}
attention <- function(q,k,v,inputs,causal=FALSE) {
  q<-split_heads(q);k<-split_heads(k);v<-split_heads(v)
  a<-tf$matmul(q,k,transpose_b=TRUE)/8
  mask<-tf$gather(inputs,1L,axis=2L)
  a<-a+(1-tf$expand_dims(tf$expand_dims(mask,1L),1L))*(-1e4)
  if(causal) {
    T<-tf$gather(tf$shape(a),3L)
    lower<-tf$linalg$band_part(tf$ones(tf$stack(list(T,T)),dtype=a$dtype),-1L,0L)
    a<-a+(1-lower)*(-1e4)
  }
  merge_heads(tf$matmul(tf$nn$softmax(a),v))
}
gelu <- function(x,approximate=FALSE) {
  # TensorFlow's native exact-GELU path casts 1/sqrt(2) through float32;
  # that loses float64 derivative accuracy. Write the same formula directly.
  if(!approximate)return(x*scalar(.5,x)*(1+tf$math$erf(x/tf$sqrt(scalar(2,x)))))
  # GPT-2's declared tanh approximation. Construct constants directly in the
  # operand dtype: tf.cast(Python-float,float64) first rounds through float32.
  x*scalar(.5,x)*(1+tf$tanh(scalar(sqrt(2/pi),x)*(x+scalar(.044715,x)*x^3)))
}

# %% Load public pretrained weights without hiding a different training program

# %% Load the public pretrained transformer weights
load_public_weights <- function(model_id) {
  hub<-import("huggingface_hub",convert=FALSE)
  safe<-import("safetensors.numpy",convert=FALSE)
  revision<-switch(model_id,
    "openai-community/gpt2"="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "FacebookAI/roberta-base"="e2da8e2f811d1448a5b465c236feacd80ffbac7b")
  stopifnot(!is.null(revision))
  path<-as_r(hub$hf_hub_download(model_id,"model.safetensors",revision=revision,token=FALSE))
  raw<-safe$load_file(path)
  list(get=function(name) {
    a<-raw$get(name)
    if(is.null(a)) stop("Missing public weight: ",name)
    tf$convert_to_tensor(a,dtype=tf$float32)
  },path=path)
}

# %% Complete GPT-2 rank-4 LoRA construction
build_gpt2_lora <- function(rank=4L,seed=41L,public=NULL,dtype=tf$float32) {
  if(is.null(public)) public<-load_public_weights("openai-community/gpt2")
  tf$random$set_seed(as.integer(seed));g<-new_graph(rank,seed);g$training<-FALSE
  g$prior_center<-"zero"
  W<-function(name)tf$cast(public$get(name),dtype)
  emb<-W("wte.weight");pos<-W("wpe.weight")
  x<-node(g,"embedding",1L,function(xs,ws) {
    ids<-tf$cast(tf$gather(xs[[1]],0L,axis=2L),tf$int32)
    n<-tf$gather(tf$shape(ids),1L)
    tf$gather(emb,ids)+tf$gather(pos,tf$range(n))
  })
  frozen_ln<-function(x,p) {
    a<-W(paste0(p,".weight"));b<-W(paste0(p,".bias"))
    node(g,p,x,function(xs,ws)layer_norm(xs[[1]],a,b,1e-5))
  }
  frozen_dense<-function(x,a,b,p) {
    a<-tf$identity(a);b<-tf$identity(b)
    node(g,p,x,function(xs,ws)tf$matmul(xs[[1]],a)+b)
  }
  for(i in 0:11) {
    p<-paste0("h.",i);r<-x;x<-frozen_ln(x,paste0(p,".ln_1"))
    a<-W(paste0(p,".attn.c_attn.weight"));b<-W(paste0(p,".attn.c_attn.bias"))
    aa<-as_r(tf$split(a,3L,axis=1L));bb<-as_r(tf$split(b,3L,axis=0L))
    qkv<-integer(3)
    for(j in 1:3) {
      a0<-aa[[j]];b0<-bb[[j]];pn<-paste0(p,c(".query",".key",".value")[[j]])
      if(j==2L) qkv[[j]]<-frozen_dense(x,a0,b0,pn) else {
        ia<-weight(g,paste0(pn,"_A"),normal(c(768L,4L),.02,dtype),1e-6,1/2500)
        ib<-weight(g,paste0(pn,"_B"),tf$zeros(c(4L,768L),dtype=dtype),1e-6,1/2500)
        qkv[[j]]<-node(g,pn,x,local({a0<-a0;b0<-b0;function(xs,ws) {
          tf$matmul(xs[[1]],a0)+b0+8*tf$matmul(tf$matmul(xs[[1]],ws[[1]]),ws[[2]])
        }}),c(ia,ib),kind="lora")
        g$nodes[[qkv[[j]]-1L]]$frozen_weight<-a0
        g$nodes[[qkv[[j]]-1L]]$lora_scale<-8
      }
    }
    x<-node(g,paste0(p,".attention"),c(qkv,1L),function(xs,ws)attention(xs[[1]],xs[[2]],xs[[3]],xs[[4]],TRUE))
    x<-frozen_dense(x,W(paste0(p,".attn.c_proj.weight")),W(paste0(p,".attn.c_proj.bias")),paste0(p,".attn.out"))
    x<-add_node(g,r,x,paste0(p,".attention_residual"));r<-x
    x<-frozen_ln(x,paste0(p,".ln_2"))
    x<-frozen_dense(x,W(paste0(p,".mlp.c_fc.weight")),W(paste0(p,".mlp.c_fc.bias")),paste0(p,".mlp.in"))
    x<-node(g,paste0(p,".gelu"),x,function(xs,ws)gelu(xs[[1]],TRUE))
    x<-frozen_dense(x,W(paste0(p,".mlp.c_proj.weight")),W(paste0(p,".mlp.c_proj.bias")),paste0(p,".mlp.out"))
    x<-add_node(g,r,x,paste0(p,".mlp_residual"))
  }
  x<-frozen_ln(x,"ln_f")
  # These token IDs are obtained and checked from the public tokenizer below.
  tokenizer<-import("transformers",convert=FALSE)$AutoTokenizer$from_pretrained("openai-community/gpt2",
    revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",token=FALSE)
  tokenizer$pad_token<-tokenizer$eos_token;tokenizer$padding_side<-"right"
  tokens<-lapply(c(" negative"," positive"),function(s)as.integer(as_r(tokenizer$encode(s,add_special_tokens=FALSE))))
  stopifnot(all(lengths(tokens)==1L));g$verbalizer<-unlist(tokens)
  V<-tf$gather(emb,as.integer(g$verbalizer))
  node(g,"binary_verbalizer",c(x,1L),function(xs,ws) {
    mask<-tf$gather(xs[[2]],1L,axis=2L)
    index<-tf$cast(tf$reduce_sum(mask,axis=1L)-1,tf$int32)
    h<-tf$gather(xs[[1]],index,axis=1L,batch_dims=1L)
    tf$matmul(h,V,transpose_b=TRUE)
  })
  g$architecture<-"gpt2_small_qv_lora4";g$tokenizer<-tokenizer;g$public_path<-public$path;g
}

# %% RoBERTa-base: full 12-block encoder and 150-class CLINC head

# %% RoBERTa backbone and rank-8 LoRA construction
build_roberta <- function(rank=4L,seed=41L,public=NULL,dtype=tf$float32) {
  if(is.null(public)) public<-load_public_weights("FacebookAI/roberta-base")
  tf$random$set_seed(as.integer(seed));g<-new_graph(rank,seed);g$training<-FALSE
  W<-function(name)tf$cast(public$get(name),dtype)
  parameter<-function(p,transpose=FALSE) {
    a<-W(p);if(transpose) a<-tf$transpose(a)
    i<-weight(g,gsub("\\.","_",p),a,variance=1e-8,prior_variance=1e-6)
    g$weights[[i]]$source_name<-p;g$weights[[i]]$source_transpose<-transpose;i
  }
  ids<-sapply(c("roberta.embeddings.word_embeddings.weight","roberta.embeddings.position_embeddings.weight",
               "roberta.embeddings.token_type_embeddings.weight"),parameter)
  x<-node(g,"embedding",1L,function(xs,ws) {
    t<-tf$cast(tf$gather(xs[[1]],0L,axis=2L),tf$int32)
    mask<-tf$cast(tf$not_equal(t,1L),tf$int32)
    pos<-tf$cumsum(mask,axis=1L)*mask+1L
    tf$gather(ws[[1]],t)+tf$gather(ws[[2]],pos)+tf$gather(ws[[3]],0L)
  },ids)
  norm_node<-function(x,p) {
    ids<-c(parameter(paste0(p,".weight")),parameter(paste0(p,".bias")))
    node(g,p,x,function(xs,ws)layer_norm(xs[[1]],ws[[1]],ws[[2]],1e-5),ids)
  }
  linear<-function(x,p) {
    ids<-c(parameter(paste0(p,".weight"),TRUE),parameter(paste0(p,".bias")))
    node(g,p,x,function(xs,ws)tf$matmul(xs[[1]],ws[[1]])+ws[[2]],ids)
  }
  x<-norm_node(x,"roberta.embeddings.LayerNorm")
  for(i in 0:11) {
    p<-paste0("roberta.encoder.layer.",i);skip<-x
    q<-linear(x,paste0(p,".attention.self.query"))
    k<-linear(x,paste0(p,".attention.self.key"))
    v<-linear(x,paste0(p,".attention.self.value"))
    x<-node(g,paste0(p,".attention"),c(q,k,v,1L),function(xs,ws)attention(xs[[1]],xs[[2]],xs[[3]],xs[[4]],FALSE))
    x<-linear(x,paste0(p,".attention.output.dense"))
    x<-norm_node(add_node(g,skip,x,paste0(p,".residual1")),paste0(p,".attention.output.LayerNorm"));skip<-x
    x<-linear(x,paste0(p,".intermediate.dense"))
    x<-node(g,paste0(p,".gelu"),x,function(xs,ws)gelu(xs[[1]]))
    x<-linear(x,paste0(p,".output.dense"))
    x<-norm_node(add_node(g,skip,x,paste0(p,".residual2")),paste0(p,".output.LayerNorm"))
  }
  x<-node(g,"CLS",x,function(xs,ws)tf$gather(xs[[1]],0L,axis=1L))
  linear_head<-function(x,input,output,name,key) {
    iw<-weight(g,paste0(name,"_weight"),normal(c(input,output),.02,dtype),1e-8,1e-6)
    ib<-weight(g,paste0(name,"_bias"),tf$zeros(c(output),dtype=dtype),1e-8,1e-6)
    g$weights[[iw]]$source_name<-paste0(key,".weight");g$weights[[iw]]$source_transpose<-TRUE
    g$weights[[ib]]$source_name<-paste0(key,".bias");g$weights[[ib]]$source_transpose<-FALSE
    node(g,name,x,function(xs,ws)tf$matmul(xs[[1]],ws[[1]])+ws[[2]],c(iw,ib))
  }
  x<-linear_head(x,768L,768L,"classifier_dense","classifier.dense")
  x<-node(g,"classifier_tanh",x,function(xs,ws)tf$tanh(xs[[1]]))
  # The historical RoBERTa contract uses the shared first-order factors through
  # the full classifier, not the CIFAR conditional-Gaussian readout closure.
  linear_head(x,768L,150L,"classifier_out","classifier.out_proj")
  g$architecture<-"roberta_base_full";g$public_path<-public$path
  g$tokenizer<-import("transformers",convert=FALSE)$AutoTokenizer$from_pretrained("FacebookAI/roberta-base",
    revision="e2da8e2f811d1448a5b465c236feacd80ffbac7b",token=FALSE)
  g
}
# RoBERTa-LoRA: fixed backbone, Gaussian A, point B and point classifier.
# For a row activation x, a query/value map is x W0 + b0 + (x A) B.
# A is 768-by-r; B is r-by-768. Here alpha/r = 8/8 = 1.
# The same graph supports point_forward() and the shared-factor moment pass.
build_roberta_lora <- function(rank=4L,seed=41L,public=NULL,dtype=tf$float32) {
  if(is.null(public))public<-load_public_weights("FacebookAI/roberta-base")
  tf$random$set_seed(as.integer(seed))
  g<-new_graph(rank,seed);g$training<-FALSE;g$prior_center<-"zero"
  g$point_weights<-list()
  fixed<-function(name,transpose=FALSE) {
    a<-tf$cast(public$get(name),dtype)
    if(transpose)a<-tf$transpose(a)
    tf$Variable(a,trainable=FALSE,name=gsub("\\.","_",name))
  }
  point_parameter<-function(name,a) {
    w<-tf$Variable(a,name=gsub("\\.","_",name))
    g$point_weights[[name]]<-w;w
  }
  emb<-fixed("roberta.embeddings.word_embeddings.weight")
  pos<-fixed("roberta.embeddings.position_embeddings.weight")
  typ<-fixed("roberta.embeddings.token_type_embeddings.weight")
  x<-node(g,"embedding",1L,function(xs,ws) {
    ids<-tf$cast(tf$gather(xs[[1]],0L,axis=2L),tf$int32)
    mask<-tf$cast(tf$not_equal(ids,1L),tf$int32)
    positions<-tf$cumsum(mask,axis=1L)*mask+1L
    tf$gather(emb,ids)+tf$gather(pos,positions)+tf$gather(typ,0L)
  })
  norm<-function(x,name) {
    force(x)
    a<-fixed(paste0(name,".weight"));b<-fixed(paste0(name,".bias"))
    node(g,name,x,function(xs,ws)layer_norm(xs[[1]],a,b,1e-5))
  }
  linear<-function(x,name,lora=FALSE) {
    force(x)
    a<-fixed(paste0(name,".weight"),TRUE);b<-fixed(paste0(name,".bias"))
    if(!lora)return(node(g,name,x,function(xs,ws)tf$matmul(xs[[1]],a)+b))
    # PEFT's A initializer; B starts at zero. rho^2 is the standard deviation.
    A<-tf$random$uniform(c(768L,8L),minval=-1/sqrt(768),maxval=1/sqrt(768),dtype=dtype)
    i<-weight(g,paste0(name,"_A"),A,variance=1e-6,prior_variance=.2^2)
    rho<-tf$random$uniform(c(768L,8L),minval=.05/sqrt(2),maxval=.05,dtype=dtype)
    g$weights[[i]]$lv$assign(4*tf$math$log(rho))
    g$weights[[i]]$source_name<-paste0(name,".lora_A")
    B<-point_parameter(paste0(name,".lora_B"),tf$zeros(c(8L,768L),dtype=dtype))
    node(g,name,x,function(xs,ws)
      tf$matmul(xs[[1]],a)+b+tf$matmul(tf$matmul(xs[[1]],ws[[1]]),B),i)
  }
  x<-norm(x,"roberta.embeddings.LayerNorm")
  for(i in 0:11) {
    p<-paste0("roberta.encoder.layer.",i);skip<-x
    q<-linear(x,paste0(p,".attention.self.query"),TRUE)
    k<-linear(x,paste0(p,".attention.self.key"))
    v<-linear(x,paste0(p,".attention.self.value"),TRUE)
    x<-node(g,paste0(p,".attention"),c(q,k,v,1L),function(xs,ws)
      attention(xs[[1]],xs[[2]],xs[[3]],xs[[4]],FALSE))
    x<-linear(x,paste0(p,".attention.output.dense"))
    x<-norm(add_node(g,skip,x,paste0(p,".residual1")),paste0(p,".attention.output.LayerNorm"));skip<-x
    x<-linear(x,paste0(p,".intermediate.dense"))
    x<-node(g,paste0(p,".gelu"),x,function(xs,ws)gelu(xs[[1]]))
    x<-linear(x,paste0(p,".output.dense"))
    x<-norm(add_node(g,skip,x,paste0(p,".residual2")),paste0(p,".output.LayerNorm"))
  }
  x<-node(g,"CLS",x,function(xs,ws)tf$gather(xs[[1]],0L,axis=1L))
  head<-function(x,input,output,name) {
    force(x)
    a<-point_parameter(paste0(name,".weight"),normal(c(input,output),.02,dtype))
    b<-point_parameter(paste0(name,".bias"),tf$zeros(c(output),dtype=dtype))
    node(g,name,x,function(xs,ws)tf$matmul(xs[[1]],a)+b)
  }
  x<-head(x,768L,768L,"classifier.dense")
  x<-node(g,"classifier.tanh",x,function(xs,ws)tf$tanh(xs[[1]]))
  head(x,768L,150L,"classifier.out_proj")
  g$architecture<-"roberta_lora8";g$public_path<-public$path;g
}

