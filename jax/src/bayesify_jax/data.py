"""Public data -> NumPy arrays. Download once; persist exact split indices."""
from pathlib import Path
import gzip,json,hashlib,urllib.request,pickle,tarfile
import numpy as np


def download(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        part=path.with_suffix(path.suffix+'.part');urllib.request.urlretrieve(url,part);part.replace(path)
    return path


def stratified(labels,count,seed):
    rng=np.random.default_rng(seed);train=[];validation=[]
    for c in np.unique(labels):
        ids=rng.permutation(np.flatnonzero(labels==c));validation.extend(ids[:count]);train.extend(ids[count:])
    return rng.permutation(train),rng.permutation(validation)


def mnist(root='data'):
    root=Path(root);file=download('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',root/'mnist.npz')
    with np.load(file) as a:x=a['x_train'];y=a['y_train'];xt=a['x_test'];yt=a['y_test']
    split=root/'mnist_split.npz'
    if not split.exists():
        import shutil
        shutil.copyfile(Path(__file__).parent/'assets/mnist_split.npz',split)
    with np.load(split) as a:fit=a['fit'];val=a['validation']
    data={'train':(x[fit,...,None]/np.float32(255),y[fit].astype('int32')),'validation':(x[val,...,None]/np.float32(255),y[val].astype('int32')),'test':(xt[...,None]/np.float32(255),yt.astype('int32'))}
    return data


def fashion(root='data'):
    root=Path(root);base='https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
    a=download(base+'t10k-images-idx3-ubyte.gz',root/'fashion-images.gz')
    b=download(base+'t10k-labels-idx1-ubyte.gz',root/'fashion-labels.gz')
    x=np.frombuffer(gzip.decompress(a.read_bytes()),dtype=np.uint8,offset=16).reshape(-1,28,28,1)/np.float32(255)
    y=np.frombuffer(gzip.decompress(b.read_bytes()),dtype=np.uint8,offset=8).astype('int32')
    return x,y


def cifar10(root='data'):
    root=Path(root);archive=download('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',root/'cifar10.tar.gz')
    # Read trusted public batch members directly; no filesystem tar extraction.
    with tarfile.open(archive) as tar:
        def batch(name):
            a=pickle.load(tar.extractfile('cifar-10-batches-py/'+name),encoding='bytes')
            return a[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1).astype('float32')/255,np.array(a[b'labels'],dtype='int32')
        pairs=[batch(f'data_batch_{i}') for i in range(1,6)];xt,yt=batch('test_batch')
    x=np.concatenate([p[0] for p in pairs]);y=np.concatenate([p[1] for p in pairs]);split=root/'cifar10_split.npz'
    if not split.exists():
        fit,val=stratified(y,500,20261021);np.savez(split,fit=fit,validation=val)
    with np.load(split) as a:fit,val=a['fit'],a['validation']
    mean=np.array([.4914,.4822,.4465],dtype='float32');std=np.array([.247,.2435,.2616],dtype='float32')
    return {'train':((x[fit]-mean)/std,y[fit]),'validation':((x[val]-mean)/std,y[val]),'test':((xt-mean)/std,yt)}


def svhn(root='data'):
    """SVHN test images with the CIFAR training transform; no OOD fitting."""
    from scipy.io import loadmat
    path=download('https://ufldl.stanford.edu/housenumbers/test_32x32.mat',Path(root)/'svhn_test.mat')
    if hashlib.md5(path.read_bytes()).hexdigest()!='eb5a983be6a315427106f1b164d9cef3':
        raise ValueError('SVHN public test archive checksum mismatch')
    a=loadmat(path);x=a['X'].transpose(3,0,1,2).astype('float32')/255
    mean=np.array([.4914,.4822,.4465],dtype='float32');std=np.array([.247,.2435,.2616],dtype='float32')
    return (x-mean)/std,a['y'].ravel().astype('int32')%10


def text_data(task,root='data'):
    """Tokenize with the original prompt and explicit stopping/report separation."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    root=Path(root)/task;root.mkdir(parents=True,exist_ok=True)
    cache=root/'tokens.npz'
    if cache.exists():
        with np.load(cache) as f:return {k:f[k] for k in f.files}
    tokenizer=AutoTokenizer.from_pretrained('gpt2' if task=='sst2' else 'roberta-base',revision='607a30d783dfa663caf39e06633721c8d4cfcd7e' if task=='sst2' else 'e2da8e2f811d1448a5b465c236feacd80ffbac7b')
    result={}
    if task=='sst2':
        tokenizer.pad_token=tokenizer.eos_token;ds=load_dataset('stanfordnlp/sst2',revision='8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb')
        prefix=tokenizer.encode('Review: great acting and direction\nSentiment: positive.\n\nReview: boring and predictable\nSentiment: negative.\n\n',add_special_tokens=False)
        suffix=tokenizer.encode('\nSentiment:',add_special_tokens=False)
        for name,part in [('train','train'),('report','validation')]:
            texts=ds[part]['sentence'];ids=np.full((len(texts),128),tokenizer.eos_token_id,dtype='int32');mask=np.zeros_like(ids)
            for i,t in enumerate(texts):
                row=prefix+tokenizer.encode('Review: '+t,add_special_tokens=False)[:128-len(prefix)-len(suffix)]+suffix
                ids[i,:len(row)]=row;mask[i,:len(row)]=1
            result[name+'_ids']=ids;result[name+'_mask']=mask;result[name+'_labels']=np.asarray(ds[part]['label'],dtype='int32')
        result['target_ids']=np.array([tokenizer.encode(x,add_special_tokens=False)[0] for x in [' negative',' positive']],dtype='int32')
    else:
        ds=load_dataset('clinc/clinc_oos','plus',revision='155b9c710419136e17307b80d0a13e68cd46b4ec');names=ds['train'].features['intent'].names;oos=names.index('oos');classes=[i for i in range(len(names)) if i!=oos]
        for name,part,inside in [('alltrain','train',True),('development','validation',True),('report','test',True),('ood','test',False)]:
            rows=[r for r in ds[part] if (r['intent']!=oos)==inside]
            t=tokenizer([r['text'] for r in rows],padding='max_length',truncation=True,max_length=128,return_tensors='np')
            result[name+'_ids']=t['input_ids'].astype('int32');result[name+'_mask']=t['attention_mask'].astype('int32');result[name+'_labels']=np.array([classes.index(r['intent']) if inside else -1 for r in rows],dtype='int32')
        groups={}
        for i,row in enumerate(result['alltrain_ids']):groups.setdefault(tuple(row[:result['alltrain_mask'][i].sum()].tolist()),[]).append(i)
        order=sorted(groups,key=lambda k:hashlib.sha256(('20261023:'+','.join(map(str,k))).encode()).digest());counts=np.zeros(150,int);stop=[];labels=result['alltrain_labels']
        for key in order:
            group=groups[key]
            if any(counts[labels[i]]<10 for i in group):
                stop.extend(group)
                for i in group:counts[labels[i]]+=1
        stop=np.array(sorted(stop));fit=np.array(sorted(set(range(len(labels)))-set(stop)))
        for tag,inds in [('train',fit),('stop',stop)]:
            for suffix in ['ids','mask','labels']:result[tag+'_'+suffix]=result['alltrain_'+suffix][inds]
        for key in ['alltrain_ids','alltrain_mask','alltrain_labels']:del result[key]
        result['target_ids']=np.array([0,1],dtype='int32');np.savez(root/'split.npz',fit=fit,stop=stop)
    np.savez_compressed(cache,**result)
    (root/'data_receipt.json').write_text(json.dumps({'sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),'task':task,'shapes':{k:list(v.shape) for k,v in result.items()}},indent=2))
    return result


def tensor_archive(path):
    """Read the authors' tensor-only .pt archive as NumPy, without PyTorch.

    Only tensor rebuilding, primitive storage types and OrderedDict are allowed.
    This is a data-format adapter, not a model or inference dependency.
    """
    import io,zipfile,collections
    with zipfile.ZipFile(path) as z:
        def rebuild(storage,offset,shape,strides,*unused):
            return np.ndarray(shape,dtype=storage.dtype,buffer=storage,offset=offset*storage.itemsize,strides=tuple(i*storage.itemsize for i in strides)).copy()
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                if module=='torch._utils' and name=='_rebuild_tensor_v2':return rebuild
                if module=='torch' and name in ('FloatStorage','LongStorage'):return np.dtype('<f4' if name=='FloatStorage' else '<i8')
                if module=='collections' and name=='OrderedDict':return collections.OrderedDict
                raise ValueError('Unsupported tensor archive global '+module+'.'+name)
            def persistent_load(self,item):
                kind,dtype,key,location,size=item
                if kind!='storage':raise ValueError(kind)
                a=np.frombuffer(z.read('archive/data/'+key),dtype=dtype)
                if a.size!=size:raise ValueError('Storage size mismatch')
                return a
        return Reader(io.BytesIO(z.read('archive/data.pkl'))).load()


def ambiguous(root='data'):
    root=Path(root);cache=root/'ambiguous.npz'
    if not cache.exists():
        base='https://github.com/BlackHC/ddu_dirty_mnist/releases/download/data-v1.0.0/'
        files=[]
        for name,checksum in [('amnist_samples.pt','4f7865093b1d28e34019847fab917722'),('amnist_labels.pt','3bfc055a9f91a76d8d493e8b898c3c95')]:
            p=download(base+name,root/name)
            if hashlib.md5(p.read_bytes()).hexdigest()!=checksum:raise ValueError('Public data checksum mismatch')
            files.append(tensor_archive(p))
        x,y=files;np.savez_compressed(cache,images=x.transpose(0,2,3,1),labels=y.astype('int32'))
    with np.load(cache) as a:return a['images'],a['labels']


def dirty_mnist(root='data'):
    clean=mnist(root);x,y=ambiguous(root);order=np.random.default_rng(20261107).permutation(6000)
    for name,inds in [('train',order[:5500]),('validation',order[5500:])]:
        cx,cy=clean[name];clean[name]=(np.concatenate((cx,np.repeat(x[inds],10,axis=0))),np.concatenate((cy,y[inds].ravel())))
    clean['ambiguous_test']=(x[6000:],y[6000:]);return clean
