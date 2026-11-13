"""The same public image data, split indices and ambiguity protocol as R/JAX.

Arrays remain NHWC here; the native PyTorch training loop transposes to NCHW.
No framework's predictions or plotting coordinates are imported.
"""
from pathlib import Path
import gzip,json,hashlib,urllib.request,pickle,tarfile
import numpy as np
import torch

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


def ambiguous(root='data'):
    root=Path(root);cache=root/'ambiguous.npz'
    if not cache.exists():
        base='https://github.com/BlackHC/ddu_dirty_mnist/releases/download/data-v1.0.0/'
        files=[]
        for name,checksum in [('amnist_samples.pt','4f7865093b1d28e34019847fab917722'),('amnist_labels.pt','3bfc055a9f91a76d8d493e8b898c3c95')]:
            p=download(base+name,root/name)
            if hashlib.md5(p.read_bytes()).hexdigest()!=checksum:raise ValueError('Public data checksum mismatch')
            files.append(torch.load(p, map_location="cpu", weights_only=True).numpy())
        x,y=files;np.savez_compressed(cache,images=x.transpose(0,2,3,1),labels=y.astype('int32'))
    with np.load(cache) as a:return a['images'],a['labels']


def dirty_mnist(root='data'):
    clean=mnist(root);x,y=ambiguous(root);order=np.random.default_rng(20261107).permutation(6000)
    for name,inds in [('train',order[:5500]),('validation',order[5500:])]:
        cx,cy=clean[name];clean[name]=(np.concatenate((cx,np.repeat(x[inds],10,axis=0))),np.concatenate((cy,y[inds].ravel())))
    clean['ambiguous_test']=(x[6000:],y[6000:]);return clean
