import unittest
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as j
from bayesify_jax.gaussian import *
from bayesify_jax import layers as L
from bayesify_jax import heads,messages,models,covariance
from bayesify_jax.gaussian_reference import chain_reference,condition,cavi
from bayesify_jax.gaussian_chain import candidate

class Core(unittest.TestCase):
    def setUp(self):self.rng=np.random.default_rng(4)
    def arr(self,shape):return j.asarray(self.rng.normal(size=shape))
    def q(self,shape):return Moments(self.arr(shape),j.exp(self.arr(shape)-2))
    def same(self,a,b,atol=1e-10):
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):np.testing.assert_allclose(x,y,atol=atol,rtol=1e-8)
    def layer_check(self,f,literal,qs):
        y=f(*qs);g=Moments(self.arr(y.mean.shape),self.arr(y.variance.shape));d=encode(y,g)
        self.same(literal(*qs,d),L.projected(f,qs,d))
        # Check an ordinary directional derivative against central differences.
        eps=1e-5;u=jax.tree_util.tree_map(lambda z:j.ones_like(z)*.01,qs)
        scalar=lambda ps:j.sum(f(*ps).mean*g.mean+f(*ps).variance*g.variance)
        _,dphi=jax.jvp(scalar,(qs,),(u,))
        plus=jax.tree_util.tree_map(lambda x,t:x+eps*t,qs,u);minus=jax.tree_util.tree_map(lambda x,t:x-eps*t,qs,u)
        np.testing.assert_allclose(dphi,(scalar(plus)-scalar(minus))/(2*eps),rtol=1e-5,atol=1e-7)
    def test_diagonal_roundtrip(self):
        q=self.q((3,4));g=Moments(self.arr((3,4)),self.arr((3,4)));self.same(g,decode(q,encode(q,g)))
    def test_deterministic_and_tiny(self):
        q=Moments(j.array([2.,3.]),j.array([0.,1e-197]));g=Moments(j.array([4.,-2.]),j.array([1.,7.]));self.same(g,decode(q,encode(q,g)))
    def test_full_roundtrip(self):
        a=self.arr((5,5));q=FullMoments(self.arr((5,)),a@a.T+j.eye(5));g=self.arr((5,5));g=(g+g.T)/2
        adj=(self.arr((5,)),g);self.same(adj,decode_full(q,encode_full(q,adj)))
    def test_dense(self):self.layer_check(L.dense,L.dense_backward,(self.q((3,4)),self.q((4,5)),self.q((5,))))
    def test_convolution(self):
        f=lambda *q:L.convolution(*q,2,1);b=lambda x,w,c,d:L.convolution_backward(x,w,c,d,2,1)
        self.layer_check(f,b,(self.q((2,5,5,2)),self.q((3,3,2,3)),self.q((3,))))
    def test_relu(self):self.layer_check(L.relu,L.relu_backward,(self.q((3,7)),))
    def test_max_pool(self):self.layer_check(L.max_pool,L.max_pool_backward,(self.q((2,6,6,2)),))
    def test_average_pool(self):self.layer_check(L.avg_pool,L.avg_pool_backward,(self.q((2,6,6,2)),))
    def test_normalization(self):
        m=self.arr((3,));v=j.exp(self.arr((3,)))
        f=lambda *x:L.normalization(*x,m,v);b=lambda x,g,c,d:L.normalization_backward(x,g,c,d,m,v)
        self.layer_check(f,b,(self.q((2,4,4,3)),self.q((3,)),self.q((3,))))
    def test_linear_layers(self):
        for f,qs in [(L.flatten,(self.q((2,3,4,2)),)),(L.global_pool,(self.q((2,3,4,2)),)),(L.add,(self.q((2,4)),self.q((2,4)))),(L.concatenate,(self.q((2,4)),self.q((2,3)))),(L.duplicate,(self.q((2,4)),))]:
            self.layer_check(f,lambda *args:L.linear_literal(f,args[:-1],args[-1]),qs)
    def test_scalar_negative_control(self):
        q=Moments(j.zeros(1),j.ones(1));_,ep=heads.gaussian_head(q,j.zeros(1),1.,'ep');_,vi=heads.gaussian_head(q,j.zeros(1),1.,'mfvi')
        self.same(ep_moment_update(q,ep).variance,j.array([.5]));self.same(mfvi_update(q,vi,q).variance,j.array([.5]))
        self.same(ep_moment_update(q,vi).variance,j.zeros(1))
    def test_covariance_quadratic_term(self):
        q=self.q((3,));d=encode(q,Moments(j.ones(3),j.zeros(3)));u=ep_moment_update(q,d,.2)
        self.same(u.variance,q.variance-.04*d.mean**2)
        self.assertGreater(float(j.max(j.abs(u.variance-(q.variance+.2*d.centered)))),1e-6)
    def test_chain(self):
        dims=[3,4,2,5];layers=[];weights=[]
        for a,b in zip(dims[:-1],dims[1:]):
            layers.append((self.arr((b,a))*.2,self.arr((b,b)),self.arr((b,))))
            w=self.arr((b,b));weights.append(FullMoments(self.arr((b,)),w@w.T+j.eye(b)))
        initial=FullMoments(self.arr((3,)),j.eye(3));H=self.arr((2,5));y=self.arr((2,));R=j.eye(2)*.7
        _,acts,ws=candidate(layers,weights,initial,H,y,R)
        convert=lambda t:jax.tree_util.tree_map(np.asarray,t)
        ref=chain_reference(*convert((layers,weights,initial,H,y,R)))
        for i,q in enumerate(acts):self.same(q,FullMoments(ref['activation_mean'][i],ref['activation_covariance'][i]))
        for i,q in enumerate(ws):self.same(q,FullMoments(ref['weight_mean'][i],ref['weight_covariance'][i]))
        self.same(j.concatenate([acts[0].mean]+[q.mean for q in ws]),ref['dense_mean'])
    def test_cavi_each_update(self):
        X=self.arr((6,4));y=self.arr((6,));prior=Moments(j.zeros(4),j.ones(4));q=self.q((4,));precision=j.eye(4)+X.T@X;information=X.T@y
        objective=lambda q:-.5*j.sum((y-X@q.mean)**2+(X*X)@q.variance)
        ref=np.asarray(q.mean).copy()
        for _ in range(5):
            for i in range(4):
                # The candidate uses only current-belief likelihood autodiff.
                g=jax.grad(objective)(q);proposed=mfvi_update(q,encode(q,g),prior)
                q=Moments(q.mean.at[i].set(proposed.mean[i]),q.variance.at[i].set(proposed.variance[i]))
                ref[i]=(information[i]-precision[i]@ref+precision[i,i]*ref[i])/precision[i,i]
                np.testing.assert_allclose(q.mean,ref,atol=1e-10)
    def test_shared_residual(self):
        x=covariance.Shared(j.zeros((2,3)),self.arr((5,2,3)))
        out=covariance.push(lambda a:a+a,(x,));self.same(out.variance,4*x.variance)
    def test_structured_woodbury(self):
        q=covariance.StructuredGaussian(j.zeros(6),j.exp(self.arr((6,))),self.arr((2,6)))
        b=self.arr((6,));self.same(q.solve(b),j.linalg.solve(q.covariance(),b));self.same(q.logdet(),j.linalg.slogdet(q.covariance())[1])
    def test_complete_mlp_messages(self):
        m=models.mlp(inputs=4,width=5,classes=3,dtype=j.float64);x=self.q((2,4));states,record=messages.forward(m.nodes,m.snapshot(),x)
        for method in ('ep','mfvi'):
            value,d=heads.categorical_head(states[m.nodes[-1].name],j.array([0,2]),method)
            a=messages.backward(record,d,literal=False);b=messages.backward(record,d,literal=True);self.same(a,b)
            f=lambda qs:heads.categorical_head(messages.forward(m.nodes,qs,x)[0][m.nodes[-1].name],j.array([0,2]),method)[0]
            g=jax.grad(f)(m.snapshot())
            for n,q in m.snapshot().items():self.same(decode(q,a[2][n]),g[n])
    def test_shared_parameter_accumulation(self):
        nodes=(messages.Node('a','dense',('input',),('w','b')),messages.Node('b','dense',('input',),('w','b')),messages.Node('sum','add',('a','b')))
        qs={'w':self.q((3,2)),'b':self.q((2,))};x=self.q((2,3));states,record=messages.forward(nodes,qs,x)
        adj=Moments(j.ones((2,2)),j.ones((2,2)));out=messages.backward(record,encode(states['sum'],adj),literal=True)[2]
        loss=lambda q:sum(j.sum(z) for z in messages.forward(nodes,q,x)[0]['sum'])
        for n,g in jax.grad(loss)(qs).items():self.same(decode(qs[n],out[n]),g)
    def test_literal_custom_vjp(self):
        args=(self.q((2,3)),self.q((3,4)),self.q((4,)))
        literal=L.literal_autodiff(L.dense,L.dense_backward)
        scalar=lambda f,*ps:j.sum(f(*ps).mean**2+f(*ps).variance)
        a=jax.grad(lambda *ps:scalar(literal,*ps),argnums=(0,1,2))(*args)
        b=jax.grad(lambda *ps:scalar(L.dense,*ps),argnums=(0,1,2))(*args)
        self.same(a,b)
    def test_jit_update(self):
        q=self.q((3,));g=Moments(j.ones(3),-j.ones(3));p=Moments(j.zeros(3),j.ones(3))
        self.same(jax.jit(mfvi_update)(q,encode(q,g),p),mfvi_update(q,encode(q,g),p))
    def test_image_architecture_float32_with_x64_enabled(self):
        # The strict mathematics suite enables x64 globally. A missing dtype
        # on a deterministic zero bias must not promote a float32 CNN midway.
        for name,size in [('mlp',28),('lenet',28),('resnet20',32),('densenet40',32)]:
            model=getattr(models,name)();channels=1 if size==28 else 3
            output=jax.eval_shape(model,j.zeros((1,size,size,channels),j.float32))
            self.assertEqual(output.dtype,j.float32);self.assertEqual(output.shape,(1,10))
            for q in list(model.snapshot().values())+list(model.priors().values()):
                self.assertEqual(q.mean.dtype,j.float32)
                self.assertEqual(q.variance.dtype,j.float32)
            x=Moments(j.zeros((1,size,size,channels),j.float32),j.zeros((1,size,size,channels),j.float32))
            output=jax.eval_shape(lambda q:messages.forward(model.nodes,q,x,model.statistics())[0][model.nodes[-1].name],model.snapshot())
            self.assertEqual(output.mean.dtype,j.float32)
            self.assertEqual(output.covariance.dtype,j.float32)

if __name__=='__main__':unittest.main()
