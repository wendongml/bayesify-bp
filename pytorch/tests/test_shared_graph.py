"""Independent checks of the correlation extension of the native layer graph."""
import unittest
import torch
from bayesify_torch.correlation import (
    shared_forward, shared_likelihood, gaussian_readout, root_nodes, root_objective)
from bayesify_torch.inference import named_gaussian_parameters
from bayesify_torch.models import TinyResidualMLP
from bayesify_torch.graph import MomentDAG
from bayesify_torch.layers import (
    MomentLinear, MomentAdd, MomentConv2d, MomentBatchNorm2d,
    MomentReLU, MomentGlobalAvgPool2d)


class SharedGraphTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(29)
        torch.set_num_threads(2)

    def check_graph(self, graph, x, labels):
        beliefs = {n:(p.mean,p.variance) for n,p in named_gaussian_parameters(graph)}
        for method in ('mfvi','ep'):
            with self.subTest(method=method):
                value, weights, _, _, passed = shared_likelihood(
                    graph,x,labels,rank=3,seed=61,method=method,power=.7,beliefs=beliefs)
                # The reference differentiates the whole declared graph. It
                # never calls the explicit local-message reverse traversal.
                def objective(q):
                    z=shared_forward(graph,x,3,61,beliefs=q).output
                    return root_objective(z,labels,method,.7)
                direct=torch.func.grad(objective)(beliefs)
                for name in beliefs:
                    for got,expected in zip(weights[name],direct[name]):
                        torch.testing.assert_close(got,expected,atol=1e-10,rtol=1e-8)
                torch.testing.assert_close(value,objective(beliefs))
                self.assertTrue(all(torch.isfinite(a).all() for p in weights.values() for a in p))
        return passed

    def test_residual_mlp_local_vjps_and_readout(self):
        graph=TinyResidualMLP(4,5,3).network.double()
        passed=self.check_graph(graph,torch.randn(3,4),torch.tensor([0,1,2]))
        node=graph.nodes[-1];head=graph.layers[node.name]
        hidden=passed.states[node.parents[0]]
        mean,cov=gaussian_readout(hidden,head.weight.mean,head.weight.variance,
                                 head.bias.mean,head.bias.variance)
        torch.testing.assert_close(passed.output.mean,mean)
        torch.testing.assert_close(passed.output.covariance,cov)
        # Contrast cubature reproduces the represented mean and covariance,
        # including the 1/rank factor; softmax is invariant to the removed logit.
        nodes=root_nodes(passed.output)
        means=nodes.mean(0);centered=nodes-means
        actual=torch.einsum('rbi,rbj->bij',centered,centered)/len(nodes)
        expected=cov-cov[:,:,-1:]-cov[:,-1:,:]+cov[:,-1:,-1:]
        torch.testing.assert_close(means,mean-mean[:,-1:])
        torch.testing.assert_close(actual,expected)

    def test_repeated_parent_keeps_cross_covariance_and_reverse_sum(self):
        graph=MomentDAG([
            ('hidden',MomentLinear(4,5),('input',)),
            ('twice',MomentAdd(),('hidden','hidden')),
            ('head',MomentLinear(5,3),('twice',))]).double()
        passed=self.check_graph(graph,torch.randn(2,4),torch.tensor([0,2]))
        torch.testing.assert_close(passed.states['twice'].variance,
                                   4*passed.states['hidden'].variance)

    def test_convolution_norm_reverse_and_frozen_running_state(self):
        graph=MomentDAG([
            ('conv',MomentConv2d(1,2,3,padding=1),('input',)),
            ('norm',MomentBatchNorm2d(2),('conv',)),
            ('relu',MomentReLU(),('norm',)),
            ('pool',MomentGlobalAvgPool2d(),('relu',)),
            ('head',MomentLinear(2,3),('pool',))]).double()
        before={n:a.clone() for n,a in graph.named_buffers()}
        self.check_graph(graph,torch.randn(2,1,4,4),torch.tensor([1,2]))
        for n,a in graph.named_buffers():torch.testing.assert_close(a,before[n])

    def test_correlated_ep_cavity_field_and_finite_updates(self):
        # Ordinary autodiff sees the complete scalar energy, including the
        # posterior-to-cavity chain rule and the single external field.
        import copy
        from bayesify_torch.training import moment_ep_gradients
        from bayesify_torch.optimizers import power_cavity, EPAdam
        graph=TinyResidualMLP(4,5,3).network.double()
        reference=copy.deepcopy(graph)
        x,y=torch.randn(3,4),torch.tensor([0,1,2])
        ps=dict(named_gaussian_parameters(graph));rs=dict(named_gaussian_parameters(reference))
        a,b=EPAdam(list(ps.values()),1e-4,2e-4),EPAdam(list(rs.values()),1e-4,2e-4)
        for _ in range(3):
            got=moment_ep_gradients(graph,x,y,53,.14,.7,1/3,rank=3)
            cavity=power_cavity(rs,53,.14)
            q={n:(v.mean,v.variance) for n,v in cavity.beliefs.items()}
            z=shared_forward(reference,x,3,41,q).output
            loss=-root_objective(z,y,'ep',.7,1/3)-cavity.log_field/.7
            gradients=torch.autograd.grad(loss,tuple(v for p in rs.values()
                                                    for v in (p.mean,p.log_variance)))
            gm=list(gradients[::2]);gv=[g/p.variance for g,p in zip(gradients[1::2],rs.values())]
            torch.testing.assert_close(got.loss,loss,atol=1e-10,rtol=1e-8)
            for i,n in enumerate(ps):
                torch.testing.assert_close(got.mean[n],gm[i],atol=1e-10,rtol=1e-8)
                torch.testing.assert_close(got.variance[n],gv[i],atol=1e-10,rtol=1e-8)
            a.step(list(got.mean.values()),list(got.variance.values()))
            b.step(gm,gv)
            for n in ps:
                torch.testing.assert_close(ps[n].mean,rs[n].mean,atol=1e-10,rtol=1e-8)
                torch.testing.assert_close(ps[n].variance,rs[n].variance,atol=1e-12,rtol=1e-8)

    def test_batchnorm_covariance_reverse_against_finite_differences(self):
        # Agreement of two autodiff traversals alone misses a faulty primitive's
        # higher derivative. Check d/dm of a loss depending on its covariance
        # JVP directly against central finite differences.
        from bayesify_torch.correlation import shared_layer
        layer=MomentBatchNorm2d(2).double().train()
        m=torch.randn(2,2,3,3);U=torch.randn(3,2,2,3,3)*.2
        q=((layer.weight.mean,layer.weight.variance),(layer.bias.mean,layer.bias.variance))
        noise=(torch.randn(3,2),torch.randn(3,2))
        am,au=torch.randn_like(m),torch.randn_like(U)
        def objective(a):
            mean,factors=shared_layer(layer,((a,U),),q,noise)
            return (am*mean).sum()+(au*factors).sum()
        gradient=torch.func.grad(objective)(m);direction=torch.randn_like(m)
        exact=(gradient*direction).sum()
        for h in (1e-4,1e-5,1e-6):
            finite=(objective(m+h*direction)-objective(m-h*direction))/(2*h)
            torch.testing.assert_close(exact,finite,atol=2e-7,rtol=2e-7)

    def test_diagonal_hidden_full_readout_reverse(self):
        from bayesify_torch.geometry import DiagonalMoments
        from bayesify_torch.training import moment_likelihood_gradients
        graph=TinyResidualMLP(4,5,3).network.double()
        x,y=torch.randn(2,4),torch.tensor([0,1])
        q=DiagonalMoments(x,torch.zeros_like(x))
        for method in ('mfvi','ep'):
            result=moment_likelihood_gradients(graph,x,y,method,full_readout=True)
            z=graph.forward_moments(q,full_readout=True).output
            objective=root_objective(z,y,method)
            ps=dict(named_gaussian_parameters(graph))
            gs=torch.autograd.grad(objective,tuple(a for p in ps.values() for a in (p.mean,p.log_variance)))
            for i,(name,p) in enumerate(ps.items()):
                torch.testing.assert_close(result.mean[name],-gs[2*i],atol=1e-10,rtol=1e-8)
                torch.testing.assert_close(result.variance[name],-gs[2*i+1]/p.variance,atol=1e-10,rtol=1e-8)


if __name__=='__main__':unittest.main()
