"""Build the standalone JAX reader: mathematics -> exact source -> native output."""
from pathlib import Path
import ast,base64,io,json,textwrap,zipfile,re
import nbformat

ROOT=Path(__file__).parent;SOURCE=ROOT/'src'/'bayesify_jax';cells=[]

def md(s):cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s,hidden=False):
    cell=nbformat.v4.new_code_cell(textwrap.dedent(s).strip())
    if hidden:cell.metadata.update(tags=['hide-input'],jupyter={'source_hidden':True},cellView='form')
    cells.append(cell)
def excerpt(file,names,visible=True):
    source=(SOURCE/file).read_text();tree=ast.parse(source);parts={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    text='\n\n'.join(parts[n] for n in names)
    heading=f'Implementation: `{file}`'
    if visible:md('**'+heading+'**\n\n```python\n'+text+'\n```')
    else:md('<details><summary>'+heading+'</summary>\n\n```python\n'+text+'\n```\n\n</details>')
    cells[-1].metadata['tags']=['key-source' if visible else 'support-source']
    cells[-1].metadata['source_label']=file+': '+', '.join(names)
def math(name):
    source=(ROOT/'tutorial_derivations.md').read_text();md(source.split('<!-- '+name+' -->',1)[1].split('<!--',1)[0])
def payload():
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for path in SOURCE.rglob('*'):
            if path.is_file() and path.suffix in ('.py','.npz'):z.write(path,Path('src/bayesify_jax')/path.relative_to(SOURCE))
        seen=set()
        for pattern in ['tests/*.py','configs/*.json','assets/architectures/*.txt','assets/architectures/*.json','assets/architectures/onnx/*.onnx','assets/architectures/onnx/*.json','assets/architectures/netron/*.png','assets/architectures/netron/*.pdf','results/**/*.json','results/**/*.csv','results/**/*predictions.npz','results/semantics/*.npz','results/semantics/*.png','results/semantics/*.pdf','results/learned_fidelity/*','figures/*.png','figures/*.pdf']:
            for path in ROOT.glob(pattern):
                if path.name=='execution_scope.json':continue
                if 'local_checks' in path.parts:continue
                if path.name.startswith('epoch') and path.name.endswith('_predictions.npz'):continue
                if path.name.startswith('serving_') and path.suffix=='.npz':continue
                name=str(path.relative_to(ROOT))
                if name not in seen:
                    z.write(path,name);seen.add(name)
        for name in ['run_training.py','run_text.py','run_image_suite.py','run_text_suite.py','run_validation.py','run_gaussian_text.py','learned_fidelity.py','evaluate_text_serving.py','evaluate_cifar_ood.py','export_models.py','export_transformers.py','pyproject.toml','MIRROR_MAP.md']:
            z.write(ROOT/name,name)
        z.write(ROOT.parent/'site/compare_approximations.py','matched/compare_approximations.py')
        z.write(ROOT.parent/'site/compare_text_beliefs.py','matched/compare_text_beliefs.py')
        for path in (ROOT/'results/transferred_gaussian_text').glob('*_gradients.npz'):
            z.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT.parent/'site/evidence').iterdir():
            if path.suffix in ('.npz','.json','.csv'):z.write(path,'matched/evidence/'+path.name)
    return base64.b64encode(b.getvalue()).decode()

md(r'''# Bayesian backpropagation in JAX

Gaussian moment layers, explicit backward messages, and reproducible uncertainty experiments.

Companion implementation for *Bayesian Backprop as Belief Propagation:
Single-Pass Predictive Uncertainty*, Wen Dong, NeurIPS 2026
([paper page](https://neurips.cc/virtual/2026/poster/154383)).

The implementation uses **JAX** for numerical operations, **Flax NNX** for readable model modules,
and **Optax** where an Adam recurrence is intended. Exact Gaussian inference, EP, and MFVI share
forward layers and the Gaussian coordinate bridge. Their objectives and finite updates remain distinct.

The notebook is self-contained: its collapsed setup cell carries the source, fixed data partition,
validation scripts, and small numerical evidence. It contains no trained checkpoints. Full runs download
public data/model assets when their switches are enabled. The R and PyTorch implementations are separate
references; their outputs are explicitly labeled when compared with native JAX.
''')
code('''RUN_CHECKS = True
RUN_SEMANTICS = True
RUN_MNIST = False       # all five methods, MLP40 / LeNet30 epochs
RUN_DIRTY_MNIST = False # all five methods, LeNet30 epochs on ambiguous digits
RUN_CIFAR = False       # all five methods, 100 epochs per architecture
RUN_TEXT = False        # original sampled and conditional-moment transfer protocol
DOWNLOAD_PUBLIC_DATA = False
''')
code("""#@title Source and environment (expand to inspect)
import base64,io,zipfile,pathlib,sys,subprocess,importlib.util,os,gc,json
ROOT=pathlib.Path.cwd()/"bayesify_jax_reader"
ROOT.mkdir(exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode("""+repr(payload())+"""))).extractall(ROOT)
from importlib.metadata import version,PackageNotFoundError
def installed(name):
    try:return version(name)
    except PackageNotFoundError:return None
expected={'jax':'0.4.38','flax':'0.10.4','optax':'0.2.4'}
gpu=subprocess.run(['sh','-c','command -v nvidia-smi'],capture_output=True).returncode==0
if any(installed(n)!=v for n,v in expected.items()) or (gpu and installed('jax-cuda12-plugin')!='0.4.38'):
    if any(n in sys.modules for n in expected):
        raise RuntimeError('Restart the kernel before running this standalone notebook setup.')
    packages=['jax[cuda12]==0.4.38' if gpu else 'jax==0.4.38','flax==0.10.4','optax==0.2.4']
    subprocess.check_call([sys.executable,'-m','pip','install','--quiet',*packages])
if gpu:
    if installed('jax-cuda13-plugin') is not None:
        subprocess.check_call([sys.executable,'-m','pip','uninstall','-y','jax-cuda13-plugin','jax-cuda13-pjrt'])
for package in ['scipy','numpy','matplotlib','pandas','nbformat']:
    if importlib.util.find_spec(package) is None:subprocess.check_call([sys.executable,'-m','pip','install','--quiet',package])
if DOWNLOAD_PUBLIC_DATA:
    subprocess.check_call([sys.executable,'-m','pip','install','--quiet','datasets==4.8.5','transformers==5.22.1','huggingface-hub==1.33.0','safetensors==0.8.0'])
os.environ.setdefault('XLA_FLAGS','--xla_disable_hlo_passes=constant_folding')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT)
TRAIN_ENV={**os.environ,'PYTHONPATH':str(ROOT/'src'),
           'JAX_ENABLE_X64':'false','JAX_DEFAULT_MATMUL_PRECISION':'default'}
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision','highest')
import jax.numpy as jnp
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from IPython.display import display,Image,HTML
from bayesify_jax import gaussian,layers,heads,messages,models,covariance,inference
from bayesify_jax import experiments,results,inspection
def artifact(path,label):
    # Colab downloads through its kernel; ordinary Jupyter serves the extracted
    # directory. Static HTML/PDF builders use the same recorded relative file.
    import html,json
    path=str(path);click=''
    try:
        from google.colab import output,files
        callback='bayesify.download.'+path
        output.register_callback(callback,lambda:files.download(str(ROOT/path)))
        click='google.colab.kernel.invokeFunction('+json.dumps(callback)+', [], {}); return false;'
    except ImportError:
        pass
    display(HTML('<a href="bayesify_jax_reader/'+html.escape(path,quote=True)+'" data-reader-file="'+html.escape(path,quote=True)+'" onclick="'+html.escape(click,quote=True)+'">'+html.escape(label+pathlib.Path(path).name)+'</a>'))
print('JAX',jax.__version__,'devices',jax.devices())
""",True)
md(r'''## 1. Gaussian coordinates and shared layers

Observed inputs are fixed tensors, not Gaussian random data. Weight beliefs carry uncertainty;
nonlinear activation distributions receive an explicitly declared approximation. The first bridge
below changes coordinates of a derivative. It is not, by itself, a posterior update.''')
math('gaussian');excerpt('gaussian.py',['Moments','Signal','FullMoments','encode','decode','encode_full','decode_full'])
math('literal_dense');excerpt('layers.py',['dense','dense_backward','convolution','relu_terms','relu_backward'])
md(r'''The literal and projected paths use the same forward map. The literal path implements its local
message algebra directly. The projected path uses a local JAX VJP. A custom-VJP wrapper converts back
to ordinary cotangents before returning to JAX: a natural signal must never masquerade as a Euclidean gradient.''')
excerpt('layers.py',['projected','literal_autodiff']);excerpt('messages.py',['forward','backward'])
md(r'''**From equations to modules.** `GaussianParameter` owns a mean, log variance,
and a fixed prior. `MomentLayer` calls the same mathematical kernels above;
`MomentModel` connects those layers. Its ordinary call exposes the mean network
to Flax inspection. `forward_moments` and the explicit reverse traversal use an
immutable belief snapshot. Installing an updated belief happens after the whole
reverse pass, so no layer observes a partially updated parent.''')
excerpt('models.py',['GaussianParameter','MomentLayer','MomentModel'])
excerpt('models.py',['Builder'],False)
md(r'''## 2. Gaussian, EP and MFVI message equivalence

The exact example freezes a **64 → 12 → 8 → 10 tanh MLP** and its local Jacobians. The latent variables
are weight perturbations. Thus the observation passed to the perturbation graph is the label minus the
checkpoint output. Independent conditioning and RTS smoothing provide activation and weight references.
EP comparisons remove and replace explicit sites. MFVI comparisons use the current candidate state after
every coordinate update. Diagonal MFVI is compared with reverse-KL VI, not posterior marginal variances.''')
math('gaussian_smoothing');excerpt('gaussian_chain.py',['candidate'])
math('ep');math('gaussian_heads');excerpt('heads.py',['gaussian_head']);excerpt('inference.py',['ep_replace'])
excerpt('gaussian.py',['ep_moment_update','mfvi_update']);math('power_notation')
code('''if RUN_SEMANTICS:
    comparison=experiments.exact_equivalence('results/semantics')
    nonlinear=experiments.nonlinear_equivalence('results/semantics')
else:
    import json
    comparison=json.loads(pathlib.Path('results/semantics/equivalence.json').read_text())
    nonlinear=json.loads(pathlib.Path('results/semantics/nonlinear_equivalence.json').read_text())
display(pd.DataFrame(comparison))
display(pd.DataFrame(nonlinear))
for name in ['gaussian_equivalence','ep_nonlinear_equivalence','mfvi_nonlinear_equivalence']:
    display(Image(filename=f'results/semantics/{name}.png'))
''')
md(r'''The identity line is the target. The exact Gaussian and explicit-site tables use independent
inference references. The EP/MFVI plots instead use a **64 → 12 → 8 → 10 ReLU moment MLP**:
they compare handwritten layer messages with local VJPs of the same declared nonlinear closure.
The mean signal is $v\partial\Phi/\partial m$; variance sensitivity is
$2v^2\partial\Phi/\partial v$. These are backward signals, not posterior variances.
Their agreement validates message transport, not exact nonlinear Bayesian inference.
The different covariance targets of diagonal EP and MFVI remain mathematically different.''')
md(r'''## 3. Correlation, Monte Carlo fidelity, and computation

The main accuracy refinement in this section is **sampling-assisted moment
propagation**: integrate a frozen Gaussian surrogate, then use paired network
draws to estimate its nonlinear remainder. The weight belief stays one Gaussian.
The correction is stochastic; it is not the uncorrected 64-direction closure.

Read the residual identity and native implementation first, then the corrected
messages against Monte Carlo. The retained-direction plots are **ablations**:
they explain why covariance retention alone is insufficient. The exact Gaussian,
EP and MFVI message-equivalence checks in Section 2 retain their separate role.

The layers accept diagonal moments or shared covariance directions. A shared direction keeps its identity
through residual additions, concatenations, layers, and examples. Correlated weights, represented by a
diagonal residual plus low-rank factors, are a separate modeling choice.''')
math('covariance');excerpt('covariance.py',['Shared','StructuredGaussian','relu_shared','relu_shared_jvp','shared_local','structured_kl'])
math('mc_definitions');math('residual_integration');excerpt('covariance.py',['shared_records','shared_likelihood','residual_integral']);excerpt('residual.py',['residual_backward'],False)
code('''if RUN_SEMANTICS:
    fidelity=experiments.approximation_study('results/semantics',samples=16384)
else:
    import json
    fidelity=json.loads(pathlib.Path('results/semantics/approximation.json').read_text())
display(pd.DataFrame(fidelity['rows']).round(6))
''')
md(r'''This study uses a common nonlinear Gaussian-weight belief and 16,384 complete-network draws. The
backward comparison differentiates the same MFVI expectation or EP log normalizer with respect to the
same Gaussian mean/variance coordinates, then encodes both as moment signals. It does not compare a
sampled point-weight gradient directly with a pseudo-gradient.

Increasing rank reduces covariance-sketch error, but does not restore omitted nonlinear Hermite terms.
Accordingly, rank need not improve every error monotonically. These plots measure approximation quality
and execution cost on a small fixture; classification performance is measured separately below.

This small ablation uses `shared_forward`: it sketches every affine map, including
the readout. The trained CNN path uses `shared_records`, whose Gaussian readout
adds the exact conditional weight-noise term to the retained hidden covariance.
The learned-ResNet comparison later in this section uses that trained path. These are
declared readout choices within the common layer implementation, not separate
EP and MFVI layer libraries.''')
md(r'''**Correcting nonlinear residuals.** The identity above also supplies a direct
accuracy control: integrate the Gaussian surrogate, then estimate the difference
between the nonlinear network and that surrogate with shared draws. This example
uses 32,768 independent reference draws. The table counts actual nonlinear
evaluations, including antithetic and shifted evaluations. Execution timings in
the saved record have different setup boundaries, so no speedup is inferred from
them. At finite budgets, direct sampling and residual integration both retain MC error.''')
code('''if RUN_SEMANTICS:
    residual=experiments.residual_study('results/semantics',samples=32768)
else:
    import json
    residual=json.loads(pathlib.Path('results/semantics/residual.json').read_text())['rows']
display(pd.DataFrame(residual)[['method','scheme','draws','mean_relative_error','centered_relative_error']].round(5))
''')
md('## Conditional readout: implementation')
excerpt('heads.py',['sampled_readout_adjoint'])
md(r'''### What the R/TensorFlow repair established

The earlier R/TensorFlow experiment applied this residual correction to 236
selected parameter coordinates spanning the ResNet-20 blocks. On the first of
four predeclared CIFAR-10 inputs, 2,048 correction evaluations were compared with
an independent 8,192-network reference. Relative errors were:

| Method | Mean signal: baseline → corrected ↓ | Raw-second signal: baseline → corrected ↓ |
|---|---:|---:|
| MFVI | 68.30% → 4.64% | 68.09% → 4.34% |
| EP | 50.29% → 9.09% | 52.13% → 9.14% |

Mean and raw-second errors decreased on all four inputs. This is the successful
sampling-assisted repair; it is not evidence that a 64-direction closure alone
is accurate. The raw-second signal is $d_s=2m\odot d_m+c$, whereas the centered
variance signal is $c=2v^2\odot g_v$. A large mean contribution can hide errors
in $c$. In this earlier experiment the corrected centered-signal error was about
133% for both methods, with very large MC uncertainty. The readout experiment
below therefore checks mean and centered variance separately with a larger,
independent reference. The two studies have different coordinates and references;
their percentages are not a common learning curve.

The source CSV, including every input and its MC standard error, is
`r_residual_confirmation.csv` in the supplied matched evidence. These are recorded
R/TensorFlow results, not newly executed full-network experiments in every backend.

## Sampling-assisted readout messages against Monte Carlo

Keep the learned ResNet-20 MFVI or EP belief and CIFAR-10 test input 0 fixed.
An independent **524,288-network** reference measures mean and centered-variance
signals, $d_m=v\odot g_m$ and $c=2v^2\odot g_v$. These are backward sensitivities,
not posterior variances. Both plot axes use the same units and scale; the diagonal
is ideal agreement.

The first figure shows the paired residual correction at 4,096 and 16,384
evaluations. The next two figures are closure ablations: forward variance against
MC, then forward/backward error versus retained directions. They do not represent
the repaired method. Antithetic-group reference standard errors are about 0.25%
for logit variance, 0.88%/0.75% for MFVI mean/variance signals and 2.31%/1.39% for EP.

The residual readout values and complete-network reference were computed in
PyTorch and are shared numerical evidence across the readers. Each framework
renders its own plots. The closure ablations use independently computed native
backend values: 80 checks agree within $1.82\times10^{-13}$ in float64.
Implementation agreement does not imply approximation accuracy. The selected-
coordinate R study above and this readout-only study establish different scopes;
neither establishes a matched all-layer corrected training run in all three backends.''')
code('''import importlib.util
spec=importlib.util.spec_from_file_location('matched_comparison',ROOT/'matched/compare_approximations.py')
matched=importlib.util.module_from_spec(spec);spec.loader.exec_module(matched)
comparison=matched.compare();matched.figures(comparison,backend='jax')
for name in ('readout_agreement','forward_agreement','covariance_and_message_error'):
    display(Image(filename=str(matched.ROOT/(name+'.png'))))
''')
md(r'''### Accuracy and measured work

| Readout calculation | MFVI mean / variance error ↓ | EP mean / variance error ↓ |
|---|---:|---:|
| Uncorrected 64-direction closure (ablation) | 60.30% / 211.49% | 44.37% / 41.83% |
| Paired correction: 4,096 evaluations | 10.1% / 17.2% | 2.73% / 2.09% |
| Paired correction: 16,384 evaluations | 2.79% / 5.85% | 0.98% / 0.97% |
| Paired correction: 65,536 evaluations | 1.45% / 2.92% | 1.29% / 0.70% |

The correction substantially improves readout agreement for both objectives.
Errors near the reference's MC standard error should not be ranked: the last
two EP rows do not establish that fewer evaluations are intrinsically better.

| Correction evaluations | MFVI seconds ↓ | EP seconds ↓ |
|---:|---:|---:|
| 4,096 | 3.75 | 2.06 |
| 16,384 | 9.34 | 7.69 |
| 65,536 | 31.80 | 30.15 |

These saved L4 timings include Jacobian/surrogate preparation and readout
correction. MFVI ran first and paid more setup overhead. They measure one-input
diagnostics, not training or serving. The correction uses float64 and returns
readout derivatives; the original reverse pass returns all weight derivatives.
They therefore do not establish a 2–3× training-cost claim. No benchmark below
has been retrained or credited with this correction.

**Why the closure remains an ablation.** With 64 directions, forward covariance
errors are still 23.3% (MFVI) and 28.2% (EP). Refining only the Gaussian head from
18 nodes to 262,144 Sobol nodes changes mean/variance errors to 49.2%/38.8% and
42.7%/31.6%. Thus neither adding directions nor integrating that approximate
head more accurately removes the missing nonlinear response. The paired residual
directly measures this missing contribution. It is the supported refinement for
this example; its budget and variance accuracy must be measured, not assumed.''')
md(r'''## 4. MNIST: MLP and LeNet

The MLP has 478,410 parameters and widths 784 → 400 → 400 → 10. LeNet has 61,706 parameters:
5×5 convolutions with 6/16 channels, pooling, and 120/84/10 dense layers. Gaussian pooling precedes ReLU,
as in the R implementation. Both use full 10×10 readout covariance; moment EP's MLP includes the rank-four
hidden covariance correction. Observed pixels have zero probabilistic variance.''')
excerpt('models.py',['mlp','lenet','correlated_mlp']);excerpt('layers.py',['max_pair','max_pool','readout','correlated_readout'])
math('classifier');math('optimizer_complete');excerpt('training.py',['Protocol','protocol','adam_recurrence']);excerpt('inference.py',['vadam_step'])
md(r'''Each update executes the actual forward/reverse graph. Sampled MFVI uses the gradient's sampled
second moment. Moment MFVI uses a declared Hessian-vector approximation with a cached contribution,
refreshed at the source protocol's interval. EP uses its cavity and external field, followed by the
explicitly labeled Adam mean/log-variance variant. These optimizer choices are separate from the exact
canonical MFVI and explicit-site EP checks in Section 2.''')
excerpt('training.py',['make_step'],False)
math('metrics');excerpt('uncertainty.py',['uncertainty_metrics','ood_metrics']);excerpt('plots.py',['calibration_plot','risk_coverage_plot','ood_roc_plot'],False)
md(r'''**Reading the native summaries.** The MLP and LeNet summaries show every
layer; the larger models show selected stage or transformer-block boundaries.
The complete, unfiltered layer summaries and expanded graphs are linked beside
each model. Run the setup cell to activate their Colab download links; the HTML
and PDF use the corresponding files in the extracted supplement. NNX counts stored arrays: Gaussian means and log variances are
trainable arrays, while priors and normalization statistics are additional state.
These totals differ from the number of scalar weights in the mean network.''')
code('''# Execute the real NNX modules; no pretrained download is needed for inspection.
inspection.all_summaries('assets/architectures')
for name in ['mlp','lenet']:
    print(pathlib.Path(f'assets/architectures/{name}_compact.txt').read_text())
    artifact(f'assets/architectures/{name}.txt','Complete Flax NNX summary: ')
    artifact(f'assets/architectures/onnx/{name}.onnx','Actual graph for Netron: ')
    artifact(f'assets/architectures/netron/{name}.pdf','Expanded graph PDF: ')
''',True)
code('''if RUN_MNIST:
    assert DOWNLOAD_PUBLIC_DATA, 'Enable public downloads before a full run.'
    jax.clear_caches();gc.collect()
    subprocess.check_call([sys.executable,'run_image_suite.py','--architectures','mlp','lenet'],env=TRAIN_ENV)
frame=results.plots('results','figures')
if len(frame):
    for architecture in ['mlp','lenet']:
        display(HTML('<h3>'+{'mlp':'MLP / MNIST','lenet':'LeNet / MNIST'}[architecture]+'</h3>'))
        display(results.protocol_table(frame,architecture).round(5).fillna('—'))
        display(results.reader_table(frame,architecture).round(5).fillna("—"))
        for name in ['learning','uncertainty']:
            path=pathlib.Path(f'figures/{architecture}_{name}.png')
            if path.exists():display(Image(filename=str(path)))
''')
md(r'''**What the MNIST runs demonstrate.** In the saved native runs, both moment MFVI
and moment EP improve accuracy and predictive NLL over the ordinary baseline
for both architectures. Their measured training costs are below three times the
ordinary cost, and one-pass moment prediction is below 1.4 times the ordinary
inference cost. The tables also retain the sampled alternatives, allowing quality
and computation to be compared directly on the same data.

Training uses the fixed original 55,000/5,000 partition and the official 10,000-image test set.
Both MNIST models use batch size 128, with 40 epochs for MLP and 30 for LeNet.
The reported checkpoint is selected on validation NLL. Pre-run compilation is recorded separately; training
time includes the forward pass, reverse pass, update, and any additional first-use compilation after warm-up. Moment prediction uses one moment pass.
Training seconds cover the complete fixed budget; `selected_epoch` identifies the checkpoint evaluated.
Sampled prediction averages class probabilities, with the draw count stored in each run configuration.
One seed is an implementation demonstration, not strong statistical evidence of a universal Bayesian win.
The full native runs use Consumer Colab NVIDIA L4 GPUs and float32 arithmetic.
Some independent methods run on different L4 runtimes; the recorded wall-time
ratios include host overhead and are measurements of these runs, not hardware-independent FLOP ratios.
The protocol table reports the base learning rates before their schedules.
OOD comparisons use Fashion-MNIST for MNIST, SVHN for CIFAR-10, and the official
CLINC out-of-scope test split. SST-2 reports ID error detection rather than an
additional OOD task; its OOD entries are marked with a dash.
Vadam derives variance from its curvature state and has no independent variance learning rate.
Full training runs in fresh processes, separating its recorded float32 protocol
from the reader's float64 mathematical checks. Text training explicitly requests
highest matrix-multiplication precision in its own implementation.
The main table shows the core uncertainty measures; `results/benchmark_table.csv`
also retains AUPRC, FPR95, excess AURC and risks at fixed coverage where defined.

**Dirty-MNIST.** The optional loader uses the authors' ambiguous images and ten labels per image,
without added noise. Splits group repeated images to prevent leakage. Fashion-MNIST is the OOD task;
ambiguous MNIST is an in-domain ambiguity test. They answer different uncertainty questions.''')
excerpt('data.py',['ambiguous','dirty_mnist'],False)
code('''if RUN_DIRTY_MNIST:
    if not DOWNLOAD_PUBLIC_DATA:raise RuntimeError('Enable public-data downloads for training.')
    subprocess.run([sys.executable,'run_image_suite.py','--architectures','lenet','--dataset','dirty'],env=TRAIN_ENV,check=True)
frame=results.plots('results','figures')
dirty=frame[(frame.architecture=='lenet')&(frame.dataset=='dirty')]
assert len(dirty)==5,'The complete native Dirty-MNIST evidence must be bundled.'
display(results.reader_table(frame,'lenet','dirty').round(5))
display(results.protocol_table(frame,'lenet','dirty'))
display(Image(filename='figures/lenet_dirty_uncertainty.png'))
''')
md(r'''The five native Dirty-MNIST runs use 30 epochs, 110,000 training observations
and a fixed 10,000-observation validation set. Proper scores retain all ten labels
per ambiguous image, while the OOD ranking counts each image once. The shared
inference timer uses 10,000 clean test images; it does not count repeated labels
as additional forward evaluations. No new hyperparameter search is used here.
The native JAX results are separate from the R/PyTorch runs, with their own
random draws and compilation overhead. One seed is an implementation comparison,
not strong statistical evidence of a universal Bayesian advantage.''')
md(r'''## 5. CIFAR-10: residual and dense connections

ResNet-20 uses three stages of three residual blocks and option-A shortcuts. DenseNet-40 uses growth 12
and 12 units per dense block. Their parameter counts are 269,722 and 1,019,722. A skip connection does not
make the forward computational graph cyclic. It does create correlated branches; adding their independent
variances would omit cross terms. Shared covariance factors retain those terms.
Each CIFAR run uses 45,000 training images, a fixed 5,000-image validation split,
and the official 10,000-image test set: batch size 125, 100 epochs, random crops
from four-pixel padding, and horizontal flips during training. All methods use
the same partition and seed 41; validation NLL selects the reported checkpoint.''')
excerpt('models.py',['resnet20','densenet40']);math('training')
md(r'''The full readout covariance is retained as a square root, $C=RR^\top$.
For the softmax contrast matrix $D$, a thin QR factorization $(DR)^\top=QT$
gives $DCD^\top=T^\top T$. Choosing the positive diagonal gives the same Cholesky
quadrature nodes, while avoiding loss of small positive directions when a dense
Gram matrix is formed. This changes the numerical representation, not the Gaussian model.''')
code('''for name in ['resnet20','densenet40']:
    print(pathlib.Path(f'assets/architectures/{name}_compact.txt').read_text())
    artifact(f'assets/architectures/{name}.txt','Complete Flax NNX summary: ')
    artifact(f'assets/architectures/onnx/{name}.onnx','Actual graph for Netron: ')
    artifact(f'assets/architectures/netron/{name}.pdf','Expanded graph PDF: ')
if RUN_CIFAR:
    assert DOWNLOAD_PUBLIC_DATA
    jax.clear_caches();gc.collect()
    subprocess.check_call([sys.executable,'run_image_suite.py','--architectures','resnet20','densenet40'],env=TRAIN_ENV)
    for script in ['evaluate_cifar_ood.py','learned_fidelity.py']:
        subprocess.check_call([sys.executable,script],env=TRAIN_ENV)
frame=results.plots('results','figures')
if len(frame):
    for architecture in ['resnet20','densenet40']:
        rows=frame[frame.architecture==architecture]
        if len(rows):
            display(HTML('<h3>'+{'resnet20':'ResNet-20 / CIFAR-10','densenet40':'DenseNet-40 / CIFAR-10'}[architecture]+'</h3>'))
            display(results.protocol_table(frame,architecture).round(5).fillna('—'))
            display(results.reader_table(frame,architecture).round(5).fillna("—"))
            for name in ['learning','uncertainty']:
                path=pathlib.Path(f'figures/{architecture}_{name}.png')
                if path.exists():display(Image(filename=str(path)))
''')
md(r'''**What the CIFAR runs demonstrate.** Both moment methods improve accuracy and
predictive NLL over ordinary training on both architectures in these native runs.
The cost depends on the architecture and update: ResNet moment EP takes about
three times the training time and 2.5 times the inference time, while DenseNet
moment MFVI has a larger training cost. The tables retain these differences rather
than implying one overhead factor for every model. OOD evaluation uses the same
selected beliefs and is assessed separately from in-distribution calibration.

**Same-belief approximation check.** Section 3 separately compares the native
learned-belief forward and backward approximation with a high-precision MC
reference. Predictive performance and local moment fidelity are different
measurements; neither is substituted for the other.''')
md(r'''## 6. GPT-2 LoRA: attention and Gaussian message boundaries

GPT-2-small retains its pretrained backbone and uses rank-four query/value adapters. SST-2 is evaluated
through the probabilities of the negative/positive answer tokens. The model does not average the two
answer logits and call that Bayesian model averaging. Gaussian-A+B and conditional Gaussian-A/point-B
are distinct families and are labeled separately.''')
math('transformer');math('transformer_calculus');excerpt('transformers.py',['attention','block','readout','tangent']);excerpt('transformers.py',['moment_likelihood'],False)
md(r'''### The same Gaussian adapters in two independent implementations

This check transfers the selected MFVI and EP beliefs, with both Gaussian LoRA
factors, into native JAX without training again. Both implementations use the same
frozen GPT-2 weights, 872 SST-2 inputs and 20-node scalar Gaussian integral.
For the logit contrast $z(w)$, the delta approximation is $m_z=z(m)$ and
$v_z=\sum_j(\partial_jz(m))^2v_j$. Differentiating this map includes the
dependence of its Jacobian on the mean. It does not replace that Jacobian by a
constant during the backward pass.
This check uses the fully differentiated map; a frozen-Jacobian training variant
is a separate approximation.

The four-example likelihood comparison checks all adapter mean and variance
derivatives in matching coordinates. It establishes implementation agreement for
this approximation. It is separate from accuracy against nonlinear network samples,
which is the question addressed by the earlier Monte Carlo study.''')
excerpt('transformers.py',['scalar_moments','scalar_likelihood'])
code('''folder=pathlib.Path('results/transferred_gaussian_text')
subprocess.run([sys.executable,'matched/compare_text_beliefs.py',str(folder)],
               check=True,capture_output=True,text=True)
agreement=json.loads((folder/'native_agreement.json').read_text())
assert agreement['passed'],'Investigate the matched-belief discrepancy before reporting equivalence.'
records=[]
for method in ['mfvi','ep']:
    for suffix,framework in [('', 'JAX'),('_torch','PyTorch reference')]:
        r=json.loads((folder/(method+suffix+'.json')).read_text())
        records.append({'Belief':method.upper(),'Implementation':framework,
                        'Accuracy ↑':r['accuracy'],'NLL ↓':r['nll'],'ECE ↓':r['ece']})
display(pd.DataFrame(records).round(7))
display(pd.DataFrame(agreement['rows']).rename(columns={
    'max_abs':'Maximum absolute error ↓','relative_l2':'Relative L2 error ↓'}))
''')
md(r'''**Full-weight RoBERTa counterpart.** Setting `family='full'` gives a Gaussian belief to every embedding, normalization, attention, feed-forward and classifier parameter. Local embedding JVPs seed the activation covariance; repeated token IDs reuse the same uncertain embedding rows. This is separate from the fixed-backbone LoRA model. Token IDs and attention masks remain observed tensors.''')
excerpt('transformers.py',['full_weight_beliefs']);excerpt('gaussian_text.py',['initialize','frozen_statistics','make_step'],False)
code('''for name in ['gpt2','roberta']:
    print(pathlib.Path(f'assets/architectures/{name}_compact.txt').read_text())
    artifact(f'assets/architectures/{name}.txt','Complete Flax NNX summary: ')
    artifact(f'assets/architectures/onnx/{name}.onnx','Actual graph for Netron: ')
    artifact(f'assets/architectures/netron/{name}.pdf','Expanded graph PDF: ')
''',True)
md(r'''## 7. Conditional moment MFVI: GPT-2 and RoBERTa LoRA

This path mirrors the BLoB-inspired source experiment. It uses Gaussian A with variance $\rho^4$,
point B, and a full point classifier for CLINC150. It is not full-weight Bayesian RoBERTa, nor EP.
The split optimizer preserves the source ordering: a likelihood AdamW step on means, followed by SGD
of the retained variance adjoint and a KL evaluated after the mean step.

Both text protocols use batches of 32 and at most 128 tokens. GPT-2 uses 67,349
SST-2 training sentences, three epochs, and the official 872-sentence validation
set as the reporting cohort. Its fixed two-example sentiment prompt and answer
tokens are defined in `data.text_data`; no hidden-test result is claimed.
CLINC uses 13,500 fitting utterances and a 1,500-utterance stopping split taken
from the training set, six epochs, and 4,500 official in-scope test utterances.
The 1,000 official out-of-scope test utterances are used only for OOD evaluation.
Duplicate token sequences stay in the same fitting/stopping group. The ordinary
and Bayesian variants share the pretrained backbone, adapter initialization,
data partitions, and reporting protocol.

For unit vector $a$, source Gaussian noise $\xi$ and independent Gaussian $Z_i$, define
$\Xi_i=a_i\xi+Z_i-a_i\sum_j a_j Z_j$. Conditional on $a$, the joint covariance is the identity,
and $\sum_i a_i\Xi_i=\xi$. Antithetic evaluations form local means and odd/even deviations.
Spherical recompression preserves their conditional first two moments. It does not preserve all higher
moments. With one direction, the source-coupled construction recovers the source output and derivatives
exactly; this is an executable exact-limit check for the approximation.''')
excerpt('blob.py',['coupled_nodes','transition','source_kl','split_update']);excerpt('text_training.py',['make_step'],False)
code('''# Native tests include K=1 output/derivative identity and actual split updates.
import unittest
from bayesify_jax import blob
if RUN_TEXT:
    assert DOWNLOAD_PUBLIC_DATA
    jax.clear_caches();gc.collect()
    subprocess.check_call([sys.executable,'run_text_suite.py','--tasks','clinc','sst2'],env=TRAIN_ENV)
    subprocess.check_call([sys.executable,'evaluate_text_serving.py'],
        env=TRAIN_ENV)
''')
md(r'''The primary text tables below use ten source-style Flipout probability predictions for both learned Bayesian beliefs, so they compare training outcomes under a common predictor. This is not a one-pass moment inference claim. Flipout uses token-dependent signs; it is not one complete sampled weight matrix shared across all tokens. The serving diagnostic also includes `global` sampling, which draws Gaussian adapter matrices shared throughout each sequence. The separate same-belief serving table compares sampled and conditional-moment prediction and records both draw counts and elapsed time. In that table, `sampled` means source Flipout, `hybrid` means conditional moments with two antithetic pairs per block, and `global` means shared Gaussian adapter draws. CLINC uses its training-only stopping split; SST-2 reports the fixed final epoch. In these text protocols, the second learning rate updates $\rho$ (variance is $\rho^4$); decoupled decay applies to ordinary weights and the point classifier, not Gaussian adapter means. The graph downloads open in [Netron](https://netron.app); their numerical export checks are in `assets/architectures/onnx/*manifest.json`.''')
code('''frame=results.plots('results','figures')
for task in ['sst2','clinc']:
    if len(frame) and (frame.architecture==task).any():
        display(HTML('<h3>'+{'sst2':'GPT-2 LoRA / SST-2','clinc':'RoBERTa LoRA / CLINC150'}[task]+'</h3>'))
        display(results.protocol_table(frame,task).round(5).fillna('—'))
        display(results.reader_table(frame,task).round(5).fillna("—"))
        for name in ['learning','uncertainty']:
            p=pathlib.Path(f'figures/{task}_{name}.png')
            if p.exists():display(Image(filename=str(p)))
p=pathlib.Path('results/text_serving.json')
if p.exists():
    import json
    serving=pd.DataFrame(json.loads(p.read_text()))
    display(serving[['task','training','prediction','draws','accuracy','nll','ece','inference_seconds']].round(5))
''')
md(r'''**What the text examples demonstrate.** The native conditional-moment training
path improves predictive NLL and ECE over the ordinary baseline in both datasets.
CLINC also improves accuracy; SST-2's accuracy is slightly lower. These are distinct
outcomes, and the OOD table is assessed separately. The same-belief serving table
shows how prediction choices change both quality and cost. In particular, the
ten-pass primary predictor is not evidence of one-pass transformer deployment.
The faster one-pass MNIST result and this transformer result therefore support
different computational claims.''')
md(r'''## 8. Reproduction and code reading

Start with `gaussian.py`, `layers.py`, and `messages.py`. Then read the method-specific heads and
updates in `heads.py` and `inference.py`. `covariance.py` changes the retained moment representation,
not the definition of EP or MFVI. `models.py` and `transformers.py` define the architectures.
`training.py` and `text_training.py` execute the minibatch protocols. `data.py`, `uncertainty.py`,
`plots.py`, and `results.py` supply the shared data and evaluation path.

Exact conditioning and site references are independent NumPy/SciPy code. Every figure is generated
from saved numeric arrays or newly executed JAX computations. The source-only supplement excludes
checkpoints and large datasets. The complete source remains in the notebook's collapsed bootstrap cell. `gaussian_text.py` supplies the original independent-Gaussian EP/MFVI adapter or full-weight updates; the conditional Gaussian MFVI path is separately named.''')
code('''if RUN_CHECKS:
    completed=subprocess.run([sys.executable,'run_validation.py','--output','results/checks'],
        env={**os.environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
    pathlib.Path('results/checks/execution.log').write_text(completed.stdout+'\\n'+completed.stderr)
    import json
    receipt=json.loads(pathlib.Path('results/checks/validation_receipt.json').read_text())
    display(pd.DataFrame([{k:receipt[k] for k in ['status','tests_run','failures','errors','seconds','jax']}]))
    artifact('results/checks/validation_receipt.json','Versions, device and tested source hash: ')
    if completed.returncode:
        print(completed.stderr)
        raise RuntimeError('JAX validation failed')
''',True)
md(r'''Implementation references: [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html),
[Flax NNX summaries](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/summary.html),
[ONNX export and Netron](https://enpasos.github.io/jax2onnx/), and
[BLoB source](https://github.com/Wang-ML-Lab/bayesian-peft/tree/7d960082efa3a680b76aac79d2a4f13d7f0f651a).
''')
nb=nbformat.v4.new_notebook(cells=cells);nb.metadata.kernelspec={'display_name':'Python 3 (JAX)','language':'python','name':'python3'}
for number,cell in enumerate(cells,1):
    if cell.cell_type=='code':compile(cell.source,f'notebook cell {number}','exec')
import sys
sys.path.insert(0, str(ROOT.parent/'site'))
from tutorial_content import restore_topics
nb=nbformat.from_dict(restore_topics(nb,'jax'));cells=nb.cells
# A build is unexecuted. Keep the validated published reader intact until the
# candidate has been run and its native outputs checked.
nbformat.write(nb,ROOT/'README.full.ipynb')
# Human-readable cells/dependencies, generated from the actual reader.
sections=[];current=None
for number,cell in enumerate(cells,1):
    source=cell.source
    if cell.cell_type=='markdown' and re.match(r'^#{1,2} [1-8]\. ',source):
        if current:sections.append(current)
        current={'title':source.splitlines()[0][3:],'start':number,'code':0,'key':0,'support':0}
    if current:
        current['end']=number
        tags=cell.metadata.get('tags',[])
        if cell.cell_type=='code':current['code']+=len(source.splitlines()) if 'hide-input' not in tags else 0
        if 'key-source' in tags:current['key']+=sum(len(x.splitlines()) for x in source.split('```python\n')[1:])
        if 'support-source' in tags:current['support']+=sum(len(x.splitlines()) for x in source.split('```python\n')[1:])
if current:sections.append(current)
lines=['# Reader map','', 'Visible source excerpts are copied directly from the implemented files. The hidden bootstrap embeds the complete executable source; no private directory is required.','','| Section | Cells | Visible executable lines | Expanded source lines | Collapsed source lines |','|---|---:|---:|---:|---:|']
for x in sections:lines.append(f"| {x['title']} | {x['start']}–{x['end']} | {x['code']} | {x['key']} | {x['support']} |")
lines+=['','Dependency order: Gaussian geometry → shared layers/reverse traversal → method-specific heads and finite updates → covariance representation → architectures → training and shared evaluation. The transformer chapters reuse the same Gaussian and covariance objects.','', 'Compact architecture summaries are shown inline; fully expanded native graphs are downloadable PDFs and ONNX files. Full summaries are regenerated by calling the actual NNX modules. ONNX graphs were separately exported and numerically checked; their optional exporter scripts require jax2onnx and ONNX Runtime in an isolated environment.','', 'The complete R-to-JAX file/function map is in MIRROR_MAP.md.']
(ROOT/'READER_MAP.md').write_text('\n'.join(lines)+'\n')
print('Built',len(cells),'cells')
