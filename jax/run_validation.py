"""python run_validation.py; add --figures for native semantic/approximation plots."""
import argparse,unittest,json,time,platform,hashlib
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
p=argparse.ArgumentParser();p.add_argument('--figures',action='store_true');p.add_argument('--only-figures',action='store_true');p.add_argument('--output',default='results');a=p.parse_args()
root=Path(__file__).parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
if not a.only_figures:
 begin=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(root/'tests')))
 digest=hashlib.sha256()
 for f in sorted((root/'src').rglob('*.py')):digest.update(str(f.relative_to(root)).encode());digest.update(f.read_bytes())
 receipt={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'seconds':time.perf_counter()-begin,'jax':jax.__version__,'versions':{n:version(n) for n in ['jax','flax','optax','numpy','scipy']},'python':platform.python_version(),'source_sha256':digest.hexdigest(),'utc':datetime.now(timezone.utc).isoformat(),'devices':[str(d) for d in jax.devices()],'status':'PASS' if result.wasSuccessful() else 'FAIL'}
 (out/'validation_receipt.json').write_text(json.dumps(receipt,indent=2))
 if not result.wasSuccessful():raise SystemExit(1)
if a.figures or a.only_figures:
 from bayesify_jax.experiments import exact_equivalence,nonlinear_equivalence,approximation_study
 print(exact_equivalence(out/'semantics'),flush=True);print(nonlinear_equivalence(out/'semantics'),flush=True);print(approximation_study(out/'semantics'),flush=True)
