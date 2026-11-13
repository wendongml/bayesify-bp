"""Run the included original transformer sources in their own pinned environment.

Only public pretrained models/data are downloaded. Outputs stay under python/.
No private bucket, project path, cloud backup or API credential is required.
"""
from pathlib import Path
import argparse,subprocess,sys,tarfile,urllib.request
ROOT=Path(__file__).resolve().parent
COMMIT='7d960082efa3a680b76aac79d2a4f13d7f0f651a'
PACKAGES=['torch==2.4.1','transformers==4.45.2','peft==0.13.0','numpy==2.1.2',
          'scipy==1.14.1','datasets==3.0.1','accelerate==1.0.0','torchmetrics==1.4.2',
          'bitsandbytes==0.44.1','ipdb==0.13.13','tqdm','scikit-learn==1.5.2']

def environment():
    import shutil
    if not shutil.which('uv'):subprocess.run([sys.executable,'-m','pip','install','uv'],check=True)
    env=ROOT/'.training-env';python=env/'bin/python'
    if not python.exists():
        subprocess.run(['uv','venv','--python','3.12',str(env)],check=True)
    subprocess.run(['uv','pip','install','--python',str(python),*PACKAGES],check=True)
    official=ROOT/'blob_mirror/official'
    if not official.exists():
        archive=ROOT/'blob_source.tar.gz'
        urllib.request.urlretrieve(f'https://codeload.github.com/Wang-ML-Lab/bayesian-peft/tar.gz/{COMMIT}',archive)
        with tarfile.open(archive) as tf:
            top=tf.getnames()[0].split('/')[0]
            for member in tf.getmembers():
                name=Path(member.name)
                if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                    raise ValueError('Unexpected upstream archive path')
            tf.extractall(ROOT/'blob_mirror',filter='data')
        (ROOT/'blob_mirror'/top).rename(official)
        archive.unlink()
    return str(python)

def run(task):
    python=environment()
    if task=='gpt2_selected':
        for method in ['adamw','sampled_mfvi','moment_ep']:
            subprocess.run([python,'run_selected.py','--method',method],cwd=ROOT/'gpt2',check=True)
    else:
        for method in ['ordinary','hybrid']:
            subprocess.run([python,'train_fixed_transfer.py','--task',task,'--method',method],cwd=ROOT/'blob_mirror',check=True)
        subprocess.run([python,'predict.py','--task',task],cwd=ROOT/'blob_mirror',check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['gpt2_selected','sst2','clinc','setup']);a=p.parse_args()
    environment() if a.task=='setup' else run(a.task)
