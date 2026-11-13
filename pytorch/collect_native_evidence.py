"""Package raw *PyTorch* predictions, never figures or model checkpoints.

One-time archive recovery; readers use the small packaged files. The source
path is an argument so no personal filesystem path enters the release.
"""
from pathlib import Path
import argparse, csv, hashlib, json, shutil
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("archive", type=Path, help="directory containing the original projects")
a = p.parse_args()
root = Path(__file__).resolve().parent
out = root / "evidence"
out.mkdir(exist_ok=True)
workflow = a.archive / "moment_vadam_mnist/uncertainty_workflow_20261027/executed_remote/results"
costs = {r["id"]: r for r in csv.DictReader((workflow / "optimization_and_costs.csv").open())}
labels = {"adam": "Ordinary Adam", "adamw": "Ordinary AdamW",
          "vadam_s1": "Sampled MFVI", "sampled_vadam": "Sampled MFVI",
          "sampled_ep_s4": "Sampled power-EP", "sampled_ep": "Sampled power-EP",
          "moment_k1": "Moment MFVI", "moment_vadam": "Moment MFVI",
          "moment_ep": "Moment power-EP"}
manifest = []
for meta_file in sorted((workflow / "predictions").glob("*__clean.json")):
    meta = json.loads(meta_file.read_text())
    family = meta["family"]
    if family not in ("mlp", "lenet", "resnet20", "densenet40"):
        continue
    if meta.get("prediction_repeat", 0) != 0 or meta["method"] not in labels:
        continue
    if family in ("resnet20", "densenet40") and meta.get("cohort") != "scratch":
        continue
    ident = meta["id"]
    if ident not in costs:
        raise ValueError(f"missing original costs: {ident}")
    source = meta_file.with_suffix(".npz")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == meta["sha256"]
    with np.load(source) as raw:
        data = {"probabilities": raw["probabilities"], "labels": raw["labels"]}
    ood = source.with_name(source.name.replace("__clean", "__ood"))
    if ood.exists():
        with np.load(ood) as raw:
            data["ood_probabilities"] = raw["probabilities"]
        paired_id = source.with_name(source.name.replace("__clean", "__noise0"))
        if not paired_id.exists():
            raise ValueError(f"missing predeclared OOD ID cohort: {paired_id}")
        with np.load(paired_id) as raw:
            data["ood_id_probabilities"] = raw["probabilities"]
    dest = out / f"{ident}.npz"
    np.savez_compressed(dest, **data)
    c = costs[ident]
    manifest.append(dict(file=dest.name, family=family, label=labels[meta["method"]],
        seed=meta["seed"], framework="PyTorch", source=source.relative_to(a.archive).as_posix(),
        source_sha256=meta["sha256"], sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        split=meta.get("evaluation_split"), n=len(data["labels"]),
        ood="Fashion-MNIST" if family in ("mlp", "lenet") else "SVHN",
        configuration=meta.get("configuration", json.loads(c["configuration"])),
        costs={k: c[k] for k in ("training_seconds", "selected_training_seconds", "inference_seconds",
                  "hardware", "epochs", "selected_epoch", "optimizer", "learning_rate",
                  "variance_learning_rate", "regularization", "inference_n")},
        training_source=meta.get("run", meta.get("checkpoint", "")),
        prediction_mode=meta.get("prediction_mode"), prediction_draws=meta.get("prediction_draws")))

# The selected transformer results were produced by the included original
# Python code. Keep the early EP/MFVI comparison separate from the BLoB fit.
tutorial = a.archive / "bayesian_backprop_advanced_tutorial"
source = tutorial / "advanced_results/gpt2_selected_campaign/original_predictions.npz"
with np.load(source) as raw:
    for key, label in [("adamw", "Ordinary AdamW"), ("moment_ep", "Moment power-EP"),
                       ("moment_mfvi_accuracy", "Moment MFVI (accuracy)"),
                       ("moment_mfvi_nll", "Moment MFVI (NLL)"),
                       ("sampled_mfvi", "Sampled MFVI")]:
        filename = f"sst2_{key}.npz"
        np.savez_compressed(out / filename, probabilities=raw[f"{key}_probabilities"], labels=raw["labels"])
        tables = root.parent / "results/gpt2_selected_campaign"
        cost = next(r for r in csv.DictReader((tables / "recorded_costs.csv").open()) if r["key"] == key)
        config = next(r for r in csv.DictReader((tables / "configurations.csv").open()) if r["key"] == key)
        manifest.append(dict(file=filename, family="sst2", label=label, seed=41, framework="PyTorch",
            source=source.relative_to(a.archive).as_posix(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            sha256=hashlib.sha256((out/filename).read_bytes()).hexdigest(), n=len(raw["labels"]),
            split="SST-2 official validation, selected checkpoint comparison", configuration=config,
            costs={**cost, "hardware": "A100", "inference_n": len(raw["labels"])},
            training_source="python/gpt2", prediction_draws=int(cost["network_draws"])))

v7 = tutorial / "analysis/blob_replication_20261102/revision_v7/evidence"
for task in ("sst2", "clinc"):
    for method, label in [("ordinary", "Ordinary AdamW"), ("hybrid", "Moment MFVI (BLoB)")]:
        source = v7 / f"{task}_final_graphs/{method}.npz"
        with np.load(source) as raw:
            # Preserve complete probabilities; no plots or rounded metrics.
            data = {k: raw[k] for k in raw.files}
        print(task, method, {k:v.shape for k,v in data.items()})
        prob = next(data[k] for k in ("p", "probabilities", "probability") if k in data)
        y = next(data[k] for k in ("y", "labels") if k in data)
        filename = f"compact_{task}_{method}.npz"
        np.savez_compressed(out/filename, probabilities=prob, labels=y)
        manifest.append(dict(file=filename, family=f"compact_{task}", label=label, seed=41,
            framework="PyTorch", source=source.relative_to(a.archive).as_posix(),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            sha256=hashlib.sha256((out/filename).read_bytes()).hexdigest(), n=len(y),
            split="fixed epoch-3" if task=="sst2" else "stopping-selected epoch-6; official ID test",
            training_source="python/blob_mirror/train_fixed_transfer.py"))

for folder in ("gpt2_selected_campaign", "compact_transformers"):
    dest = out / folder
    dest.mkdir(exist_ok=True)
    for path in (root.parent / "results" / folder).glob("*"):
        if path.suffix in (".csv", ".md"):
            shutil.copy2(path, dest/path.name)
shutil.copy2(workflow/"learning_curves.csv", out/"learning_curves.csv")
(out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(len(manifest), "native prediction records; bytes", sum(f.stat().st_size for f in out.rglob('*') if f.is_file()))
