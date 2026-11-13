"""Inspect real transformer mean architectures without pretrained downloads.

Random weights validate shapes only. These models cannot produce the archived
benchmark predictions. GPT-2 uses the original QVLoRA class. RoBERTa uses the
same Transformers/PEFT constructor and Q/V target list as transfer_model.py;
the BLoB variance wrapper is absent from this *mean-architecture* inspection.
"""
from pathlib import Path
import importlib.util
import torch
from torch import nn
from .inspect import model_summary


def transformer_model(repository, task):
    from transformers import GPT2Config, GPT2Model, RobertaConfig, RobertaForSequenceClassification
    from peft import LoraConfig, get_peft_model

    if task == "sst2":
        path = Path(repository) / "python/gpt2/model.py"
        spec = importlib.util.spec_from_file_location("original_gpt2_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Assemble the exact class without its pretrained/tokenizer I/O.
        # IDs are shape-test rows only, not the trained verbalizer tokens.
        model = module.GPT2LoRA.__new__(module.GPT2LoRA)
        nn.Module.__init__(model)
        cfg=GPT2Config(n_layer=12,n_embd=768,n_head=12,use_cache=False)
        cfg._attn_implementation='eager'
        model.backbone = GPT2Model(cfg)
        for p in model.backbone.parameters(): p.requires_grad_(False)
        for block in model.backbone.h:
            block.attn.c_attn = module.QVLoRA(block.attn.c_attn, rank=4, alpha=32)
        model.register_buffer("label_ids", torch.tensor([1, 2]))
        inputs = {"ids": torch.tensor([[3, 4, 5, 6]]), "mask": torch.ones(1, 4, dtype=torch.long)}
    elif task == "clinc":
        cfg=RobertaConfig(num_labels=150);cfg._attn_implementation='eager'
        base = RobertaForSequenceClassification(cfg)
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=8, lora_dropout=0.,
                                               target_modules=["query", "value"]))
        for p in model.get_base_model().classifier.parameters(): p.requires_grad_(True)
        inputs = {"input_ids": torch.tensor([[0, 5, 6, 2]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    else:
        raise ValueError("task must be sst2 or clinc")
    model = model.float().eval()
    return model,inputs


def transformer_summary(repository, task):
    model,inputs=transformer_model(repository,task)
    with torch.no_grad():
        output = model(**inputs)
        logits = output if isinstance(output, torch.Tensor) else output.logits
        summary = model_summary(model, inputs)
    counts = dict(task=task, output_shape=list(logits.shape),
                  trainable_scalars=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  frozen_scalars=sum(p.numel() for p in model.parameters() if not p.requires_grad))
    return summary, counts


def export_transformer(repository,task,path):
    """Real mean architecture; graph weights are inputs, not bundled assets.

    ONNX Runtime receives the live module state to verify the exported graph.
    The downloadable graph contains no checkpoint or pretrained parameter data.
    """
    import inspect,json,numpy as np,onnx,onnxruntime as ort
    model,inputs=transformer_model(repository,task)
    class MeanModel(nn.Module):
        def __init__(self):super().__init__();self.model=model
        def forward(self,ids,mask):
            out=self.model(ids,mask) if task=='sst2' else self.model(input_ids=ids,attention_mask=mask)
            return out if isinstance(out,torch.Tensor) else out.logits
    wrapper=MeanModel().eval();example=tuple(inputs.values());path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    flags={'dynamo':False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}
    with torch.no_grad():
        expected=wrapper(*example).numpy()
        torch.onnx.export(wrapper,example,str(path),export_params=False,keep_initializers_as_inputs=True,
            do_constant_folding=False,opset_version=17,input_names=['token_ids','attention_mask'],output_names=['logits'],**flags)
    graph=onnx.load(path);onnx.checker.check_model(graph,full_check=True)
    assert not graph.graph.initializer,'The inspection graph must not contain parameter tensors'
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    state={n:v.detach().numpy() for n,v in wrapper.state_dict().items()}
    state.update(token_ids=example[0].numpy(),attention_mask=example[1].numpy())
    actual=session.run(None,{a.name:state[a.name] for a in session.get_inputs()})[0]
    np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-4)
    receipt={'task':task,'bytes':path.stat().st_size,'maximum_absolute_error':float(abs(actual-expected).max()),
             'parameter_storage':'explicit graph inputs; no weights included','random_weights_for_inspection':True}
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2));return receipt
