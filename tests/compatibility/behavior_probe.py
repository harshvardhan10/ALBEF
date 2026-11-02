import importlib,json,random,tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from dataset import create_dataset
from models.vit import VisionTransformer
from models.xbert import BertConfig,BertForMaskedLM

def capture():
    torch.set_num_threads(2)
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    out={}
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);Image.new('RGB',(48,48),(100,130,150)).save(root/'image.png')
        (root/'data.json').write_text(json.dumps([{'image':str(root/'image.png'),'caption':'A test image.'}]))
        ds=create_dataset('pretrain',{'train_file':[str(root/'data.json')],'image_res':32})
        image,caption=ds[0];out['caption']=caption;out['image']=image.flatten().tolist()
        vision=VisionTransformer(img_size=32,patch_size=16,embed_dim=32,depth=2,num_heads=4)
        features=vision(image.unsqueeze(0));config=BertConfig(vocab_size=32,hidden_size=32,num_hidden_layers=2,num_attention_heads=4,intermediate_size=64)
        config.fusion_layer=1;config.encoder_width=32
        bert=BertForMaskedLM(config);ids=torch.tensor([[2,5,6,3]])
        result=bert(input_ids=ids,attention_mask=torch.ones_like(ids),encoder_hidden_states=features,encoder_attention_mask=torch.ones(1,5),labels=ids,return_dict=True)
        result.loss.backward();out['features']=features.detach().flatten().tolist();out['loss']=result.loss.item();out['vision_gradient']=vision.patch_embed.proj.weight.grad.flatten().tolist();out['vision_state_keys']=list(vision.state_dict());out['bert_state_keys']=list(bert.state_dict())
    cases=[('mean','SimpleMeanFusion',{}),('gated','ImageLevelGatedFusion',{'hidden_dim':32,'gate_hidden_dim':16}),('residual','OriginalAnchoredResidualFusion',{'hidden_dim':32,'gate_hidden_dim':16}),('transformer','ViewTransformerFusion',{'hidden_dim':32,'num_heads':4}),('cross_attention','ViewCrossAttentionFusion',{'hidden_dim':32,'num_heads':4})]
    for family,name,kwargs in cases:
        torch.manual_seed(42);module=importlib.import_module('models.model_pretrain_multiview_'+family+'_fusion');model=getattr(module,name)(**kwargs)
        inputs=[torch.randn(2,5,32,requires_grad=True) for _ in range(3)]
        output=model(*inputs);output.square().mean().backward()
        out[family]={'output':output.detach().flatten().tolist(),'input_gradients':[v.grad.flatten().tolist() for v in inputs],'state_keys':list(model.state_dict())}
    return out

def summarize(value):
    if isinstance(value, dict):
        return {k: summarize(v) for k, v in value.items()}
    if isinstance(value, list):
        if value and isinstance(value[0], (int, float)):
            array = np.asarray(value, dtype=np.float64)
            return {'size': int(array.size), 'first': array[:16].tolist(),
                    'sum': float(array.sum()), 'norm': float(np.linalg.norm(array))}
        return [summarize(v) for v in value]
    return value
