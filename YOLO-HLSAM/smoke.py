"""Synthetic CPU checks only: genuine upstream modules and pipeline contracts."""
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from model import build_detector, predict, Segmentor, read_state, file_sha256, Detector
from decoder import MaskDecoderHQ
from train import detector_sample
from data import read_manifest
from upstream import SOURCE

def main():
    torch.set_num_threads(4); torch.manual_seed(42)
    from utils.loss import ComputeLoss
    detector=build_detector().train()
    detector.hyp=yaml.safe_load((SOURCE/'data/hyp.scratch.p5.yaml').read_text());detector.gr=1.
    image=np.zeros((80,120,3),dtype=np.uint8)
    image[20:40,50:60]=255
    x,target=detector_sample(image,np.array([[50,20,60,40]],dtype=np.float32),128,'cpu')
    loss,_=ComputeLoss(detector)(detector(x),target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in detector.parameters() if p.grad is not None)
    detector.eval()
    with torch.no_grad(): pred=detector(x)[0]
    assert pred.ndim==3 and pred.shape[-1]==6 and torch.isfinite(pred).all()
    del detector,loss,x,pred
    decoder=MaskDecoderHQ('vit_b')
    # Full official decoder, spatially reduced synthetic embeddings for CPU.
    kwargs=dict(image_embeddings=torch.randn(1,256,8,8),image_pe=torch.randn(1,1,256,8,8),
                sparse_prompt_embeddings=[torch.randn(2,2,256)],
                dense_prompt_embeddings=[torch.randn(2,256,8,8)],
                multimask_output=False,hq_token_only=True,
                interm_embeddings=[torch.randn(1,8,8,768)])
    masks=decoder(**kwargs)
    assert masks.shape==(2,1,32,32) and torch.isfinite(masks).all()
    masks.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters() if p.requires_grad)
    class NoSegment:
        def logits(self,*args):raise AssertionError('No proposals must skip segmentation')
    prob,boxes,scores=predict(image,lambda x:(np.empty((0,4)),np.empty(0)),NoSegment())
    assert not prob.any() and prob.shape==(80,120)
    class FakeSegment:
        def logits(self,im,boxes):
            m=torch.full((1,1,*im.shape[:2]),-20.)
            x1,y1,x2,y2=boxes[0].astype(int)
            m[:,:,y1:y2,x1:x2]=20.
            return m
    prob,boxes,scores=predict(image,lambda x:(np.array([[50,20,60,40]]),np.array([.9])),FakeSegment())
    assert (prob>=.5).sum()==200 and prob[20,50]>.99 and prob[19,50]<.01
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        Image.fromarray(image).save(root/'im.png');Image.fromarray(image[:,:,0]).save(root/'mask.png')
        rows=[{'id':'a','patient_id':'p','split':'train','image':'im.png','mask':'mask.png'},
              {'id':'b','patient_id':'p','split':'test','image':'im.png','mask':'mask.png'}]
        (root/'manifest.json').write_text(json.dumps(rows))
        try:read_manifest(root/'manifest.json')
        except ValueError as exc:assert 'overlap' in str(exc)
        else:raise AssertionError('Patient leakage accepted')
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        (root/'sam_a').write_bytes(b'sam-a'); (root/'sam_b').write_bytes(b'sam-b')
        metadata={'settings':{'stage':'segmentor','model_type':'vit_b'},'sam_sha256':file_sha256(root/'sam_a')}
        torch.save({'state_dict':{'weight':torch.ones(1)},'metadata':metadata},root/'decoder.pt')
        read_state(root/'decoder.pt',stage='segmentor',sam_checkpoint=root/'sam_a',model_type='vit_b')
        try:read_state(root/'decoder.pt',stage='segmentor',sam_checkpoint=root/'sam_b',model_type='vit_b')
        except ValueError as exc:assert 'original SAM' in str(exc)
        else:raise AssertionError('Mismatched SAM accepted')
    del decoder,masks,kwargs
    segmentor=Segmentor().eval()
    with torch.no_grad():
        encoded=segmentor.encode(image)
        output=segmentor.decode(encoded,np.array([[50,20,60,40]],np.float32))
    assert output.shape==(1,1,80,120) and torch.isfinite(output).all()
    # Native geometry round-trip from detector letterbox (128 input, 80x120 image).
    from utils.datasets import letterbox
    _,ratio,pad=letterbox(image,new_shape=128,auto=False)
    native=torch.tensor([[50.,20.,60.,40.]])
    b=native*torch.tensor([ratio[0],ratio[1],ratio[0],ratio[1]])+torch.tensor([pad[0],pad[1],pad[0],pad[1]])
    proposal=torch.cat([(b[:,:2]+b[:,2:])/2,b[:,2:]-b[:,:2],torch.ones(1,2)],dim=1)[None]
    class FixedDetector(torch.nn.Module):
        def forward(self,x):return (proposal,)
    d=object.__new__(Detector);d.model=FixedDetector();d.device='cpu';d.size=128;d.confidence=.25;d.iou=.45;d.max_detections=10
    returned,scores=d(image)
    assert np.allclose(returned,native.numpy(),atol=1e-5) and np.allclose(scores,1)
    probability,returned,_=predict(image,d,segmentor)
    assert probability.shape==(80,120) and np.isfinite(probability).all()
    result={'scope':'synthetic CPU smoke; not trained performance',
            'upstream_detector_forward_backward':True,'upstream_decoder_forward_backward':True,
            'empty_proposals_zero_mask':True,'predicted_boxes_geometry':True,'patient_split_leakage_rejected':True,
            'full_sam_encoder_and_decoder_random_weights':True, 'detector_letterbox_round_trip':True,
            'full_pipeline_fixed_detector_proposal':True, 'sam_checkpoint_mismatch_rejected':True,
            'trained_checkpoint_performance':'not evaluated', 'torch':str(torch.__version__)}
    (Path(__file__).parent/'smoke_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
