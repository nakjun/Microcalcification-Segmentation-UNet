"""Official-release decoder connected to image-only detector proposals.

This implements the published repository's SAM/HQ decoder, not a claim that
unreleased paper-specific operations have been reconstructed.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from upstream import activate, SOURCE
activate()
from segment_anything_training import sam_model_registry
from decoder import MaskDecoderHQ


def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_state(path, stage=None, sam_checkpoint=None, model_type=None):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        metadata=state.get('metadata')
        if metadata is not None and stage is not None:
            settings=metadata.get('settings',{})
            if settings.get('stage') != stage:
                raise ValueError('Checkpoint training stage mismatch')
            if stage == 'segmentor':
                if settings.get('model_type') != model_type:
                    raise ValueError('Checkpoint SAM model type mismatch')
                if not sam_checkpoint or metadata.get('sam_sha256') != file_sha256(sam_checkpoint):
                    raise ValueError('Decoder checkpoint requires its original SAM weights')
        state = state['state_dict']
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a tensor state_dict checkpoint')
    return state


class Segmentor(nn.Module):
    def __init__(self, model_type='vit_b', sam_checkpoint=None, decoder_checkpoint=None):
        super().__init__()
        if model_type not in ('vit_b', 'vit_l', 'vit_h'):
            raise ValueError('Expected vit_b, vit_l or vit_h')
        self.model_type = model_type
        self.sam = sam_model_registry[model_type](checkpoint=None)
        if sam_checkpoint:
            self.sam.load_state_dict(read_state(sam_checkpoint), strict=True)
        self.decoder = MaskDecoderHQ(model_type, self.sam.mask_decoder.state_dict())
        if decoder_checkpoint:
            self.decoder.load_state_dict(read_state(decoder_checkpoint, stage="segmentor", sam_checkpoint=sam_checkpoint, model_type=model_type), strict=True)
        self.sam.requires_grad_(False)
        self.sam.eval()

    def train(self, mode=True):
        super().train(mode)
        self.sam.eval()
        return self

    @torch.no_grad()
    def encode(self, image):
        """Cache one image embedding for all detector proposals."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected uint8 RGB HWC image')
        device = next(self.parameters()).device
        h,w = image.shape[:2]
        size = self.sam.image_encoder.img_size
        scale = size/max(h,w)
        nh,nw = max(1,int(h*scale+0.5)),max(1,int(w*scale+0.5))
        x=torch.as_tensor(image.copy(),device=device).permute(2,0,1)[None].float()
        x=F.interpolate(x,size=(nh,nw),mode='bilinear',align_corners=False,antialias=True)[0]
        embedding,intermediate=self.sam.image_encoder(self.sam.preprocess(x)[None])
        return embedding,intermediate,(nh,nw),(h,w)

    def decode(self, encoded, boxes):
        embedding,intermediate,(nh,nw),(h,w)=encoded
        boxes=torch.as_tensor(boxes,dtype=torch.float32,device=embedding.device).reshape(-1,4)
        if not len(boxes):return torch.empty(0,1,h,w,device=embedding.device)
        if not torch.isfinite(boxes).all() or (boxes[:,2:]<=boxes[:,:2]).any():
            raise ValueError('Invalid boxes')
        if (boxes<0).any() or (boxes[:,[0,2]]>w).any() or (boxes[:,[1,3]]>h).any():
            raise ValueError('Boxes outside native image')
        transformed=boxes*boxes.new_tensor([nw/w,nh/h,nw/w,nh/h])
        with torch.no_grad():
            sparse,dense=self.sam.prompt_encoder(points=None,boxes=transformed,masks=None)
        logits=self.decoder(image_embeddings=embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe()[None],
            sparse_prompt_embeddings=[sparse],dense_prompt_embeddings=[dense],
            multimask_output=False,hq_token_only=True,interm_embeddings=intermediate)
        return self.sam.postprocess_masks(logits,(nh,nw),(h,w))

    def logits(self, image, boxes):
        """RGB uint8 HWC image and native pixel xyxy prompts; no masks."""
        return self.decode(self.encode(image),boxes)


def build_detector(config=None):
    from models.yolo import Model
    config=str(config or SOURCE/'cfg/training/CBAM3.yaml')
    return Model(config,ch=3,nc=1)


class Detector:
    def __init__(self, checkpoint, device='cpu', size=640, confidence=.25, iou=.45, max_detections=100):
        if size < 32 or size % 32 or not 0 <= confidence <= 1 or not 0 <= iou <= 1 or max_detections < 1:
            raise ValueError('Invalid detector settings')
        self.model=build_detector().to(device).eval()
        self.model.load_state_dict(read_state(checkpoint, stage="detector"),strict=True)
        self.device=device; self.size=size; self.confidence=confidence
        self.iou=iou; self.max_detections=max_detections

    @torch.inference_mode()
    def __call__(self,image):
        from torchvision.ops import nms
        from utils.datasets import letterbox
        h,w=image.shape[:2]
        resized,ratio,pad=letterbox(image,new_shape=self.size,auto=False)
        x=torch.from_numpy(np.ascontiguousarray(resized.transpose(2,0,1))).to(self.device).float()[None]/255
        pred=self.model(x)[0][0]
        scores=pred[:,4]*pred[:,5]
        keep=scores>=self.confidence
        pred,scores=pred[keep],scores[keep]
        boxes=torch.cat((pred[:,:2]-pred[:,2:4]/2,pred[:,:2]+pred[:,2:4]/2),1)
        idx=nms(boxes,scores,self.iou)[:self.max_detections]
        boxes,scores=boxes[idx],scores[idx]
        boxes-=boxes.new_tensor([pad[0],pad[1],pad[0],pad[1]])
        boxes/=boxes.new_tensor([ratio[0],ratio[1],ratio[0],ratio[1]])
        boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,w)
        boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,h)
        valid=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
        return boxes[valid].cpu().numpy(),scores[valid].cpu().numpy()


def predict(image, detector, segmentor):
    boxes,scores=detector(image)
    probability=np.zeros(image.shape[:2],dtype=np.float32)
    with torch.inference_mode():
        # Bound full-resolution mask memory independently of detection count.
        encoded=segmentor.encode(image) if len(boxes) and hasattr(segmentor,'encode') else None
        for box in boxes:
            logits=segmentor.decode(encoded,box[None]) if encoded is not None else segmentor.logits(image,box[None])
            mask=logits.sigmoid()[0,0].cpu().numpy()
            np.maximum(probability,mask,out=probability)
    return probability,boxes,scores
