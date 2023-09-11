from torch import Tensor
from clip import clip


from .. import almodel
from ..model import ALModel


@almodel("ALCLIP")
class ALCLIP(ALModel):
    
    def __init__(self, backbone) -> None:
        super().__init__()

        _support_backbones = list(clip._MODELS.keys())

        if backbone['name'] not in _support_backbones:
            raise ValueError(f"CLIP only support following backbones: {_support_backbones}.")

        self.clip_model, self.vis_preprocess = clip.load(backbone['name'])

    def encode_text(self, text_inputs: Tensor, norm=True) -> Tensor:
        device = next(self.clip_model.parameters()).device
        tokenized_text_inputs = clip.tokenize(text_inputs).to(device)
        text_features = self.clip_model.encode_text(tokenized_text_inputs)
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def encode_image(self, image_inputs: Tensor, norm=True) -> Tensor:
        image_features = self.clip_model.encode_image(image_inputs)
        if norm:
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features


    



