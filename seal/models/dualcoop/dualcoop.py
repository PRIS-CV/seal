"""Acknowledgement: This code is modified from DualCoop, a multi-label classification model, 
we modify DualCoop for attribute recognition. 
Paper: https://arxiv.org/abs/2206.09541
Code: https://github.com/sunxm2357/DualCoOp
"""

import torch
import torch.nn as nn
import logging

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F

_tokenizer = _Tokenizer()


from .. import almodel
from ..model import ALModel
from .dualcoop_clip import build_model_conv_proj
from ..loss import build_loss
from ...dataset.utils import load_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_clip_to_cpu(backbone_name, input_size, dir_cache, feature_aggregation):
    
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root=dir_cache)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if feature_aggregation: 
        model = build_model_conv_proj(state_dict or model.state_dict(), input_size)
    else:
        model, _ = clip.load(backbone_name, device="cpu")
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MLCPromptLearner(nn.Module):
    def __init__(
        self,
        n_ctx_pos: int, n_ctx_neg: int, 
        classnames: list, clip_model, csc: bool,
        ctx_init_pos=None, ctx_init_neg=None, 
    ):
        super().__init__()
        n_cls = len(classnames)
        # ctx_init_pos = cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT.strip()
        # ctx_init_neg = cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT.strip()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = clip.tokenize(ctx_init_pos)
            prompt_neg = clip.tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if csc:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if csc:
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(n_cls, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_cls, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)

        print(f'Initial positive context: "{prompt_prefix_pos}"')
        print(f'Initial negative  context: "{prompt_prefix_neg}"')
        print(f"Number of positive context words (tokens): {n_ctx_pos}")
        print(f"Number of negative context words (tokens): {n_ctx_neg}")

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + name + "." for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        for p_pos, p_neg in zip(prompts_pos, prompts_neg):
            tokenized_prompts_pos.append(clip.tokenize(p_pos))
            tokenized_prompts_neg.append(clip.tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx_neg:, :])

        self.n_cls = n_cls
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens

    def forward(self, cls_id=None):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg

        if ctx_pos.dim() == 2:
            if cls_id is None:
                ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_pos = ctx_pos.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_pos = ctx_pos[cls_id]

        if ctx_neg.dim() == 2:
            if cls_id is None:
                ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_neg = ctx_neg.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_neg = ctx_neg[cls_id]

        if cls_id is None:
            prefix_pos = self.token_prefix_pos
            prefix_neg = self.token_prefix_neg
            suffix_pos = self.token_suffix_pos
            suffix_neg = self.token_suffix_neg
        else:
            prefix_pos = self.token_prefix_pos[cls_id]
            prefix_neg = self.token_prefix_neg[cls_id]
            suffix_pos = self.token_suffix_pos[cls_id]
            suffix_neg = self.token_suffix_neg[cls_id]


        prompts_pos = torch.cat(
            [
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts = torch.cat([prompts_neg, prompts_pos], dim=0)

        if cls_id is not None:
            tokenized_prompts_pos = self.tokenized_prompts[self.n_cls:][cls_id]
            tokenized_prompts_neg = self.tokenized_prompts[:self.n_cls][cls_id]
            tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)
        else:
            tokenized_prompts = self.tokenized_prompts


        return prompts, tokenized_prompts


@almodel("DualCoOp")
class DualCoOp(ALModel):
    def __init__(
        self, 
        input_size: int, backbone: dict, loss: dict, mlc_prompt_learner: dict, f_attribute_index: list, logit_scale: float, 
        finetune_backbone: bool, finetune_attn: bool, finetune_text_encoder: bool, 
        optimizer: dict, dir_cache: str, fp16: bool = True, feature_aggregation: bool = False
    ):
        super().__init__()

        classnames = list(load_json(f_attribute_index).keys())

        self.loss_fn = build_loss(loss['name'])(**loss)

        # self.clip_model = load_clip_to_cpu(backbone["name"], input_size, dir_cache).float()

        self.feature_aggregation = feature_aggregation

        clip_model = load_clip_to_cpu(backbone["name"], input_size, dir_cache, feature_aggregation)
        
        if not fp16:
            clip_model.float()

        self.optim_set = optimizer

        self.visual_encoder_type = backbone["name"]

        self.dtype = clip_model.dtype
        
        logger.info(f"Using dtype: {self.dtype}")

        self.prompt_learner = MLCPromptLearner(**mlc_prompt_learner, classnames=classnames, clip_model=clip_model)
        
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = logit_scale
        
        self.start_device = 1
        self.num_devices = torch.cuda.device_count() - self.start_device
        self.num_prompts = len(classnames) * 2
        
        self.devices = [torch.device(f'cuda:{i + self.start_device}') for i in range(self.num_devices)]

        if not finetune_backbone:
            logger.info('Freeze the backbone weights')
            backbone_params = self.backbone_params()
            for param in backbone_params:
                param.requires_grad_(False)

        if not finetune_attn:
            logger.info('Freeze the attn weights')
            attn_params = self.attn_params()
            for param in attn_params:
                param.requires_grad_(False)

        if not finetune_text_encoder:
            logger.info('Freeze the text encoder weights')
            text_encoder_params = self.text_encoder_params()
            for param in text_encoder_params:
                param.requires_grad_(False)

    def to(self, device):
        model = super().to(device)
        model.text_encoder = model.text_encoder.to(torch.device("cuda"))
        model.text_encoder = nn.DataParallel(model.text_encoder)
        return model

    
    def distributed_prompts(self, prompts, tokenized_prompts):
        dist_prompts, dist_tokenized_prompts = [], []
        num_prompt_per_device = prompts.shape[0] // self.num_devices
        num_rest_prompt = prompts.shape[0] - self.num_devices * num_prompt_per_device
        group_bin = [[i * num_prompt_per_device, (i + 1) * num_prompt_per_device] for i in range(self.num_devices)]
        group_bin[-1][-1] += num_rest_prompt
        
        for i, gb in enumerate(group_bin):
            dp = prompts[gb[0]: gb[1]]
            dtp = tokenized_prompts[gb[0]: gb[1]]
            dist_prompts.append(dp)
            dist_tokenized_prompts.append(dtp)     
        
        return dist_prompts, dist_tokenized_prompts       # 7000 MB
    
    def distributed_extract_text_features(self, dist_prompts, dist_tokenized_prompts):
        encoder_device = torch.device('cuda:0')
        distributed_text_features = []
        
        for i, (dp, dtp) in enumerate(zip(dist_prompts, dist_tokenized_prompts)):

            dtf = self.text_encoder(dp.to(self.devices[i]), dtp.to(self.devices[i]))
            dtf = dtf / dtf.norm(dim=-1, keepdim=True)

            distributed_text_features.append(dtf)

            dp = dp.to(encoder_device)
            dtp = dtp.to(encoder_device)

        return distributed_text_features

    def distributed_similarity_computation(self, image_features, distributed_text_features):
        imf_device = image_features
        # from IPython import embed
        # embed()
        # exit()
        outputs = []
        for dtf in distributed_text_features:
            dtf_device = dtf.device
            image_features = image_features.to(dtf_device)
            
            if self.feature_aggregation:
                output = 20 * F.conv1d(image_features, dtf[:, :, None]).to(imf_device)
            else:
                output = image_features @ dtf.T
            
            outputs.append(output)
        
        return torch.cat(outputs, dim=1)


    def infer(self, data, cls_id=None):
        # get image and text features
        image = data['i']
        
        if self.feature_aggregation:
            image_features, attn_weights = self.image_encoder(image.type(self.dtype))
        else:
            image_features = self.image_encoder(image.type(self.dtype))
        
        prompts, tokenized_prompts = self.prompt_learner(cls_id)

        dist_prompts, dist_tokenized_prompts = self.distributed_prompts(prompts, tokenized_prompts)

        distributed_text_features = self.distributed_extract_text_features(dist_prompts, dist_tokenized_prompts)

        # text_features = self.text_encoder(prompts, tokenized_prompts)
        # normalize features
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)

        # Class-Specific Region Feature Aggregation
        # output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])

        output = self.distributed_similarity_computation(image_features_norm, distributed_text_features)

        b, c = output.shape[0], output.shape[1]
        # output_half = output[:,  c // 2:]
        # w_half = F.softmax(output_half, dim=-1)
        # w = torch.cat([w_half, w_half], dim=1)
        # output = 5 * (output * w).sum(-1)

        # convert the shape of logits to [b, 2, num_class]
        logits = output.resize(b, 2, c//2)

        if self.training:
            return logits
    
        return logits[:, 0, :].squeeze(1)

    def text_encoder_params(self):
        params = []
        for name, param in self.named_parameters():
            if "text_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
        return params

    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)
        return params
    
    def get_params(self):
        names, params = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:
                params.append(p)
                names.append(n)
        logger.info(f"Training parameters: {names}.")
        return params

    def get_optimizer(self):
        params = self.get_params()
        return torch.optim.Adam(params=params, lr=self.optim_set["lr"], weight_decay=0)
        # return torch.optim.SGD(params=params, lr=self.optim_set["lr"], weight_decay=0)
    
    def compute_loss(self, pred, target):
        loss = self.loss_fn(pred, target)
        loss_dict = {self.loss_fn.name: loss.item()}
        return loss, loss_dict
    
    def forward(self, data):
        if self.training:
            return self.train_model(data)
        else:
            return self.infer(data)

    def train_model(self, data):
        target = data['t']
        pred = self.infer(data).float()
        return self.compute_loss(pred, target)
    