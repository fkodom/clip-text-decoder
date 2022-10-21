from __future__ import annotations

from enum import Enum
from typing import Callable, Optional, Tuple, Union

import clip
import torch
from clip.model import CLIP
from lavis.models import BlipFeatureExtractor, load_model_and_preprocess
from PIL import Image
from torch import Tensor, nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

PreprocessorType = Callable[[Image.Image], Tensor]


class LanguageModels(Enum):
    distilgpt2: str = "distilgpt2"
    gpt2: str = "gpt2"
    gpt2_medium: str = "gpt2-medium"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def check_language_model(name: str) -> None:
    allowed = LanguageModels.list()
    if name not in allowed:
        raise ValueError(f"Unsupported language model '{name}'. Allowed: {allowed}.")


def load_language_model(
    name: str, device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    check_language_model(name)
    config = GPT2Config.from_pretrained(name, add_cross_attention=True)
    model = GPT2LMHeadModel.from_pretrained(name, config=config)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    return model


def load_tokenizer(name: str) -> GPT2Tokenizer:
    check_language_model(name)
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class VisionBackbones(Enum):
    blip_base: str = "blip:base"
    clip_rn50: str = "clip:RN50"
    clip_rn101: str = "clip:RN101"
    clip_vit_b32: str = "clip:VIT-B/32"
    clip_vit_b16: str = "clip:VIT-B/16"
    clip_vit_l14: str = "clip:VIT-L/14"
    clip_vit_l14_336px: str = "clip:VIT-L/14@336px"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def check_vision_backbone(backbone: str) -> None:
    allowed = VisionBackbones.list()
    if backbone not in allowed:
        raise ValueError(f"Unsupported backbone '{backbone}'. Allowed: {allowed}.")


def load_vision_backbone(
    backbone: str, device: Optional[Union[str, torch.device]] = None
) -> Tuple[nn.Module, PreprocessorType]:
    check_vision_backbone(backbone)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if backbone == VisionBackbones.blip_base.value:
        model, preprocessors, _ = load_model_and_preprocess(
            "blip_feature_extractor", model_type="base", is_eval=True, device=device
        )
        return model, preprocessors["eval"]
    else:
        # Currently, all other supported backbones are CLIP
        _, name = backbone.split(":")
        return clip.load(name, device=device, jit=False)


def encode_image_tensor(image: Tensor, backbone: nn.Module) -> Tensor:
    if isinstance(backbone, BlipFeatureExtractor):
        features = backbone.extract_features({"image": image}, mode="image")
        return features.image_embeds[:, 0]
    else:
        # Currently, all other supported backbones are CLIP
        assert isinstance(backbone, CLIP)
        return backbone.encode_image(image)
