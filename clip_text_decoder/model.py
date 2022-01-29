from __future__ import annotations

import os
import tempfile
from typing import Callable, Optional, Tuple, Union

import clip
import gdown
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import LightningModule
from torch import Tensor, optim
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

PRETRAINED_INFERENCE_MODEL_PATH = (
    "https://drive.google.com/uc?id=1oXPhrXMqRO_Q1UFe4NAs_RvDXR1AGoL2"
    # https://drive.google.com/file/d/1oXPhrXMqRO_Q1UFe4NAs_RvDXR1AGoL2/view?usp=sharing
)


class ClipDecoder(LightningModule):
    def __init__(self, gpt2_type: str = "distilgpt2"):
        super().__init__()
        self.config = GPT2Config.from_pretrained(gpt2_type, add_cross_attention=True)
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_type, config=self.config)

    def forward(
        self,
        input_ids: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ):
        batch_size, _, num_features = encoder_hidden_states.shape
        # TODO: Check if we can get '768' (num_features) from the GPT2 model.
        hidden = torch.zeros(
            size=(batch_size, 1, 768),
            dtype=encoder_hidden_states.dtype,
            device=encoder_hidden_states.device,
        )
        hidden[:, :, :num_features] = encoder_hidden_states

        return self.gpt(
            input_ids=input_ids,
            encoder_hidden_states=hidden,
            attention_mask=attention_mask,
            labels=labels,
        )

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.98))

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], *_) -> Tensor:
        encoder_hidden_states, input_ids, attention_mask = batch
        result = self.forward(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        self.log("training_loss", result.loss, on_step=False, on_epoch=True)
        return result.loss

    @torch.no_grad()
    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], *_) -> Tensor:
        encoder_hidden_states, input_ids, attention_mask = batch
        result = self.forward(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        self.log("validation_loss", result.loss, on_step=False, on_epoch=True)
        return result.loss


class ClipDecoderInferenceModel:
    _model_path = "model.pt"
    _tokenizer_path = "tokenizer.pkl"

    def __init__(
        self,
        model: ClipDecoder,
        tokenizer: GPT2Tokenizer,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device: torch.device) -> ClipDecoderInferenceModel:
        self.model.to(device)
        return self

    def save(self, path: str):
        # Save a copy of the current model weights, and cast to FP16 for storage
        model_state_dict = self.model.state_dict()
        # Avoid saving any cached properties of this class or its subclasses :)
        obj = self.__class__(model=self.model.half(), tokenizer=self.tokenizer)
        torch.save(obj, path)
        # Restore the original model weights
        self.model.load_state_dict(model_state_dict)

    @classmethod
    def load(cls, path: str) -> ClipDecoderInferenceModel:
        temp = torch.load(path)
        # Just in case we change any of the class methods here, unpack the model
        # and tokenizer, and pass them into a new instance of this class.
        return cls(model=temp.model.float(), tokenizer=temp.tokenizer)

    @classmethod
    def download_pretrained(cls, dest: str = None) -> ClipDecoderInferenceModel:
        with tempfile.TemporaryDirectory() as tempdir:
            if dest is None:
                dest = os.path.join(tempdir, "model.zip")
            gdown.download(PRETRAINED_INFERENCE_MODEL_PATH, dest, quiet=False)
            return cls.load(dest)

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def __call__(
        self, x: Tensor, max_len: int = 64, temperature: float = 1e-8, topk: int = 1
    ) -> str:
        embedding_size = x.size(-1)
        encoder_hidden_states = x.reshape(1, -1, embedding_size).to(self.device)
        input_ids = torch.tensor(
            self.tokenizer.bos_token_id,
            device=self.device,
        ).reshape(1, -1)

        for _ in range(max_len - 1):
            outputs = self.model(input_ids, encoder_hidden_states)
            logits: Tensor = outputs.logits[0, -1]

            topk_logits = logits.topk(k=topk, dim=-1)
            probs = F.softmax(topk_logits.values / temperature, dim=-1)
            idx = torch.argmax(probs * torch.rand_like(probs))
            pred = topk_logits.indices[idx]
            if pred.item() == self.tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, pred.reshape(1, 1)], dim=1)

        return self.tokenizer.decode(input_ids.flatten(), skip_special_tokens=True)


class ImageCaptionInferenceModel(ClipDecoderInferenceModel):
    def __init__(self, model: ClipDecoder, tokenizer: GPT2Tokenizer):
        super().__init__(model, tokenizer)
        self._clip_model: Optional[torch.nn.Module] = None
        self._clip_preprocessor: Optional[Callable] = None

    def _load_clip(self):
        self._clip_model, self._clip_preprocessor = clip.load(
            "ViT-B/32", device=self.device, jit=False
        )

    @property
    def clip_model(self) -> torch.nn.Module:
        if self._clip_model is None:
            self._load_clip()
        assert self._clip_model is not None, "Could not load CLIP model."
        return self._clip_model

    @property
    def clip_preprocessor(self) -> Callable:
        if self._clip_preprocessor is None:
            self._load_clip()
        assert self._clip_preprocessor is not None, "Could not load CLIP model."
        return self._clip_preprocessor

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, Image.Image],
        max_len: int = 64,
        temperature: float = 1e-8,
        topk: int = 1,
    ) -> str:
        if isinstance(image, str):
            image = Image.open(image)

        preprocessed = self.clip_preprocessor(image).to(self.device)
        encoded = self.clip_model.encode_image(preprocessed.unsqueeze(0))
        return super().__call__(
            encoded, max_len=max_len, temperature=temperature, topk=topk
        )
