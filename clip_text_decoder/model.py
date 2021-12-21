from __future__ import annotations
import os
import tempfile
from typing import Optional, Tuple

import gdown
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
from torch import Tensor, optim
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from clip_text_decoder.tokenizer import SPECIALS_STOI

PADDING_VALUE = SPECIALS_STOI["PAD"]
PRETRAINED_INFERENCE_MODEL_PATH = (
    "https://drive.google.com/uc?id=1bYAog3ZFLiBZEPRLqBcy8J-gXp7NTPAY"
    # https://drive.google.com/file/d/1bYAog3ZFLiBZEPRLqBcy8J-gXp7NTPAY/view?usp=sharing
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
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> ClipDecoderInferenceModel:
        temp = torch.load(path)
        # Just in case we change any of the class methods here, unpack the model
        # and tokenizer, and pass them into a new instance of this class.
        return cls(model=temp.model, tokenizer=temp.tokenizer)

    @classmethod
    def download_pretrained(cls, dest: str = None) -> ClipDecoderInferenceModel:
        with tempfile.TemporaryDirectory() as tempdir:
            if dest is None:
                dest = os.path.join(tempdir, "model.zip")
            gdown.download(PRETRAINED_INFERENCE_MODEL_PATH, dest)
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
