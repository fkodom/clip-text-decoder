from __future__ import annotations

import os
import tempfile
from typing import Callable, List, Optional, Tuple, Union

import gdown
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import LightningModule
from torch import Tensor, nn, optim
from transformers import GPT2Tokenizer

from clip_text_decoder.common import (
    check_language_model,
    check_vision_backbone,
    encode_image_tensor,
    load_language_model,
    load_vision_backbone,
)

PRETRAINED_INFERENCE_MODEL_PATH = (
    "https://drive.google.com/uc?id=1bEAyV2279C4V4iYMaJahREiM58vjy6G1"
    # https://drive.google.com/file/d/1bEAyV2279C4V4iYMaJahREiM58vjy6G1/view?usp=sharing
)


class Decoder(LightningModule):
    def __init__(
        self,
        vision_backbone: str = "blip:base",
        language_model: str = "distilgpt2",
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.save_hyperparameters()
        check_vision_backbone(vision_backbone)
        self.vision_backbone = vision_backbone
        check_language_model(language_model)
        self.language_model = load_language_model(language_model, device=device)

        self.to(device)

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

        return self.language_model(
            input_ids=input_ids,
            encoder_hidden_states=hidden,
            attention_mask=attention_mask,
            labels=labels,
        )

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-4)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], *_) -> Tensor:
        encoder_hidden_states, input_ids, attention_mask = batch
        result = self.forward(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        self.log(
            "training_loss",
            result.loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
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

        self.log(
            "validation_loss",
            result.loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return result.loss


class DecoderInferenceModel:
    _model_path = "model.pt"
    _tokenizer_path = "tokenizer.pkl"

    def __init__(
        self,
        model: Decoder,
        tokenizer: GPT2Tokenizer,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device: torch.device) -> DecoderInferenceModel:
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
    def load(cls, path: str) -> DecoderInferenceModel:
        temp = torch.load(path)
        # Just in case we change any of the class methods here, unpack the model
        # and tokenizer, and pass them into a new instance of this class.
        return cls(model=temp.model.float(), tokenizer=temp.tokenizer)

    @classmethod
    def download_pretrained(cls, dest: str = None) -> DecoderInferenceModel:
        with tempfile.TemporaryDirectory() as tempdir:
            if dest is None:
                dest = os.path.join(tempdir, "model.zip")
            gdown.download(PRETRAINED_INFERENCE_MODEL_PATH, dest, quiet=False)
            return cls.load(dest)

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def __call__(self, x: Tensor, max_len: int = 64, beam_size: int = 1) -> str:
        """Inference using beam search. For beam search, we predict one token per step.
        After each step, we keep only the 'beam_size' output sequences with the highest
        end-to-end confidence score. Repeat this process until at most 'max_len' tokens
        have been generated.
        """
        encoder_hidden_states = x.reshape(1, 1, -1).to(self.device)
        # Since we haven't performed any beam search steps yet, we just have one
        # set of input IDs (with a single "start" token). We use 'None' for the log
        # probability of this sequence, since it's not being predicted by the model.
        input_ids = [torch.tensor([self.tokenizer.bos_token_id], device=self.device)]
        beam_logprobs: Optional[List[float]] = None

        def _get_beam_outputs(_input_ids: Tensor) -> Tuple[List[Tensor], Tensor]:
            """Performs inference on the 'input_ids' Tensor, and collects the top
            'beam_size' results by score. Returns a list of output Tensors, and
            their respective log-probabilities.
            """
            outputs = self.model(_input_ids.unsqueeze(0), encoder_hidden_states)
            logits: Tensor = outputs.logits[0, -1]
            logprobs = F.log_softmax(logits, dim=-1)

            topk_logprobs = logprobs.topk(k=beam_size)
            indices = topk_logprobs.indices
            logprobs = topk_logprobs.values
            output_ids = [
                torch.cat([_input_ids, idx.reshape(-1)], dim=0) for idx in indices
            ]

            return output_ids, logprobs

        for _ in range(max_len - 1):
            output_ids: List[Tensor] = []
            logprobs: List[float] = []
            beams_done: List[bool] = []

            # Collect the top 'beam_size' results from each beam individually.
            for beam_idx, ids in enumerate(input_ids):
                # If 'beam_logprobs' is already defined, then we've predicted at least
                # one token already. And if the last token is equal to the "stop" token,
                # we don't need to perform inference with this beam anymore.
                if beam_logprobs and ids[-1].item() == self.tokenizer.eos_token_id:
                    output_ids.append(ids)
                    logprobs.append(beam_logprobs[beam_idx])
                    beams_done.append(True)
                    continue

                _output_ids, _logprobs = _get_beam_outputs(ids)
                if beam_logprobs is not None:
                    # Sum the log-probabilities of the existing beam and our predicted
                    # token to get the total log-probability.
                    _logprobs += beam_logprobs[beam_idx]

                # Append the results from this beam to the aggregate lists.
                output_ids += _output_ids
                logprobs += _logprobs.tolist()
                beams_done.append(False)

            if all(beams_done):
                # All search beams are done generating text.
                break

            # Keep only the top 'beam_size' beams by total log-probability.
            indices = torch.tensor(logprobs).topk(k=beam_size).indices
            input_ids = [output_ids[idx] for idx in indices]
            beam_logprobs = [logprobs[idx] for idx in indices]

        # Find the predicted beam with highest overall log-probability.
        best_beam_idx: int = torch.tensor(beam_logprobs).argmax().item()  # type: ignore
        # Decode the predicted token IDs into a text string.
        return self.tokenizer.decode(input_ids[best_beam_idx], skip_special_tokens=True)


class ImageCaptionInferenceModel(DecoderInferenceModel):
    def __init__(self, model: Decoder, tokenizer: GPT2Tokenizer):
        super().__init__(model, tokenizer)
        self._vision_backbone: Optional[nn.Module] = None
        self._preprocessor: Optional[Callable] = None

    def _load_vision_backbone(self):
        backbone, preprocessor = load_vision_backbone(self.model.vision_backbone)
        self._vision_backbone = backbone
        self._preprocessor = preprocessor

    @property
    def vision_backbone(self) -> nn.Module:
        if self._vision_backbone is None:
            self._load_vision_backbone()
        assert self._vision_backbone is not None
        return self._vision_backbone

    @property
    def preprocessor(self) -> Callable:
        if self._preprocessor is None:
            self._load_vision_backbone()
        assert self._preprocessor is not None
        return self._preprocessor

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, Image.Image],
        max_len: int = 64,
        beam_size: int = 1,
    ) -> str:
        if isinstance(image, str):
            image = Image.open(image)

        preprocessed: Tensor = self.preprocessor(image).to(self.device)
        encoded = encode_image_tensor(preprocessed.unsqueeze(0), self.vision_backbone)
        return super().__call__(encoded, max_len=max_len, beam_size=beam_size)
