from __future__ import annotations
import io
import math
import os
import pickle5 as pickle
import tempfile
from typing import Optional, Tuple
from zipfile import ZipFile

import gdown
from pytorch_lightning import LightningModule
import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score

from clip_text_decoder.tokenizer import Tokenizer, SPECIALS_STOI

PADDING_VALUE = SPECIALS_STOI["PAD"]
PRETRAINED_INFERENCE_MODEL_PATH = (
    "https://drive.google.com/uc?id=1bYAog3ZFLiBZEPRLqBcy8J-gXp7NTPAY"
    # https://drive.google.com/file/d/1bYAog3ZFLiBZEPRLqBcy8J-gXp7NTPAY/view?usp=sharing
)


def positional_encoding(
    sequence_length: int,
    embedding_size: int,
    device: torch.device = None,
    batch_first: bool = False,
) -> Tensor:
    pos = torch.arange(sequence_length, dtype=torch.float, device=device)
    dim = torch.arange(embedding_size, dtype=torch.float, device=device)

    pos = pos.reshape(1, -1, 1) if batch_first else pos.reshape(-1, 1, 1)
    dim = dim.reshape(1, 1, -1)

    phase = pos * torch.exp(-dim * math.log(10000) / embedding_size)
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        dropout: float = 0.1,
        max_len: int = 1024,
        batch_first: bool = False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout)

        encodings = positional_encoding(
            max_len, embedding_size, batch_first=batch_first
        )
        self.register_buffer("encodings", encodings)

    def forward(self, token_embedding: Tensor):
        seq_len = token_embedding.size(0)
        encodings = self.encodings[:seq_len]
        return self.dropout(token_embedding + encodings)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * (self.embedding_size ** 0.5)


class ClipDecoder(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int = 6,
        embedding_size: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.BatchNorm1d(embedding_size)
        self.embedding = TokenEmbedding(vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=dropout)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embedding_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            num_layers=num_layers,
        )
        self.linear = nn.Linear(embedding_size, vocab_size)

        # Initialize parameters using the 'Xavier uniform' distribution.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        memory_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        embeddings = self.embedding(tgt)
        encodings = self.positional_encoding(embeddings)
        memory = self.norm(memory.permute(1, 2, 0)).permute(2, 0, 1)

        decoded = self.transformer(
            encodings,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.linear(decoded)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.98))

    def training_step(self, batch: Tuple[Tensor, Tensor], *_) -> Tensor:
        memory, tgt = batch
        tgt_input, tgt_out = tgt[:-1], tgt[1:]
        tgt_mask, tgt_padding_mask = create_tgt_masks(tgt_input)

        logits = self.forward(
            tgt_input,
            memory,
            tgt_mask,
            None,
            tgt_padding_mask,
            None,
        )
        loss = F.cross_entropy(
            input=logits.reshape(-1, logits.size(-1)),
            target=tgt_out.reshape(-1),
        )
        bleu = compute_bleu_score(logits.argmax(dim=-1), tgt_out)

        self.log("training_loss", loss, on_step=False, on_epoch=True)
        self.log("training_bleu", bleu, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Tuple[Tensor, Tensor], *_) -> Tensor:
        memory, tgt = batch
        tgt_input, tgt_out = tgt[:-1], tgt[1:]
        tgt_mask, tgt_padding_mask = create_tgt_masks(tgt_input)

        logits = self.forward(
            tgt_input,
            memory,
            tgt_mask,
            None,
            tgt_padding_mask,
            None,
        )
        loss = F.cross_entropy(
            input=logits.reshape(-1, logits.size(-1)),
            target=tgt_out.reshape(-1),
        )
        bleu = compute_bleu_score(logits.argmax(dim=-1), tgt_out)

        self.log("validation_loss", loss, on_step=False, on_epoch=True)
        self.log("validation_bleu", bleu, on_step=False, on_epoch=True)
        return loss


class ClipDecoderInferenceModel:
    _model_path = "model.pt"
    _tokenizer_path = "tokenizer.pkl"

    def __init__(
        self,
        model: ClipDecoder,
        tokenizer: Tokenizer,
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
        model = torch.jit.script(self.model)

        with tempfile.TemporaryDirectory() as tempdir:
            model_path = os.path.join(tempdir, self._model_path)
            tokenizer_path = os.path.join(tempdir, self._tokenizer_path)

            model.save(model_path)
            with open(tokenizer_path, "wb") as f:
                # Protocol < 5 for compatibility with lower Python versions (Colab)
                pickle.dump(self.tokenizer, f, protocol=4)

            with ZipFile(path, "w") as zipfile:
                zipfile.write(model_path, arcname=self._model_path)
                zipfile.write(tokenizer_path, arcname=self._tokenizer_path)

    @classmethod
    def load(cls, path: str) -> ClipDecoderInferenceModel:
        with ZipFile(path, "r") as f:
            model_buffer = io.BytesIO(f.read(cls._model_path))
            tokenizer_buffer = io.BytesIO(f.read(cls._tokenizer_path))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(
            model=torch.jit.load(model_buffer, map_location=device),
            tokenizer=pickle.load(tokenizer_buffer),
        )

    @classmethod
    def download_pretrained(cls, dest: str = None) -> ClipDecoderInferenceModel:
        with tempfile.TemporaryDirectory() as tempdir:
            if dest is None:
                dest = os.path.join(tempdir, "model.zip")
            gdown.download(PRETRAINED_INFERENCE_MODEL_PATH, dest)

            try:
                return cls.load(dest)
            except ModuleNotFoundError:
                import sys
                import clip_text_decoder

                # For backwards compatibility -- early releases and Colab notebooks
                # used the 'src' namespace, so 'pickle' fails to load the underlying
                # 'Tokenizer' object.  Register 'clip_text_decoder/' as 'src/' here.
                sys.modules["src"] = clip_text_decoder
                return cls.load(dest)

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def __call__(self, x: Tensor, max_len: int = 32) -> str:
        embedding_size = x.size(-1)
        memory = x.reshape(-1, 1, embedding_size).to(self.device)
        tgt = self.tokenizer.tokenize("BOS", device=self.device).reshape(-1, 1)

        for _ in range(max_len - 1):
            tgt_mask = get_subsequent_mask(tgt.size(0), device=self.device).bool()
            probs = self.model(tgt, memory, tgt_mask)[-1]
            pred = torch.argmax(probs, dim=-1)

            if pred.item() == self.tokenizer.stoi["EOS"]:
                break
            else:
                tgt = torch.cat([tgt, pred.reshape(1, 1)], dim=0)

        return self.tokenizer.untokenize(tgt[1:])


def get_subsequent_mask(size, device: torch.device = None):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.to(device)


def create_tgt_masks(tgt: Tensor) -> Tuple[Tensor, Tensor]:
    seq_len = tgt.shape[0]
    device = tgt.device

    tgt_mask = get_subsequent_mask(seq_len, device=device)
    tgt_padding_mask = (tgt == PADDING_VALUE).transpose(0, 1)

    return tgt_mask, tgt_padding_mask


def compute_bleu_score(predictions: Tensor, targets: Tensor) -> float:
    def _single_bleu_score(pred: Tensor, target: Tensor) -> float:
        candidate = [str(x) for x in pred[pred != PADDING_VALUE].tolist()]
        reference = [str(x) for x in target[target != PADDING_VALUE].tolist()]
        return bleu_score([candidate], [[reference]])

    return sum(
        _single_bleu_score(prediction, target)
        for prediction, target in zip(predictions.T, targets.T)
    ) / len(predictions)
