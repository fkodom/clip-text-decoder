import os
import tempfile

import torch
import pytest

from clip_text_decoder.model import ClipDecoder, ClipDecoderInferenceModel
from clip_text_decoder.tokenizer import Tokenizer


DUMMY_TEXTS = [
    "This is a dummy sentence.",
    "This is also a dummy sentence.",
]


@pytest.mark.parametrize("vocab_size", [1, 100])
@pytest.mark.parametrize("num_layers", [3, 6])
@pytest.mark.parametrize("embedding_size", [512])
@pytest.mark.parametrize("nhead", [4, 8])
@pytest.mark.parametrize("dim_feedforward", [128, 256])
def test_model_forward(
    vocab_size: int,
    num_layers: int,
    embedding_size: int,
    nhead: int,
    dim_feedforward: int,
):
    model = ClipDecoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        embedding_size=embedding_size,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
    ).eval()

    SEQ_LEN, BATCH_SIZE = 16, 1
    tgt = torch.randint(0, vocab_size, size=(SEQ_LEN, 1))
    memory = torch.randn(1, 1, embedding_size)
    tgt_mask = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.bool)

    out = model.forward(tgt, memory, tgt_mask)
    seq_len, batch_size, out_size = out.shape
    assert seq_len == SEQ_LEN
    assert batch_size == BATCH_SIZE
    assert out_size == vocab_size


def test_model_script():
    model = ClipDecoder(vocab_size=100).eval()
    _ = torch.jit.script(model)


@pytest.mark.parametrize("embedding_size", [512])
def test_inference_model_forward(embedding_size: int):
    tokenizer = Tokenizer.from_texts(DUMMY_TEXTS)
    model = ClipDecoder(vocab_size=tokenizer.num_tokens, embedding_size=embedding_size)
    inference_model = ClipDecoderInferenceModel(model=model, tokenizer=tokenizer)

    memory = torch.randn(1, 1, embedding_size)
    text = inference_model(memory)
    assert isinstance(text, str) and len(text) > 4


def test_inference_model_save():
    tokenizer = Tokenizer.from_texts(DUMMY_TEXTS)
    model = ClipDecoder(vocab_size=tokenizer.num_tokens)
    inference_model = ClipDecoderInferenceModel(model=model, tokenizer=tokenizer)

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "model.zip")
        inference_model.save(path)


def test_inference_model_save_load():
    tokenizer = Tokenizer.from_texts(DUMMY_TEXTS)
    model = ClipDecoder(vocab_size=tokenizer.num_tokens)
    inference_model = ClipDecoderInferenceModel(model=model, tokenizer=tokenizer)

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "model.zip")
        inference_model.save(path)
        _ = ClipDecoderInferenceModel.load(path)


def test_inference_model_download_pretrained():
    _ = ClipDecoderInferenceModel.download_pretrained()
