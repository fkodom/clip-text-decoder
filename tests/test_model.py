import os
import tempfile
from functools import lru_cache

import pytest
import torch
from PIL import Image
from transformers import GPT2Tokenizer

from clip_text_decoder.model import (
    ClipDecoder,
    ClipDecoderInferenceModel,
    ImageCaptionInferenceModel,
)

GPT2_TYPES = ["distilgpt2"]
DUMMY_TEXTS = [
    "This is a dummy sentence.",
    "This is also a dummy sentence.",
]


@lru_cache()
def get_tokenizer(gpt2_type: str = "distilgpt2") -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.mark.parametrize("gpt2_type", GPT2_TYPES)
def test_model_forward(gpt2_type: str):
    model = ClipDecoder(gpt2_type=gpt2_type).eval()
    tokenizer = get_tokenizer(gpt2_type=gpt2_type)

    vocab_size = len(tokenizer)
    BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM = 1, 16, 512
    input_ids = torch.randint(0, vocab_size, size=(BATCH_SIZE, SEQ_LEN))
    encoder_hidden_states = torch.randn(BATCH_SIZE, 1, EMBEDDING_DIM)

    out = model.forward(
        input_ids=input_ids,
        encoder_hidden_states=encoder_hidden_states,
    )
    batch_size, seq_len, out_size = out.logits.shape
    assert batch_size == BATCH_SIZE
    assert seq_len == SEQ_LEN
    assert out_size == vocab_size


@pytest.mark.parametrize("gpt2_type", GPT2_TYPES)
def test_inference_model_forward(gpt2_type: str):
    model = ClipDecoder(gpt2_type=gpt2_type).eval()
    tokenizer = get_tokenizer(gpt2_type=gpt2_type)
    inference_model = ClipDecoderInferenceModel(model=model, tokenizer=tokenizer)

    EMBEDDING_SIZE = 512
    memory = torch.randn(1, 1, EMBEDDING_SIZE)
    text = inference_model(memory)
    assert isinstance(text, str)


@pytest.mark.parametrize("gpt2_type", GPT2_TYPES)
def test_inference_model_save(gpt2_type: str):
    model = ClipDecoder(gpt2_type=gpt2_type).eval()
    tokenizer = get_tokenizer(gpt2_type=gpt2_type)
    inference_model = ClipDecoderInferenceModel(model=model, tokenizer=tokenizer)

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "model.pt")
        inference_model.save(path)


@pytest.mark.parametrize("gpt2_type", GPT2_TYPES)
def test_inference_model_save_load(gpt2_type: str):
    model = ClipDecoder(gpt2_type=gpt2_type).eval()
    tokenizer = get_tokenizer(gpt2_type=gpt2_type)
    inference_model = ClipDecoderInferenceModel(model=model, tokenizer=tokenizer)

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "model.pt")
        inference_model.save(path)
        _ = ClipDecoderInferenceModel.load(path)


@pytest.mark.slow
def test_inference_model_download_pretrained():
    _ = ClipDecoderInferenceModel.download_pretrained()


@pytest.mark.slow
def test_image_caption_model_predict():
    image = Image.new("RGB", (224, 224))
    model = ImageCaptionInferenceModel.download_pretrained()
    pred = model(image)
    assert isinstance(pred, str)
    assert len(pred) > 0
