import os
import tempfile
from functools import lru_cache

import pytest
import torch
from PIL import Image

from clip_text_decoder.common import load_tokenizer
from clip_text_decoder.model import (
    Decoder,
    DecoderInferenceModel,
    ImageCaptionInferenceModel,
)

get_tokenizer = lru_cache()(load_tokenizer)
LANGUAGE_MODELS = ["distilgpt2"]
DUMMY_TEXTS = [
    "This is a dummy sentence.",
    "This is also a dummy sentence.",
]


@pytest.mark.parametrize("language_model", LANGUAGE_MODELS)
def test_model_forward(language_model: str):
    model = Decoder(language_model=language_model).eval()
    tokenizer = get_tokenizer(language_model)

    device = model.device
    vocab_size = len(tokenizer)
    BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM = 1, 16, 512
    input_ids = torch.randint(0, vocab_size, size=(BATCH_SIZE, SEQ_LEN), device=device)
    encoder_hidden_states = torch.randn(BATCH_SIZE, 1, EMBEDDING_DIM, device=device)

    out = model.forward(
        input_ids=input_ids,
        encoder_hidden_states=encoder_hidden_states,
    )
    batch_size, seq_len, out_size = out.logits.shape
    assert batch_size == BATCH_SIZE
    assert seq_len == SEQ_LEN
    assert out_size == vocab_size


@pytest.mark.parametrize("language_model", LANGUAGE_MODELS)
def test_inference_model_forward(language_model: str):
    model = Decoder(language_model=language_model).eval()
    tokenizer = get_tokenizer(language_model)
    inference_model = DecoderInferenceModel(model=model, tokenizer=tokenizer)

    EMBEDDING_SIZE = 512
    memory = torch.randn(1, 1, EMBEDDING_SIZE)
    text = inference_model(memory)
    assert isinstance(text, str)


@pytest.mark.parametrize("language_model", LANGUAGE_MODELS)
def test_inference_model_save(language_model: str):
    model = Decoder(language_model=language_model).eval()
    tokenizer = get_tokenizer(language_model)
    inference_model = DecoderInferenceModel(model=model, tokenizer=tokenizer)

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "model.pt")
        inference_model.save(path)


@pytest.mark.parametrize("language_model", LANGUAGE_MODELS)
def test_inference_model_save_load(language_model: str):
    model = Decoder(language_model=language_model).eval()
    tokenizer = get_tokenizer(language_model)
    inference_model = DecoderInferenceModel(model=model, tokenizer=tokenizer)

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "model.pt")
        inference_model.save(path)
        _ = DecoderInferenceModel.load(path)


@pytest.mark.slow
def test_inference_model_download_pretrained():
    _ = DecoderInferenceModel.download_pretrained()


@pytest.mark.slow
def test_image_caption_model_predict():
    image = Image.new("RGB", (224, 224))
    model = ImageCaptionInferenceModel.download_pretrained()
    pred = model(image)
    assert isinstance(pred, str)
    assert len(pred) > 0
