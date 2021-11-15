import torch

from clip_text_decoder.tokenizer import Tokenizer, SPECIALS_STOI


DUMMY_TEXTS = [
    "This is a dummy sentence.",
    "This is also a dummy sentence.",
]


def test_tokenizer_from_texts():
    tokenizer = Tokenizer.from_texts(DUMMY_TEXTS)
    # There are 6 unique words in the dummy text, plus '.' punctuations,
    # and any added special tokens.
    assert len(tokenizer.stoi) == (7 + len(SPECIALS_STOI))


def test_tokenizer_tokenize_untokenize():
    tokenizer = Tokenizer.from_texts(DUMMY_TEXTS)
    # There are 6 unique words in the dummy text, plus '.' punctuations,
    # and 4 added special tokens ['UNK', 'PAD', 'BOS', 'EOS']
    assert len(tokenizer.stoi) == 11

    tokenized = tokenizer.tokenize(DUMMY_TEXTS[0])
    assert isinstance(tokenized, torch.Tensor)
    assert tokenizer.untokenize(tokenized) == DUMMY_TEXTS[0]
