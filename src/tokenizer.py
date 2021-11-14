from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import dill
import torch
from torch import Tensor
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer as _get_tokenizer_torchtext


StringTokenizer = Callable[[str], List[str]]
SPECIALS_STOI = {"UNK": 0, "PAD": 1, "BOS": 2, "EOS": 3}


class Tokenizer:
    def __init__(self):
        self._str_tokenizer: Optional[StringTokenizer] = None
        self._stoi: Dict = {}
        self._itos: Dict = {}

    @property
    def stoi(self) -> Dict[str, int]:
        if not self._stoi and self._itos:
            self._stoi = {s: i for i, s in self._itos.items()}
        return self._stoi

    @property
    def itos(self) -> Dict[int, str]:
        if not self._itos and self._stoi:
            self._itos = {i: s for s, i in self._stoi.items()}
        return self._itos

    @property
    def num_tokens(self):
        return len(self.stoi)

    @classmethod
    def from_iterator(
        cls,
        iterator: Iterator[Iterable[str]],
        str_tokenizer: str = "spacy",
        language: str = "en",
        min_freq: int = 1,
        specials: Sequence[str] = list(SPECIALS_STOI),
    ) -> Tokenizer:
        tokenizer = Tokenizer()
        tokenizer._str_tokenizer = _get_tokenizer_torchtext(
            tokenizer=str_tokenizer,
            language=language,
        )
        _vocab = build_vocab_from_iterator(
            iterator,
            min_freq=min_freq,
            specials=specials,
            special_first=True,
        )
        tokenizer._stoi = _vocab.get_stoi()
        return tokenizer

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        str_tokenizer: str = "spacy",
        language: str = "en",
        min_freq: int = 1,
        specials: Sequence[str] = list(SPECIALS_STOI),
    ) -> Tokenizer:
        _str_tokenizer = _get_tokenizer_torchtext(
            tokenizer=str_tokenizer,
            language=language,
        )
        return cls.from_iterator(
            (_str_tokenizer(text) for text in texts),
            str_tokenizer=str_tokenizer,
            language=language,
            min_freq=min_freq,
            specials=specials,
        )

    def save(self, path: str):
        with open(path, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        with open(path, "rb") as f:
            return dill.load(f)

    def tokenize(
        self,
        text: str,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        if self.stoi is None:
            raise ValueError("Cannot tokenize texts when 'stoi' is None.")
        elif self._str_tokenizer is None:
            raise ValueError("Cannot tokenize texts when '_str_tokenizer' is None.")

        unk_idx = self.stoi["UNK"]
        return torch.as_tensor(
            [self.stoi.get(token, unk_idx) for token in self._str_tokenizer(text)],
            dtype=torch.long,
            device=device,
        )

    def untokenize(self, tokens: Union[Tensor, Iterable[int]]) -> str:
        if self.itos is None:
            raise ValueError("Cannot tokenize texts when 'itos' is None.")

        text = " ".join([self.itos.get(int(i), "UNK") for i in tokens])
        patterns = [
            re.compile(r"\s([.,?!')])"),  # Leading space before certain punctuation
            re.compile(r"([(])(?:\s)"),  # Trailing space after open parenthesis
        ]
        for pattern in patterns:
            text = re.sub(pattern, r"\1", text)

        return text
