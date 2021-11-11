from __future__ import annotations
from functools import lru_cache
import json
import os
import pickle
import requests
from typing import List, Tuple
from zipfile import ZipFile

import clip
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
)


@lru_cache()
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClipCocoCaptionsDataset(Dataset):
    def __init__(self, root: str = "./coco-captions"):
        super().__init__()
        self.root = root
        self.zip_path = os.path.join(root, "annotations.zip")
        self.cache_path = os.path.join(root, "cache.pkl")

        self.data: List[Tuple[Tensor, str]] = self._load()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        return self.data[idx]

    def _download(self, force: bool = False):
        os.makedirs(self.root, exist_ok=True)
        zip_path = os.path.join(self.root, "annotations.zip")

        if force or not os.path.exists(zip_path):
            print("Downloading COCO Captions annotations...")
            response = requests.get(COCO_ANNOTATIONS_URL)
            with open(zip_path, "wb") as f:
                f.write(response.content)

        with ZipFile(zip_path, mode="r") as zip:
            zip.extract("annotations/captions_train2014.json", path=self.root)
            zip.extract("annotations/captions_val2014.json", path=self.root)

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def _build(self, cache: bool = True):
        self._download()
        with open(os.path.join(self.root, "annotations/captions_train2014.json")) as f:
            train_captions = json.load(f)
        with open(os.path.join(self.root, "annotations/captions_val2014.json")) as f:
            val_captions = json.load(f)
        captions = train_captions["annotations"] + val_captions["annotations"]
        texts: List[str] = [a["caption"] for a in captions]

        print("Loading CLIP model...")
        clip_model, _ = clip.load("ViT-B/32", device=get_device(), jit=False)

        print(f"Getting CLIP input tokens...")
        tokens = clip.tokenize(texts)

        print(f"Getting CLIP text encodings...")
        encodings = []
        loader = DataLoader(tokens, batch_size=1024)
        for tokens_batch in tqdm(loader, desc="CLIP Encodings"):
            x = tokens_batch.to(device=get_device())
            y = clip_model.encode_text(x).float().cpu()
            encodings += list(y.squeeze().numpy())

        data = [(encoding, text) for encoding, text in zip(encodings, texts)]
        if cache:
            with open(self.cache_path, "wb") as fb:
                pickle.dump(data, fb)

    def _load(self):
        if not os.path.exists(self.cache_path):
            print(
                "Building CLIP encodings for COCO Captions dataset. "
                "This will take several minutes (with a GPU) the first time this "
                "happens. The result will be cached so that subsequent calls are fast."
            )
            self._build()

        with open(self.cache_path, "rb") as f:
            return pickle.load(f)
