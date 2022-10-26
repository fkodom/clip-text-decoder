from __future__ import annotations

import os
import pickle
from typing import List, Tuple

import gdown
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from clip_text_decoder.common import check_vision_backbone
from clip_text_decoder.datapipes import ParallelImageEncoder, coco_captions_datapipe

COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
)
CACHE_URLS = {
    "blip:base": {
        "train": "https://drive.google.com/uc?id=1vzmE9dyHeTyL0S7pYj8RurY82W53SGqE",
        # https://drive.google.com/file/d/1vzmE9dyHeTyL0S7pYj8RurY82W53SGqE/view?usp=sharing
        "val": "https://drive.google.com/uc?id=1aeUKoWOSjHd2K_SAKZUhmrzreXdiuydT",
        # https://drive.google.com/file/d/1aeUKoWOSjHd2K_SAKZUhmrzreXdiuydT/view?usp=sharing
    },
    "clip:ViT-B/32": {
        "train": "https://drive.google.com/uc?id=1e-K7UIgVsvsHZEkZguzhTqEoMAUfu538",
        # https://drive.google.com/file/d/1e-K7UIgVsvsHZEkZguzhTqEoMAUfu538/view?usp=sharing
        "val": "https://drive.google.com/uc?id=11l6b9rol53FAZe4EhvlgiwrsD4qfUUdj",
        # https://drive.google.com/file/d/11l6b9rol53FAZe4EhvlgiwrsD4qfUUdj/view?usp=sharing
    },
}
CACHE_URL = {
    "train": "https://drive.google.com/uc?id=1e-K7UIgVsvsHZEkZguzhTqEoMAUfu538",
    # https://drive.google.com/file/d/1e-K7UIgVsvsHZEkZguzhTqEoMAUfu538/view?usp=sharing
    "val": "https://drive.google.com/uc?id=11l6b9rol53FAZe4EhvlgiwrsD4qfUUdj",
    # https://drive.google.com/file/d/11l6b9rol53FAZe4EhvlgiwrsD4qfUUdj/view?usp=sharing
}

BUILD_DATASET_MESSAGE = """
Building encodings for {dataset} dataset. This may take an hour or more
(with a GPU). The result will be cached so that subsequent calls are fast.
"""


class CocoCaptionsDataset(Dataset):
    def __init__(
        self,
        vision_backbone: str = "blip:base",
        root: str = "./coco-captions",
        split: str = "train",
        force_rebuild: bool = False,
    ):
        super().__init__()
        check_vision_backbone(vision_backbone)
        self.vision_backbone = vision_backbone
        name = vision_backbone.lower().replace("/", "").replace(":", "-")

        self.root = root
        self.dir = os.path.join(root, name)
        self.split = split
        self.zip_path = os.path.join(self.dir, "annotations.zip")
        self.cache_path = os.path.join(self.dir, f"cache-{split}.pkl")

        self.data: List[Tuple[Tensor, str]] = self._load(force_rebuild=force_rebuild)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        return self.data[idx]

    def _download_cache(self):
        url_by_split = CACHE_URLS[self.vision_backbone]
        if not os.path.exists(self.cache_path):
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            gdown.download(url_by_split[self.split], self.cache_path, quiet=False)

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def _build(self, force: bool = False, cache: bool = True):
        if force or not os.path.exists(self.cache_path):
            print(BUILD_DATASET_MESSAGE.format(dataset="CocoCaptions"))
            data = _compile_encoded_data(
                vision_backbone=self.vision_backbone,
                cache_dir=self.root,
                split=self.split,
            )
            if cache:
                with open(self.cache_path, "wb") as fb:
                    pickle.dump(data, fb)

    def _load(self, cache: bool = True, force_rebuild: bool = False):
        if not force_rebuild and self.vision_backbone in CACHE_URLS:
            self._download_cache()
        self._build(force=force_rebuild, cache=cache)
        with open(self.cache_path, "rb") as f:
            return pickle.load(f)


def _compile_encoded_data(
    vision_backbone: str, cache_dir: str, split: str
) -> List[Tuple[np.ndarray, List[str]]]:
    print("Compiling encodings with text captions...")
    pipe = coco_captions_datapipe(cache_dir=cache_dir, split=split)
    pipe = pipe.batch(32)
    pipe = ParallelImageEncoder(pipe, vision_backbone=vision_backbone)
    pipe = pipe.unbatch()

    return [(encoding, captions) for encoding, captions in tqdm(pipe)]


if __name__ == "__main__":
    ds = CocoCaptionsDataset(split="train", force_rebuild=True)
    ds = CocoCaptionsDataset(split="val", force_rebuild=True)
    # ds = ClipCocoCaptionsDataset(split="test")
