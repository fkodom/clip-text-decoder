from __future__ import annotations

import os
import pickle
from tempfile import TemporaryDirectory
from typing import Any, Iterable, List, Tuple

import gdown
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchdata.datapipes.iter import IterableWrapper
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


class CachedDataset(Dataset):
    def __init__(self, data: List[Tuple[Tensor, Any]]):
        super().__init__()
        self.data = data

    @classmethod
    def load(cls, path: str) -> CachedDataset:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return CachedDataset(data=data)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.data, f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        return self.data[idx]


def build_cached_dataset(
    data_iterable: Iterable[Image.Image, Any],
    vision_backbone: str,
) -> CachedDataset:
    print("Compiling encodings with text captions...")
    pipe = IterableWrapper(iter(data_iterable))
    pipe = pipe.batch(32)
    pipe = ParallelImageEncoder(pipe, vision_backbone=vision_backbone)
    pipe = pipe.unbatch()

    data = [(encoding, captions) for encoding, captions in tqdm(pipe)]
    return CachedDataset(data=data)


class CocoCaptionsDataset(CachedDataset):
    @staticmethod
    def download(url: str) -> CocoCaptionsDataset:
        with TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "cache.pkl")
            gdown.download(url, path, quiet=False)
            with open(path, "rb") as f:
                data = pickle.load(f)

        return CocoCaptionsDataset(data=data)

    @classmethod
    def build(
        cls,
        vision_backbone: str = "blip:base",
        root: str = "./coco-captions",
        split: str = "train",
        force_rebuild: bool = False,
    ) -> CocoCaptionsDataset:
        check_vision_backbone(vision_backbone)
        if force_rebuild or vision_backbone not in CACHE_URLS:
            pipe = coco_captions_datapipe(cache_dir=root, split=split)
            cached_dataset = build_cached_dataset(pipe, vision_backbone=vision_backbone)
            return CocoCaptionsDataset(data=cached_dataset.data)
        else:
            url_by_split = CACHE_URLS[vision_backbone]
            url = url_by_split[split]
            return cls.download(url)
