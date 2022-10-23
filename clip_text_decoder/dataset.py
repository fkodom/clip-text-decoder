from __future__ import annotations

import io
import json
import os
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import gdown
import numpy as np
import requests
import torch
import wget
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clip_text_decoder.common import (
    check_vision_backbone,
    encode_image_tensor,
    load_vision_backbone,
)

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
        self.root = os.path.join(root, name)
        self.split = split
        self.zip_path = os.path.join(self.root, "annotations.zip")
        self.cache_path = os.path.join(self.root, f"cache-{split}.pkl")

        self.data: List[Tuple[Tensor, str]] = self._load(force_rebuild=force_rebuild)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        return self.data[idx]

    def _download_coco_captions(self, force: bool = False):
        os.makedirs(self.root, exist_ok=True)
        zip_path = os.path.join(self.root, "annotations.zip")

        if force or not os.path.exists(zip_path):
            print("Downloading COCO Captions annotations.")
            wget.download(COCO_ANNOTATIONS_URL, zip_path)

        with ZipFile(zip_path, mode="r") as zip:
            zip.extract("annotations/captions_train2014.json", path=self.root)
            zip.extract("annotations/captions_val2014.json", path=self.root)

    def _load_coco_captions(self) -> Dict:
        print("Downloading and extracting caption data...")
        self._download_coco_captions()
        path = os.path.join(self.root, f"annotations/captions_{self.split}2014.json")
        with open(path) as f:
            return json.load(f)

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
            labels = self._load_coco_captions()
            images, annotations = labels["images"], labels["annotations"]
            encodings = _build_clip_encodings(images)
            data = _compile_clip_data(images, annotations, encodings)

            if cache:
                with open(self.cache_path, "wb") as fb:
                    pickle.dump(data, fb)

    def _load(self, cache: bool = True, force_rebuild: bool = False):
        if not force_rebuild and self.vision_backbone in CACHE_URLS:
            self._download_cache()
        self._build(force=force_rebuild, cache=cache)
        with open(self.cache_path, "rb") as f:
            return pickle.load(f)


@lru_cache()
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache()
def _load_vision_backbone(backbone: str = "BLIP"):
    print("Loading model...")
    return load_vision_backbone(backbone)


def _group_captions_by_image_id(annotations: Sequence[Dict]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = defaultdict(list)
    for ann in annotations:
        image_id = ann["image_id"]
        out[image_id].append(ann["caption"])

    return out


def _get_image_from_url(url: str) -> Tensor:
    response = requests.get(url)
    bytes_io = io.BytesIO(response.content)
    _, preprocessor = _load_vision_backbone()
    return preprocessor(Image.open(bytes_io).convert("RGB"))


def _encode_coco_images(urls: List[str]) -> Tensor:
    pool = ThreadPoolExecutor()
    images = list(pool.map(_get_image_from_url, urls))
    device = get_device()
    image_tensor = torch.stack([image for image in images], dim=0).to(device)
    backbone, _ = _load_vision_backbone()

    return encode_image_tensor(image_tensor, backbone)


def _build_clip_encodings(
    images: Sequence[Dict], batch_size: int = 16
) -> List[Optional[np.ndarray]]:
    _ = _load_vision_backbone()

    urls: List[str] = [i["coco_url"] for i in images]
    loader = DataLoader(
        urls,
        batch_size=batch_size,
        num_workers=cpu_count(),
        prefetch_factor=8,
    )

    print("Computing CLIP image encodings...")
    encodings = []
    for urls_batch in tqdm(loader):
        try:
            encodings += list(_encode_coco_images(urls_batch).cpu().numpy())
        except Exception:
            encodings += len(urls_batch) * [None]

    return encodings


def _compile_clip_data(
    images: Sequence[Dict],
    annotations: Sequence[Dict],
    encodings: Sequence[Optional[np.ndarray]],
) -> List[Tuple[np.ndarray, List[str]]]:
    print("Compiling encodings with text captions...")
    captions = _group_captions_by_image_id(annotations)
    return [
        (encoding, captions[image["id"]])
        for encoding, image in zip(encodings, images)
        if encoding is not None
    ]


if __name__ == "__main__":
    ds = CocoCaptionsDataset(split="train", force_rebuild=True)
    ds = CocoCaptionsDataset(split="val", force_rebuild=True)
    # ds = ClipCocoCaptionsDataset(split="test")
