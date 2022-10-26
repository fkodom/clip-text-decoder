from __future__ import annotations

import asyncio
import io
import json
import os
from collections import defaultdict
from typing import Dict, Generator, List, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import torch
import wget
from PIL import Image
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from tqdm import tqdm

from clip_text_decoder.common import encode_image_tensor, load_vision_backbone
from clip_text_decoder.utils.fileio import async_batch_get_request

COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
)
TRAIN_JSON = "annotations/captions_train2014.json"
VAL_JSON = "annotations/captions_val2014.json"


def _download_coco_captions_json(cache_dir: str):
    zip_path = os.path.join(cache_dir, "annotations.zip")
    if not os.path.exists(zip_path):
        print("Downloading COCO Captions annotations.")
        wget.download(COCO_ANNOTATIONS_URL, zip_path)

    with ZipFile(zip_path, mode="r") as zip:
        if not os.path.exists(os.path.join(cache_dir, TRAIN_JSON)):
            zip.extract(TRAIN_JSON, path=cache_dir)
        if not os.path.exists(os.path.join(cache_dir, VAL_JSON)):
            zip.extract(VAL_JSON, path=cache_dir)


def _load_coco_captions_json(cache_dir: str, split: str) -> Dict[str, List]:
    _download_coco_captions_json(cache_dir)

    if split == "train":
        json_file = TRAIN_JSON
    elif split == "val":
        json_file = VAL_JSON
    else:
        raise ValueError(f"Invalid split '{split}'. Available: ['train', 'val'].")

    with open(os.path.join(cache_dir, json_file)) as f:
        return json.load(f)


def _get_captions_by_image_id(annotations: Sequence[Dict]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = defaultdict(list)
    for ann in annotations:
        image_id = ann["image_id"]
        out[image_id].append(ann["caption"])

    return out


class ParallelSampleDownloader(IterDataPipe):
    def __init__(self, dp: IterDataPipe[str]) -> None:
        super().__init__()
        self.dp = dp

    def __iter__(self) -> Generator[Image.Image, None, None]:
        for batch in self.dp:
            images = asyncio.run(async_batch_get_request([x[0] for x in batch]))
            captions = [x[1] for x in batch]

            batch_results = []
            for _image, _captions in zip(images, captions):
                if _image is None:
                    continue
                try:
                    image = Image.open(io.BytesIO(_image))
                    batch_results.append((image, _captions))
                except Exception:
                    continue

            yield batch_results


class ParallelImageEncoder(IterDataPipe):
    def __init__(
        self, dp: IterDataPipe[Tuple[Image.Image, List[str]]], vision_backbone: str
    ):
        super().__init__()
        self.dp = dp
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocessor = load_vision_backbone(
            vision_backbone, device=self.device
        )

    @torch.inference_mode()
    @torch.no_grad()
    def __iter__(self) -> Generator[Tuple[np.ndarray, List[str]], None, None]:
        for batch in self.dp:
            captions = [caps for _, caps in batch]
            images = torch.stack(
                [self.preprocessor(image.convert("RGB")) for image, _ in batch]
            ).to(self.device)
            image_features = encode_image_tensor(images, self.model)

            # For legacy reasons, unsqueeze along dimension 1.
            image_features_np = image_features.unsqueeze(1).cpu().numpy()
            yield [(feats, caps) for feats, caps in zip(image_features_np, captions)]


def coco_captions_datapipe(
    cache_dir: str = "./coco-captions",
    split: str = "train",
    buffer_size: int = 128,
) -> IterDataPipe:
    captions_json = _load_coco_captions_json(cache_dir=cache_dir, split=split)
    images, annotations = captions_json["images"], captions_json["annotations"]
    captions_by_image_id = _get_captions_by_image_id(annotations)
    images_with_captions = [
        (image["coco_url"], captions_by_image_id[image["id"]]) for image in images
    ]

    pipe: IterDataPipe = IterableWrapper(images_with_captions)
    pipe = pipe.batch(buffer_size)
    pipe = ParallelSampleDownloader(pipe)
    pipe = pipe.unbatch()

    return pipe


if __name__ == "__main__":
    pipe = coco_captions_datapipe()

    for _ in tqdm(pipe.batch(32)):
        pass
