from __future__ import annotations

import multiprocessing as mp
import os
import random
from functools import lru_cache
from typing import List, Tuple

import evaluate
import torch
from pytorch_lightning import Trainer, callbacks, seed_everything, strategies
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from clip_text_decoder.common import load_tokenizer
from clip_text_decoder.dataset import CachedDataset, CocoCaptionsDataset
from clip_text_decoder.model import Decoder, DecoderInferenceModel

get_tokenizer = lru_cache()(load_tokenizer)


@lru_cache()
def load_coco_captions(
    vision_backbone: str = "blip:base", split: str = "train"
) -> CocoCaptionsDataset:
    return CocoCaptionsDataset.build(vision_backbone=vision_backbone, split=split)


def collate_fn(
    batch: List[Tuple[Tensor, str]],
    gpt2_type: str = "distilgpt2",
    max_length: int = 1024,
) -> Tuple[Tensor, Tensor, Tensor]:
    tokenizer = get_tokenizer(gpt2_type)
    bos, eos = tokenizer.bos_token, tokenizer.eos_token
    encoded = tokenizer.batch_encode_plus(
        [f"{bos}{random.choice(y)}{eos}" for _, y in batch],
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    encoder_hidden_states = torch.stack(
        [torch.from_numpy(x) for x, _ in batch],
        dim=0,
    ).reshape(len(batch), 1, -1)

    return (
        encoder_hidden_states.float(),
        encoded["input_ids"],
        encoded["attention_mask"],
    )


def get_dataloader(dataset: CachedDataset, batch_size: int = 64, shuffle: bool = False):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=mp.cpu_count(),
        collate_fn=collate_fn,
        shuffle=shuffle,
    )


def show_sample_predictions(
    model: DecoderInferenceModel,
    dataset: CachedDataset,
    num_samples: int = 25,
    beam_size: int = 1,
):
    torch.manual_seed(0)
    idx = torch.randperm(len(dataset))[:num_samples].tolist()
    subset = Subset(dataset, indices=idx)

    for encoding, captions in subset:
        pred = model(torch.from_numpy(encoding), beam_size=beam_size)
        print(f"Pred: {pred}")
        print(f"True: {captions}")


def compute_bleu_score(
    model: DecoderInferenceModel,
    dataset: CachedDataset,
    beam_size: int = 1,
    num_samples: int = 2048,
    verbose: bool = True,
) -> float:
    torch.manual_seed(0)
    idx = torch.randperm(len(dataset))[:num_samples].tolist()
    subset = Subset(dataset, indices=idx)
    bleu = evaluate.load("bleu")

    for encoding, captions in tqdm(subset, desc="BLEU", disable=(not verbose)):
        prediction = model(torch.as_tensor(encoding), beam_size=beam_size)
        bleu.add_batch(predictions=[prediction], references=[captions])

    return bleu.compute()["bleu"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-backbone", type=str, default="blip:base")
    parser.add_argument("--language-model", type=str, default="distilgpt2")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.checkpoint:
        model = Decoder.load_from_checkpoint(args.checkpoint)
    else:
        model = Decoder(
            vision_backbone=args.vision_backbone,
            language_model=args.language_model,
        )

    if not args.eval_only:
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator="auto",
            devices="auto",
            strategy=strategies.DDPStrategy(find_unused_parameters=False),
            precision=args.precision,
            accumulate_grad_batches=args.accumulate_grad_batches,
            logger=True,
            callbacks=[
                callbacks.ModelCheckpoint(monitor="validation_loss"),
                callbacks.EarlyStopping(
                    monitor="validation_loss", patience=args.patience
                ),
            ],
        )

        train_dataset = load_coco_captions(args.vision_backbone, split="train")
        val_dataset = load_coco_captions(args.vision_backbone, split="val")
        # Train the model, and then load the best-performing state dictionary.
        trainer.fit(
            model,
            get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True),
            get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False),
        )
        assert trainer.checkpoint_callback is not None
        checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
        model.load_state_dict(checkpoint["state_dict"])

    # Build a self-contained inference model and generate a bunch of sample predictions.
    decoder = DecoderInferenceModel(
        model=model,
        tokenizer=get_tokenizer(args.language_model),
    )

    if not args.eval_only:
        # Save the inference model to our experiment logs directory.
        assert trainer.log_dir is not None
        inference_model_path = os.path.join(trainer.log_dir, "model.pt")
        decoder.save(inference_model_path)

    # Get sample predictions, and compute the BLEU score for the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device=device)

    val_dataset = load_coco_captions(vision_backbone=args.vision_backbone, split="val")
    show_sample_predictions(decoder, val_dataset, beam_size=args.beam_size)
    bleu = compute_bleu_score(decoder, dataset=val_dataset, beam_size=args.beam_size)
    print(f"BLEU score: {bleu:.4f}")
