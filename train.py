from __future__ import annotations

import multiprocessing as mp
import os
import random
from functools import lru_cache
from typing import List, Tuple

import torch
from datasets import load_metric
from pytorch_lightning import Trainer, callbacks, seed_everything
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from clip_text_decoder.common import load_tokenizer
from clip_text_decoder.dataset import CocoCaptionsDataset
from clip_text_decoder.model import Decoder, DecoderInferenceModel

get_tokenizer = lru_cache()(load_tokenizer)


@lru_cache()
def load_dataset(
    vision_backbone: str = "blip:base", split: str = "train"
) -> CocoCaptionsDataset:
    return CocoCaptionsDataset(vision_backbone=vision_backbone, split=split)


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


def get_dataloader(vision_backbone: str, batch_size: int = 64, split: str = "train"):
    return DataLoader(
        dataset=load_dataset(vision_backbone=vision_backbone, split=split),
        batch_size=batch_size,
        num_workers=mp.cpu_count(),
        collate_fn=collate_fn,
        shuffle=(split == "train"),
    )


def show_sample_predictions(
    model: DecoderInferenceModel, n: int = 25, beam_size: int = 1
):
    ds = load_dataset(vision_backbone=model.model.vision_backbone, split="val")
    random.seed(0)
    for _ in range(n):
        encoding, text = ds[random.randint(0, len(ds)) - 1]
        pred = model(torch.from_numpy(encoding), beam_size=beam_size)
        print(f"Pred: {pred}")
        print(f"True: {text}")


def compute_bleu_score(
    model: DecoderInferenceModel,
    beam_size: int = 1,
    num_samples: int = 2048,
    verbose: bool = True,
) -> float:
    torch.manual_seed(0)

    ds = load_dataset(vision_backbone=model.model.vision_backbone, split="val")
    idx = torch.randperm(len(ds))[:num_samples].tolist()
    subset = Subset(ds, indices=idx)
    bleu = load_metric("bleu")

    for x, y in tqdm(subset, desc="BLEU", disable=(not verbose)):
        output = model(torch.as_tensor(x), beam_size=beam_size)
        prediction = model.tokenizer.tokenize(output)
        reference = [model.tokenizer.tokenize(ref) for ref in y]
        bleu.add_batch(predictions=[prediction], references=[reference])

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
        # state_dict = torch.load(
        #     "lightning_logs/blip-distilgpt2/checkpoints/epoch=17-step=11646.ckpt"
        # )["state_dict"]
        # for key in list(state_dict.keys()):
        #     if key.startswith("gpt"):
        #         new_key = key.replace("gpt", "language_model")
        #         state_dict[new_key] = state_dict.pop(key)
        # model.load_state_dict(state_dict)

    if not args.eval_only:
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=-1,
            strategy="ddp",
            precision=args.precision,
            accumulate_grad_batches=args.accumulate_grad_batches,
            logger=True,
            callbacks=[
                callbacks.ModelCheckpoint(monitor="validation_loss"),
                callbacks.EarlyStopping(
                    monitor="validation_loss", patience=args.patience
                ),
            ],
            limit_train_batches=1,
            limit_val_batches=1,
        )
        # Train the model, and then load the best-performing state dictionary.
        trainer.fit(
            model,
            get_dataloader(split="train", batch_size=args.batch_size),
            get_dataloader(split="val", batch_size=args.batch_size),
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
    show_sample_predictions(decoder, n=25, beam_size=args.beam_size)
    print(f"BLEU score: {compute_bleu_score(decoder, beam_size=args.beam_size):.4f}")
