from __future__ import annotations
from functools import lru_cache
import multiprocessing as mp
import os
import random
from typing import List, Tuple

from pytorch_lightning import callbacks, seed_everything, Trainer
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

from clip_text_decoder.dataset import ClipCocoCaptionsDataset
from clip_text_decoder.model import ClipDecoder, ClipDecoderInferenceModel
from clip_text_decoder.tokenizer import Tokenizer


@lru_cache()
def load_dataset(split: str = "train") -> ClipCocoCaptionsDataset:
    return ClipCocoCaptionsDataset(split=split)


@lru_cache()
def get_tokenizer() -> Tokenizer:
    ds = load_dataset(split="train")
    return Tokenizer.from_texts(
        texts=(text for i in range(len(ds)) for text in ds[i][1]),
        language="en_core_web_sm",
    )


def collate_fn(
    batch: List[Tuple[Tensor, str]],
    max_len: int = 128,
) -> Tuple[Tensor, Tensor]:
    def to_tensor(text: str) -> Tensor:
        line = text.rstrip("\n\r")
        tensor = get_tokenizer().tokenize(f"BOS {line} EOS")
        return tensor[:max_len]

    src = torch.stack(
        [torch.from_numpy(x) for x, _ in batch],
        dim=0,
    ).reshape(1, len(batch), -1)
    padding_value = get_tokenizer().stoi["PAD"]
    tgt = pad_sequence(
        [to_tensor(random.choice(y)) for _, y in batch],
        padding_value=padding_value,
    )
    return src, tgt


def get_dataloader(batch_size: int = 64, split: str = "train"):
    return DataLoader(
        dataset=load_dataset(split=split),
        batch_size=batch_size,
        num_workers=mp.cpu_count(),
        collate_fn=collate_fn,
        shuffle=(split == "train"),
    )


def show_sample_predictions(model: ClipDecoderInferenceModel, n: int = 10):
    ds = load_dataset(split="val")
    for i in range(n):
        encoding, text = ds[i]
        pred = model(torch.from_numpy(encoding))
        print(f"Pred: {pred}")
        print(f"True: {text}")


def compute_bleu_score(model: ClipDecoderInferenceModel, verbose: bool = True) -> float:
    ds = load_dataset(split="val")
    tokenizer = model.tokenizer._str_tokenizer
    assert tokenizer is not None

    bleu = 0
    for x, y in tqdm(ds, desc="BLEU", disable=(not verbose)):
        candidate = tokenizer(model(torch.as_tensor(x)))
        candidate = [x for x in candidate if not " " in x]
        references = [tokenizer(ref) for ref in y]
        bleu += bleu_score([candidate], [references])

    return bleu / len(ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--accumulate-grad-batches", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)

    model = ClipDecoder(
        vocab_size=get_tokenizer().num_tokens,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
    )

    # model = ClipDecoder.load_from_checkpoint(
    #     "lightning_logs/version_9/checkpoints/epoch=49-step=16199.ckpt",
    #     vocab_size=get_tokenizer().num_tokens,
    # )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=True,
        callbacks=[
            callbacks.ModelCheckpoint(monitor="validation_bleu", mode="max"),
            callbacks.EarlyStopping(monitor="validation_bleu", mode="max", patience=5),
        ],
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
    decoder = ClipDecoderInferenceModel(model=model, tokenizer=get_tokenizer())
    # Save the inference model to our experiment logs directory.
    assert trainer.log_dir is not None
    inference_model_path = os.path.join(trainer.log_dir, "model.zip")
    decoder.save(inference_model_path)

    # Get sample predictions, and compute the BLEU score for the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device=device)
    show_sample_predictions(decoder, n=10)
    print(f"BLEU score: {compute_bleu_score(decoder):.4f}")
