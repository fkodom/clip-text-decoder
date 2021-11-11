from __future__ import annotations
from functools import lru_cache
import multiprocessing as mp
import os
from typing import List, Tuple

from pytorch_lightning import callbacks, plugins, seed_everything, Trainer
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset import ClipCocoCaptionsDataset
from src.model import ClipDecoder, ClipDecoderInferenceModel
from src.tokenizer import Tokenizer


@lru_cache()
def load_dataset(training: bool = True):
    dataset: Dataset = ClipCocoCaptionsDataset()
    dataset = Subset(dataset, torch.arange(50000).tolist())
    # Deterministically split into train/val subsets
    torch.manual_seed(0)
    indices = torch.randperm(len(dataset)).tolist()
    num_train = int(0.9 * len(dataset))
    return Subset(dataset, (indices[:num_train] if training else indices[num_train:]))


@lru_cache()
def get_tokenizer() -> Tokenizer:
    ds = load_dataset(training=True)
    return Tokenizer.from_texts(
        texts=(ds[i][1] for i in range(len(ds))),
        language="en",
    )


def collate_fn(batch: List[Tuple[Tensor, str]]) -> Tuple[Tensor, Tensor]:
    def to_tensor(text: str) -> Tensor:
        line = text.rstrip("\n\r")
        return get_tokenizer().tokenize(f"BOS {line} EOS")

    src = torch.stack([torch.from_numpy(sample[0]) for sample in batch], dim=0).reshape(
        1, len(batch), -1
    )
    padding_value = get_tokenizer().stoi["PAD"]
    tgt = pad_sequence(
        [to_tensor(sample[1]) for sample in batch], padding_value=padding_value
    )
    return src, tgt


def get_dataloader(batch_size: int = 128, training: bool = True):
    return DataLoader(
        dataset=load_dataset(training=training),
        batch_size=batch_size,
        num_workers=mp.cpu_count(),
        collate_fn=collate_fn,
    )


def show_sample_predictions(model: ClipDecoderInferenceModel, n: int = 10):
    ds = load_dataset(training=False)
    for i in range(n):
        encoding, text = ds[i]
        pred = model(torch.from_numpy(encoding))
        print(f"True: {text}")
        print(f"Pred: {pred}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)

    model = ClipDecoder(
        vocab_size=get_tokenizer().num_tokens,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
    )
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        precision=args.precision,
        logger=True,
        callbacks=[
            callbacks.ModelCheckpoint(monitor="validation_loss"),
            callbacks.EarlyStopping(monitor="validation_loss"),
        ],
        plugins=[
            plugins.DDPPlugin(find_unused_parameters=False),
        ],
    )

    # Train the model, and then load the best-performing state dictionary.
    trainer.fit(model, get_dataloader(training=True), get_dataloader(training=False))
    assert trainer.checkpoint_callback is not None
    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint["state_dict"])

    # Build a self-contained inference model and generate a bunch of sample predictions.
    decoder = ClipDecoderInferenceModel(model=model, tokenizer=get_tokenizer())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device=device)
    show_sample_predictions(decoder, n=25)

    # Save the inference model to our experiment logs directory.
    assert trainer.log_dir is not None
    inference_model_path = os.path.join(trainer.log_dir, "model.zip")
    decoder.save(inference_model_path)
