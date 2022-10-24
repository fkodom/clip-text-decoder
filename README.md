# clip-text-decoder

Train an image captioner with 0.323 BLEU on COCO Captions in under one hour! ([0.352 BLEU with beam search](#ablation-beam-size) 🙂)

Generates text captions for images from their embeddings. **Now includes [BLIP](https://github.com/salesforce/BLIP) as an available vision backbone!**


## Example Predictions

Computed using the pretrained model mentioned below.

<p align="center">
    <img src="http://farm6.staticflickr.com/5028/5654757697_bcdd8088da_z.jpg" height=224/><br>
    "A man riding a wave on top of a surfboard."
</p>

<br>

<p align="center">
    <img src="http://farm4.staticflickr.com/3202/2697603492_fbb44f6d2d_z.jpg" height=224/><br>
    "A baseball player is swinging a bat at a ball."
</p>

<br>

<p align="center">
    <img src="http://farm3.staticflickr.com/2544/3715539092_f070a36b22_z.jpg" height=224/><br>
    "A dog jumping in the air to catch a frisbee."
</p>


## Installation

Using `pip`:
```bash
pip install "clip @ git+https://github.com/openai/CLIP.git"
pip install "lavis @ git+https://github.com/salesforce/LAVIS.git"
pip install clip-text-decoder
```

From source:
```bash
pip install "clip @ git+https://github.com/openai/CLIP.git"
pip install "lavis @ git+https://github.com/salesforce/LAVIS.git"
git clone https://github.com/fkodom/clip-text-decoder.git
cd clip-text-decoder
pip install .
```

## Inference

### Pretrained Model
```python
from PIL import Image
import torch

from clip_text_decoder.model import ImageCaptionInferenceModel

model = ImageCaptionInferenceModel.download_pretrained()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = Image.open("path/to/image.jpeg")
# The beam_size argument is optional. Larger beam_size is slower, but has
# slightly higher accuracy. Recommend using beam_size <= 3.
caption = model(image, beam_size=1)
```

To cache the pretrained model locally, so that it's not re-downloaded each time:
```python
model = ImageCaptionInferenceModel.download_pretrained("path/to/model.pt")
```

### Custom Trained Model

Training produces a `model.pt` archive, containing a `Tokenizer` and model parameters.  To reload the trained inference model:
```python
from clip_text_decoder.model import ImageCaptionInferenceModel

model = ImageCaptionInferenceModel.load("path/to/model.pt").to(device)
# Load image and get predictions like above...
```

## Ablation: Beam Size

Measuring the BLEU-4 score for different `beam_size` arguments.  By default, the inference model uses a beam size of 1:

```python
from clip_text_decoder.model import ImageCaptionInferenceModel

model = ImageCaptionInferenceModel.load("path/to/model.pt")
caption = model(image, beam_size=1)
```

Using larger `beam_size` gives better BLEU score with a trade-off of slower inference speeds. The metrics below were collected from the same model, which uses a BLIP vision backbone and was trained for 10 epochs (roughly 1 hour on a T4 GPU):

Beam size   | BLEU-4
------------|-------
1 (default) | 0.323
2           | 0.343
3           | 0.350
4           | 0.352

## Training

Launch your own training session using `train.py`:
```bash
python train.py --max-epochs 10
```

Training CLI arguments, along with their default values:
```bash
--vision-backbone blip:base  # (str)
--language-model distilgpt2  # (str)
--max-epochs 10  # (int)
--beam-size 1  # (int)
--batch-size 32  # (int)
--accumulate-grad-batches 4  # (int)
--precision 16  # (16 or 32)
--seed 0  # (int)
```

One epoch takes about 5-6 minutes using a T4 GPU, which is usually free in Google Colab (depending on availability).  After about 10 training epochs, you'll reach a BLEU-4 score just over 0.30 (without beam search).  So, in under an hour, you can train a pretty good image captioning model. 😎

### Notes

BLEU doesn't increase much beyond 1 hour of training. Training and validation loss will continue to decrease, but the resulting image captions are effectively equivalent. 

This appears to be a limitation of the image embeddings, rather than a limitation of the language model. Changing the vision backbone gives the biggest improvement in BLEU score. (BLIP gets 5-10% better BLEU than CLIP backbones using the same language model head.) Larger language models (e.g. GPT-2 Large) don't improve the BLEU score by much.

## TODO

* Plan to train on Conceptual Captions for more generic image captioning.
