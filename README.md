# clip-text-decoder

Generate text captions for images from their CLIP embeddings.  Includes PyTorch model code and example training script.

## Training

Launch your own training session using the provided script (`train.py`):
```bash
python train.py --max-epochs 5
```

Training CLI arguments, along with their default values:
```bash
--max-epochs 5  # (int)
--num-layers 6  # (int)
--dim-feedforward 256  # (int)
--precision 16  # (16 or 32)
--seed 0  # (int)
```


## Inference

The training script will produce a `model.zip` archive, containing the `Tokenizer` and trained model parameters.  To perform inference with it:
```python
import clip
from PIL import Image
import torch

from src.model import ClipDecoderInferenceModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ClipDecoderInferenceModel.load("path/to/model.zip").to(device)
clip_model, clip_preprocessor = clip.load("ViT-B/32", device=device, jit=False)

# Create a blank dummy image
dummy_image = Image.new("RGB", (224, 224))
preprocessed = clip_preprocessor(dummy_image).to(device)
# Add a batch dimension using '.unsqueeze(0)'
encoded = clip_model.encode_image(preprocessed.unsqueeze(0))
text = model(encoded)

print(text)
# Probably some nonsense, because we used a dummy image.
```


## Pretrained Models

I've pretrained a few models, which are hosted in Google Drive:
* [https://drive.google.com/file/d/1-6rt3Yb4y-F84wSwzkqeOj6mdfHv0dG0/view?usp=sharing](clip-text-decoder-5-epochs.zip)
<!-- * [https://drive.google.com/file/d/1-6rt3Yb4y-F84wSwzkqeOj6mdfHv0dG0/view?usp=sharing](clip-text-decoder-10-epochs.zip)
* [https://drive.google.com/file/d/1-6rt3Yb4y-F84wSwzkqeOj6mdfHv0dG0/view?usp=sharing](clip-text-decoder-20-epochs.zip) -->


## Shortcomings

* Only works well with COCO-style images.  If you go outside the distribution of COCO objects, you'll get nonsense text captions.
* Relatively short training time.  Even within the COCO domain, you'll occasionally see incorrect captions.  Quite a few captions will have bad grammar, repetitive descriptors, etc.
* 
