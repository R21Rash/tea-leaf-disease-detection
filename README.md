# Tea Leaf Disease Detection (Sri Lanka) — SE4050

Supervised learning project to classify Sri Lankan tea leaf diseases from images.

Dataset: [TeaLeafBD — Kaggle](https://www.kaggle.com/datasets/bmshahriaalam/tealeafbd-tea-leaf-disease-detection)

---

## Models (as required by assignment)

- **Baseline CNN** (from scratch)
- **VGG16** (transfer learning)
- **ResNet50** (transfer learning)
- **MobileNetV2** (transfer learning) ✅ trained and tested

---

## Dataset layout

Use an **ImageFolder** style directory (created after splitting):

```
data/
  train/
    Healthy/
    BrownBlight/
    GrayBlight/
    RedSpider/
    ...
  val/
    Healthy/
    BrownBlight/
    ...
  test/
    Healthy/
    BrownBlight/
    ...
```

> The Kaggle dataset provides raw class folders.  
> Use the provided `scripts/split_dataset.py` to split into `train/val/test`.

---

## Quickstart

```bash
# 1) Create venv
python -m venv .venv
# On Linux/Mac
source .venv/bin/activate
# On Windows
.venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Download dataset (manual or API)
# Place the Kaggle zip under datasets/ and extract:
kaggle datasets download -d bmshahriaalam/tealeafbd-tea-leaf-disease-detection -p datasets
unzip datasets/tealeafbd-tea-leaf-disease-detection.zip -d data/raw

# 4) Split into train/val/test
python scripts/split_dataset.py --src data/raw --dst data

# 5) Train (MobileNetV2, frozen backbone first)
python src/train.py --data_dir data --model mobilenet_v2 --epochs 10 --freeze_backbone

# (example run on Windows with AMP)
python src\train.py --data_dir data --model mobilenet_v2 --epochs 3 --freeze_backbone --amp

# 6) Fine-tune (unfreeze last blocks)
python src/train.py --data_dir data --model mobilenet_v2 --epochs 10 --unfreeze --ckpt outputs/best_model.pth

# 7) Evaluate on test set
python src/train.py --data_dir data --model mobilenet_v2 --evaluate --ckpt outputs/best_model.pth
```

---

## Reproducible configs

You can also run with a YAML config:

```bash
python src/train.py --config config.yaml
```

---

## Repo structure

```
src/
  train.py
  models/
    factory.py
    baseline_cnn.py
scripts/
  split_dataset.py
data/
  raw/
  train/
  val/
  test/
outputs/
reports/
notebooks/
experiments/
```

---

## License

MIT
