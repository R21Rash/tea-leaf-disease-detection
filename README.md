# Tea Leaf Disease Detection (Sri Lanka) — SE4050

Supervised learning project to classify Sri Lankan tea leaf diseases from images.

## Models (as required by assignment)
- Baseline CNN (from scratch)
- VGG16 (transfer learning)
- ResNet50 (transfer learning)
- MobileNetV2 (transfer learning)

## Dataset layout
Use an `ImageFolder` style directory:
```
data/
  train/
    Healthy/
    BlisterBlight/
    RedLeafSpot/
    ... (other classes)
  val/
    Healthy/
    BlisterBlight/
    ...
  test/
    Healthy/
    BlisterBlight/
    ...
```
> You can name classes as appropriate for your dataset.

## Quickstart
```bash
# 1) Create venv
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train (MobileNetV2, frozen backbone first)
python src/train.py --data_dir data --model mobilenet_v2 --epochs 10 --freeze_backbone

# 4) Fine-tune (unfreeze last blocks)
python src/train.py --data_dir data --model mobilenet_v2 --epochs 10 --unfreeze --ckpt outputs/best_model.pth

# 5) Evaluate on test set
python src/train.py --data_dir data --model mobilenet_v2 --evaluate --ckpt outputs/best_model.pth
```

## Reproducible configs
You can also run with a YAML config:
```bash
python src/train.py --config config.yaml
```

## Repo structure
```
src/
  train.py
  models/
    factory.py
    baseline_cnn.py
notebooks/
data/
scripts/
reports/
experiments/
```

## License
MIT
