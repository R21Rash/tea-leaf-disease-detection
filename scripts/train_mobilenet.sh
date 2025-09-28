#!/usr/bin/env bash
set -euo pipefail
python src/train.py --data_dir data --model mobilenet_v2 --epochs 10 --batch_size 32 --freeze_backbone --amp
