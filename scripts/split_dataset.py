import argparse, os, random, shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(root: Path):
    classes = []
    files = []
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        classes.append(cls_dir.name)
        for img in cls_dir.rglob("*"):
            if img.suffix.lower() in IMG_EXTS:
                files.append((cls_dir.name, img))
    return classes, files

def split_copy(files, classes, dst: Path, train=0.8, val=0.1, test=0.1, seed=42):
    random.seed(seed)
    by_cls = {c: [] for c in classes}
    for c, p in files:
        by_cls[c].append(p)

    for c in classes:
        imgs = by_cls[c]
        random.shuffle(imgs)
        n = len(imgs)
        n_tr = int(n * train)
        n_va = int(n * val)
        parts = {
            "train": imgs[:n_tr],
            "val":   imgs[n_tr:n_tr+n_va],
            "test":  imgs[n_tr+n_va:],
        }
        for split, items in parts.items():
            outdir = dst / split / c
            outdir.mkdir(parents=True, exist_ok=True)
            for src in items:
                shutil.copy(src, outdir / src.name)  # faster than copy2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets", help="folder with class subfolders")
    ap.add_argument("--dst", default="data", help="output root (creates train/val/test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    classes, files = collect_images(src)
    print(f"Found {len(files)} images across {len(classes)} classes: {classes}")
    split_copy(files, classes, dst, train=args.train, val=args.val, test=args.test, seed=args.seed)
    print(f"Done. Structure ready under: {dst}/train|val|test/<class>/")
