#!/usr/bin/env python3
"""Predict the aircraft family of one or more images with the exported ONNX model.

Usage:
    python predict.py samples/                      # every image in a folder
    python predict.py photo.jpg another.png --top 5
    python predict.py photo.jpg --json              # machine-readable output

Only needs numpy, Pillow and onnxruntime (no PyTorch).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "weights" / "aircraft_family_efficientnet_b0.onnx"
DEFAULT_LABELS = ROOT / "weights" / "labels.json"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT)
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: {p} not found, skipping", file=sys.stderr)
    return files


def preprocess(path, cfg, crop_banner=False):
    """Same preprocessing as training/evaluation: (optional banner crop), resize to 224x224, ImageNet normalisation."""
    im = Image.open(path).convert("RGB")
    if crop_banner:  # only for original FGVC-Aircraft images, which carry a 20 px copyright banner
        w, h = im.size
        im = im.crop((0, 0, w, h - 20))
    size = cfg["input_size"]
    im = im.resize((size, size), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - np.array(cfg["mean"], dtype=np.float32)) / np.array(cfg["std"], dtype=np.float32)
    return x.transpose(2, 0, 1)[None]  # NCHW


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files and/or folders")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--top", type=int, default=5, help="number of predictions to show")
    ap.add_argument("--crop-banner", action="store_true", help="crop the 20 px banner (original FGVC-Aircraft images only)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args()

    cfg = json.load(open(args.labels))
    classes = cfg["classes"]
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    files = collect_images(args.images)
    if not files:
        sys.exit("no images found")

    results = []
    for f in files:
        probs = softmax(sess.run(None, {"input": preprocess(f, cfg, args.crop_banner)})[0])[0]
        order = probs.argsort()[::-1][: args.top]
        results.append({"image": str(f), "predictions": [{"family": classes[i], "probability": round(float(probs[i]), 4)} for i in order]})

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"\n{r['image']}")
            for k, p in enumerate(r["predictions"], 1):
                print(f"  {k}. {p['family']:<24} {p['probability'] * 100:6.2f}%")


if __name__ == "__main__":
    main()
