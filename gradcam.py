#!/usr/bin/env python3
"""Generate Grad-CAM heatmaps for aircraft images using the trained EfficientNet-B0 weights (.pth).

Usage:
    python gradcam.py samples/ --out gradcam_out
    python gradcam.py photo.jpg --target "Boeing 737"   # explain a specific family instead of the top prediction
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model(num_classes):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) on the last convolutional block."""

    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.acts = out.detach()
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        logits[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().numpy(), logits.detach()[0].softmax(0), class_idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", default=str(ROOT / "weights" / "aircraft_family_efficientnet_b0.pth"))
    ap.add_argument("--labels", default=str(ROOT / "weights" / "labels.json"))
    ap.add_argument("--out", default="gradcam_out")
    ap.add_argument("--target", default=None, help="family name to explain (default: top prediction)")
    args = ap.parse_args()

    cfg = json.load(open(args.labels))
    classes, size = cfg["classes"], cfg["input_size"]
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)

    model = build_model(len(classes))
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    cam_engine = GradCAM(model, model.features[-1])

    target = None
    if args.target:
        if args.target not in classes:
            sys.exit(f"unknown family '{args.target}'. Choose from: {', '.join(classes)}")
        target = classes.index(args.target)

    files = []
    for p in map(Path, args.images):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p]
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for f in files:
        img = Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR)
        rgb = np.asarray(img, dtype=np.float32) / 255.0
        x = ((torch.from_numpy(rgb).permute(2, 0, 1) - mean) / std).unsqueeze(0)
        cam, prob, ci = cam_engine(x, target)
        heat = plt.cm.jet(cam)[..., :3]
        overlay = np.clip(0.5 * rgb + 0.5 * heat, 0, 1)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4.2))
        ax[0].imshow(rgb); ax[0].axis("off"); ax[0].set_title("input")
        ax[1].imshow(overlay); ax[1].axis("off")
        ax[1].set_title(f"Grad-CAM: {classes[ci]} ({prob[ci] * 100:.1f}%)")
        out = Path(args.out) / f"{f.stem}_gradcam.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"{f} -> {out}  [{classes[ci]} {prob[ci] * 100:.1f}%]")


if __name__ == "__main__":
    main()
