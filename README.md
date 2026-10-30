# Aircraft Family Recognition Using Deep Learning and Grad-CAM

Classifies an aircraft photo into one of **70 aircraft families** (e.g. Boeing 737, Airbus A320, Cessna 172) with a fine-tuned **EfficientNet-B0**, and explains each prediction with **Grad-CAM** heatmaps.

**Result:** 79.96% top-1 and 94.06% top-5 accuracy on the official, untouched test set (3,333 images).

## Project links

| Item | Link |
|---|---|
| Live web app (Vercel) | https://aircraft-family-gradcam.vercel.app/ |
| Project report | https://aircraft-family-gradcam.vercel.app/report/Aircraft_Family_Recognition_Report.pdf |
| Presentation (slides) | https://aircraft-family-gradcam.vercel.app/slides/Aircraft_Family_Recognition_Slides.pptx |
| YouTube video | https://www.youtube.com/watch?v=cD7ZXYaKIM8 |
| LinkedIn post | YOUR_LINKEDIN_URL |
| Dataset (official source) | https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ |

## Dataset

- **FGVC-Aircraft** (release 2013b), a public benchmark by Maji, Rahtu, Kannala, Blaschko and Vedaldi (Oxford Visual Geometry Group), arXiv:1306.5151. Secondary dataset, downloaded automatically by `torchvision.datasets.FGVCAircraft(..., annotation_level="family", download=True)`.
- Label level: **family** (70 classes). Official split used unchanged: **3,334 train / 3,333 validation / 3,333 test** images (about 48 training images per family on average; family sizes vary, see `results/class_distribution.png`).
- Images are about 1-2 megapixels. Each carries a 20-pixel copyright banner at the bottom, which is **cropped before training** so the model cannot read it.
- The dataset is not stored in this repository (about 2.6 GB). It downloads in the notebook's first data cell. Images are for **non-commercial research and education only** (see the official page).

## Method

| Step | Detail |
|---|---|
| Models compared | MobileNetV2, ResNet18, EfficientNet-B0 (ImageNet-pretrained, all layers fine-tuned) |
| Input | banner cropped, resized to 224x224, ImageNet mean/std normalisation |
| Training augmentation | RandomResizedCrop(224, scale 0.5-1.0), horizontal flip, light colour jitter |
| Optimiser | AdamW, max lr 5e-4, weight decay 1e-2, OneCycleLR schedule |
| Loss | cross-entropy with label smoothing 0.1 |
| Epochs / batch | 20 epochs, batch size 64, mixed precision (AMP) |
| Seed | 42 |
| Model selection | best **validation** accuracy (the test set was not used for any decision) |
| Hardware | Google Colab, Tesla T4 GPU, PyTorch 2.11 |

## Results (test set, 3,333 images)

| Model | Val acc | Test top-1 | Test top-5 | Macro F1 | Params (M) | Train time |
|---|---|---|---|---|---|---|
| MobileNetV2 | 77.17% | 78.04% | 93.52% | 76.42% | 2.31 | 26.3 min |
| ResNet18 | 76.36% | 77.02% | 91.84% | 75.16% | 11.21 | 25.5 min |
| **EfficientNet-B0 (selected)** | **78.64%** | **79.96%** | **94.06%** | **78.21%** | 4.10 | 26.1 min |

- Approximate 95% confidence interval for the selected model's test accuracy: **78.6% to 81.3%** (normal approximation, n = 3,333).
- Validation (78.6%) and test (80.0%) accuracy are close, so the validation-based selection did not overfit.
- Full numbers: [`results/results.json`](results/results.json). Per-class precision/recall/F1: [`results/classification_report.txt`](results/classification_report.txt).

### Where the model makes mistakes

The most confused pairs are visually or historically close designs:

| True family | Predicted | Count |
|---|---|---|
| C-47 | DC-3 | 16 |
| Boeing 777 | Boeing 767 | 11 |
| A320 | A330 | 9 |
| MD-90 / MD-80 | MD-80 / MD-90 | 7 / 7 |
| MD-11 | DC-10 | 7 |
| Boeing 707 | Boeing 747 | 7 |
| A320 | Boeing 737 | 7 |

C-47 vs DC-3 is largely a labelling ambiguity: the C-47 is the military version of the DC-3, so the two look almost identical. See `results/confusion_matrix.png`.

## Explainability (Grad-CAM)

Grad-CAM is computed on the last convolutional block of EfficientNet-B0. Example overlays: `results/gradcam_correct.png`, `results/gradcam_errors.png`, and `gradcam_examples/`.

**Deletion test** (300 correctly classified test images): hiding the 25% of pixels Grad-CAM marks as most important reduced the true-class probability by **0.738** on average (from 0.771), while hiding the 25% marked least important reduced it by only **0.007**. The most-important region hurt more in **300 of 300** images. This supports that the heatmaps point at evidence the model actually uses. It is a sanity check, not a proof: masking also changes the image statistics, and Grad-CAM is a coarse (7x7) explanation.

## Quick start

```bash
git clone https://github.com/Muhammad-Ishtiaq-Haider/aircraft-family-gradcam
cd aircraft-family-gradcam
pip install -r requirements.txt

# predict (ONNX, CPU, no PyTorch needed at run time)
python predict.py samples/
python predict.py my_photo.jpg --top 5 --json

# Grad-CAM heatmaps (PyTorch weights)
python gradcam.py samples/ --out gradcam_out
python gradcam.py my_photo.jpg --target "Boeing 737"
```

Use `--crop-banner` with `predict.py` only for original FGVC-Aircraft images.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── predict.py                  # inference with the ONNX model
├── gradcam.py                  # Grad-CAM with the PyTorch weights
├── notebook/
│   └── Aircraft_Family_Recognition_GradCAM.ipynb   # data, training, evaluation, Grad-CAM, export
├── weights/
│   ├── aircraft_family_efficientnet_b0.pth         # PyTorch weights
│   ├── aircraft_family_efficientnet_b0.onnx        # ONNX (also used by the web app)
│   └── labels.json                                  # 70 class names + preprocessing settings
├── results/                    # results.json, classification report, all plots
├── samples/                    # 12 test images for quick tests
└── gradcam_examples/           # Grad-CAM overlays
```

## Reproduce

Open the notebook in Google Colab, select a T4 GPU, and run all cells (about 40 minutes). It downloads the dataset, trains the three models, evaluates once on the test set and writes every file listed above.

The exported ONNX model matches the PyTorch model on 100 of 100 test images (largest logit difference 4.1e-05).

## Limitations

- One training run with one seed. Differences of about one percentage point between models are within noise.
- All numbers come from the FGVC-Aircraft test split. Photos from other cameras, angles or lighting may give lower accuracy, and the model was not tested on outside images.
- Family-level labels group several variants. Near-identical designs (777 vs 767, MD-80 vs MD-90) remain the main error source.
- Input is squashed to 224x224, which distorts the aspect ratio of wide aircraft photos. Higher resolution would likely help.

## References

1. S. Maji, E. Rahtu, J. Kannala, M. Blaschko, A. Vedaldi. *Fine-Grained Visual Classification of Aircraft.* arXiv:1306.5151, 2013.
2. R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
3. M. Tan, Q. Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
4. M. Sandler et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
5. K. He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.

## License

Code: MIT. The FGVC-Aircraft images remain under their original non-commercial research licence.
