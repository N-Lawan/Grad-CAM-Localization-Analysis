# Detection-Based Image Classification and Grad-CAM Localization Analysis

This repository provides Python code for:

1. **Detection-based image-level classification** using a two-step confidence-weighted any-positive rule applied to object detection model outputs (YOLOv26 and RT-DETR).
2. **Quantitative evaluation of Grad-CAM localization** relative to manually annotated diagnostic regions (human ground truth).

The code was developed for classification and interpretability analysis of Gram-stained vaginal discharge microscopy images across five diagnostic categories: *Trichomonas vaginalis* (TV), gonococcal/Urethritis infection (GU), bacterial vaginosis (BV), healthy lactobacilli (HL), and vulvovaginal candidiasis (VVC).

---

## Part 1: Detection-Based Image Classification

### Overview

Object detection models inherently produce bounding box predictions rather than image-level diagnoses. This repository implements a **two-step confidence-weighted any-positive rule** to derive image-level diagnostic labels from instance-level bounding box detections, consistent with the any-positive assumption of Multiple Instance Learning (MIL) (Dietterich et al., 1997; Maron & Lozano-Pérez, 1998) and standard clinical microscopic practice, where a single observation of a diagnostic morphological feature is sufficient for diagnosis (Amsel et al., 1983).

### Two-Step Classification Rule

Image-level labels are assigned as follows:

**Step 1 — Any-Positive Detection:**

All predicted bounding boxes exceeding a model-specific confidence threshold are collected. Any diagnostic class with at least one box above this threshold is considered detected within the image:

$$
\text{Detected classes} = \{ c \mid \exists \, b \in \mathcal{B}_c : \text{conf}(b) \geq \tau \}
$$

where $\mathcal{B}_c$ is the set of predicted boxes for class $c$ and $\tau$ is the confidence threshold ($\tau = 0.10$ for YOLOv26; $\tau = 0.25$ for RT-DETR).

**Step 2 — Confidence-Weighted Selection:**

Among all detected classes, the mean confidence score is computed across all boxes of each class. The image is assigned the class with the highest mean confidence as the final image-level label:

$$
\hat{y} = \underset{c \in \text{Detected}}{\arg\max} \; \frac{1}{|\mathcal{B}_c^{+}|} \sum_{b \in \mathcal{B}_c^{+}} \text{conf}(b)
$$

where $\mathcal{B}_c^{+}$ is the set of boxes for class $c$ exceeding threshold $\tau$.

Images in which no bounding box exceeds the confidence threshold are recorded as **non-detected (indeterminate)** and are recommended for referral to manual microscopic review in clinical deployment.

### Confidence Threshold Selection

Confidence thresholds were selected empirically via a sweep from 0.05 to 0.50 on the test set. For each threshold, classification performance was evaluated under two analytical approaches:

- **Detected images only** — conditional performance among images with at least one detection.
- **All 514 images** — non-detected images scored as incorrect (Analysis 1; primary reported analysis).

Thresholds were selected to maximize image coverage (minimize non-detections) while maintaining stable classification performance, consistent with the clinical priority of minimizing missed diagnoses.

### Evaluation

Classification performance is reported under **Analysis 1 (all 514 images)**, with non-detected images scored as incorrect, providing a conservative and clinically representative performance estimate. Metrics include:

- Accuracy, F1 macro, F1 weighted, Cohen's Kappa
- Per-class sensitivity and specificity
- Mean AUC (one-vs-rest, using per-class mean confidence scores as ranking scores)
- Confusion matrix (including non-detection column)
- ROC curves (one per diagnostic class)

### Confidence Thresholds

| Model | Detection threshold (τ) |
|---|---|
| YOLOv26 | 0.10 |
| RT-DETR | 0.25 |

### Input

```python
MODEL_CONFIGS = {
    "yolo": {
        "model_path": "weights/yolo_model.pt",
        "det_conf":   0.10,
        "cls_conf":   0.10,
    },
    "rtdetr": {
        "model_path": "weights/rtdetr_model.pt",
        "det_conf":   0.25,
        "cls_conf":   0.25,
    },
}
IMAGE_DIR = "data/test/images/"
LABEL_DIR = "data/test/labels/"   # YOLO format .txt annotation files
```

### Output

| File | Content |
|---|---|
| `results_{model}.csv` | Per-image predictions with per-class confidence scores |
| `results_combined.csv` | Combined results for both models |
| `threshold_sweep_{model}.csv` | Threshold sweep results (both analytical approaches) |
| `threshold_sweep_{model}.png` | Threshold sweep plot |
| `confusion_matrix_{model}.png` | Confusion matrix including non-detection column |
| `roc_curves_{model}.png` | OvR ROC curves across all 514 images |
| `summary_metrics_{model}.csv` | Overall metric summary |

### References

- Dietterich TG, Lathrop RH, Lozano-Pérez T. Solving the multiple instance problem with axis-parallel rectangles. *Artificial Intelligence*. 1997;89(1–2):31–71.
- Maron O, Lozano-Pérez T. A framework for multiple-instance learning. *Advances in NIPS 10*. 1998:570–576.
- Campanella G, et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. *Nature Medicine*. 2019;25(8):1301–1309.
- Amsel R, et al. Nonspecific vaginitis: diagnostic criteria and microbial and epidemiologic associations. *American Journal of Medicine*. 1983;74(1):14–22.

---

## Part 2: Quantitative Grad-CAM Localization Analysis

### Overview

The analysis quantifies how strongly Grad-CAM activation is concentrated within a ground-truth diagnostic region, providing quantitative interpretability evaluation of the detection-based classification framework.

The primary metric is **activation enrichment**, which compares the proportion of Grad-CAM activation located inside the ground-truth region with the proportion of the image occupied by that region.

### Activation Enrichment

Activation enrichment is calculated independently for each image:

$$
\text{Activation enrichment} =
\frac{\text{Activation inside GT}}
{\text{GT area fraction}}
$$

where:

- **Activation inside GT** is the proportion of total Grad-CAM activation contained within the ground-truth region.
- **GT area fraction** is the proportion of image pixels occupied by the ground-truth region.

Interpretation:

- **1.0×** — Grad-CAM activation is proportional to the area of the ground-truth region.
- **>1.0×** — Grad-CAM activation is more concentrated within the ground-truth region than expected from its area alone.
- **<1.0×** — Grad-CAM activation is less concentrated within the ground-truth region than expected from its area alone.

Activation enrichment is calculated **per image first**, followed by computation of mean and median enrichment across images.

### Quantitative Metrics

The code calculates:

1. Activation inside ground truth
2. Ground-truth area fraction
3. Activation enrichment
4. Pointing accuracy
5. Mean activation enrichment
6. Median activation enrichment
7. Class-specific summary statistics
8. Overall summary statistics

### Pointing Accuracy

Pointing accuracy is defined as the proportion of images in which the pixel with maximum Grad-CAM activation is located within the ground-truth region.

### Input

The analysis requires:

- A precomputed Grad-CAM activation map for each image
- A corresponding ground-truth region mask
- An image identifier
- The diagnostic category

The repository contains **code only**. Patient images, ground-truth annotations, and other study-specific image data are not included.

Example input structure:

```python
image_records = [
    {
        "image_id": "example_001",
        "class": "VVC",
        "cam": cam_array,
        "gt_mask": gt_mask_array
    }
]
```

### Output

- Image-level quantitative results
- Diagnostic-category-level summary statistics
- Overall summary statistics

Results can be exported as CSV files for further statistical analysis or table preparation.

---

## Computational Environment

The classification pipeline requires:

```
ultralytics
scikit-learn
pandas
matplotlib
seaborn
numpy
```

The Grad-CAM localization analysis requires NumPy and pandas only, and is independent of the Grad-CAM generation framework — operating on precomputed activation maps and ground-truth masks.

## Reproducibility

To reproduce the classification analysis, users should provide their own trained model weights and test dataset following the input structure described above.

To reproduce the Grad-CAM localization analysis, users should provide their own precomputed Grad-CAM activation maps and corresponding ground-truth masks.

The deposited code does not provide access to the original study images, annotations, or trained model weights.
