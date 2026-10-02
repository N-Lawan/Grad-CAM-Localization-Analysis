"""
detection_based_classification.py
==================================
Detection-Based Image-Level Classification using a
Two-Step Confidence-Weighted Any-Positive Rule.

This code accompanies the manuscript:
    "Interpretable Detection-Based Image Classification for Differential Diagnosis of Vaginal Discharge Using Smartphone-Assisted Microscopy"

Description
-----------
Object detection models (YOLO, RT-DETR) are repurposed for
image-level diagnostic classification using a two-step rule
consistent with the any-positive assumption of Multiple Instance
Learning (MIL):

    Step 1 — Any-Positive Detection:
        All predicted bounding boxes exceeding a model-specific
        confidence threshold are collected. Any diagnostic class
        with at least one box above this threshold is considered
        detected within the image.

    Step 2 — Confidence-Weighted Selection:
        The mean confidence score is computed across all detected
        boxes for each class. The image is assigned the class
        with the highest mean confidence as the final image-level
        label.

Classes
-------
    0: TV  — Trichomonas vaginalis
    1: GU  — Gonococcal/Urethritis infection
    2: BV  — Bacterial vaginosis
    3: HL  — Healthy lactobacilli (normal flora)
    4: VVC — Vulvovaginal candidiasis

Confidence Thresholds (empirically selected via threshold sweep)
---------------------------------------------------------------
    YOLO   : conf >= 0.10
    RT-DETR: conf >= 0.25

Requirements
------------
    pip install ultralytics scikit-learn pandas matplotlib seaborn

References
----------
    Dietterich et al. (1997). Artificial Intelligence, 89(1-2), 31-71.
    Maron & Lozano-Perez (1998). Advances in NIPS 10, 570-576.
    Campanella et al. (2019). Nature Medicine, 25(8), 1301-1309.
    Amsel et al. (1983). American Journal of Medicine, 74(1), 14-22.


"""

# ============================================================
# Imports
# ============================================================

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.preprocessing import label_binarize


# ============================================================
# Configuration
# ============================================================

# Diagnostic class mapping (YOLO format: integer index -> class name)
CLASS_NAMES = {
    0: "TV",
    1: "GU",
    2: "BV",
    3: "HL",
    4: "VVC",
}

# Model-specific confidence thresholds selected via threshold sweep
# (see sweep_confidence_thresholds() below)
MODEL_CONFIGS = {
    "yolo": {
        "model_path": "YOUR BEST MODEL.pt",
        "det_conf":   0.10,   # Detection inference threshold
        "cls_conf":   0.10,   # Classification aggregation threshold
    },
    "rtdetr": {
        "model_path": "YOUR BEST MODEL.pt",
        "det_conf":   0.25,
        "cls_conf":   0.25,
    },
}

# Dataset paths
IMAGE_DIR  = "YOUR IMAGE "
LABEL_DIR  = "YOUR LABEL"
OUTPUT_DIR = "YOUR OUTPUT"


# ============================================================
# Step 1: Ground Truth Loading
# ============================================================

def load_ground_truth(label_path):
    """
    Load the ground-truth diagnostic class from a YOLO-format
    annotation file (.txt).

    In single-label images, the ground-truth class is determined
    by the most frequently annotated class within the image,
    consistent with the proof-of-concept single-infection design
    of this study.

    Parameters
    ----------
    label_path : str
        Path to the YOLO annotation .txt file.
        Each line format: class_id cx cy w h

    Returns
    -------
    int or None
        Most frequent class_id in the annotation file,
        or None if the file is missing or empty.
    """
    if not os.path.exists(label_path):
        return None

    class_ids = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_ids.append(int(parts[0]))

    if not class_ids:
        return None

    # Return the most frequently annotated class
    return max(set(class_ids), key=class_ids.count)


# ============================================================
# Step 2: Two-Step Confidence-Weighted Any-Positive Rule
# ============================================================

def collect_class_scores(detections):
    """
    Collect detection confidence scores grouped by class
    from all predicted bounding boxes in an image.

    This implements the foundation of the any-positive rule:
    every detected box contributes its confidence score to
    its respective class pool.

    Parameters
    ----------
    detections : ultralytics.engine.results.Boxes
        Bounding box predictions from a YOLO/RT-DETR inference result.

    Returns
    -------
    dict
        Dictionary mapping class_id (int) to a list of
        confidence scores (float) for all detected boxes
        of that class.
        Example: {0: [0.85, 0.80], 2: [0.45]}
    """
    class_scores = {}
    for box in detections:
        cls  = int(box.cls.item())
        conf = float(box.conf.item())
        class_scores.setdefault(cls, []).append(conf)
    return class_scores


def apply_confidence_threshold(class_scores, cls_conf):
    """
    Step 1 — Any-Positive Detection:
        Filter class scores to retain only classes with at least
        one bounding box whose confidence meets or exceeds the
        classification threshold.

        A class is considered "detected" in the image if at least
        one of its predicted boxes passes this threshold, consistent
        with the any-positive assumption of MIL (Dietterich et al.,
        1997; Maron & Lozano-Perez, 1998).

    Parameters
    ----------
    class_scores : dict
        Raw class scores from collect_class_scores().
    cls_conf : float
        Minimum confidence threshold for a box to be considered
        a valid detection (YOLO: 0.10, RT-DETR: 0.25).

    Returns
    -------
    dict
        Filtered dictionary containing only classes where at
        least one box meets the confidence threshold.
        Returns empty dict if no class passes the threshold
        (image will be recorded as non-detected/indeterminate).
    """
    return {
        cls: [c for c in scores if c >= cls_conf]
        for cls, scores in class_scores.items()
        if any(c >= cls_conf for c in scores)
    }


def select_image_label(eligible_class_scores):
    """
    Step 2 — Confidence-Weighted Selection:
        Among all detected classes (those passing Step 1),
        compute the mean confidence score per class and assign
        the image-level label to the class with the highest
        mean confidence.

        This step resolves multi-class detections by selecting
        the most confidently detected diagnostic class as the
        final image-level label.

    Parameters
    ----------
    eligible_class_scores : dict
        Filtered class scores from apply_confidence_threshold().
        Must be non-empty.

    Returns
    -------
    tuple (int, float)
        - best_class_id : int
            Class index of the selected image-level label.
        - best_mean_conf : float
            Mean confidence score of the selected class (rounded
            to 4 decimal places).
    """
    mean_scores = {
        cls: sum(scores) / len(scores)
        for cls, scores in eligible_class_scores.items()
    }
    best_class_id   = max(mean_scores, key=mean_scores.get)
    best_mean_conf  = round(mean_scores[best_class_id], 4)
    return best_class_id, best_mean_conf


def classify_image(detections, cls_conf):
    """
    Apply the full two-step confidence-weighted any-positive
    classification rule to a single image.

    Two-Step Rule Summary:
        Step 1: Collect all boxes above cls_conf threshold
                (any-positive detection).
        Step 2: Select class with highest mean confidence
                (confidence-weighted selection).

    If no boxes exceed the threshold, the image is recorded
    as non-detected (indeterminate) and recommended for
    manual microscopic review in clinical deployment.

    Parameters
    ----------
    detections : ultralytics.engine.results.Boxes
        Bounding box predictions from model inference.
    cls_conf : float
        Classification confidence threshold.

    Returns
    -------
    tuple (int or None, float, dict)
        - pred_class_id : int or None
            Predicted class index, or None if non-detected.
        - pred_confidence : float
            Mean confidence of predicted class (0.0 if non-detected).
        - per_class_conf : dict
            Mean confidence per class across all detected boxes
            (0.0 for classes with no detections). Used for
            proper OvR AUC computation.
    """
    # Collect raw scores for all boxes
    all_class_scores = collect_class_scores(detections)

    # Compute per-class mean confidence for all 5 classes
    # (including below-threshold classes, for AUC scoring)
    per_class_conf = {
        cls_id: round(sum(all_class_scores.get(cls_id, [])) /
                      len(all_class_scores[cls_id]), 4)
                if cls_id in all_class_scores else 0.0
        for cls_id in CLASS_NAMES.keys()
    }

    # Step 1: Apply confidence threshold (any-positive filter)
    eligible = apply_confidence_threshold(all_class_scores, cls_conf)

    # If no class passes threshold → non-detected (indeterminate)
    if not eligible:
        return None, 0.0, per_class_conf

    # Step 2: Select class with highest mean confidence
    pred_class_id, pred_confidence = select_image_label(eligible)

    return pred_class_id, pred_confidence, per_class_conf


# ============================================================
# Step 3: Inference Pipeline
# ============================================================

def run_inference(model_name, config, image_paths):
    """
    Run the full detection-based image-level classification
    pipeline for a single model across all test images.

    For each image:
        1. Load ground-truth label from YOLO annotation file.
        2. Run object detection inference.
        3. Apply two-step classification rule.
        4. Record results including per-class confidence scores
           for proper OvR AUC computation.

    Parameters
    ----------
    model_name : str
        Model identifier ("yolo" or "rtdetr").
    config : dict
        Model configuration from MODEL_CONFIGS.
    image_paths : list of Path
        List of image file paths to process.

    Returns
    -------
    list of dict
        One record per image containing: model, image filename,
        ground-truth class, predicted class, confidence,
        correctness flag, and per-class confidence scores.
    """
    print(f"\n{'='*60}")
    print(f"Model : {model_name.upper()}")
    print(f"det_conf = {config['det_conf']} | cls_conf = {config['cls_conf']}")
    print(f"{'='*60}")

    model = YOLO(config["model_path"])
    rows  = []

    for img_path in image_paths:

        # Load ground truth
        label_path = os.path.join(LABEL_DIR, img_path.stem + ".txt")
        gt_id      = load_ground_truth(label_path)
        gt_name    = CLASS_NAMES.get(gt_id, "unknown") if gt_id is not None else "unknown"

        # Run detection inference
        result   = model(str(img_path), conf=config["det_conf"], verbose=False)[0]

        # Apply two-step classification rule
        pred_id, pred_conf, per_class_conf = classify_image(
            result.boxes, config["cls_conf"]
        )

        pred_name = CLASS_NAMES.get(pred_id, "no_detection") \
                    if pred_id is not None else "no_detection"

        rows.append({
            "model":      model_name,
            "image":      img_path.name,
            "gt_class":   gt_name,
            "pred_class": pred_name,
            "confidence": pred_conf,
            # All 514 images scored: non-detected = incorrect (Analysis 1)
            "correct":    int(gt_name == pred_name),
            # Per-class confidence scores for proper OvR AUC computation
            "conf_TV":    per_class_conf[0],
            "conf_GU":    per_class_conf[1],
            "conf_BV":    per_class_conf[2],
            "conf_HL":    per_class_conf[3],
            "conf_VVC":   per_class_conf[4],
        })

        print(f"  {img_path.name:40s} | GT: {gt_name:5s} | "
              f"Pred: {pred_name:12s} | Conf: {pred_conf:.3f}")

    return rows


# ============================================================
# Step 4: Confidence Threshold Sweep
# ============================================================

def sweep_confidence_thresholds(df, class_names, model_name,
                                thresholds=None, output_dir=None):
    """
    Sweep confidence thresholds to empirically select the
    optimal operating threshold for each model.

    For each threshold, performance is evaluated under two
    analytical approaches:
        - Detected images only (conditional performance).
        - All 514 images with non-detections scored as incorrect
          (Analysis 1 — conservative, clinically representative).

    The selected threshold maximises image coverage (minimises
    non-detections) while maintaining stable classification
    performance, consistent with the clinical priority of
    minimising missed diagnoses.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe from run_inference().
    class_names : list of str
        Ordered list of class name strings.
    model_name : str
        Model identifier for labelling output.
    thresholds : list of float, optional
        Confidence thresholds to evaluate.
        Default: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    output_dir : str, optional
        Directory to save sweep CSV and plot.

    Returns
    -------
    pd.DataFrame
        Sweep results with F1, accuracy, coverage, and
        non-detection count for each threshold.
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    n_total = len(df)
    results = []

    print(f"\n{'='*60}")
    print(f"Threshold sweep: {model_name.upper()}")
    print(f"{'='*60}")

    for thresh in thresholds:

        df_t = df.copy()
        df_t.loc[df_t["confidence"] < thresh, "pred_class"] = "no_detection"

        # Conditional: detected images only
        df_det   = df_t[df_t["pred_class"] != "no_detection"]
        f1_cond  = f1_score(df_det["gt_class"], df_det["pred_class"],
                            labels=class_names, average="macro",
                            zero_division=0) if len(df_det) > 0 else 0.0
        acc_cond = accuracy_score(df_det["gt_class"],
                                  df_det["pred_class"]) if len(df_det) > 0 else 0.0

        # Analysis 1: all images, non-detection = incorrect
        y_true_all = df_t["gt_class"].tolist()
        y_pred_all = df_t["pred_class"].tolist()
        f1_all     = f1_score(y_true_all, y_pred_all,
                              labels=class_names, average="macro", zero_division=0)
        acc_all    = sum(p == g for p, g in zip(y_pred_all, y_true_all)) / n_total

        coverage = len(df_det) / n_total
        no_det   = n_total - len(df_det)

        results.append({
            "model":       model_name,
            "threshold":   thresh,
            "f1_cond":     round(f1_cond, 4),
            "acc_cond":    round(acc_cond, 4),
            "f1_all514":   round(f1_all, 4),
            "acc_all514":  round(acc_all, 4),
            "coverage":    round(coverage, 4),
            "no_det":      no_det,
        })

        print(f"  conf={thresh:.2f} | "
              f"F1(cond)={f1_cond:.4f} | Acc(cond)={acc_cond:.4f} | "
              f"F1(all514)={f1_all:.4f} | Acc(all514)={acc_all:.4f} | "
              f"Coverage={coverage:.1%} | No-det={no_det}")

    sweep_df = pd.DataFrame(results)

    # Save and plot if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        sweep_df.to_csv(
            os.path.join(output_dir, f"threshold_sweep_{model_name}.csv"),
            index=False
        )
        _plot_threshold_sweep(sweep_df, model_name, output_dir)

    return sweep_df


def _plot_threshold_sweep(sweep_df, model_name, output_dir):
    """
    Plot F1, accuracy, and image coverage across confidence
    thresholds for visual threshold selection.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output from sweep_confidence_thresholds().
    model_name : str
        Model identifier for plot title and filename.
    output_dir : str
        Directory to save the plot.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(sweep_df["threshold"], sweep_df["f1_cond"],
             "b-o",  label="F1 (detected only)")
    ax1.plot(sweep_df["threshold"], sweep_df["f1_all514"],
             "b--o", label="F1 (all 514)")
    ax1.plot(sweep_df["threshold"], sweep_df["acc_cond"],
             "g-s",  label="Acc (detected only)")
    ax1.plot(sweep_df["threshold"], sweep_df["acc_all514"],
             "g--s", label="Acc (all 514)")
    ax2.plot(sweep_df["threshold"], sweep_df["coverage"],
             "r--^", label="Coverage")

    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("F1 / Accuracy")
    ax2.set_ylabel("Image Coverage", color="red")
    ax1.set_title(f"{model_name.upper()} — Confidence Threshold Sweep")
    ax1.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"threshold_sweep_{model_name}.png"),
        dpi=150
    )
    plt.close()


# ============================================================
# Step 5: Evaluation Metrics
# ============================================================

def compute_metrics(df, class_names, model_name, output_dir):
    """
    Compute and report image-level classification performance
    metrics under Analysis 1 (all 514 images, non-detected
    images scored as incorrect).

    Metrics reported:
        - Accuracy, F1 Macro, F1 Weighted, Cohen's Kappa
        - Per-class Sensitivity and Specificity
        - Mean AUC (one-vs-rest, using per-class confidence scores)
        - Confusion matrix (including non-detection column)
        - ROC curves (one per diagnostic class)

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe from run_inference().
    class_names : list of str
        Ordered list of class name strings.
    model_name : str
        Model identifier for output filenames and titles.
    output_dir : str
        Directory to save figures and summary CSV.

    Returns
    -------
    dict
        Summary of all computed metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # All 514 images — non-detection scored as incorrect
    y_true = df["gt_class"].tolist()
    y_pred = df["pred_class"].tolist()

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------
    acc     = sum(p == g for p, g in zip(y_pred, y_true)) / len(y_true)
    kappa   = cohen_kappa_score(
                  y_true,
                  [p if p in class_names else "no_detection" for p in y_pred]
              )
    f1_mac  = f1_score(y_true, y_pred, labels=class_names,
                       average="macro", zero_division=0)
    f1_wtd  = f1_score(y_true, y_pred, labels=class_names,
                       average="weighted", zero_division=0)
    no_det  = (df["pred_class"] == "no_detection").sum()

    print(f"\n{'='*50}")
    print(f"Model            : {model_name.upper()}")
    print(f"Total images     : {len(df)}")
    print(f"Non-detected     : {no_det} ({100*no_det/len(df):.1f}%)")
    print(f"{'='*50}")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Cohen's Kappa    : {kappa:.4f}")
    print(f"F1 Macro         : {f1_mac:.4f}")
    print(f"F1 Weighted      : {f1_wtd:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred,
                                labels=class_names, zero_division=0))

    # --------------------------------------------------------
    # Per-class sensitivity and specificity
    # --------------------------------------------------------
    cm_full = confusion_matrix(y_true, y_pred,
                               labels=class_names + ["no_detection"])
    print(f"\n{'Class':<8} {'Sensitivity':>12} {'Specificity':>12} {'Support':>10}")
    print("-" * 46)
    for i, cls in enumerate(class_names):
        TP = cm_full[i, i]
        FN = cm_full[i, :].sum() - TP
        FP = cm_full[:, i].sum() - TP
        TN = cm_full.sum() - TP - FN - FP
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        print(f"{cls:<8} {sens:>12.4f} {spec:>12.4f} {(TP+FN):>10}")

    # --------------------------------------------------------
    # Confusion matrix figure
    # --------------------------------------------------------
    _plot_confusion_matrix(cm_full, class_names, model_name, output_dir)

    # --------------------------------------------------------
    # OvR ROC curves using per-class confidence scores
    # Non-detected images assigned zero confidence for all classes
    # --------------------------------------------------------
    conf_cols    = [f"conf_{cls}" for cls in class_names]
    score_matrix = df[conf_cols].fillna(0.0).values
    y_true_bin   = label_binarize(y_true, classes=class_names)

    mean_auc = _plot_roc_curves(
        y_true_bin, score_matrix, class_names, model_name, output_dir
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------
    summary = {
        "model":          model_name,
        "n_total":        len(df),
        "n_no_detection": int(no_det),
        "accuracy":       round(acc, 4),
        "kappa":          round(kappa, 4),
        "f1_macro":       round(f1_mac, 4),
        "f1_weighted":    round(f1_wtd, 4),
        "mean_auc":       round(mean_auc, 4),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, f"summary_metrics_{model_name}.csv"),
        index=False
    )
    print(f"\nMean AUC : {mean_auc:.4f}")
    print(f"Saved    : {output_dir}")

    return summary


def _plot_confusion_matrix(cm_full, class_names, model_name, output_dir):
    """
    Plot and save the confusion matrix including a
    non-detection column to represent missed images.

    Parameters
    ----------
    cm_full : np.ndarray
        Confusion matrix from sklearn including no_detection label.
    class_names : list of str
        Ordered list of class name strings.
    model_name : str
        Model identifier for title and filename.
    output_dir : str
        Directory to save the figure.
    """
    # Show GT classes as rows; predicted classes + no_det as columns
    cm_display  = cm_full[:len(class_names), :]
    col_labels  = class_names + ["no_det"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_display, annot=True, fmt="d", cmap="Blues",
                xticklabels=col_labels, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(
        f"Confusion Matrix — {model_name.upper()} (All 514 Images)",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"confusion_matrix_{model_name}.png"),
        dpi=150
    )
    plt.close()


def _plot_roc_curves(y_true_bin, score_matrix, class_names,
                     model_name, output_dir):
    """
    Plot and save one-vs-rest (OvR) ROC curves for all
    diagnostic classes using per-class mean confidence scores.

    Non-detected images are included with zero confidence
    scores for all classes, providing a conservative AUC
    estimate across all 514 test images.

    Parameters
    ----------
    y_true_bin : np.ndarray
        Binarized ground-truth labels (n_images x n_classes).
    score_matrix : np.ndarray
        Per-class confidence scores (n_images x n_classes).
    class_names : list of str
        Ordered list of class name strings.
    model_name : str
        Model identifier for title and filename.
    output_dir : str
        Directory to save the figure.

    Returns
    -------
    float
        Mean AUC across all classes.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    auc_scores = []

    for i, cls in enumerate(class_names):
        if y_true_bin[:, i].sum() == 0:
            continue
        if np.max(score_matrix[:, i]) > np.min(score_matrix[:, i]):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], score_matrix[:, i])
            auc = roc_auc_score(y_true_bin[:, i], score_matrix[:, i])
            auc_scores.append(auc)
            ax.plot(fpr, tpr, label=f"{cls} (AUC = {auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"ROC Curves — {model_name.upper()} (All 514 Images, OvR)",
        fontsize=13
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"roc_curves_{model_name}.png"),
        dpi=150
    )
    plt.close()

    return float(np.mean(auc_scores)) if auc_scores else 0.0


# ============================================================
# Main Pipeline
# ============================================================

def main():
    """
    Execute the full detection-based image classification pipeline:

        1. Load test images and ground-truth labels.
        2. Run inference for each model.
        3. Apply two-step confidence-weighted any-positive rule.
        4. Perform confidence threshold sweep.
        5. Compute and save evaluation metrics and figures.

    All results are saved to OUTPUT_DIR.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all test images
    image_paths = sorted(Path(IMAGE_DIR).glob("*.*"))
    image_paths = [
        p for p in image_paths
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    ]
    print(f"Found {len(image_paths)} test images.")

    class_name_list = list(CLASS_NAMES.values())   # ["TV","GU","BV","HL","VVC"]
    all_rows        = []
    fieldnames      = [
        "model", "image", "gt_class", "pred_class", "confidence", "correct",
        "conf_TV", "conf_GU", "conf_BV", "conf_HL", "conf_VVC"
    ]

    for model_name, config in MODEL_CONFIGS.items():

        # --------------------------------------------------
        # Run inference and apply two-step classification rule
        # --------------------------------------------------
        rows = run_inference(model_name, config, image_paths)
        all_rows.extend(rows)

        # Save per-model results CSV
        model_csv = os.path.join(OUTPUT_DIR, f"results_{model_name}.csv")
        with open(model_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        df = pd.DataFrame(rows)

        # Summary
        total   = len(rows)
        correct = df["correct"].sum()
        no_det  = (df["pred_class"] == "no_detection").sum()
        print(f"\n{model_name.upper()} — "
              f"Correct: {correct}/{total} ({100*correct/total:.1f}%) | "
              f"No detection: {no_det} ({100*no_det/total:.1f}%)")

        # --------------------------------------------------
        # Confidence threshold sweep
        # --------------------------------------------------
        sweep_confidence_thresholds(
            df, class_name_list, model_name,
            output_dir=OUTPUT_DIR
        )

        # --------------------------------------------------
        # Evaluation metrics and figures (Analysis 1: all 514)
        # --------------------------------------------------
        compute_metrics(
            df, class_name_list, model_name,
            output_dir=OUTPUT_DIR
        )

    # Save combined results CSV (both models)
    combined_csv = os.path.join(OUTPUT_DIR, "results_combined.csv")
    with open(combined_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCombined results saved: {combined_csv}")


if __name__ == "__main__":
    main()
