"""Evaluate known-class classification and OOD detection using MSP and energy scores."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score, roc_curve
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

KNOWN_CLASSES = ["airplane", "automobile", "bird", "cat", "deer"]
UNKNOWN_CLASSES = ["dog", "frog", "horse", "ship", "truck"]
ALL_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN_CLASSES))
    return model


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def collect_outputs(model, loader, device):
    logits_all, labels_all = [], []
    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits_all.append(model(images).cpu())
        labels_all.extend(labels.numpy().tolist())
    return torch.cat(logits_all), np.asarray(labels_all)


def scores_from_logits(logits, temperature=1.0):
    probs = torch.softmax(logits, dim=1)
    msp = probs.max(dim=1).values.numpy()
    # Conventional energy score: -T * logsumexp(logits / T).
    # Lower energy generally indicates more ID-like input; negate it so
    # that larger values mean "more likely in-distribution" for metric consistency.
    energy_id_score = (temperature * torch.logsumexp(logits / temperature, dim=1)).numpy()
    return msp, energy_id_score


def fpr_at_tpr(y_true, scores, target_tpr=0.95):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    eligible = np.where(tpr >= target_tpr)[0]
    return float(fpr[eligible[0]]) if len(eligible) else 1.0


def detection_metrics(id_scores, ood_scores):
    y = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    return {
        "auroc_id_vs_ood": float(roc_auc_score(y, s)),
        "aupr_id_vs_ood": float(average_precision_score(y, s)),
        "fpr_at_95_tpr_id_detection": fpr_at_tpr(y, s, 0.95),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--checkpoint", default="./outputs/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--known-acceptance", type=float, default=0.95,
                        help="Target fraction of known validation images accepted by threshold.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    transform = get_eval_transform()
    train_ds = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)

    # Known validation subset is sampled from CIFAR-10's training split. It is not used
    # to fit the classifier weights when train.py is run with the provided split logic.
    # Use deterministic indices to make threshold calibration reproducible.
    known_train_ids = [i for i, y in enumerate(train_ds.targets)
                       if test_ds.classes[y] in KNOWN_CLASSES]
    rng = np.random.default_rng(42)
    rng.shuffle(known_train_ids)
    n_val = int(len(known_train_ids) * 0.1)
    val_ids = known_train_ids[:n_val]
    val_loader = DataLoader(Subset(train_ds, val_ids), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    known_test_ids, ood_test_ids = [], []
    for i, y in enumerate(test_ds.targets):
        name = test_ds.classes[y]
        if name in KNOWN_CLASSES:
            known_test_ids.append(i)
        elif name in UNKNOWN_CLASSES:
            ood_test_ids.append(i)
    known_loader = DataLoader(Subset(test_ds, known_test_ids), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    ood_loader = DataLoader(Subset(test_ds, ood_test_ids), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    val_logits, _ = collect_outputs(model, val_loader, device)
    known_logits, known_original_labels = collect_outputs(model, known_loader, device)
    ood_logits, ood_original_labels = collect_outputs(model, ood_loader, device)

    msp_val, energy_val = scores_from_logits(val_logits, args.temperature)
    msp_known, energy_known = scores_from_logits(known_logits, args.temperature)
    msp_ood, energy_ood = scores_from_logits(ood_logits, args.temperature)

    # Thresholds are selected using known validation data only. A score below the
    # corresponding lower quantile is rejected as unknown. This controls known rejection
    # on validation data, but does not guarantee a specific test-time rejection rate.
    q = 1.0 - args.known_acceptance
    msp_threshold = float(np.quantile(msp_val, q))
    energy_threshold = float(np.quantile(energy_val, q))

    y_known = np.asarray([KNOWN_CLASSES.index(test_ds.classes[y])
                          for y in known_original_labels])
    pred_known = known_logits.argmax(1).numpy()
    known_accuracy = float(accuracy_score(y_known, pred_known))
    report = classification_report(y_known, pred_known, labels=list(range(5)),
                                    target_names=KNOWN_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_known, pred_known, labels=list(range(5)))

    results = {
        "known_classes": KNOWN_CLASSES,
        "unknown_classes": UNKNOWN_CLASSES,
        "threshold_calibration": {
            "target_known_acceptance": args.known_acceptance,
            "temperature": args.temperature,
            "msp_threshold": msp_threshold,
            "energy_id_score_threshold": energy_threshold,
            "calibration_source": "known validation images only",
        },
        "known_classification": {
            "accuracy": known_accuracy,
            "macro_f1": float(f1_score(y_known, pred_known, average="macro")),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        },
        "ood_detection": {
            "msp": detection_metrics(msp_known, msp_ood),
            "energy": detection_metrics(energy_known, energy_ood),
        },
        "rejection_behavior": {
            "msp_known_rejection_rate": float(np.mean(msp_known < msp_threshold)),
            "msp_ood_rejection_rate": float(np.mean(msp_ood < msp_threshold)),
            "energy_known_rejection_rate": float(np.mean(energy_known < energy_threshold)),
            "energy_ood_rejection_rate": float(np.mean(energy_ood < energy_threshold)),
        },
        "notes": [
            "OOD detection is evaluated on held-out CIFAR-10 classes, not arbitrary real-world unknowns.",
            "Higher MSP and energy_id_score are treated as more in-distribution-like.",
            "Thresholds use only known validation examples; no OOD validation examples are used.",
        ],
    }
    with open(out / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame([
        {"method": "MSP", **results["ood_detection"]["msp"],
         "known_rejection_rate": results["rejection_behavior"]["msp_known_rejection_rate"],
         "ood_rejection_rate": results["rejection_behavior"]["msp_ood_rejection_rate"]},
        {"method": "Energy", **results["ood_detection"]["energy"],
         "known_rejection_rate": results["rejection_behavior"]["energy_known_rejection_rate"],
         "ood_rejection_rate": results["rejection_behavior"]["energy_ood_rejection_rate"]},
    ]).to_csv(out / "ood_metrics.csv", index=False)

    # Confidence distributions
    plt.figure(figsize=(9, 5))
    plt.hist(msp_known, bins=40, alpha=0.65, density=True, label="Known test")
    plt.hist(msp_ood, bins=40, alpha=0.65, density=True, label="Held-out class test")
    plt.axvline(msp_threshold, linestyle="--", label=f"MSP threshold = {msp_threshold:.3f}")
    plt.xlabel("Maximum softmax probability (higher = more ID-like)")
    plt.ylabel("Density")
    plt.title("MSP score distributions: known vs. held-out classes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "msp_distributions.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(energy_known, bins=40, alpha=0.65, density=True, label="Known test")
    plt.hist(energy_ood, bins=40, alpha=0.65, density=True, label="Held-out class test")
    plt.axvline(energy_threshold, linestyle="--",
                label=f"Energy ID-score threshold = {energy_threshold:.3f}")
    plt.xlabel("Energy ID score (logsumexp logits; higher = more ID-like)")
    plt.ylabel("Density")
    plt.title("Energy score distributions: known vs. held-out classes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "energy_distributions.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Known-class confusion matrix")
    plt.colorbar()
    ticks = np.arange(len(KNOWN_CLASSES))
    plt.xticks(ticks, KNOWN_CLASSES, rotation=35, ha="right")
    plt.yticks(ticks, KNOWN_CLASSES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close()

    # Prediction-level audit for inspecting failures and threshold behavior.
    known_msp_rejected = msp_known < msp_threshold
    ood_msp_rejected = msp_ood < msp_threshold
    known_rows = []
    for i, (logit, score, rejected, true_y) in enumerate(
            zip(known_logits, msp_known, known_msp_rejected, y_known)):
        pred = int(logit.argmax().item())
        known_rows.append({
            "split": "known_test", "true_class": KNOWN_CLASSES[int(true_y)],
            "predicted_class": KNOWN_CLASSES[pred], "msp": float(score),
            "rejected_as_unknown": bool(rejected),
            "correct_before_rejection": bool(pred == int(true_y)),
        })
    for logit, score, rejected, original_y in zip(
            ood_logits, msp_ood, ood_msp_rejected, ood_original_labels):
        pred = int(logit.argmax().item())
        ood_rows = {
            "split": "ood_test", "true_class": test_ds.classes[int(original_y)],
            "predicted_class": KNOWN_CLASSES[pred], "msp": float(score),
            "rejected_as_unknown": bool(rejected),
            "correct_before_rejection": False,
        }
        known_rows.append(ood_rows)
    pd.DataFrame(known_rows).to_csv(out / "prediction_audit.csv", index=False)

    print(json.dumps({
        "known_accuracy": known_accuracy,
        "msp_ood_metrics": results["ood_detection"]["msp"],
        "energy_ood_metrics": results["ood_detection"]["energy"],
        "rejection_behavior": results["rejection_behavior"],
        "output_dir": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
