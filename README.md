# The Model That Doesn't Know
## Open-set image classification and out-of-distribution (OOD) detection

This project trains a classifier on five CIFAR-10 classes and evaluates whether it can flag images from five categories excluded from training. It compares a plain closed-set classifier with two score-based rejection methods: Maximum Softmax Probability (MSP) and an energy/log-sum-exp score.

> **Important:** OOD detection is not a guarantee. A detector can reject valid known images or confidently accept unfamiliar images. Results on CIFAR-10's held-out classes do not prove performance on arbitrary real-world images.

## Experimental design

### Classes
- **Known / in-distribution (ID):** airplane, automobile, bird, cat, deer
- **Held-out / OOD:** dog, frog, horse, ship, truck

The classifier's output layer has only five units. The five OOD categories are never used for classifier training or threshold calibration. They are used only for final OOD evaluation.

### Model and preprocessing
- Architecture: ResNet-18 initialized with ImageNet-pretrained weights.
- Fine-tuning: replace the final fully connected layer with a five-class layer.
- Input preprocessing: resize to 224×224, convert to tensor, normalize using ImageNet mean and standard deviation.
- Training augmentation: random crop with padding and random horizontal flip.
- Optimizer: AdamW; cosine annealing learning-rate schedule.
- Seed: 42 by default.
- Best checkpoint: selected by known-class validation accuracy.

CIFAR-10 consists of 32×32 color images. Resizing to 224×224 is done to match the pretrained model's standard input pipeline; it does not add image detail.

## Methods

### Closed-set baseline
The ordinary classifier always returns one of the five known labels, even when an input is from an excluded class.

### Maximum Softmax Probability (MSP)
For logits \(z\), calculate \(s_{MSP}(x)=\max_k softmax(z)_k\). A score below a threshold causes the prediction to be rejected as potentially unknown.

### Energy ID score
Calculate \(s_E(x)=T\log\sum_k\exp(z_k/T)\), with \(T=1\) by default. Larger scores are treated as more ID-like. This is the sign-reversed form of the commonly used energy score, so the score direction matches MSP in the metrics and plots.

### Threshold selection
Thresholds are set from the lower 5th percentile of the known validation scores by default (target known acceptance = 95%). This calibration uses no OOD examples. It aims to control rejection of known validation examples; it does not guarantee the same rate on test data. Do not tune thresholds using final test labels.

## Metrics
- **Known-class accuracy and macro-F1:** evaluate ordinary classification on known test classes.
- **AUROC:** distinguish known test images from held-out classes using an ID-oriented score (higher = more ID-like).
- **AUPR:** average precision with ID as the positive class.
- **FPR at 95% TPR:** false-positive rate when true-positive rate for ID detection reaches at least 95%. Here, positive means ID, so FPR is the proportion of OOD examples incorrectly treated as ID.
- **Known rejection rate:** fraction of known test images below the calibrated threshold.
- **OOD rejection rate:** fraction of OOD test images below the calibrated threshold.

Always read the metric names and positive-class convention before interpreting results.

## Run locally or in Google Colab

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

In Colab, select **Runtime → Change runtime type → T4 GPU** (or another available GPU).

### 2. Train
```bash
python train.py --epochs 5 --batch-size 64 --lr 0.001
```

The CIFAR-10 dataset downloads automatically. Outputs are saved in `outputs/`:
- `best_model.pt`
- `training_history.json`

Training with ImageNet weights requires an internet connection for the first weight download. If the pretrained weights cannot be downloaded, check the runtime's network access. `--no-pretrained` is available for offline experiments but changes the experiment and may reduce performance.

### 3. Evaluate
```bash
python evaluate.py
```

Evaluation creates:
- `evaluation_results.json` — full metrics and class report
- `ood_metrics.csv` — concise OOD metric comparison
- `msp_distributions.png` — MSP score distribution plot
- `energy_distributions.png` — energy score distribution plot
- `confusion_matrix.png` — known-class confusion matrix
- `prediction_audit.csv` — per-image prediction and rejection outcomes

Optional arguments:
```bash
python evaluate.py --known-acceptance 0.95 --temperature 1.0
```

**Reproducibility note:** training and threshold calibration use seed 42 by default. GPU kernels and library versions can still introduce small nondeterministic differences. Record your runtime, library versions, and any changed hyperparameters in your report.

### 4. Run the Streamlit demo
After training and evaluation:
```bash
streamlit run app.py
```
Upload a JPEG, PNG, WEBP, or BMP image. The demo reports the top known class, the selected score, the validation-derived threshold, and whether the input was accepted or rejected.

## Google Colab workflow

1. Upload this project folder or clone your repository into the Colab runtime.
2. Install dependencies with `!pip install -r requirements.txt`.
3. Train with `!python train.py --epochs 5 --batch-size 64`.
4. Evaluate with `!python evaluate.py`.
5. Start Streamlit with `!streamlit run app.py --server.port 8501`. To access the app externally, use a suitable Colab tunneling workflow if permitted by your environment; do not expose a public tunnel without understanding its access controls.
6. Download the `outputs/` directory and include the actual generated metrics/plots in the submission. Never invent or estimate results in the README.

## Suggested analysis for the recruitment report

After running the experiment, answer these questions using your generated outputs:

1. How accurate is the classifier on the five known test classes?
2. Which known classes are confused most often, according to the confusion matrix?
3. Which detector has higher AUROC and AUPR on the held-out CIFAR-10 classes?
4. At the chosen threshold, what fraction of known images are rejected? What fraction of held-out images are caught?
5. Inspect `prediction_audit.csv` for:
   - OOD images with high MSP that the detector accepts as known;
   - known images with low scores that are rejected;
   - known images that are misclassified but still accepted.
6. Do score histograms overlap? If so, explain why a single threshold cannot perfectly separate ID from OOD.

## Limitations
- The unknown set consists of five CIFAR-10 classes, not arbitrary objects or real-world distribution shifts.
- CIFAR-10 is low-resolution and small compared with deployment imagery.
- MSP is a heuristic; neural networks can be overconfident on unfamiliar inputs.
- The energy score is not a calibrated probability.
- The validation threshold is calibrated only on known validation examples; OOD false acceptance is measured only after the threshold is fixed.
- A single random split and seed do not establish robustness. Repeat with multiple seeds for stronger evidence.
- Corruptions such as blur, noise, rotation and low light should be tested separately; they can shift known images without making them semantically unknown.
- Pretrained ImageNet weights may contain prior exposure to visually related concepts, although the five CIFAR-10 output classes remain the only supervised labels in this fine-tuning setup.

## Repository structure
```text
model_that_doesnt_know/
├── train.py
├── evaluate.py
├── app.py
├── requirements.txt
├── README.md
└── outputs/                 # generated by training/evaluation
```

## Submission checklist
- [ ] Run training and preserve the actual checkpoint.
- [ ] Run evaluation and include generated metrics and figures.
- [ ] Explain threshold calibration and score direction correctly.
- [ ] Discuss at least one false acceptance and one false rejection from the audit CSV.
- [ ] Include setup commands and a reproducible environment.
- [ ] Do not claim that the detector recognizes all unknown images.
