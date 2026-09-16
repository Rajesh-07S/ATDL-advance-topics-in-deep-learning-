# ATDL Assignment I — Knowledge Distillation with Ternary-Weight Quantization-Aware Training

## ResNet34 FP32 Teacher → ResNet18 Ternary Student with KD + QAT on CIFAR-10

This README documents the complete workflow implemented in the two assignment notebooks.

```text
CIFAR-10
   │
   ├── ResNet34 FP32 Teacher
   │       └── Best Teacher Checkpoint
   │
   ├── FP32 ResNet18 Baseline
   │
   └── Ternary Quantizer + STE
           └── Ternary ResNet18
                    ├── QAT
                    └── Knowledge Distillation
                              └── Final Ternary Student
                                      ├── Verification
                                      ├── Evaluation
                                      ├── Sparsity
                                      ├── Compression
                                      ├── Weight Analysis
                                      └── KD Temperature Ablation
```

## 1. Project Objective

The assignment trains a ResNet34 FP32 teacher and a smaller ResNet18 student on CIFAR-10. The student uses ternary weights:

```text
{-α, 0, +α}
```

and is trained with Knowledge Distillation (KD) and Quantization-Aware Training (QAT).

The project contains three models:

1. **ResNet34 FP32 Teacher** — trained from scratch.
2. **FP32 ResNet18 Baseline** — trained without KD or ternary quantization.
3. **Ternary ResNet18 Student** — trained using ternary QAT + KD from the frozen teacher.

The purpose of the FP32 ResNet18 baseline is to provide a direct reference for measuring the effect of ternary quantization and KD.

---

## 2. Project Files

### Notebook 1 — Teacher

```text
Teachermodel(1).ipynb
```

This notebook performs the teacher stage:

```text
CIFAR-10
→ train/validation split
→ ResNet34 from scratch
→ 200-epoch training
→ best validation checkpoint
→ final test evaluation
```

Best teacher checkpoint:

```text
/home/sem_7th/saved_models/resnet34_cifar10_best_validation.pth
```

### Notebook 2 — Remaining Pipeline

```text
ATDL_Assignment1_Remaining_Pipeline_updated.ipynb
```

This notebook assumes the ResNet34 teacher has already been trained and performs:

1. Load/verify teacher.
2. Configure reproducibility and paths.
3. Load CIFAR-10.
4. Train FP32 ResNet18 baseline.
5. Implement ternary quantization.
6. Implement STE.
7. Build ternary ResNet18.
8. Run QAT smoke test.
9. Implement KD.
10. Train final ternary KD+QAT student.
11. Verify ternary checkpoint.
12. Evaluate teacher/baseline/student.
13. Calculate sparsity and compression.
14. Analyze weights.
15. Run temperature ablation.
16. Save results and plots.
17. Save experiment configuration.
18. Run final artifact check.

---

## 3. Environment

Recorded remaining-pipeline environment:

```text
PyTorch: 2.13.0+cu130
CUDA available: True
Device: cuda
```

Main Python packages:

```text
torch
torchvision
numpy
pandas
matplotlib
```

Installation:

```bash
pip install torch torchvision numpy pandas matplotlib
```

For an exact package lock:

```bash
pip freeze > requirements.txt
```

Run this in the same environment used for training.

---

## 4. Paths

Teacher:

```text
/home/sem_7th/saved_models/resnet34_cifar10_best_validation.pth
```

Optional final teacher checkpoint:

```text
/home/sem_7th/saved_models/resnet34_cifar10_final.pth
```

Dataset root:

```text
/home/sem_7th/saved_models
```

Project output:

```text
/home/sem_7th/assignment1_remaining
```

The notebook creates:

```text
assignment1_remaining/
├── checkpoints/
├── results/
└── plots/
```

If running on another machine, change these paths in the configuration cell before running.

---

# 5. Stage 1 — ResNet34 Teacher

## 5.1 Purpose

The ResNet34 is the high-capacity FP32 teacher. It is trained from scratch and then frozen before Knowledge Distillation.

## 5.2 Teacher Data Split

The teacher notebook uses:

```text
Train:      45,000
Validation: 5,000
Test:       10,000
```

with seed:

```text
42
```

The best teacher checkpoint is selected using validation performance.

## 5.3 Teacher Training Configuration

```text
Model:           ResNet34
Dataset:         CIFAR-10
Epochs:          200
Optimizer:       SGD
Learning rate:   0.1
Momentum:        0.9
Weight decay:    5e-4
Scheduler:       CosineAnnealingLR
T_max:           200
Batch size:      256
Seed:            42
Mixed precision: Enabled
```

## 5.4 Run the Teacher

Open:

```text
Teachermodel(1).ipynb
```

Run the notebook from beginning to end.

The expected workflow is:

```text
Set seed
→ Load CIFAR-10
→ Create train/validation/test split
→ Create transforms
→ Build ResNet34
→ Train 200 epochs
→ Track validation accuracy
→ Save best validation checkpoint
→ Evaluate best checkpoint on test set
```

Recorded teacher result:

```text
Parameters:               21,282,122
Approx. FP32 size:        81.185 MB
Best validation accuracy: ~95.52%
Test accuracy:             95.04%
Test loss:                 ~0.200994
```

The output checkpoint required by the second notebook is:

```text
/home/sem_7th/saved_models/resnet34_cifar10_best_validation.pth
```

---

# 6. Stage 2 — Load the Teacher

Open:

```text
ATDL_Assignment1_Remaining_Pipeline_updated.ipynb
```

Set:

```python
TEACHER_BEST_PATH = "/home/sem_7th/saved_models/resnet34_cifar10_best_validation.pth"
DATA_ROOT = "/home/sem_7th/saved_models"
OUTPUT_ROOT = "/home/sem_7th/assignment1_remaining"
```

The notebook loads the teacher and verifies its parameter count and size.

Then:

```python
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False
```

During student training:

```python
with torch.no_grad():
    teacher_logits = teacher(x)
```

Therefore the teacher is frozen and receives no gradients.

---

# 7. Stage 3 — CIFAR-10 Data Pipeline

The remaining notebook uses:

```text
50,000 training images
10,000 test images
10 classes
```

The classes are:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

Image size:

```text
3 × 32 × 32
```

## Training Transform

```python
RandomCrop(32, padding=4)
RandomHorizontalFlip()
ToTensor()
Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616)
)
```

## Test Transform

```python
ToTensor()
Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616)
)
```

DataLoader:

```text
Batch size: 128
Workers:    2
Shuffle train: True
Shuffle test:  False
```

The sanity check should show:

```text
Train samples: 50000
Test samples:  10000
Classes:       10
Image shape:   [128, 3, 32, 32]
```

---

# 8. Stage 4 — Reproducibility

The project uses:

```text
SEED = 42
```

The notebook sets Python, NumPy, PyTorch and CUDA seeds.

For CUDA:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Keep seed 42 when reproducing the recorded experiment.

---

# 9. Stage 5 — FP32 ResNet18 Baseline

## Purpose

The baseline answers:

> How does a normal full-precision ResNet18 perform without KD and without ternary quantization?

It uses the same ResNet18 architecture as the student but keeps FP32 weights.

## Configuration

```text
Epochs:         200
Batch size:     128
Learning rate:  0.1
Optimizer:      SGD
Momentum:       0.9
Nesterov:       True
Weight decay:   5e-4
Scheduler:      CosineAnnealingLR
Seed:           42
```

Training:

```text
Image
→ FP32 ResNet18
→ logits
→ Cross Entropy
→ backpropagation
→ SGD update
→ cosine LR update
```

The notebook records:

```text
Train loss
Train accuracy
Test loss
Test accuracy
Learning rate
```

Checkpoint:

```text
/home/sem_7th/assignment1_remaining/checkpoints/resnet18_fp32_baseline_best.pth
```

Recorded result:

```text
Parameters:        11,173,962
Test accuracy:     95.35%
Test loss:         ~0.179740
Approx. FP32 size: 42.625 MB
```

---

# 10. Stage 6 — Ternary Quantization

The student uses:

```text
Wq ∈ {-α, 0, +α}
```

The implementation uses:

```text
Alpha mode:   per_tensor
Delta factor: 0.7
```

Convolutional and fully connected weights are quantized.

BatchNorm parameters remain FP32.

---

# 11. Alpha and Delta Calculation

For each weight tensor `W`:

```text
Δ = 0.7 × mean(|W|)
```

The threshold is implemented as:

```python
abs_w = weight.detach().abs()
delta = 0.7 * abs_w.mean()
```

Weights are classified as:

```text
W > +Δ  → +α
W < -Δ  → -α
otherwise → 0
```

Alpha is derived from the selected non-zero weights:

```text
α = mean(|W_i|)
```

for weights satisfying:

```text
|W_i| > Δ
```

A small minimum value is enforced for alpha.

---

# 12. Ternary Forward Pass

The quantized tensor is constructed as:

```text
W > Δ    → +α
W < -Δ   → -α
otherwise → 0
```

Thus every quantized convolutional/fully connected weight is one of:

```text
-α
0
+α
```

The notebook smoke test confirms this behavior.

Example:

```text
Original:
[-1.2, -0.8, -0.1, 0.0, 0.1, 0.7, 1.4]

Quantized:
[-1.025, -1.025, 0, 0, 0, 1.025, 1.025]

Unique values:
[-1.025, 0, 1.025]
```

---

# 13. Latent FP32 Weights

The model retains latent FP32 weights during optimization.

Conceptually:

```text
Latent FP32 W
      ↓
Ternary quantizer
      ↓
Wq = {-α, 0, +α}
      ↓
Forward pass
```

The optimizer updates the latent FP32 weights.

The forward pass uses their ternary representation.

This is the key difference between QAT and applying ternarization only after training.

---

# 14. Stage 7 — Straight-Through Estimator

The ternary quantizer is non-differentiable.

The project therefore implements:

```text
TernaryQuantizeSTE
```

Forward:

```text
W → {-α, 0, +α}
```

Backward:

```text
approximate gradient passes through the
quantization operation using an STE mask
```

The implementation uses:

```text
|W| <= 1
```

for gradient propagation.

The smoke test produces non-zero gradients for values inside the clipping range.

---

# 15. Stage 8 — Ternary ResNet18

The student uses ResNet18.

Quantization configuration:

```text
Convolution weights:       ternary
Fully connected weights:   ternary
BatchNorm:                 FP32
Alpha:                     per_tensor
Delta factor:              0.7
```

The notebook reports:

```text
21 ternary Conv/FC layers
0 remaining FP32 Conv/FC layers
```

---

# 16. Stage 9 — QAT Smoke Test

Before the expensive student training, a short QAT smoke test is run:

```text
QAT_SMOKE_EPOCHS = 2
```

Purpose:

- Check forward pass.
- Check ternary quantization.
- Check gradient flow.
- Check STE.
- Detect training/runtime problems before 200 epochs.

Only proceed to full training after the smoke test succeeds.

---

# 17. Stage 10 — Knowledge Distillation

The ResNet34 teacher provides soft predictions.

The student receives:

```text
Ground-truth labels
+
Teacher logits
```

Teacher configuration:

```python
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False
```

Teacher logits:

```python
with torch.no_grad():
    teacher_logits = teacher(x)
```

Therefore:

```text
Teacher: frozen
Student: trainable
```

---

# 18. Temperature and KD Loss

Main KD settings:

```text
Temperature T = 4.0
Lambda KD     = 0.5
```

The KD loss uses temperature-scaled KL divergence with the temperature-squared scaling factor.

The total loss is:

```text
L = (1 - λ) LCE + λ LKD
```

where:

```text
LCE = hard-label Cross Entropy
LKD = temperature-scaled KL divergence
λ   = 0.5
```

---

# 19. Stage 11 — Joint KD + QAT Student Training

The final student combines:

```text
Ternary QAT
+
Knowledge Distillation
```

Configuration:

```text
Epochs:          200
Batch size:      128
Learning rate:   0.05
Optimizer:       SGD
Momentum:        0.9
Weight decay:    5e-4
Scheduler:       CosineAnnealingLR
Temperature:     4.0
Lambda KD:       0.5
Alpha mode:      per_tensor
Delta factor:    0.7
Seed:            42
```

Complete training flow:

```text
Input image
    │
    ├──────────────→ Frozen ResNet34
    │                       ↓
    │                 Teacher logits
    │
    └──────────────→ Ternary ResNet18
                            ↓
                      Student logits
                            │
                ┌───────────┴───────────┐
                ↓                       ↓
             CE loss                KD loss
                                        │
                                        ↓
                                Temperature T=4
                │                       │
                └───────────┬───────────┘
                            ↓
                       Total loss
                            ↓
                    Backpropagation
                            ↓
                           STE
                            ↓
                  Update latent FP32 W
```

Checkpoint:

```text
/home/sem_7th/assignment1_remaining/checkpoints/resnet18_ternary_kd_qat_best.pth
```

---

# 20. Stage 12 — Final Evaluation

The pipeline evaluates:

```text
ResNet34 Teacher
FP32 ResNet18 Baseline
Ternary KD + QAT ResNet18
```

Recorded results:

| Model | Precision | KD | Parameters | Test Accuracy |
|---|---|---|---:|---:|
| ResNet34 Teacher | FP32 | No | 21,282,122 | 95.04% |
| ResNet18 Baseline | FP32 | No | 11,173,962 | 95.35% |
| ResNet18 Student | Ternary | Yes | 11,173,962 | 95.34% |

Recorded losses:

```text
Teacher:           0.200994
FP32 ResNet18:     0.179740
Ternary KD+QAT:    0.221761
```

---

# 21. Stage 13 — Sparsity

The student has three possible weight states:

```text
-α
0
+α
```

The zero state produces sparsity.

Recorded student sparsity:

```text
49.737%
```

---

# 22. Stage 14 — Compression

The theoretical ternary storage calculation assumes:

```text
2 bits per ternary weight
+
additional storage for α
```

Recorded sizes:

```text
ResNet34 FP32:       81.185 MB
ResNet18 FP32:       42.625 MB
Ternary student:      2.662 MB
```

Theoretical compression:

```text
vs FP32 ResNet18 = 16.013×
vs ResNet34      = 30.499×
```

These are theoretical storage estimates, not measured hardware speedups.

Actual inference speed depends on the weight encoding, implementation, kernels, hardware, and ability to exploit ternary/sparse computation.

---

# 23. Stage 15 — Ternary Constraint Verification

The project verifies every ternary layer.

The verification checks:

1. Unique quantized values.
2. Alpha.
3. Delta.
4. Zero count.
5. Layer sparsity.
6. Whether all values are valid `{-α,0,+α}`.

Final result:

```text
Overall ternary constraint valid: True
```

Output:

```text
/home/sem_7th/assignment1_remaining/results/ternary_constraint_report.csv
```

---

# 24. Stage 16 — Weight Analysis

The notebook analyzes the ternary baseline and KD+QAT student weights.

Plots include:

```text
ternary_baseline_weight_distribution.png
kd_qat_weight_distribution.png
ternary_baseline_first_100_weights.png
kd_qat_first_100_weights.png
ternary_baseline_vs_kd_qat_weights.png
```

These plots show:

- concentration at `-α`, `0`, `+α`;
- zero-weight distribution;
- differences between ternary baseline and KD+QAT;
- representative first-layer weights.

---

# 25. Stage 17 — KD Temperature Ablation

The assignment requires at least one KD/quantization hyperparameter ablation.

The completed experiment varies:

```text
T = 2
T = 4
T = 8
```

while keeping:

```text
λ = 0.5
Seed = 42
Epochs = 200
Student LR = 0.05
```

Recorded results:

| Temperature | Lambda | Test Accuracy |
|---:|---:|---:|
| 2 | 0.5 | 95.17% |
| 4 | 0.5 | 95.06% |
| 8 | 0.5 | 95.02% |

Among the tested configurations, `T=2` gives the highest measured accuracy.

This does not establish universal optimality; it is only the result for the tested values.

The notebook also contains lambda-ablation infrastructure:

```text
λ = {0, 0.25, 0.5, 0.75, 1.0}
```

with:

```text
ABLATION_EPOCHS = 30
```

Only report lambda results if the experiment was actually completed.

---

# 26. Results and Artifact Files

The project saves:

```text
/home/sem_7th/assignment1_remaining/
```

Expected structure:

```text
assignment1_remaining/
│
├── checkpoints/
│   ├── resnet18_fp32_baseline_best.pth
│   └── resnet18_ternary_kd_qat_best.pth
│
├── results/
│   ├── final_model_comparison.csv
│   ├── final_summary.json
│   ├── ternary_constraint_report.csv
│   ├── kd_temperature_ablation.csv
│   ├── experiment_config.json
│   └── kd_lambda_ablation_results.csv
│
└── plots/
    ├── fp32_resnet18_accuracy.png
    ├── ternary_student_accuracy.png
    ├── ternary_student_losses.png
    ├── kd_temperature_ablation.png
    ├── ternary_baseline_weight_distribution.png
    ├── kd_qat_weight_distribution.png
    ├── ternary_baseline_first_100_weights.png
    ├── kd_qat_first_100_weights.png
    └── ternary_baseline_vs_kd_qat_weights.png
```

The final artifact-check cell should be used to confirm which files actually exist.

---

# 27. Experiment Configuration

The notebook saves:

```text
results/experiment_config.json
```

Main recorded settings:

```json
{
  "seed": 42,
  "batch_size": 128,
  "baseline_epochs": 200,
  "student_epochs": 200,
  "baseline_lr": 0.1,
  "student_lr": 0.05,
  "weight_decay": 0.0005,
  "temperature": 4.0,
  "lambda_kd": 0.5,
  "alpha_mode": "per_tensor",
  "delta_factor": 0.7,
  "device": "cuda"
}
```

---

# 28. Complete Reproduction Procedure

## Step 1 — Install environment

```bash
pip install torch torchvision numpy pandas matplotlib
```

Optionally:

```bash
pip install -r requirements.txt
```

## Step 2 — Prepare CIFAR-10

Set:

```python
DATA_ROOT = "/home/sem_7th/saved_models"
```

Run the dataset cells and confirm:

```text
50000 training images
10000 test images
10 classes
```

## Step 3 — Train teacher

Open:

```text
Teachermodel(1).ipynb
```

Run all cells.

Confirm:

```text
Teacher checkpoint exists
Teacher test accuracy ≈ 95.04%
```

## Step 4 — Configure remaining notebook

Open:

```text
ATDL_Assignment1_Remaining_Pipeline_updated.ipynb
```

Set the teacher, data, and output paths.

## Step 5 — Set seed

```text
SEED = 42
```

## Step 6 — Run dataset sanity check

Confirm image and label shapes.

## Step 7 — Train FP32 ResNet18 baseline

Run the baseline section for 200 epochs.

Expected:

```text
95.35% reported test accuracy
```

## Step 8 — Test ternary quantizer and STE

Confirm:

```text
Unique values = {-α, 0, +α}
```

and gradient propagation works.

## Step 9 — Build ternary ResNet18

Confirm convolutional and FC weights are ternary while BatchNorm remains FP32.

## Step 10 — Run QAT smoke test

Run the 2-epoch smoke test.

## Step 11 — Run KD + QAT

Use:

```text
T = 4.0
λ = 0.5
LR = 0.05
Epochs = 200
```

## Step 12 — Verify checkpoint

Run ternary verification.

Expected:

```text
Overall ternary constraint valid: True
```

## Step 13 — Evaluate models

Run the final comparison section.

## Step 14 — Calculate compression/sparsity

Confirm:

```text
Student sparsity ≈ 49.737%
Theoretical size ≈ 2.662 MB
Compression vs FP32 R18 ≈ 16.013×
Compression vs teacher ≈ 30.499×
```

## Step 15 — Run temperature ablation

Run:

```text
T = 2, 4, 8
```

## Step 16 — Generate weight plots

Run the weight analysis cells.

## Step 17 — Save experiment configuration

Run the configuration-saving cell.

## Step 18 — Run final artifact check

Confirm that the required checkpoints, CSV/JSON results and plots exist.

---

# 29. Important Evaluation Limitation

There is a dataset-protocol difference between the two notebooks.

The teacher uses:

```text
45k train
5k validation
10k test
```

The remaining pipeline uses:

```text
50k train
10k test
```

In addition, the remaining baseline/student training histories record test accuracy during training and use the best checkpoint associated with the recorded test performance.

Therefore, the baseline/student test set was not completely isolated from model selection.

This is an evaluation-protocol limitation, not a failure of the model implementation.

For a future cleaner experiment:

```text
CIFAR-10
→ fixed 45k train / 5k validation / 10k test
→ use the same split for teacher, baseline and student
→ select checkpoints using validation accuracy
→ evaluate test set only at the end
```

The current report should describe the actual experiment rather than claiming a validation protocol that was not used.

---

# 30. Final Results Summary

```text
ResNet34 Teacher
    Accuracy: 95.04%
    Parameters: 21,282,122
    FP32 size: 81.185 MB

FP32 ResNet18 Baseline
    Accuracy: 95.35%
    Parameters: 11,173,962
    FP32 size: 42.625 MB

Ternary KD + QAT ResNet18
    Accuracy: 95.34%
    Parameters: 11,173,962
    Sparsity: 49.737%
    Theoretical size: 2.662 MB
    Compression vs FP32 ResNet18: 16.013×
    Compression vs ResNet34: 30.499×
```

---

# 31. Viva / Technical Explanation

### Why ResNet34 as teacher?

It is the larger model and supplies soft predictions to the smaller student.

### Why ResNet18 as student?

It has fewer parameters and is therefore a suitable compact architecture.

### Why ternary weights?

Each weight is restricted to:

```text
-α, 0, +α
```

which reduces the number of weight states and enables compact storage.

### What is alpha?

Alpha is the scale used for the non-zero ternary values.

### What is delta?

Delta is the threshold determining whether a latent FP32 weight becomes `-α`, `0`, or `+α`.

### Why per-tensor alpha?

The implementation calculates a separate scaling value for each quantized tensor.

### Why latent FP32 weights?

They allow continuous optimization while the forward pass uses ternary weights.

### Why STE?

The ternary operation is non-differentiable, so STE provides an approximate backward gradient.

### Why QAT?

The student experiences the ternary constraint during training instead of being quantized only after training.

### Why Knowledge Distillation?

The student learns from both the ground-truth labels and the teacher's soft predictions.

### Why freeze the teacher?

The teacher is the fixed source of transferred knowledge; only the student is optimized.

### What does temperature do?

It softens the teacher/student output distributions used for KD.

### What does lambda do?

It controls the balance between hard-label CE loss and KD loss.

### Why FP32 ResNet18 baseline?

It provides a control model for comparing the effect of ternary quantization and KD.

---

# 32. Submission Checklist

## Code

- [ ] `Teachermodel(1).ipynb`
- [ ] `ATDL_Assignment1_Remaining_Pipeline_updated.ipynb`
- [ ] Data pipeline
- [ ] ResNet34 teacher
- [ ] FP32 ResNet18 baseline
- [ ] Ternary quantizer
- [ ] Alpha/delta calculation
- [ ] Latent FP32 weights
- [ ] STE
- [ ] Ternary ResNet18
- [ ] KD loss
- [ ] Frozen teacher
- [ ] Joint KD + QAT
- [ ] Evaluation
- [ ] Compression
- [ ] Sparsity
- [ ] Ablation
- [ ] Weight analysis
- [ ] Ternary verification

## Checkpoints

- [ ] ResNet34 teacher
- [ ] FP32 ResNet18
- [ ] Ternary KD+QAT ResNet18

## Results

- [ ] Final model comparison CSV
- [ ] Final summary JSON
- [ ] Ternary constraint report
- [ ] Temperature ablation CSV
- [ ] Training curves
- [ ] Weight plots
- [ ] Ablation plot
- [ ] Experiment configuration

## Reproducibility

- [ ] Seed 42
- [ ] `requirements.txt`
- [ ] README
- [ ] Correct paths
- [ ] Ternary verification

## Report

Include:

- [ ] Introduction
- [ ] Dataset
- [ ] Teacher
- [ ] FP32 baseline
- [ ] Ternary quantization
- [ ] Alpha/delta
- [ ] STE
- [ ] KD formulation
- [ ] QAT
- [ ] Training curves
- [ ] Final comparison
- [ ] Compression
- [ ] Sparsity
- [ ] Weight analysis
- [ ] Temperature ablation
- [ ] Limitations
- [ ] Conclusion
- [ ] References
- [ ] AI-use disclosure/conversation attachment

---

# 33. One-Line Pipeline

```text
ResNet34 Teacher → FP32 ResNet18 Baseline → Ternary Quantizer + STE → Ternary ResNet18 → QAT + KD → Final Student → Verification → Evaluation → Compression/Sparsity → Ablation → Report
```

