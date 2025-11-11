# 🎯 Accuracy Improvement Guide

## 📊 Current Status (Achieved)

| Stage | Accuracy | Technique |
|-------|----------|-----------|
| Initial Baseline (ResNet-18) | 88.10% | Basic augmentation |
| Improved Single Model (DenseNet-121) | 89.93% | Basic augmentation |
| Best Single Model (EfficientNet-V2-S) | 93.04% | Enhanced augmentation + tuning |
| Ensemble (Weighted) | 93.96% | Model combination |
| **🏆 Final (Ensemble + TTA)** | **94.14%** | **Combined techniques** |

**Total Improvement:** +6.04% (from 88.10% to 94.14%)

---

## ✅ Completed Experiments

This section documents the actual experiments performed and their results.

### 1. Baseline Models (Basic Augmentation)

**Notebook:** `insect_classification_transfer_learning.ipynb`

**Initial Training with Basic Augmentation:**
```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
```

**Results:**
- **ResNet-18 (Finetuned):** 88.10% test accuracy
- **DenseNet-121:** 90.11% validation, 89.93% test (baseline)
- **EfficientNet-V2-S:** 89.04% validation, 88.46% test (unstable)

---

### 2. Enhanced Data Augmentation ✓ (Tier 2)

**Notebook:** `accuracy_impro.ipynb`

**Enhanced Transforms Applied:**
```python
enhanced_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),        # NEW
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # NEW
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),           # NEW
        transforms.GaussianBlur(kernel_size=3),      # NEW
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)              # NEW
    ])
}
```

**Results:**
| Model | Baseline Val Acc | Enhanced Val Acc | Improvement |
|-------|-----------------|-----------------|-------------|
| ResNet-18 | 88.77% | 89.68% | +0.91% |
| DenseNet-121 | 90.32% | 91.05% | +0.73% |
| EfficientNet-V2-S | 89.04% | 94.70% | **+5.66%** 🎉 |

**Observations:**
- Enhanced augmentation significantly improved EfficientNet-V2-S
- DenseNet-121 showed signs of overfitting (train 97%, val 91.05%)
- ResNet-18 improvement was modest

---

### 3. Hyperparameter Tuning for DenseNet-121 ✓

**Problem:** Overfitting detected (Train Acc: 97%, Val Acc: 91.05%)

#### Experiment 3a: Increase Weight Decay
```python
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)  # from 1e-4
```
**Result:** Val Acc: 91.51%, Test Acc: 89.38%, Train Acc: 98% ❌ Still overfitting

#### Experiment 3b: Switch to AdamW with Stronger Regularization
```python
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
```
**Result:** Val Acc: 91.32%, Test Acc: 88.46%, Train Acc: 97% ❌ Marginal improvement

#### Experiment 3c: Add Dropout Layers ✓ **BEST**
```python
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
# Modified classifier with dropout
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.classifier.in_features, num_classes)
)
```
**Result:** Val Acc: 91.96%, Test Acc: **90.48%**, Train Acc: 96.51% ✓ **Best performance!**

**Final DenseNet-121 Config:**
- Optimizer: AdamW (lr=0.0001, weight_decay=0.01)
- Dropout: 0.4
- Test Accuracy: **90.84% (with TTA)**

---

### 4. Alternative Loss Function: Focal Loss

**Motivation:** Test if class imbalance could be improved

```python
from advanced_training_techniques import FocalLoss
criterion = FocalLoss(alpha=1, gamma=2)
```

**Result (DenseNet-121):**
- Val Acc: 91.51%
- Test Acc: 89.74%
- Train Acc: 97.54%

**Conclusion:** Focal Loss didn't outperform the dropout approach. ❌ Not adopted.

---

### 5. Stochastic Weight Averaging (SWA)

**Attempted:** Yes
**Result:** Abandoned (no improvement observed)
**Reason:** The model was already well-regularized with dropout and weight decay

---

### 6. Final EfficientNet-V2-S Optimization ✓

After enhanced augmentation, EfficientNet-V2-S became the best single model.

**Final Configuration:**
```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Results:**
- Val Acc: 94.70%
- **Test Acc: 93.04%** 🏆 Best single model
- Test Acc with TTA: 93.77%

**Note:** Tried increasing weight_decay to 0.02 and dropout 0.5, but no improvement (Test Acc: 92.86%)

---

### 7. Hyperparameter Tuning for ResNet-18

**Attempted:** Applied same modifications as DenseNet-121 (AdamW + dropout 0.4 + weight_decay 0.01)

**Result:**
- Val Acc: 87%
- Test Acc: 85%

**Conclusion:** ❌ Made performance worse. Kept original configuration (SGD with momentum).

---

### 8. Model Ensemble ✓ **SUCCESSFUL**

**Implementation:** `ensemble_utils.py`

**Strategy:** Weighted ensemble based on performance
```python
weights = [0.15, 0.35, 0.50]  # ResNet-18, DenseNet-121, EfficientNet-V2-S
```

**Results:**
| Ensemble Method | Test Accuracy | Improvement |
|----------------|---------------|-------------|
| Equal Weights | 93.04% | +0% |
| Weighted (Optimized) | **93.96%** | **+0.92%** |

---

### 9. Test-Time Augmentation (TTA) ✓ **PARTIALLY SUCCESSFUL**

**Implementation:** 5 augmentations per image
- Original
- Horizontal flip
- Vertical flip
- Rotate 90°
- Rotate 270°

**Results:**
| Model | Without TTA | With TTA | Change |
|-------|-------------|----------|--------|
| ResNet-18 | 88.10% | 87.18% | **-0.92%** ⚠️ |
| DenseNet-121 | 89.93% | 90.84% | **+0.91%** ✓ |
| EfficientNet-V2-S | 93.04% | 93.77% | **+0.73%** ✓ |
| Ensemble (Weighted) | 93.96% | **94.14%** | **+0.18%** ✓ |

**Important Discovery:** TTA doesn't always help! ResNet-18 performance actually decreased with TTA.

**Final Best Result:** Ensemble (Weighted) + TTA = **94.14%** 🏆

---

## 📈 Summary of Improvements

```
88.10%  ████████████████████            ResNet-18 (Baseline)
89.93%  █████████████████████           + DenseNet-121 (Basic)
93.04%  ██████████████████████████      + EfficientNet (Enhanced Aug)
93.96%  ███████████████████████████     + Ensemble (Weighted)
94.14%  ████████████████████████████ 🏆 + TTA
```

### Key Learnings

1. ✅ **Enhanced data augmentation** was critical for EfficientNet-V2-S (+5.66%)
2. ✅ **Dropout + AdamW** combination worked best for DenseNet-121
3. ✅ **Different models need different hyperparameters** - what works for one doesn't work for all
4. ✅ **Ensemble provides consistent improvement** (+0.92%)
5. ⚠️ **TTA is model-dependent** - can hurt performance (ResNet-18)
6. ❌ **Focal Loss** didn't help (dataset is relatively balanced)
7. ❌ **SWA** didn't provide additional benefits with good regularization

---

## 🔮 Future Work - Untried Techniques

These techniques were identified but not implemented. They could potentially improve accuracy further.

### Priority 1: Data & Training

#### 1. MixUp Augmentation
**Expected Gain:** +1-3%
**Effort:** Medium (3-4 hours)

```python
from advanced_training_techniques import mixup_data, mixup_criterion

# Inside training loop:
if phase == 'train':
    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.4)
    outputs = model(inputs)
    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
```

**Why try:** MixUp creates virtual training examples and can improve generalization.

---

#### 2. CutMix Augmentation
**Expected Gain:** +1-2%
**Effort:** Medium (3-4 hours)

```python
from advanced_training_techniques import cutmix_data

# Similar to MixUp but cuts and pastes image regions
inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, alpha=1.0)
```

**Why try:** Often more effective than MixUp for fine-grained classification tasks.

---

#### 3. Larger Input Size
**Expected Gain:** +0.5-1.5%
**Effort:** Low (1 hour retrain)

```python
# Current: 224x224
# Try: 256x256 or 384x384
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(256),  # or 384
        # ... rest stays same
    ])
}
```

**Why try:** Larger images preserve more details, helpful for small insect features.

**Trade-off:** Slower training, more GPU memory required.

---

#### 4. Progressive Learning Rate Decay
**Expected Gain:** +0.5-1%
**Effort:** Medium (2-3 hours)

```python
# Continue training best model with lower learning rate
for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-5  # 10x lower

# Train for additional epochs
model, history_cont = train_model(
    model, criterion, optimizer, scheduler,
    num_epochs=15, patience=7
)
```

**Why try:** Can squeeze out extra performance from already-trained models.

---

### Priority 2: Model Architecture

#### 5. Larger Model Variants
**Expected Gain:** +1-2%
**Effort:** High (full retraining, 3-4 hours each)

```python
# Try larger versions of successful architectures:

# EfficientNet-V2-M (medium)
model = models.efficientnet_v2_m(weights='IMAGENET1K_V1')

# DenseNet-169 or DenseNet-201
model = models.densenet169(weights='IMAGENET1K_V1')
model = models.densenet201(weights='IMAGENET1K_V1')

# ResNet-50
model = models.resnet50(weights='IMAGENET1K_V1')
```

**Why try:** Larger models have more capacity to learn complex patterns.

**Trade-off:** Significantly longer training time, more GPU memory.

---

#### 6. Vision Transformer (ViT)
**Expected Gain:** +1-3%
**Effort:** High (new architecture, 4-6 hours)

```python
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=12
)
```

**Why try:** Transformers have shown superior performance on many vision tasks.

**Trade-off:** Requires more data, different hyperparameter tuning strategy.

---

### Priority 3: Advanced Techniques

#### 7. Knowledge Distillation
**Expected Gain:** +0.5-1%
**Effort:** High (4-5 hours)

```python
# Use ensemble as teacher to train a better single model
teacher_models = [model_densenet, model_efficientnet]
student_model = models.efficientnet_v2_s()

# Train student to match ensemble predictions
loss = distillation_loss(student_output, teacher_ensemble_output, temperature=3)
```

**Why try:** Can transfer ensemble knowledge to a single model for faster inference.

---

#### 8. Automatic Augmentation (AutoAugment)
**Expected Gain:** +1-2%
**Effort:** Medium (2-3 hours)

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

transform = transforms.Compose([
    AutoAugment(AutoAugmentPolicy.IMAGENET),
    # ... rest of transforms
])
```

**Why try:** Automatically searches for optimal augmentation policies.

---

#### 9. Multi-Scale Training
**Expected Gain:** +0.5-1%
**Effort:** Medium (3-4 hours)

```python
# Randomly vary input size during training
scales = [224, 256, 288, 320]
random_scale = random.choice(scales)
transform = transforms.RandomResizedCrop(random_scale)
```

**Why try:** Makes model more robust to scale variations.

---

#### 10. Attention Mechanisms
**Expected Gain:** +0.5-1.5%
**Effort:** High (requires architecture modification)

```python
# Add attention modules to existing models
from attention_modules import CBAM, SEModule

# Insert attention after convolution blocks
model = add_attention_module(model, attention_type='CBAM')
```

**Why try:** Helps model focus on discriminative regions (insect features).

---

### Priority 4: Data Collection

#### 11. Additional Data Collection
**Expected Gain:** +2-5% (potentially)
**Effort:** Very High (days to weeks)

**Approaches:**
- Collect more images for underperforming classes
- Use data augmentation to generate synthetic samples
- Leverage semi-supervised learning with unlabeled data

**Why try:** More data almost always helps, especially for underrepresented classes.

---

#### 12. External Dataset Transfer
**Expected Gain:** +1-3%
**Effort:** High (finding compatible datasets)

```python
# Pre-train on related insect dataset before fine-tuning
# Example: iNaturalist insects subset
model = pretrain_on_inaturalist()
model = finetune_on_target_dataset()
```

**Why try:** Domain-specific pretraining can provide better initialization.

---

## 📊 Expected Impact Summary

| Technique | Priority | Effort | Expected Gain | Cumulative Potential |
|-----------|----------|--------|---------------|---------------------|
| **Current Achievement** | - | - | - | **94.14%** |
| MixUp | High | Medium | +1-3% | 95-97% |
| Larger Models (EfficientNet-V2-M) | High | High | +1-2% | 95-96% |
| Larger Input Size (384x384) | Medium | Low | +0.5-1.5% | 94.6-95.6% |
| CutMix | Medium | Medium | +1-2% | 95-96% |
| AutoAugment | Medium | Medium | +1-2% | 95-96% |
| Vision Transformer | Low | Very High | +1-3% | 95-97% |
| Additional Data | Low | Very High | +2-5% | 96-99% |
| **Combining Top 3** | - | - | - | **~96-97%** |

---

## 🎯 Recommended Next Steps

### If You Need 95%+:

**Option A: Quick Win Path (2-3 days)**
1. Implement MixUp augmentation
2. Train with larger input size (384x384)
3. Re-ensemble with new models

**Expected:** 95-96% accuracy

---

**Option B: Maximum Accuracy Path (1-2 weeks)**
1. Implement MixUp + CutMix
2. Train larger models (EfficientNet-V2-M, DenseNet-169)
3. Implement AutoAugment
4. Use multi-scale training
5. Final ensemble with all models + TTA

**Expected:** 96-97% accuracy

---

**Option C: Research-Grade Path (1+ months)**
1. Collect additional data for difficult classes
2. Implement Vision Transformer
3. Apply knowledge distillation
4. Use attention mechanisms
5. Extensive hyperparameter search

**Expected:** 97-98%+ accuracy

---

## 💡 Tips for Future Experiments

### Experiment Tracking
```python
# Keep detailed logs
experiment_log = {
    'date': '2024-XX-XX',
    'model': 'DenseNet-121',
    'changes': 'Added dropout 0.4, weight_decay 0.01',
    'val_acc': 0.9196,
    'test_acc': 0.9048,
    'notes': 'Reduced overfitting successfully'
}
```

### Validation Strategy
- Always validate on validation set before testing
- Use cross-validation for small datasets
- Keep test set untouched until final evaluation

### Monitoring
Watch for:
- **Overfitting:** Train acc >> Val acc → More regularization
- **Underfitting:** Both low → Larger model or train longer
- **Instability:** High variance → Lower learning rate
- **Plateau:** No improvement → Try different technique

---

## 📁 Code Files Available

| File | Purpose | Status |
|------|---------|--------|
| `improved_augmentation.py` | Enhanced augmentation | ✅ Used |
| `ensemble_utils.py` | Ensemble + TTA functions | ✅ Used |
| `advanced_training_techniques.py` | MixUp, CutMix, SWA, Focal Loss | ⚠️ Partially used |
| `load_and_ensemble.py` | Standalone ensemble script | ✅ Used |
| `test_time_augmentation.py` | TTA implementation | ✅ Used |

---

## 🎉 Conclusion

You've achieved **94.14% accuracy** through systematic experimentation:
- ✅ Enhanced data augmentation
- ✅ Careful hyperparameter tuning
- ✅ Model ensemble
- ✅ Test-time augmentation

For further improvements, the most promising untried techniques are:
1. **MixUp/CutMix** (highest expected return)
2. **Larger models** (EfficientNet-V2-M)
3. **Larger input size** (384x384)

**Congratulations on your excellent results!** 🎯
