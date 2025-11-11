# Insect Classification with Transfer Learning

> **Transfer learning project for classifying 12 insect species in agricultural environments using PyTorch and state-of-the-art CNN architectures**

**🏆 Best Accuracy: 94.14%** (Ensemble + TTA)

---

## 📊 Quick Results

| Model | Test Accuracy | Training Time (GPU) |
|-------|--------------|---------------------|
| ResNet-18 (Finetuned) | 88.10% | ~15-20 min |
| DenseNet-121 | 89.93% | ~18-25 min |
| EfficientNet-V2-S | 93.04% | ~20-30 min |
| **🏆 Ensemble + TTA** | **94.14%** | ~50-100 img/sec |

<img width="1590" height="590" alt="image" src="https://github.com/user-attachments/assets/2c12274e-b080-432e-94ae-1563048120cb" />

---

## 📁 Project Structure

```
insect-classification-transfer-learning/
├── 📂 source/                      # All code files
│   ├── insect_classification_transfer_learning.ipynb -------> benchmark models with standard data augmentations, less performance 
│   ├── accuracy_impro.ipynb                          -------> models with enhanced data augmentations, advanced training and predicting techniques, high performance 
│   ├── reorganize_dataset.py
│   ├── ensemble_utils.py
│   ├── improved_augmentation.py
│   ├── advanced_training_techniques.py
│   ├── test_time_augmentation.py
│   ├── add_visualization_comparison.py
│   └── update_comparison_cell.py
│
├── 📂 models/                      # Trained model weights (NOT INCLUDED)
│   ├── insect_classifier_finetuned.pth          (43MB - ResNet-18)
│   ├── insect_classifier_densenet121.pth        (27MB - DenseNet-121)
│   └── insect_classifier_efficientnet_v2_s.pth  (78MB - EfficientNet)
│   └── ⚠️ Train models yourself - see setup instructions
│
├── 📂 datas/                       # Dataset (NOT INCLUDED)
│   ├── train_organized/            (11,499 images)
│   ├── valid_organized/            (1,095 images)
│   └── test_organized/             (546 images)
│   └── ⚠️ Download from Kaggle - see setup instructions
│
├── 📄 README.md                    # This file
├── 📄 ENSEMBLE_TTA_GUIDE.md        # Complete ensemble & TTA guide
├── 📄 MODEL_COMPARISON.md          # Architecture details
├── 📄 ACCURACY_IMPROVEMENT_GUIDE.md # Training tips & techniques
└── 📄 requirements.txt             # Python dependencies
```

**⚠️ Note:** Due to GitHub file size limitations, `models/` and `datas/` directories are not included in this repository. See the Setup section for instructions on downloading the dataset and training models.

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd /home/ian/Projects

# Install dependencies
pip install -r requirements.txt

# Or with conda
conda create -n myenv python=3.8
conda activate myenv
pip install -r requirements.txt
```

### 2. Download Dataset

**⚠️ Dataset and model weights are not included in this repository due to size limitations.**

**Download the dataset:**
- **Source:** [Crop Pests Dataset on Kaggle](https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset)
- **Dataset details:**
  - 12 insect classes: Ants, Bees, Beetles, Caterpillars, Earthworms, Earwigs, Grasshoppers, Moths, Slugs, Snails, Wasps, Weevils
  - 13,140 total images: 11,499 train / 1,095 val / 546 test
  - Original format: YOLO (object detection)

**Prepare the dataset:**
```bash
# After downloading, extract to datas/
unzip archive.zip -d datas/

# Convert from YOLO to PyTorch ImageFolder format
python source/reorganize_dataset.py
```

This will create `datas/train_organized/`, `datas/valid_organized/`, and `datas/test_organized/` directories.

### 3. Train Models

**⚠️ You need to train the models yourself and save them as `.pth` files.**

```bash
cd source
jupyter notebook insect_classification_transfer_learning.ipynb 
jupyter notebook accuracy_impro.ipynb
```

**Training process:**
1. Run cells sequentially to train all 4 models:
   - ResNet-18 (Finetuned)
   - DenseNet-121
   - EfficientNet-V2-S

2. Models will be automatically saved as `.pth` files in the `models/` directory:
   ```
   models/
   ├── insect_classifier_finetuned.pth          (ResNet-18)
   ├── insect_classifier_densenet121.pth        (DenseNet-121)
   └── insect_classifier_efficientnet_v2_s.pth  (EfficientNet-V2-S)
   ```

3. Training time (with GPU):
   - ResNet-18: ~15-20 minutes
   - DenseNet-121: ~18-25 minutes
   - EfficientNet-V2-S: ~20-30 minutes

**Expected Results:**
- ResNet-18: 88.10% test accuracy
- DenseNet-121: 89.93% test accuracy
- EfficientNet-V2-S: 93.04% test accuracy

### 4. Evaluate with Ensemble + TTA

**One cell to get 94.14% accuracy:**

```python
from ensemble_utils import load_models, evaluate_ensemble_with_tta
import torch
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder('../datas/test_organized', test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, 
                                          shuffle=False, num_workers=0)

models = load_models(num_classes=12, model_dir='../models', device=device)

acc, _, _ = evaluate_ensemble_with_tta(
    list(models.values()), test_loader, n_augmentations=4,
    weights=[0.15, 0.35, 0.50], device=device
)

print(f"✨ FINAL ACCURACY: {acc*100:.2f}% ✨")
```

---

## 🎯 Dataset Information

### 12 Insect Classes

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Ants | 956 | 91 | 46 | 1,093 |
| Bees | 956 | 91 | 46 | 1,093 |
| Beetles | 957 | 92 | 45 | 1,094 |
| Caterpillars | 957 | 92 | 45 | 1,094 |
| Earthworms | 957 | 92 | 46 | 1,095 |
| Earwigs | 957 | 91 | 46 | 1,094 |
| Grasshoppers | 957 | 92 | 46 | 1,095 |
| Moths | 956 | 91 | 45 | 1,092 |
| Slugs | 957 | 91 | 46 | 1,094 |
| Snails | 957 | 92 | 45 | 1,094 |
| Wasps | 957 | 91 | 45 | 1,093 |
| Weevils | 957 | 89 | 45 | 1,091 |
| **Total** | **11,499** | **1,095** | **546** | **13,140** |

### Dataset Format Conversion

Original: **YOLO format** (for object detection)
```
train/
├── images/
│   └── img001.jpg
└── labels/
    └── img001.txt  (class_id x_center y_center width height)
```

Converted to: **PyTorch ImageFolder** (for classification)
```
train_organized/
├── Ants/
│   ├── img001.jpg
│   └── img002.jpg
├── Bees/
│   └── ...
└── ...
```

Conversion script: `source/reorganize_dataset.py`

---

## 🏗️ Model Architectures

### 1. ResNet-18 (Baseline)
- **Architecture**: 18 layers, residual connections
- **Parameters**: ~11.7M
- **Optimizer**: SGD (lr=0.001, momentum=0.9)
- **Scheduler**: StepLR (decay every 7 epochs)
- **Accuracy**: 88.10%

### 2. DenseNet-121
- **Architecture**: Dense connections, 121 layers
- **Parameters**: ~8M
- **Optimizer**: Adam (lr=0.0001, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Accuracy**: 89.93%

### 3. EfficientNet-V2-S (Best Single Model)
- **Architecture**: Compound scaling, Fused-MBConv blocks
- **Parameters**: ~21.5M
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.01)
- **Scheduler**: CosineAnnealingWarmRestarts
- **Loss**: CrossEntropyLoss with label_smoothing=0.1
- **Accuracy**: 93.04%

### 4. Ensemble + TTA (Best Overall)
- **Combination**: All 3 models + Test-Time Augmentation
- **Weights**: [0.15, 0.35, 0.50] for ResNet, DenseNet, EfficientNet
- **TTA**: 5 augmentations (original, H-flip, V-flip, rotate 90°, rotate 270°)
- **Accuracy**: **94.14%** 🏆

For detailed architecture comparisons, see `MODEL_COMPARISON.md`

---

## 🎓 Transfer Learning Approaches

### Approach 1: Finetuning (ResNet-18, DenseNet, EfficientNet)
**Train all layers**
```python
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 12)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```
- ✅ Best accuracy
- ❌ Slower training
- 🎯 Use when: You have enough data and compute

### Approach 2: Feature Extraction (ResNet-18 FE)
**Freeze backbone, train only final layer**
```python
model = models.resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 12)
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
```
- ✅ Fast training (5-10 min)
- ❌ Lower accuracy (~82-88%)
- 🎯 Use when: Quick prototyping, limited compute

---

## 🔧 Training Techniques

### Data Augmentation
```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])
```

### Regularization
- **Dropout**: 0.3-0.5 in classifiers
- **Weight Decay**: 1e-4 to 0.01 depending on model
- **Label Smoothing**: 0.1 for EfficientNet
- **Early Stopping**: Patience of 5 epochs

### Learning Rate Schedules
- **StepLR**: For ResNet (simple, proven)
- **CosineAnnealingLR**: For DenseNet (smooth decay)
- **CosineAnnealingWarmRestarts**: For EfficientNet (periodic resets)

---

## 🤝 Ensemble & Test-Time Augmentation

### Ensemble Learning
**Combine multiple models for better predictions**

```
Model 1 (88%)  ┐
Model 2 (90%)  ├─→ Weighted Average ─→ 94% (Better!)
Model 3 (93%)  ┘
```

**Why it works:**
- Different models make different mistakes
- Averaging reduces individual errors
- More robust predictions

### Test-Time Augmentation (TTA)
**Apply multiple augmentations at inference**

```
Original    ┐
H-Flip      │
V-Flip      ├─→ Average Predictions ─→ +0.5-1% accuracy
Rotate 90°  │
Rotate 270° ┘
```

### Combined: Ensemble + TTA
```
Ensemble (93.96%) + TTA (+0.18%) = 94.14% 🏆
```

**For complete guide, see:** `ENSEMBLE_TTA_GUIDE.md`

---

## 📊 Experimental Results

### Accuracy Progression

```
88.10%  ████████████████████            ResNet-18 (Baseline)
89.93%  █████████████████████           + DenseNet-121
93.04%  ██████████████████████████      + EfficientNet-V2-S
93.96%  ███████████████████████████     + Ensemble (Weighted)
94.14%  ████████████████████████████ 🏆 + TTA
```

### Training Experiments

| Experiment | Best Val Acc | Test Acc | Notes |
|-----------|-------------|----------|-------|
| ResNet-18 (baseline) | 88.77% | 88.10% | SGD, StepLR |
| + Enhanced augmentation | 89.68% | 88.28% | More aggressive transforms |
| DenseNet-121 (baseline) | 90.32% | 89.93% | Adam, CosineAnnealing |
| + Enhanced augmentation | 91.05% | 88.46% | Overfitting detected |
| + Weight decay (1e-3) | 91.51% | 89.38% | Better regularization |
| + Dropout (0.4) | 91.96% | 90.84% | Best DenseNet result |
| EfficientNet-V2-S | 94.70% | 93.04% | AdamW, label smoothing |
| EfficientNet + TTA | - | 93.77% | TTA improves by +0.73% |
| Ensemble (Equal) | - | 93.04% | Equal weights |
| Ensemble (Weighted) | - | 93.96% | Weights: [0.15, 0.35, 0.50] |
| Ensemble (Equal) + TTA | - | 93.41% | Equal weights + TTA |
| **Ensemble (Weighted) + TTA** | - | **94.14%** | **Best result** 🏆 |

### Per-Class Performance (Ensemble + TTA)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Ants | 0.96 | 0.98 | 0.97 |
| Bees | 0.98 | 0.96 | 0.97 |
| Beetles | 0.94 | 0.93 | 0.94 |
| Caterpillars | 0.93 | 0.95 | 0.94 |
| Earthworms | 0.95 | 0.96 | 0.96 |
| Earwigs | 0.94 | 0.92 | 0.93 |
| Grasshoppers | 0.96 | 0.97 | 0.97 |
| Moths | 0.92 | 0.91 | 0.92 |
| Slugs | 0.95 | 0.94 | 0.95 |
| Snails | 0.94 | 0.96 | 0.95 |
| Wasps | 0.97 | 0.95 | 0.96 |
| Weevils | 0.93 | 0.94 | 0.94 |
| **Macro Avg** | **0.95** | **0.95** | **0.95** |

---

## 💾 Using Trained Models

### Load a Single Model

```python
import torch
from torchvision import models

# Load ResNet-18
model = models.resnet18()
model.fc = torch.nn.Linear(512, 12)
model.load_state_dict(torch.load('models/insect_classifier_finetuned.pth'))
model.eval()

# Load EfficientNet-V2-S
from ensemble_utils import load_efficientnet_v2_s
model = load_efficientnet_v2_s('models/insect_classifier_efficientnet_v2_s.pth', 
                                num_classes=12, device='cuda:0')
```

### Inference on Custom Image

```python
from PIL import Image
import torch
from torchvision import transforms

# Prepare transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and predict
img = Image.open('path/to/insect.jpg')
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    class_name = class_names[predicted.item()]
    
print(f"Predicted: {class_name}")
```

### Ensemble Prediction

```python
from ensemble_utils import load_models, evaluate_ensemble

models = load_models(num_classes=12, model_dir='models', device='cuda:0')
acc, predictions, labels = evaluate_ensemble(
    list(models.values()), 
    test_loader,
    weights=[0.15, 0.35, 0.50],
    device='cuda:0'
)
```

---

## 🛠️ Requirements

### System Requirements
- Python 3.8+
- GPU with CUDA support (recommended) or CPU
- 8GB RAM minimum (16GB recommended)
- 5GB disk space for models and data

### Python Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=9.0.0
scikit-learn>=1.0.0
seaborn>=0.11.0
jupyter>=1.0.0
pandas>=1.3.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📚 Documentation Files

| File | Description | Size |
|------|-------------|------|
| `README.md` | **This file** - Complete project overview | 6.3KB |
| `ENSEMBLE_TTA_GUIDE.md` | **Complete ensemble & TTA guide** | 15KB |
| `MODEL_COMPARISON.md` | Architecture details and comparisons | 8.5KB |
| `ACCURACY_IMPROVEMENT_GUIDE.md` | Training tips and techniques | 8.8KB |

---

## 🐛 Troubleshooting

### "CUDA out of memory"
**Solution:** Reduce batch size
```python
# In notebook, change:
batch_size = 16  # or 8
```

### "Training is very slow"
**Solution:** Check GPU is being used
```python
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))
```
Or use Feature Extractor approach (faster)

### "RuntimeError: DataLoader worker"
**Solution:** Set `num_workers=0` in Jupyter
```python
DataLoader(dataset, batch_size=32, num_workers=0)
```

### "Model not converging"
**Solutions:**
- Check learning rate (try 1e-4 to 1e-3)
- Increase training epochs
- Reduce data augmentation intensity
- Check for data loading issues

---

## 🎯 Project Achievements

✅ **Dataset Preparation**
- Converted 13K images from YOLO to ImageFolder format
- Organized into 12 balanced classes

✅ **Model Training**
- Trained 4 architectures with optimized hyperparameters
- Implemented proper regularization (dropout, weight decay, label smoothing)
- Achieved 93%+ on best single model

✅ **Advanced Techniques**
- Ensemble learning with optimal weighted averaging
- Test-time augmentation for inference boost
- Combined methods for 94.14% accuracy

✅ **Production-Ready Code**
- Clean utility module (`ensemble_utils.py`)
- Well-documented functions
- Easy-to-use Jupyter notebooks

✅ **Comprehensive Documentation**
- 4 detailed guides
- Quick-start tutorials
- Complete API reference

---

## 🚀 Next Steps

### Improve Accuracy Further
1. **Collect more data** for underperforming classes
2. **Try larger models** (EfficientNet-V2-M, ResNet-50)
3. **Advanced techniques** (SWA, MixUp, CutMix) - see `source/advanced_training_techniques.py`
4. **Fine-tune ensemble weights** for your specific test set
5. **Add more TTA variations** (brightness, contrast adjustments)

### Deployment
1. **Export to ONNX** for production inference
2. **Quantization** for mobile deployment
3. **FastAPI/Flask** for REST API
4. **Docker containerization**
5. **Model versioning** with MLflow

### Research Extensions
1. **Object detection** (return bounding boxes)
2. **Multi-label classification** (multiple insects per image)
3. **Few-shot learning** for new insect classes
4. **Active learning** for efficient labeling

---

## 📄 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

- Transfer learning tutorial adapted from [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- Pretrained weights from ImageNet
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- DenseNet: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- EfficientNet: [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)

---

## 📞 Support

For questions or issues:
1. Check `ENSEMBLE_TTA_GUIDE.md` for ensemble/TTA usage
2. Check `ACCURACY_IMPROVEMENT_GUIDE.md` for training tips
3. Check `MODEL_COMPARISON.md` for architecture details
4. Review experimental results in `source/expreimental_summary.txt`

---

**🎉 You're ready to classify insects with 94.14% accuracy!**

**Start here:**
1. Open `source/insect_classification_transfer_learning.ipynb`
2. Train models (or use pre-trained from `models/`)
3. Run ensemble + TTA using `ENSEMBLE_TTA_GUIDE.md`
4. Achieve 94.14% accuracy! 🏆
