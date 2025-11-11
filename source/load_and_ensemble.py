"""
Load Saved Models and Perform Ensemble Prediction
==================================================
This script shows how to:
1. Load saved model weights
2. Recreate model architectures
3. Perform ensemble prediction on test set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# ============================================
# STEP 1: Setup Device and Data
# ============================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load test data (same transforms as validation)
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'datas'
test_dataset = datasets.ImageFolder(
    f'{data_dir}/test_organized', 
    test_transforms
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=0  # Set to 0 for Jupyter compatibility
)

class_names = test_dataset.classes
num_classes = len(class_names)

print(f"\nDataset Info:")
print(f"Number of classes: {num_classes}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {class_names}")


# ============================================
# STEP 2: Load ResNet-18 (Finetuned)
# ============================================

def load_resnet18(weights_path, num_classes):
    """Load ResNet-18 finetuned model"""
    print(f"\nLoading ResNet-18 from {weights_path}...")
    
    # Recreate the exact architecture
    model = models.resnet18(weights=None)  # Don't load pretrained weights
    
    # Replace final layer to match saved model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load saved weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("✓ ResNet-18 loaded successfully")
    return model


# ============================================
# STEP 3: Load DenseNet-121
# ============================================

def load_densenet121(weights_path, num_classes):
    """Load DenseNet-121 model with dropout"""
    print(f"\nLoading DenseNet-121 from {weights_path}...")
    
    # Recreate the exact architecture
    model = models.densenet121(weights=None)
    
    # Replace classifier with dropout (match your training setup)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, num_classes)
    )
    
    # Load saved weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("✓ DenseNet-121 loaded successfully")
    return model


# ============================================
# STEP 4: Load EfficientNet-V2-S
# ============================================

def load_efficientnet_v2_s(weights_path, num_classes):
    """Load EfficientNet-V2-S model"""
    print(f"\nLoading EfficientNet-V2-S from {weights_path}...")
    
    # Recreate the exact architecture
    model = models.efficientnet_v2_s(weights=None)
    
    # Replace classifier WITHOUT dropout (match your saved model)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # Load saved weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("✓ EfficientNet-V2-S loaded successfully")
    return model


# ============================================
# STEP 5: Main Execution
# ============================================

def main():
    """Main function to run ensemble evaluation"""
    
    print("\n" + "="*60)
    print("LOADING ALL SAVED MODELS")
    print("="*60)

    # Load each model
    model_resnet = load_resnet18('insect_classifier_finetuned.pth', num_classes)
    model_densenet = load_densenet121('insect_classifier_densenet121.pth', num_classes)
    model_efficientnet = load_efficientnet_v2_s('insect_classifier_efficientnet_v2_s.pth', num_classes)

    print("\n✓ All models loaded successfully!")
    
    return model_resnet, model_densenet, model_efficientnet


# ============================================
# STEP 6: Evaluate Individual Models
# ============================================

def evaluate_model(model, dataloader, model_name):
    """Evaluate a single model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"{model_name:25s} Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return accuracy


# ============================================
# STEP 7: Ensemble Prediction
# ============================================

def ensemble_predict(models, inputs, weights=None):
    """
    Ensemble prediction using multiple models
    
    Args:
        models: List of models
        inputs: Input tensor
        weights: Optional weights for each model
    
    Returns:
        Ensemble predictions
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    ensemble_probs = None
    
    for model, weight in zip(models, weights):
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            
            if ensemble_probs is None:
                ensemble_probs = weight * probs
            else:
                ensemble_probs += weight * probs
    
    return ensemble_probs


def evaluate_ensemble(models, dataloader, weights=None, ensemble_name="Ensemble"):
    """Evaluate ensemble of models"""
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Get ensemble predictions
        ensemble_probs = ensemble_predict(models, inputs, weights)
        _, preds = torch.max(ensemble_probs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"{ensemble_name:25s} Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return accuracy, all_preds, all_labels


# ============================================
# STEP 8: Run Everything
# ============================================

if __name__ == '__main__':
    # Load all models
    model_resnet, model_densenet, model_efficientnet = main()
    
    # Evaluate individual models
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*60)
    
    acc_resnet = evaluate_model(model_resnet, test_loader, "ResNet-18 (Finetuned)")
    acc_densenet = evaluate_model(model_densenet, test_loader, "DenseNet-121")
    acc_efficientnet = evaluate_model(model_efficientnet, test_loader, "EfficientNet-V2-S")
    
    # Ensemble predictions
    print("\n" + "="*60)
    print("ENSEMBLE PREDICTIONS")
    print("="*60)
    
    # Strategy 1: Equal weights (simple average)
    models_list = [model_resnet, model_densenet, model_efficientnet]
    acc_ensemble_equal, _, _ = evaluate_ensemble(
        models_list, 
        test_loader,
        weights=None,
        ensemble_name="Ensemble (Equal Weights)"
    )
    
    # Strategy 2: Weighted by validation performance
    # Give more weight to better models
    # Based on your results: EfficientNet (93.04%), DenseNet (90.48%), ResNet (~88%)
    weights_performance = [0.15, 0.35, 0.50]  # ResNet, DenseNet, EfficientNet
    acc_ensemble_weighted, _, _ = evaluate_ensemble(
        models_list,
        test_loader,
        weights=weights_performance,
        ensemble_name="Ensemble (Weighted)"
    )
    
    # Strategy 3: Best 2 models only (DenseNet + EfficientNet)
    best_models = [model_densenet, model_efficientnet]
    acc_ensemble_best2, _, _ = evaluate_ensemble(
        best_models,
        test_loader,
        weights=None,
        ensemble_name="Ensemble (Best 2)"
    )
    
    # Final comparison
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    
    results = [
        ("ResNet-18 (Finetuned)", acc_resnet),
        ("DenseNet-121", acc_densenet),
        ("EfficientNet-V2-S", acc_efficientnet),
        ("─" * 40, None),
        ("Ensemble (Equal Weights)", acc_ensemble_equal),
        ("Ensemble (Weighted)", acc_ensemble_weighted),
        ("Ensemble (Best 2)", acc_ensemble_best2),
    ]
    
    for name, acc in results:
        if acc is None:
            print(name)
        else:
            improvement = ""
            if "Ensemble" in name:
                best_single = max(acc_resnet, acc_densenet, acc_efficientnet)
                diff = (acc - best_single) * 100
                improvement = f"  (+{diff:.2f}% improvement)" if diff > 0 else f"  ({diff:.2f}%)"
            print(f"{name:40s} {acc:.4f} ({acc*100:.2f}%){improvement}")
    
    print("="*60)
    
    # Determine best approach
    best_acc = max(acc_ensemble_equal, acc_ensemble_weighted, acc_ensemble_best2)
    if best_acc == acc_ensemble_weighted:
        print("🏆 Best: Weighted Ensemble")
    elif best_acc == acc_ensemble_best2:
        print("🏆 Best: Best 2 Models Ensemble")
    else:
        print("🏆 Best: Equal Weights Ensemble")
    
    print(f"\n✨ Final Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("="*60)

