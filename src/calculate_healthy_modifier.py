# healthy_modifier_from_eval.py
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader
from src.dataset import Nutrition5kDataset
from src.model import ImageOnlyNutritionModel
import json
import os

# Paths and config
CSV_PATH = "data/Healthy Food/Healthy Labels.csv"
IMG_DIR = "data/Healthy Food/Images"
BATCH_SIZE = 32
NUTRIENTS = ["calories", "protein", "fat", "carbs"]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load normalization stats from training
with open("checkpoints/norm_stats.json", "r") as f:
    stats = json.load(f)
    MEAN = torch.tensor(stats["mean"]).to(DEVICE) 
    STD = torch.tensor(stats["std"]).to(DEVICE)

# Load dataset (assume .jpeg extension for healthy food images, no normalization)
dataset = Nutrition5kDataset(CSV_PATH, IMG_DIR, image_extension=".jpeg", normalize=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = ImageOnlyNutritionModel().to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE))
model.eval()

all_preds = []
all_targets = []

# Predict
with torch.no_grad():
    for images, targets in dataloader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs = model(images)
        preds = outputs * STD + MEAN  # denormalize predictions
        targs = targets  # already unnormalized
        all_preds.append(preds.cpu())
        all_targets.append(targs.cpu())

# Combine predictions and targets
all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

# Compute healthy modifier as predicted / actual average ratio
modifier = {}
for i, nutrient in enumerate(NUTRIENTS):
    pred_mean = all_preds[:, i].mean()
    actual_mean = all_targets[:, i].mean()
    ratio = actual_mean / pred_mean if pred_mean > 0 else 1.0
    modifier[nutrient] = round(float(ratio), 3)  # ensure JSON-serializable float
    print(f"{nutrient.capitalize():<8} | Modifier: {ratio:.3f} (Actual: {actual_mean:.2f}, Predicted: {pred_mean:.2f})")

# Save to JSON
os.makedirs("checkpoints", exist_ok=True)
with open("checkpoints/healthy_modifier.json", "w") as f:
    json.dump(modifier, f, indent=4)

# Plot
for i, nutrient in enumerate(NUTRIENTS):
    y_true = all_targets[:, i]
    y_pred = all_preds[:, i]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"{nutrient.capitalize():<8} | R2: {r2:.3f} | MAE: {mae:.2f}")

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel(f"True {nutrient}")
    plt.ylabel(f"Predicted {nutrient}")
    plt.title(f"Healthy Food Prediction vs. Actual: {nutrient}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()