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
import argparse


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

# Load healthy modifier
with open("checkpoints/healthy_modifier.json", "r") as f:
    healthy_modifier = json.load(f)
    
# Argument parsing
parser = argparse.ArgumentParser(description="Evaluate adjusted predictions with healthy modifier")
parser.add_argument("--nutrient", type=str, help="Specify nutrient to apply modifier (e.g. protein, fat). Use 'all' or leave blank for all.")
args = parser.parse_args()

# Determine which nutrients to apply modifier to
if args.nutrient is None or args.nutrient.lower() == "all":
    modified_nutrients = list(healthy_modifier.keys())  # Apply all from modifier
else:
    modified_nutrients = [args.nutrient.lower()]
    print(f"\nℹ️  Applying healthy modifier only to: {modified_nutrients}")


# Apply healthy modifier to selected nutrients
adjusted_preds = all_preds.copy()
for i, nutrient in enumerate(NUTRIENTS):
    if nutrient in modified_nutrients and nutrient in healthy_modifier:
        adjusted_preds[:, i] *= healthy_modifier[nutrient]


# Recalculate calories from adjusted macros: calories = fat*9 + protein*4 + carbs*4
fat_idx, protein_idx, carbs_idx = NUTRIENTS.index("fat"), NUTRIENTS.index("protein"), NUTRIENTS.index("carbs")
adjusted_preds[:, 0] = (
    adjusted_preds[:, fat_idx] * 9 +
    adjusted_preds[:, protein_idx] * 4 +
    adjusted_preds[:, carbs_idx] * 4
)

# Print average true nutrient values
avg_nutrients = all_targets.mean(axis=0)
print("\nAverage True Nutrient Values:")
for nutrient, avg in zip(NUTRIENTS, avg_nutrients):
    print(f"{nutrient.capitalize():<8}: {avg:.2f}")
    
avg_preds = adjusted_preds.mean(axis=0) 
print("\nAverage Adjusted Predicted Nutrient Values:") 
for nutrient, avg in zip(NUTRIENTS, avg_preds): 
    print(f"{nutrient.capitalize():<8}: {avg:.2f}")

# Plot adjusted predictions
for i, nutrient in enumerate(NUTRIENTS):
    y_true = all_targets[:, i]
    y_pred = adjusted_preds[:, i]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"{nutrient.capitalize():<8} | R2: {r2:.3f} | MAE: {mae:.2f}")

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel(f"True {nutrient}")
    plt.ylabel(f"Predicted {nutrient} (Adjusted)")
    plt.title(f"Adjusted Prediction vs. Actual: {nutrient}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
