# evaluate_model.py
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader
from src.dataset import Nutrition5kDataset
from src.model import ImageOnlyNutritionModel
import json

# Paths and config
CSV_PATH = "data/trimmed_nutrition5k/labels.csv"
IMG_DIR = "data/trimmed_nutrition5k/images"
BATCH_SIZE = 64  # increased for speed
NUTRIENTS = ["calories", "protein", "fat", "carbs"]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load normalization stats
with open("checkpoints/norm_stats.json", "r") as f:
    stats = json.load(f)
    MEAN = torch.tensor(stats["mean"]).to(DEVICE)
    STD = torch.tensor(stats["std"]).to(DEVICE)

# Load dataset
dataset = Nutrition5kDataset(CSV_PATH, IMG_DIR)
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
        preds = outputs * STD + MEAN
        targs = targets * STD + MEAN
        all_preds.append(preds.cpu())
        all_targets.append(targs.cpu())

# Combine predictions and targets
all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

# Evaluate and plot
for i, nutrient in enumerate(NUTRIENTS):
    y_true = all_targets[:, i]
    y_pred = all_preds[:, i]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"{nutrient.capitalize():<8} | R2: {r2:.3f} | MAE: {mae:.2f}")

    # Commented out plotting for speed
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel(f"True {nutrient}")
    plt.ylabel(f"Predicted {nutrient}")
    plt.title(f"Prediction vs. Actual: {nutrient}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
