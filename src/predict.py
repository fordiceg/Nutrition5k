# predict.py
import argparse
import torch
import json
from PIL import Image
from torchvision import transforms
from src.model import ImageOnlyNutritionModel
import numpy as np

# Paths
MODEL_PATH = "checkpoints/best_model.pth"
NORM_STATS_PATH = "checkpoints/norm_stats.json"
MODIFIER_PATH = "checkpoints/healthy_modifier.json"

# Nutrients
NUTRIENTS = ["calories", "protein", "fat", "carbs"]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load normalization stats
with open(NORM_STATS_PATH, "r") as f:
    stats = json.load(f)
    MEAN = torch.tensor(stats["mean"]).to(DEVICE)
    STD = torch.tensor(stats["std"]).to(DEVICE)

# Load healthy modifier
with open(MODIFIER_PATH, "r") as f:
    healthy_modifier = json.load(f)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to image")
parser.add_argument("--modifier", type=str, choices=["all", "fat", "protein", "carbs"], default=None, help="Which modifier to apply")
args = parser.parse_args()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open(args.image).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(DEVICE)

# Load model
model = ImageOnlyNutritionModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Predict
with torch.no_grad():
    output = model(image_tensor)[0]  # shape: [4]
    pred = output * STD + MEAN  # denormalize
    pred = pred.cpu().numpy()

# Apply modifier if specified
if args.modifier:
    modified_pred = pred.copy()
    if args.modifier == "all":
        for i, nutrient in enumerate(NUTRIENTS):
            if nutrient in healthy_modifier:
                modified_pred[i] *= healthy_modifier[nutrient]
    else:
        if args.modifier in healthy_modifier:
            idx = NUTRIENTS.index(args.modifier)
            modified_pred[idx] *= healthy_modifier[args.modifier]

        # Recalculate calories if any macro modified
        if args.modifier in ["fat", "protein", "carbs"]:
            fat = modified_pred[NUTRIENTS.index("fat")]
            protein = modified_pred[NUTRIENTS.index("protein")]
            carbs = modified_pred[NUTRIENTS.index("carbs")]
            modified_pred[0] = fat * 9 + protein * 4 + carbs * 4

    print("\n=== Prediction with Healthy Modifier ===")
    for k, v in zip(NUTRIENTS, modified_pred):
        print(f"{k.capitalize():>8}: {v:.2f}")
else:
    print("\n=== Prediction without Modifier ===")
    for k, v in zip(NUTRIENTS, pred):
        print(f"{k.capitalize():>8}: {v:.2f}")
