import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import json

from src.dataset import Nutrition5kDataset
from src.model import ImageOnlyNutritionModel

# === Config ===
CSV_PATH = "data/trimmed_nutrition5k/labels.csv"
IMG_DIR = "data/trimmed_nutrition5k/images"
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    df = pd.read_csv(CSV_PATH)

    # Outlier removal (optional but can improve stability)
    # for nutrient in ["calories", "protein", "fat", "carbs"]:
    #     lower = df[nutrient].quantile(0.01)
    #     upper = df[nutrient].quantile(0.99)
    #     df = df[(df[nutrient] >= lower) & (df[nutrient] <= upper)]
    
    # df = df[df["calories"] < 2500]  # or whatever your outlier threshold is
    # df = df[df["carbs"] < 300]

    # Save mean and std for denormalization
    mean = df[["calories", "protein", "fat", "carbs"]].mean().astype("float32").tolist()
    std = df[["calories", "protein", "fat", "carbs"]].std().astype("float32").tolist()
    
    with open("checkpoints/norm_stats.json", "w") as f:
        json.dump({"mean": mean, "std": std}, f)
    print("✅ Saved normalization statistics.")

    filtered_path = "data/trimmed_nutrition5k/filtered_labels.csv"
    df.to_csv(filtered_path, index=False)

    full_dataset = Nutrition5kDataset(filtered_path, IMG_DIR)
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    model = ImageOnlyNutritionModel()
    model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE).float()

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}")

        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)

                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("✅ Best model updated.")

            with open("checkpoints/val_predictions.pkl", "wb") as f:
                pickle.dump({
                    "targets": np.vstack(val_targets),
                    "predictions": np.vstack(val_preds)
                }, f)
            print("📦 Saved validation predictions.")

    print("✅ Training complete. Model saved.")

if __name__ == "__main__":
    main()
