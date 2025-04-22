@author: Graham
"""

# Nutrition5k Multimodal Estimation

This project uses images to estimate nutritional information (calories, protein, fat, carbs) using a deep learning model.
It then utilizes a smaller, healthy food dataset to produce multipliers that can be used to improve nutritional estimation.

## 🔧 Features
- EfficientNet-B0 + DistilBERT based architecture
- Trains on a trimmed subset of the Nutrition5k dataset
- Handles modifier input at inference time
- Evaluates and compares predictions with and without modifiers
- Produces diagnostic visualizations

## 🗂️ Project Structure

```
Nutrition5k/
├── assets/
├── checkpoints/
│   ├── best_model.pth
│   ├── healthy_modifier.json
│   ├── norm_stats.json
│   └── val_predictions.pkl
├── data/
│   ├── Healthy Food/
│   │   ├── Healthy Labels.csv
│   │   └── Images/
│   ├── raw/
│   └── trimmed_nutrition5k/
│       ├── labels.csv
│       └── images/
├── Model Performance/
├── src/
│   ├── calculate_healthy_modifier.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── evaluate_healthy_modifier.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── trim.py
├── README.md
└── requirements.txt

## 🥗 Healthy Modifier Workflow

This flow allows the model to learn from standard meals and generalize to healthier alternatives:

### Step 0: Install dependencies
Ensure the project originates in nutrition5k and the files are located where indicated above.
pip install -r requirements.txt

### Step 1: Trim the Original Dataset
Removes outliers and saves a cleaned version of the dataset.

```bash
python src/trim_from_metadata.py
```

### Step 2: Train the Model
Trains a vision-based regression model on the trimmed dataset.

```bash
python -m src.train
```

Normalization statistics (`mean` and `std`) are saved to be reused during inference.

### Step 3: Evaluate Model Performance
Evaluates model accuracy on the training/validation set using denormalized predictions.

```bash
python src/evaluate.py
```

Includes scatter plots and R²/MAE scores for each nutrient.

### Step 4: Build the Healthy Modifier
Uses a set of labeled images of healthy meals to calculate a per-nutrient "healthy modifier" based on average prediction error.

```bash
python src/calculate_healthy_modifier.py
```

This script saves the modifier in a JSON file, e.g.:

```json
{
  "protein": 0.85,
  "fat": 0.65,
  "carbs": 0.72
}
```

### Step 5: Evaluate the Healthy Modifier's Impact
Tests how well the healthy modifier corrects predictions on the same healthy dataset.

```bash
python src/healthy_modifier_from_eval.py --nutrient [protein|fat|carbs|all]
```

This script prints MAE, R², average predicted vs. true nutrients, and generates scatter plots.

### Step 6: Predict with Healthy Modifier
You can run prediction on a single image and optionally apply the healthy modifier to any specific nutrient:

```bash
python src/predict.py --image path/to/image.jpg --modifier protein
```

Omit --modifier to see raw predictions, or pass "all" to apply the modifier to all macros.
