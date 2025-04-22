import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# === Config ===
DISH_ID_PATH = "data/raw/nutrition5k_dataset/dish_ids/dish_ids_cafe1.txt"
METADATA_PATH = "data/raw/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv"
SOURCE_IMAGE_DIR = "data/raw/nutrition5k_dataset/imagery/realsense_overhead"
OUTPUT_DIR = "data/trimmed_nutrition5k"
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "labels.csv")
IMAGE_SIZE = (224, 224)
NUM_SAMPLES = 4500
RANDOM_SEED = 42

# === Setup output directories ===
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# === Load available dish IDs ===
dish_ids = pd.read_csv(DISH_ID_PATH, header=None)[0].astype(str).tolist()

# === Define and load metadata ===
metadata_columns = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein', 'num_ingrs']
df_meta = pd.read_csv(METADATA_PATH, header=None, names=metadata_columns, usecols=range(len(metadata_columns)))
df_meta = df_meta[df_meta['dish_id'].astype(str).isin(dish_ids)]

# === Rename and keep only necessary columns ===
df_meta = df_meta.rename(columns={
    'dish_id': 'id',
    'total_calories': 'calories',
    'total_protein': 'protein',
    'total_fat': 'fat',
    'total_carb': 'carbs'
})
df_meta = df_meta[['id', 'calories', 'protein', 'fat', 'carbs']]
df_meta = df_meta.sample(n=NUM_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

# === Process and copy images ===
all_rows = []
print(f"Processing {len(df_meta)} dishes...")
for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
    dish_id = str(row['id'])
    folder_name = f"dish_{dish_id}" if not dish_id.startswith("dish_") else dish_id
    src_img_path = os.path.join(SOURCE_IMAGE_DIR, folder_name, "rgb.png")
    dst_img_path = os.path.join(OUTPUT_IMAGE_DIR, f"{dish_id}.png")

    if os.path.exists(src_img_path):
        try:
            img = Image.open(src_img_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            img.save(dst_img_path, quality=85)

            all_rows.append(row)
        except Exception as e:
            print(f"⚠️ Failed to process {dish_id}: {e}")

# === Save final label CSV ===
if all_rows:
    pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved {len(all_rows)} entries to {OUTPUT_CSV}")
else:
    print("❌ No images were successfully processed.")