# KrishiSetu Dataset & Model Accuracy Fix

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the tiny 30-image-per-class hand-built dataset with a 100+ images-per-class web-scraped dataset, fix the inference normalization bug, and add a fine-tuning training phase.

**Architecture:**
1. A dedicated `scripts/build_dataset.py` script uses `icrawler` (Bing engine) to download images per class with multiple smart search queries per class. Existing images are kept and any new downloads supplement them.
2. `train_cnn.py` gets Phase 1 (frozen base) + Phase 2 (fine-tuning top layers) training and stronger augmentation.
3. `inference.py` gets the critical `preprocess_input` normalization fix.

**Tech Stack:** Python, TensorFlow/Keras, icrawler, PIL, MobileNetV2 Transfer Learning

---

### Task 1: Install icrawler

**Files:**
- Modify: `requirements.txt`

**Step 1:** Add icrawler to requirements.txt

```
icrawler>=0.6.6
```

**Step 2:** Install it

```bash
cd f:\KrishiSetu
.venv\Scripts\pip install icrawler
```

Expected: `Successfully installed icrawler-0.6.6`

**Step 3:** Commit

```bash
git add requirements.txt
git commit -m "deps: add icrawler for dataset building"
```

---

### Task 2: Create the Smart Dataset Builder Script

**Files:**
- Create: `scripts/build_dataset.py`

**Step 1:** Create `scripts/` directory and write the script

The script should:
- Define `SEARCH_QUERIES` dict with 3-4 Bing search terms per class (using both the waste material AND the source crop to get enough variety)
- For each class, download images into its existing folder, skipping if enough images already exist
- Use `BingImageCrawler` with `max_num=120` per query to target ~100 net images per class
- After downloading, remove corrupt/non-openable images with PIL validation
- Print a summary table at the end

```python
# scripts/build_dataset.py
import os
import sys
from pathlib import Path
from PIL import Image

# Ensure we can import from the project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "Agri_Waste_Images"

# Smart multi-query search terms per class.
# The strategy: use 2-3 queries — one for the waste form, one for the crop,
# one for up-close/texture shots — to get diverse, realistic images.
SEARCH_QUERIES = {
    "Apple_pomace": [
        "apple pomace agricultural waste",
        "apple pomace wet pulp byproduct",
        "apple cider pomace",
    ],
    "Apple_pomace_Dry": [
        "dried apple pomace",
        "dry apple pomace powder",
        "apple marc dried",
    ],
    "Bamboo_waste": [
        "bamboo waste biomass",
        "bamboo offcuts waste",
        "bamboo green stems cut",
    ],
    "Bamboo_waste_Dry": [
        "dried bamboo waste",
        "dry bamboo stems pile",
        "bamboo dry offcuts",
    ],
    "Banana_stems": [
        "banana pseudostem agricultural waste",
        "banana trunk stem cut",
        "banana plant stem fibre",
    ],
    "Cashew_nut_shells": [
        "cashew nut shells waste",
        "cashew shell pile",
        "cashew hull agricultural byproduct",
    ],
    "Coconut_shells": [
        "coconut shells waste pile",
        "coconut shell dried",
        "coconut husk shell biomass",
    ],
    "Cotton_stalks": [
        "cotton stalks field agricultural waste",
        "cotton plant stalks after harvest",
        "cotton crop stems",
    ],
    "Groundnut_shells": [
        "groundnut shells peanut shells waste",
        "peanut shell pile",
        "groundnut husk agricultural",
    ],
    "Jute_stalks": [
        "jute stalks agricultural waste",
        "jute plant stems pile",
        "jute stem fibre waste",
    ],
    "Maize_husks": [
        "maize husks corn husks waste",
        "corn husk pile agricultural",
        "dried corn husks",
    ],
    "Maize_stalks": [
        "maize stalks corn stalks waste",
        "corn stalk pile after harvest",
        "maize crop stems",
    ],
    "Mustard_stalks": [
        "mustard stalks crop residue",
        "mustard plant stems harvest waste",
        "mustard straw biomass",
    ],
    "Pineapple_leaves": [
        "pineapple leaves waste agricultural",
        "pineapple crown leaf fibre",
        "pineapple plant leaves",
    ],
    "Rice_straw": [
        "rice straw paddy straw waste",
        "rice straw field harvest",
        "paddy straw pile biomass",
    ],
    "Soybean_stalks": [
        "soybean stalks crop residue",
        "soybean plant stems agricultural waste",
        "soybean straw biomass",
    ],
    "Sugarcane_bagasse": [
        "sugarcane bagasse waste",
        "sugarcane bagasse fibre pulp",
        "sugarcane mill bagasse",
    ],
    "Wheat_straw": [
        "wheat straw agricultural waste",
        "wheat straw bale field",
        "wheat stubble harvest",
    ],
}

TARGET_PER_CLASS = 100  # target at least this many images per class
DOWNLOADS_PER_QUERY = 50  # Bing downloads per search term


def validate_and_clean(folder: Path):
    """Remove files that cannot be opened as valid images."""
    removed = 0
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            try:
                img = Image.open(f)
                img.verify()
            except Exception:
                f.unlink()
                removed += 1
    return removed


def count_images(folder: Path) -> int:
    return sum(
        1 for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    )


def download_for_class(class_name: str, queries: list, data_dir: Path):
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("ERROR: icrawler not installed. Run: pip install icrawler")
        sys.exit(1)

    class_dir = data_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    existing = count_images(class_dir)
    print(f"\n[{class_name}] Existing: {existing} images. Target: {TARGET_PER_CLASS}")

    if existing >= TARGET_PER_CLASS:
        print(f"  → Already has enough images. Skipping download.")
        return existing

    for i, query in enumerate(queries):
        current = count_images(class_dir)
        if current >= TARGET_PER_CLASS:
            break

        print(f"  → Query {i+1}/{len(queries)}: '{query}'")
        crawler = BingImageCrawler(
            storage={"root_dir": str(class_dir)},
            log_level=40,  # ERROR only — suppress verbose output
            feeder_threads=2,
            parser_threads=2,
            downloader_threads=4,
        )
        crawler.crawl(
            keyword=query,
            max_num=DOWNLOADS_PER_QUERY,
            min_size=(100, 100),
            file_idx_offset='auto',
        )

    removed = validate_and_clean(class_dir)
    final = count_images(class_dir)
    print(f"  → After download & clean: {final} images ({removed} corrupt removed)")
    return final


def main():
    print("=" * 60)
    print("KrishiSetu Dataset Builder")
    print(f"Data directory: {DATA_DIR}")
    print("=" * 60)

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        print(f"Created data directory at {DATA_DIR}")

    results = {}
    for class_name, queries in SEARCH_QUERIES.items():
        results[class_name] = download_for_class(class_name, queries, DATA_DIR)

    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    total = 0
    for class_name, count in results.items():
        status = "✓" if count >= TARGET_PER_CLASS else "⚠ LOW"
        print(f"  {status:6} {class_name:30} {count:4} images")
        total += count
    print(f"\n  Total: {total} images across {len(results)} classes")
    print("=" * 60)
    print("\nDone! Now retrain the model: python src/models/train_cnn.py")


if __name__ == "__main__":
    main()
```

**Step 2:** Commit

```bash
git add scripts/build_dataset.py
git commit -m "feat: add smart web dataset builder script"
```

---

### Task 3: Fix the Critical Inference Normalization Bug

**Files:**
- Modify: `src/utils/inference.py`

**Step 1:** Fix `predict_image_class` to add `preprocess_input` normalization

In `inference.py`, the function currently feeds raw `[0, 255]` pixel values to a model trained on `[-1, 1]` scaled values (MobileNetV2 standard). Fix: add `preprocess_input` call.

Change line 1-8 imports to add:
```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
```

Change the prediction block inside `predict_image_class`:
```python
# BEFORE (broken):
img_array = tf.keras.preprocessing.image.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)
predictions = model.predict(img_array)

# AFTER (correct):
img_array = tf.keras.preprocessing.image.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)
img_array = preprocess_input(img_array)   # scale to [-1,1] to match training
predictions = model.predict(img_array)
```

**Step 2:** Commit

```bash
git add src/utils/inference.py
git commit -m "fix: add preprocess_input normalization in inference - critical accuracy fix"
```

---

### Task 4: Rewrite train_cnn.py with Fine-Tuning + Better Augmentation

**Files:**
- Modify: `src/models/train_cnn.py`

**Step 1:** Replace train_cnn.py contents with the improved version

Key improvements:
- **Phase 1:** Train only the top head for 15 epochs (base frozen)
- **Phase 2:** Unfreeze top 50 layers of MobileNetV2, fine-tune for 15 epochs with `lr=1e-5`
- **Better augmentation:** Add `RandomBrightness`, `RandomContrast` layers
- **Class weights:** Compute and pass class weights to `model.fit` for imbalanced classes

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "Agri_Waste_Images"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "custom_cnn_model.keras")
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 15


def get_data():
    """Loads and splits the dataset."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds_perf = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds_perf = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Compute class weights using raw labels (before caching/shuffling)
    all_labels = np.concatenate([y.numpy() for _, y in train_ds])
    class_weights_values = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=all_labels
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights_values)}
    print(f"Class weights: {class_weight_dict}")

    return train_ds_perf, val_ds_perf, num_classes, class_weight_dict


def build_model(num_classes):
    """Builds MobileNetV2 transfer learning model with strong augmentation."""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")

    base_model = applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Phase 1: freeze all

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    x = applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model, base_model


def get_callbacks(label):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    return [checkpoint, early_stop, reduce_lr]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    train_ds, val_ds, num_classes, class_weights = get_data()

    print("Building model...")
    model, base_model = build_model(num_classes)
    model.summary()

    # --- Phase 1: Train head only ---
    print(f"\n{'='*50}")
    print("PHASE 1: Training classification head (base frozen)")
    print(f"{'='*50}")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("phase1")
    )

    # --- Phase 2: Fine-tune top layers ---
    print(f"\n{'='*50}")
    print("PHASE 2: Fine-tuning top 50 layers of MobileNetV2")
    print(f"{'='*50}")

    base_model.trainable = True
    # Freeze everything except the last 50 layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    # Recompile with a very low learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("phase2")
    )

    print(f"\nTraining complete. Best model saved to {MODEL_PATH}")
    print("Now restart the Streamlit app to use the improved model.")


if __name__ == "__main__":
    main()
```

**Step 2:** Commit

```bash
git add src/models/train_cnn.py
git commit -m "feat: add fine-tuning phase + stronger augmentation + class weights to CNN training"
```

---

### Task 5: Run Dataset Builder

**Step 1:** Run the dataset builder (this will take 10-20 minutes)

```bash
cd f:\KrishiSetu
.venv\Scripts\python scripts/build_dataset.py
```

Expected: Downloads ~100 images per class. Final summary table shows all classes ≥ 100 images.

---

### Task 6: Retrain the Model

**Step 1:** Run training (this will take 20-40 min depending on GPU/CPU)

```bash
cd f:\KrishiSetu
.venv\Scripts\python src/models/train_cnn.py
```

Expected:
- Phase 1: 15 epochs, val_accuracy climbing from ~10% to ~50%+
- Phase 2: Fine-tuning, val_accuracy climbing to ~70-85%

**Step 2:** Verify the model was saved

```bash
dir f:\KrishiSetu\models\custom_cnn_model.keras
```

---

### Task 7: Verify Inference Works Correctly

**Step 1:** Run inline diagnostic

```bash
cd f:\KrishiSetu
.venv\Scripts\python -c "
from pathlib import Path
from src.utils.inference import load_cnn_model, predict_image_class, CLASS_NAMES
print('Classes:', CLASS_NAMES)
model = load_cnn_model()
# Test with a real image from each folder
for folder in sorted(Path('Agri_Waste_Images').iterdir()):
    imgs = list(folder.glob('*.jpg'))[:1] + list(folder.glob('*.png'))[:1]
    if imgs:
        pred, conf = predict_image_class(str(imgs[0]), model)
        match = '✓' if pred == folder.name else '✗'
        print(f'  {match} True: {folder.name:30} | Pred: {pred:30} {conf:.1f}%')
"
```

Expected: Most classes show ✓ with confidence > 50%.

**Step 2:** Launch app and do a live test

```bash
cd f:\KrishiSetu
.venv\Scripts\streamlit run app.py
```

Upload an image from `Agri_Waste_Images/` and verify the predicted class label is correct.

**Step 3:** Final commit

```bash
git add -A
git commit -m "feat: dataset rebuilt, inference normalized, fine-tuning enabled - accuracy fix complete"
```
