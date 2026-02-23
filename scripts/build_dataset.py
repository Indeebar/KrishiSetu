"""
KrishiSetu Dataset Builder
==========================
Automatically downloads images from Bing for each agricultural waste class.
Strategy: 3 search queries per class (waste form + source crop + texture shot)
to ensure visual diversity. Validates all downloaded files with PIL.

Usage:
    cd f:\KrishiSetu
    .venv\Scripts\python scripts/build_dataset.py

Requirements:
    pip install icrawler
"""

import os
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "Agri_Waste_Images"

# Smart multi-query search terms per class.
# Each list has 3 queries: [waste-specific, crop-related, texture/close-up]
# This gives visual variety — the model needs to learn texture, not just context.
SEARCH_QUERIES = {
    "Apple_pomace": [
        "apple pomace agricultural waste byproduct",
        "apple cider pomace wet pulp",
        "apple pomace close up texture",
    ],
    "Apple_pomace_Dry": [
        "dried apple pomace powder",
        "dry apple marc byproduct",
        "dried apple pomace close up",
    ],
    "Bamboo_waste": [
        "bamboo waste biomass green stems",
        "bamboo offcuts green pile",
        "bamboo waste close up texture",
    ],
    "Bamboo_waste_Dry": [
        "dried bamboo waste stalks pile",
        "dry bamboo stems agricultural waste",
        "dry bamboo close up texture",
    ],
    "Banana_stems": [
        "banana pseudostem agricultural waste",
        "banana trunk stem cut pile",
        "banana plant stem fibre close up",
    ],
    "Cashew_nut_shells": [
        "cashew nut shells waste pile",
        "cashew shell agricultural byproduct",
        "cashew shells close up texture",
    ],
    "Coconut_shells": [
        "coconut shells waste pile",
        "dried coconut shell biomass",
        "coconut shell halves close up",
    ],
    "Cotton_stalks": [
        "cotton stalks field after harvest",
        "cotton plant stems agricultural waste",
        "cotton stalk pile close up",
    ],
    "Groundnut_shells": [
        "groundnut shells peanut shells waste",
        "peanut shell pile agricultural",
        "groundnut husk close up texture",
    ],
    "Jute_stalks": [
        "jute stalks agricultural waste",
        "jute plant stems fibre waste",
        "jute stalk pile close up",
    ],
    "Maize_husks": [
        "maize husks corn husks waste pile",
        "corn husk agricultural byproduct",
        "dried corn husks close up",
    ],
    "Maize_stalks": [
        "maize stalks corn stalks field waste",
        "corn stalk pile after harvest",
        "maize stem close up texture",
    ],
    "Mustard_stalks": [
        "mustard stalks crop residue pile",
        "mustard plant stems harvest waste",
        "mustard straw biomass close up",
    ],
    "Pineapple_leaves": [
        "pineapple leaves waste agricultural",
        "pineapple crown leaf fibre",
        "pineapple leaf close up texture",
    ],
    "Rice_straw": [
        "rice straw paddy straw agricultural waste",
        "rice straw bale harvest field",
        "paddy straw close up texture",
    ],
    "Soybean_stalks": [
        "soybean stalks crop residue pile",
        "soybean plant stems agricultural waste",
        "soybean straw close up texture",
    ],
    "Sugarcane_bagasse": [
        "sugarcane bagasse agricultural waste",
        "sugarcane bagasse fibre pulp",
        "sugarcane bagasse close up texture",
    ],
    "Wheat_straw": [
        "wheat straw agricultural waste pile",
        "wheat straw bale field harvest",
        "wheat stubble close up texture",
    ],
}

TARGET_PER_CLASS = 100   # minimum images per class we want
DOWNLOADS_PER_QUERY = 45  # Bing downloads per search term (3 queries * 45 = 135 attempts)


def validate_and_clean(folder: Path) -> int:
    """Remove files that cannot be opened as valid images. Returns count removed."""
    removed = 0
    for f in list(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'):
            try:
                with Image.open(f) as img:
                    img.verify()
            except Exception:
                try:
                    f.unlink()
                    removed += 1
                except Exception:
                    pass
    return removed


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1 for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    )


def download_for_class(class_name: str, queries: list, data_dir: Path) -> int:
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("ERROR: icrawler not installed. Run: pip install icrawler")
        sys.exit(1)

    class_dir = data_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    existing = count_images(class_dir)
    print(f"\n[{class_name}]")
    print(f"  Existing images: {existing} | Target: {TARGET_PER_CLASS}")

    if existing >= TARGET_PER_CLASS:
        print(f"  → Already at target. Skipping download.")
        return existing

    for i, query in enumerate(queries):
        current = count_images(class_dir)
        if current >= TARGET_PER_CLASS:
            break

        need = TARGET_PER_CLASS - current
        to_download = min(DOWNLOADS_PER_QUERY, need + 10)  # download a few extra to account for failures
        print(f"  → Query {i+1}/{len(queries)}: \"{query}\" (downloading up to {to_download})")

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": str(class_dir)},
                log_level=40,  # ERROR only — suppress verbose Bing parser output
                feeder_threads=2,
                parser_threads=2,
                downloader_threads=4,
            )
            crawler.crawl(
                keyword=query,
                max_num=to_download,
                min_size=(100, 100),
                file_idx_offset='auto',
            )
        except Exception as e:
            print(f"  ⚠ Query failed: {e}")

    removed = validate_and_clean(class_dir)
    final = count_images(class_dir)
    if removed > 0:
        print(f"  Cleaned {removed} corrupt files.")
    print(f"  Final count: {final} images")
    return final


def main():
    print("=" * 60)
    print("  KrishiSetu Dataset Builder")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Target: {TARGET_PER_CLASS} images per class")
    print("=" * 60)

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
        print(f"Created data directory.")

    results = {}
    total_classes = len(SEARCH_QUERIES)
    for idx, (class_name, queries) in enumerate(SEARCH_QUERIES.items(), 1):
        print(f"\n[{idx}/{total_classes}]", end="")
        results[class_name] = download_for_class(class_name, queries, DATA_DIR)

    print("\n" + "=" * 60)
    print("  FINAL DATASET SUMMARY")
    print("=" * 60)
    total_images = 0
    low_classes = []
    for class_name, count in results.items():
        status = "✓" if count >= TARGET_PER_CLASS else "⚠ LOW"
        print(f"  {status:6} {class_name:30} {count:4} images")
        total_images += count
        if count < TARGET_PER_CLASS:
            low_classes.append(class_name)

    print(f"\n  Total: {total_images} images across {len(results)} classes")
    if low_classes:
        print(f"\n  ⚠ Classes still below {TARGET_PER_CLASS} images:")
        for c in low_classes:
            print(f"    - {c}: {results[c]} images")
        print("  Consider re-running the script or adding images manually.")
    else:
        print(f"\n  ✓ All classes meet the {TARGET_PER_CLASS}-image target!")

    print("=" * 60)
    print("\nNext step: retrain the model")
    print("  .venv\\Scripts\\python src/models/train_cnn.py")


if __name__ == "__main__":
    main()
