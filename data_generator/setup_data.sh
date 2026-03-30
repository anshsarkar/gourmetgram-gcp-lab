#!/bin/bash
set -e

# Configuration
DATASET_DIR="${DATASET_DIR:-$HOME/Food-11}"

echo "=== Step 1: Download Food-11 dataset ==="
mkdir -p "$DATASET_DIR" && cd "$DATASET_DIR"

if [ -d "training" ] && [ -d "validation" ] && [ -d "evaluation" ]; then
    echo "Dataset already exists, skipping download."
else
    echo "Downloading dataset..."
    curl -L https://nyu.box.com/shared/static/m5kvcl25hfpljul3elhu6xz8y2w61op4.zip -o Food-11.zip
    echo "Extracting dataset..."
    unzip -q Food-11.zip
    rm -f Food-11.zip
fi

echo ""
echo "=== Step 2: Organize into class directories ==="
python3 -c "
import os
import shutil

dataset_dir = '$DATASET_DIR'
subdirs = ['training', 'validation', 'evaluation']
classes = [
    'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
    'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit'
]

moved = 0
for subdir in subdirs:
    dir_path = os.path.join(dataset_dir, subdir)
    if not os.path.exists(dir_path):
        continue
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(dir_path, f'class_{i:02d}')
        os.makedirs(class_dir, exist_ok=True)
        for f in os.listdir(dir_path):
            if f.startswith(f'{i}_'):
                shutil.move(os.path.join(dir_path, f), os.path.join(class_dir, f))
                moved += 1

print(f'Organized {moved} files into class directories.')
"

echo ""
echo "=== Done ==="
echo "Dataset ready at $DATASET_DIR"
