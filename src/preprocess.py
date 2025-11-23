import os
from pathlib import Path
import cv2
import random
from tqdm import tqdm
import chardet

#config
BASE_DIR = Path("data")
ANNOT_DIR = BASE_DIR / "Annotations"
OUTPUT_DIR = BASE_DIR / "lisa"

#target class names
CLASSES = ["red", "yellow", "green", "inactive"]

#mapping from LISA tags to our class labels
TAG_MAP = {
    "stop": "red",
    "go": "green",
    "warning": "yellow",
    "off": "inactive"
}

IMG_SIZE = (32, 32)
TRAIN_SPLIT = 0.85

#output folders
for split in ["train", "val"]:
    for cls in CLASSES:
        os.makedirs(OUTPUT_DIR / split / cls, exist_ok=True)

#helper functions

#reads all csv in annotations 
def get_csv_files():
    return list(ANNOT_DIR.rglob("*.csv"))

#read csv with auto detected encoding 
def read_csv_lines(csv_path):
    with open(csv_path, "rb") as f:
        raw = f.read()
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    try:
        return raw.decode(enc).splitlines()
    except Exception:
        return raw.decode("utf-8", errors="ignore").splitlines()

#parse all annotations from csv
def load_annotations():
    rows = []
    csv_files = get_csv_files()
    print(f"Found {len(csv_files)} CSV files.")

    if not csv_files:
        print("No CSV files found.")
        return rows

    for csv_file in csv_files:
        lines = read_csv_lines(csv_file)
        if not lines:
            continue

        header = lines[0].strip().replace('"', '').split(";")
        header = [h.strip().lower() for h in header]

        def find_index(keywords):
            for kw in keywords:
                for i, h in enumerate(header):
                    if kw in h:
                        return i
            return None

        idx_filename = find_index(["filename"])
        idx_tag = find_index(["annotation tag"])
        idx_x1 = find_index(["upper left corner x"])
        idx_y1 = find_index(["upper left corner y"])
        idx_x2 = find_index(["lower right corner x"])
        idx_y2 = find_index(["lower right corner y"])

        if None in [idx_filename, idx_tag, idx_x1, idx_y1, idx_x2, idx_y2]:
            print(f"{csv_file.name}: column mismatch-skip")
            continue

        for line in lines[1:]:
            parts = line.strip().replace('"', '').split(";")
            if len(parts) <= idx_y2:
                continue

            cls_raw = parts[idx_tag].lower().strip()
            cls = TAG_MAP.get(cls_raw)
            if cls is None:
                continue

            filename = parts[idx_filename].strip()
            try:
                x1 = int(float(parts[idx_x1]))
                y1 = int(float(parts[idx_y1]))
                x2 = int(float(parts[idx_x2]))
                y2 = int(float(parts[idx_y2]))
            except ValueError:
                continue

            rows.append((filename, cls, x1, y1, x2, y2))

    print(f"total valid annotaitons: {len(rows)}")
    return rows

#preprocessings

def preprocess_and_save():
    all_rows = load_annotations()
    if not all_rows:
        print("No valid rows found")
        return

    random.shuffle(all_rows)
    split_idx = int(TRAIN_SPLIT * len(all_rows))
    train_rows = all_rows[:split_idx]
    val_rows = all_rows[split_idx:]

    for split, rows in [("train", train_rows), ("val", val_rows)]:
        print(f"\nProcessing {split} ({len(rows)} samples)")
        for filename, cls, x1, y1, x2, y2 in tqdm(rows):
            possible_matches = list(BASE_DIR.rglob(Path(filename).name))
            if len(possible_matches) == 0:
                continue
            img_path = possible_matches[0]

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            #making sure bounds are valud
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, IMG_SIZE)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            out_file = OUTPUT_DIR / split / cls / Path(filename).name
            cv2.imwrite(str(out_file), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    print("completed - cropped images saved in data/lisa/train and data/lisa/val")



if __name__ == "__main__":
    preprocess_and_save()


