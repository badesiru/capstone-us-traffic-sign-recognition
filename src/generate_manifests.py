import csv
from pathlib import Path

BASE_DIR = Path("data/lisa")
CLASSES = ["red", "yellow", "green", "inactive"]

def write_manifest(split):
    rows = []
    split_dir = BASE_DIR / split

    for cls in CLASSES:
        class_dir = split_dir / cls
        for img_path in class_dir.glob("*"):
            rows.append([str(img_path), cls])

    csv_path = BASE_DIR / f"{split}_manifest.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} entries to {csv_path}")


if __name__ == "__main__":
    write_manifest("train")
    write_manifest("val")
    print("Done")
