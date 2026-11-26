import argparse
from pathlib import Path
from src.preprocessing import verify_real_dataset, CLASSES
import tensorflow_datasets as tfds
from PIL import Image


LABEL_MAP = {
    4: "Healthy",
    3: "Mosaic_Diseased",
}


def ingest_from_tfds(data_dir="data", max_train_per_class=500, max_test_per_class=125):
    data_path = Path(data_dir)
    splits = [
        ("train", "train", max_train_per_class),
        ("test", "validation", max_test_per_class),
    ]

    for split_name, tfds_split, limit in splits:
        dataset = tfds.load("cassava", split=tfds_split, as_supervised=True)
        saved = {c: 0 for c in CLASSES}

        for img_arr, label_val in tfds.as_numpy(dataset):
            lbl = int(label_val)
            if lbl not in LABEL_MAP:
                continue

            cls_name = LABEL_MAP[lbl]
            if saved[cls_name] >= limit:
                if all(v >= limit for v in saved.values()):
                    break
                continue

            out_dir = data_path / split_name / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{cls_name.lower()}_{split_name}_{saved[cls_name]:04d}.jpg"
            Image.fromarray(img_arr).save(out_dir / filename, quality=95)
            saved[cls_name] += 1

    return verify_real_dataset(data_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["verify", "tfds", "kaggle"], default="verify")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--kaggle-dir", default="")
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--max-test", type=int, default=125)
    args = parser.parse_args()

    if args.mode == "tfds":
        counts = ingest_from_tfds(args.data_dir, args.max_train, args.max_test)
    else:
        counts = verify_real_dataset(args.data_dir)

    print("Dataset ready:", counts)




