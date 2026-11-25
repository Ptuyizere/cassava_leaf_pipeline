import io
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

IMAGE_SIZE = (64, 64)
PATCH_SIZE = (16, 16)

INTERPRETABLE_FEATURE_NAMES = [
    "chlorosis_yellow_ratio",
    "excess_green_index",
    "laplacian_texture_energy",
    "hsv_hue_std",
    "rgb_red_mean",
    "rgb_green_mean",
    "rgb_blue_mean",
    "hsv_saturation_mean",
]

CLASSES = ["Healthy", "Mosaic_Diseased"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_image_from_bytes(raw_bytes):
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_image_from_path(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def apply_clahe_rgb(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge((l_eq, a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def augment_image(img_rgb, rng):
    out = img_rgb.copy()
    if rng.random() > 0.5:
        out = np.fliplr(out)
    if rng.random() > 0.5:
        out = np.flipud(out)
    brightness = rng.uniform(0.92, 1.08)
    out = np.clip(out.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return out


def extract_interpretable_features(img_rgb):
    norm = img_rgb.astype(np.float32) / 255.0
    r = norm[:, :, 0]
    g = norm[:, :, 1]
    b = norm[:, :, 2]

    yellow_mask = (r > 0.52) & (g > 0.48) & (b < 0.35)
    chlorosis_yellow_ratio = float(np.mean(yellow_mask))

    exg = 2.0 * g - r - b
    excess_green_index = float(np.mean(exg))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_texture_energy = float(np.var(laplacian))

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    hsv_hue_std = float(np.std(hsv[:, :, 0]))
    hsv_saturation_mean = float(np.mean(hsv[:, :, 1]))

    return {
        "chlorosis_yellow_ratio": chlorosis_yellow_ratio,
        "excess_green_index": excess_green_index,
        "laplacian_texture_energy": laplacian_texture_energy,
        "hsv_hue_std": hsv_hue_std,
        "rgb_red_mean": float(np.mean(r)),
        "rgb_green_mean": float(np.mean(g)),
        "rgb_blue_mean": float(np.mean(b)),
        "hsv_saturation_mean": hsv_saturation_mean,
    }


def extract_feature_vector(img_rgb, use_clahe=True):
    proc = apply_clahe_rgb(img_rgb) if use_clahe else img_rgb
    feats = extract_interpretable_features(proc)
    interp_vec = np.array([feats[k] for k in INTERPRETABLE_FEATURE_NAMES], dtype=np.float32)

    small = cv2.resize(proc, PATCH_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    spatial_vec = small.flatten()
    return np.concatenate([interp_vec, spatial_vec])


class FeatureScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def preprocess_dataset(image_paths, labels=None, augment=False, augment_copies=1, seed=101):
    rng = np.random.default_rng(seed)
    features = []
    out_labels = []
    interp_records = []

    for i, path in enumerate(image_paths):
        img = load_image_from_path(path)
        interp_records.append(extract_interpretable_features(img))
        features.append(extract_feature_vector(img))
        if labels is not None:
            out_labels.append(labels[i])

        if augment:
            for _ in range(augment_copies):
                aug_img = augment_image(img, rng)
                features.append(extract_feature_vector(aug_img))
                if labels is not None:
                    out_labels.append(labels[i])

    X = np.vstack(features).astype(np.float32)
    y = np.array(out_labels, dtype=np.float32) if labels is not None else None
    return X, y, interp_records

def verify_real_dataset(data_dir="data", min_per_class=10):
    data_path = Path(data_dir)
    counts = {"train": {}, "test": {}}
    missing = []

    for split in ["train", "test"]:
        for cls_name in CLASSES:
            cls_dir = data_path / split / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            imgs = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            counts[split][cls_name] = len(imgs)
            if len(imgs) < min_per_class:
                missing.append(f"{split}/{cls_name} ({len(imgs)} found, need {min_per_class})")

    if missing:
        msg = "Missing real cassava leaf images in: " + ", ".join(missing)
        msg += "\nRun: python -m src.data_acquisition --mode tfds --data-dir data"
        msg += "\nOr:  python -m src.data_acquisition --mode kaggle --kaggle-dir <folder> --data-dir data"
        raise FileNotFoundError(msg)

    return counts

def load_image_paths(split_dir):
    split_path = Path(split_dir)
    paths = []
    labels = []

    for idx, cls_name in enumerate(CLASSES):
        cls_dir = split_path / cls_name
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
                labels.append(idx)

    return paths, labels



