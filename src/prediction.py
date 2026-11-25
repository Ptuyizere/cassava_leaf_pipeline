from pathlib import Path

from src.preprocessing import CLASSES
from src.model import CassavaNeuralClassifier
from src.preprocessing import (
    extract_feature_vector,
    extract_interpretable_features,
    load_image_from_bytes,
    load_image_from_path,
)


class CassavaPredictor:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = CassavaNeuralClassifier.load(self.model_path)

    def reload(self):
        self.model = CassavaNeuralClassifier.load(self.model_path)

    def predict_array(self, img_rgb):
        feats_vec = extract_feature_vector(img_rgb).reshape(1, -1)
        prob_diseased = float(self.model.predict_proba(feats_vec)[0])
        pred_idx = int(prob_diseased >= 0.5)
        confidence = prob_diseased if pred_idx == 1 else (1.0 - prob_diseased)
        interp = extract_interpretable_features(img_rgb)

        return {
            "predicted_class": CLASSES[pred_idx],
            "class_index": pred_idx,
            "probability_mosaic_diseased": round(prob_diseased, 4),
            "confidence": round(confidence, 4),
            "model_version": self.model.version,
            "interpretable_features": {k: round(v, 4) for k, v in interp.items()},
        }

    def predict_bytes(self, raw_bytes):
        img_rgb = load_image_from_bytes(raw_bytes)
        return self.predict_array(img_rgb)

    def predict_file(self, file_path):
        img_rgb = load_image_from_path(file_path)
        return self.predict_array(img_rgb)
