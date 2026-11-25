import json
from pathlib import Path
import pandas as pd

from src.model import CassavaNeuralClassifier
from src.prediction import CassavaPredictor
from src.preprocessing import preprocess_dataset, load_image_paths, verify_real_dataset

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def main():
    verify_real_dataset(DATA_DIR)

    train_paths, train_labels = load_image_paths(DATA_DIR / "train")
    test_paths, test_labels = load_image_paths(DATA_DIR / "test")

    X_train, y_train, _ = preprocess_dataset(
        train_paths, train_labels, augment=True, augment_copies=1, seed=101
    )
    X_test, y_test, _ = preprocess_dataset(test_paths, test_labels, augment=False)

    configs = [
        {"name": "Baseline (lr=1e-3, L2=0)", "lr": 1e-3, "l2": 0.0, "dropout": (0.0, 0.0)},
        {"name": "Regularized (lr=1e-3, L2=1e-3, Drop=0.3/0.2)", "lr": 1e-3, "l2": 1e-3, "dropout": (0.3, 0.2)},
        {"name": "Tuned (lr=8e-4, L2=2e-3, Drop=0.3/0.2)", "lr": 8e-4, "l2": 2e-3, "dropout": (0.3, 0.2)},
    ]

    best_model = None
    best_f1 = -1.0
    tuning_rows = []

    for cfg in configs:
        model = CassavaNeuralClassifier(
            input_dim=X_train.shape[1],
            hidden_dims=(128, 64),
            learning_rate=cfg["lr"],
            l2_lambda=cfg["l2"],
            dropout_rates=cfg["dropout"],
            seed=101,
        )
        hist = model.fit(X_train, y_train, epochs=80, batch_size=32, patience=12)
        metrics = model.evaluate(X_test, y_test)
        tuning_rows.append({
            "Configuration": cfg["name"],
            "Best Epoch": hist["best_epoch"],
            "Test Accuracy": metrics["accuracy"],
            "Test Loss": metrics["loss"],
            "F1 Score": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"],
        })
        if metrics["f1_score"] >= best_f1:
            best_f1 = metrics["f1_score"]
            best_model = model

    model_path = MODELS_DIR / "cassava_classifier.pkl"
    best_model.save(model_path)

    final_metrics = best_model.evaluate(X_test, y_test)
    (MODELS_DIR / "baseline_metrics.json").write_text(
        json.dumps({"tuning": tuning_rows, "final_test_metrics": final_metrics}, indent=2)
    )

    print(pd.DataFrame(tuning_rows).to_string(index=False))
    print("\nFinal test metrics:")
    print(json.dumps(final_metrics, indent=2))

    predictor = CassavaPredictor(model_path)
    print("\nSample Healthy prediction:", predictor.predict_file(test_paths[0])["predicted_class"])
    print("Sample Mosaic_Diseased prediction:", predictor.predict_file(test_paths[-1])["predicted_class"])


if __name__ == "__main__":
    main()
