import json
import uuid
from pathlib import Path
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.database import Database
from src.model import retrain_from_pretrained
from src.prediction import CassavaPredictor
from src.preprocessing import preprocess_dataset, load_image_paths, verify_real_dataset, CLASSES

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "retrain_uploads"
MODEL_PATH = BASE_DIR / "models" / "cassava_classifier.pkl"
METRICS_PATH = BASE_DIR / "models" / "baseline_metrics.json"
DB_PATH = DATA_DIR / "mlops.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"

verify_real_dataset(DATA_DIR, min_per_class=1)
db = Database(DB_PATH)
predictor = CassavaPredictor(MODEL_PATH)

app = FastAPI(title="Cassava Leaf Disease API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.get("/version")
def get_version():
    predictor.ensure_latest()
    return {"model_version": predictor.model.version}

@app.get("/visualizations")
def get_visualizations():
    train_paths, train_labels = load_image_paths(DATA_DIR / "train")
    _, _, interp_records = preprocess_dataset(train_paths, train_labels, augment=False)

    feature_names = [
        "chlorosis_yellow_ratio",
        "excess_green_index",
        "laplacian_texture_energy",
        "hsv_hue_std",
    ]
    feature_summary = {}
    for feat in feature_names:
        healthy_vals = [r[feat] for r, lbl in zip(interp_records, train_labels) if lbl == 0]
        diseased_vals = [r[feat] for r, lbl in zip(interp_records, train_labels) if lbl == 1]
        feature_summary[feat] = {
            "Healthy_mean": round(float(np.mean(healthy_vals)), 4),
            "Healthy_std": round(float(np.std(healthy_vals)), 4),
            "Mosaic_Diseased_mean": round(float(np.mean(diseased_vals)), 4),
            "Mosaic_Diseased_std": round(float(np.std(diseased_vals)), 4),
        }

    baseline_metrics = {}
    if METRICS_PATH.exists():
        baseline_metrics = json.loads(METRICS_PATH.read_text())

    return {
        "dataset_size": {
            "train_healthy": int(sum(1 for y in train_labels if y == 0)),
            "train_mosaic_diseased": int(sum(1 for y in train_labels if y == 1)),
        },
        "features": feature_summary,
        "interpretations": [
            {
                "feature": "chlorosis_yellow_ratio",
                "title": "1. Yellow Patch Ratio",
                "story": "Leaves with Cassava Mosaic Disease develop yellow patches, so their share of yellow pixels is higher than healthy green leaves.",
            },
            {
                "feature": "excess_green_index",
                "title": "2. Excess Green Index (2G - R - B)",
                "story": "Healthy leaves have stronger green color overall, while diseased leaves lose green where yellow spots appear.",
            },
            {
                "feature": "laplacian_texture_energy",
                "title": "3. Leaf Texture Energy",
                "story": "Mosaic spots and damaged edges create sharper color changes across the leaf surface, which raises the texture score.",
            },
        ],
        "model_metrics": baseline_metrics,
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    try:
        result = predictor.predict_bytes(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    db.log_prediction(
        filename=file.filename or "uploaded_leaf.jpg",
        predicted_class=str(result["predicted_class"]),
        confidence=float(result["confidence"]),
        model_version=str(result["model_version"]),
    )
    return result


@app.post("/upload-retrain-data")
async def upload_retrain_data(
    label: str = Form(...),
    files: list[UploadFile] = File(...),
):
    if label not in CLASSES:
        raise HTTPException(status_code=400, detail=f"Label must be one of {CLASSES}.")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    label_idx = CLASSES.index(label)
    target_dir = UPLOAD_DIR / label
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_records = []
    for up in files:
        raw = await up.read()
        if not raw:
            continue
        safe_name = f"{uuid.uuid4().hex[:8]}_{Path(up.filename or 'leaf.jpg').name}"
        dest = target_dir / safe_name
        dest.write_bytes(raw)
        row_id = db.add_retrain_upload(safe_name, str(dest), label, label_idx)
        saved_records.append({"id": row_id, "filename": safe_name, "label": label})

    return {
        "uploaded_count": len(saved_records),
        "label": label,
        "records": saved_records,
        **db.get_all_uploads_count(),
    }


@app.post("/retrain")
def trigger_retraining():
    pending = db.get_pending_uploads()
    if not pending:
        raise HTTPException(
            status_code=400,
            detail="No pending uploaded images found. Upload labeled images first.",
        )

    up_paths = [Path(r["file_path"]) for r in pending]
    up_labels = [int(r["label_idx"]) for r in pending]

    base_paths, base_labels = load_image_paths(DATA_DIR / "train")
    combined_paths = up_paths + base_paths[:40] + base_paths[-40:]
    combined_labels = up_labels + base_labels[:40] + base_labels[-40:]

    X_retrain, y_retrain, _ = preprocess_dataset(
        combined_paths, combined_labels, augment=True, augment_copies=1
    )

    _, retrain_info = retrain_from_pretrained(
        pretrained_path=MODEL_PATH,
        X_new=X_retrain,
        y_new=y_retrain,
        epochs=25,
        fine_tune_lr=3e-4,
        save_path=MODEL_PATH,
    )

    test_paths, test_labels = load_image_paths(DATA_DIR / "test")
    X_test, y_test, _ = preprocess_dataset(test_paths, test_labels, augment=False)
    predictor.reload()
    test_metrics = predictor.model.evaluate(X_test, y_test)

    db.mark_uploads_used([int(r["id"]) for r in pending])
    db.log_retrain_run(
        model_version=str(retrain_info["version"]),
        samples_used=len(pending),
        accuracy=float(test_metrics["accuracy"]),
        f1_score=float(test_metrics["f1_score"]),
        loss=float(test_metrics["loss"]),
    )

    return {
        "status": "retrained",
        "new_samples_processed": len(pending),
        "model_version": retrain_info["version"],
        "epochs_ran": retrain_info["epochs_ran"],
        "test_metrics_after_retrain": test_metrics,
    }
