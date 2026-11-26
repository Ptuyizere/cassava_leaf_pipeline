# Cassava Leaf Disease Classification MLOps Pipeline

This project classifies the cassava leaf images as either **Healthy** or **Mosaic_Diseased** (Cassava Mosaic Disease/CMD). The model is trained on actual field images of the dataset provided by the Makerere University AI Lab using TensorFlow datasets or Kaggle and involves preprocessing, a 3-layer neural network, FastAPI backend with SQLite logs, UI dashboard, and Locust load testing.

## Links
- **GitHub Repository:** `https://github.com/Ptuyizere/cassava_leaf_pipeline.git`
- **Deployed App URL:** ``
- **Video Demo:** ``

## Project Structure
```text
cassava_leaf_mlops_pipeline/
├── README.md
├── requirements.txt
├── train.py
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── locustfile.py
├── notebook/
│   └── cassava_classification.ipynb
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── app/
│   ├── __init__.py
│   ├── database.py
│   ├── main.py
│   └── static/
│       └── index.html
├── data/
│   ├── train/
│   │   ├── Healthy/
│   │   └── Mosaic_Diseased/
│   └── test/
│       ├── Healthy/
│       └── Mosaic_Diseased/
└── models/
```

## Setup and Running Locally

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Download real cassava leaf images:
```bash
# TensorFlow datasets
python -m src.data_acquisition --mode tfds --data-dir data --max-train 500 --max-test 125
```

3. Train the model (or run `notebook/cassava_classification.ipynb`):
```bash
python train.py
```

4. Start the FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser to view uptime, feature charts, single-leaf predictions, bulk uploads, and model retraining.

## Docker and Scaling
```bash
docker compose up --build -d

# Scale to 1, 2, or 4 API containers
docker compose up -d --scale ml-api=4
```

## Load Testing with Locust
```bash
locust -f locustfile.py --host http://localhost:8000 --users 50 --spawn-rate 10 --run-time 1m --headless
```

