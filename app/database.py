import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class Database:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS retrain_uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    label_name TEXT NOT NULL,
                    label_idx INTEGER NOT NULL,
                    used_in_retrain INTEGER DEFAULT 0,
                    uploaded_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS retrain_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    samples_used INTEGER NOT NULL,
                    accuracy REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    loss REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add_retrain_upload(self, filename, file_path, label_name, label_idx):
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO retrain_uploads (filename, file_path, label_name, label_idx, uploaded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (filename, file_path, label_name, label_idx, now),
            )
            return int(cur.lastrowid)

    def get_pending_uploads(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM retrain_uploads WHERE used_in_retrain = 0 ORDER BY id ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_uploads_count(self):
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM retrain_uploads").fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM retrain_uploads WHERE used_in_retrain = 0"
            ).fetchone()[0]
            return {"total_uploads": int(total), "pending_uploads": int(pending)}

    def mark_uploads_used(self, ids):
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE retrain_uploads SET used_in_retrain = 1 WHERE id IN ({placeholders})",
                ids,
            )

    def log_prediction(self, filename, predicted_class, confidence, model_version):
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_logs (filename, predicted_class, confidence, model_version, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (filename, predicted_class, confidence, model_version, now),
            )

    def log_retrain_run(self, model_version, samples_used, accuracy, f1_score, loss):
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO retrain_runs (model_version, samples_used, accuracy, f1_score, loss, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (model_version, samples_used, accuracy, f1_score, loss, now),
            )

    def get_stats(self):
        with self._connect() as conn:
            pred_count = conn.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]
            runs = conn.execute(
                "SELECT model_version, samples_used, accuracy, f1_score, loss, created_at "
                "FROM retrain_runs ORDER BY id DESC LIMIT 5"
            ).fetchall()
        counts = self.get_all_uploads_count()
        return {
            "total_predictions": int(pred_count),
            "total_uploads": counts["total_uploads"],
            "pending_uploads": counts["pending_uploads"],
            "recent_retrain_runs": [dict(r) for r in runs],
        }
