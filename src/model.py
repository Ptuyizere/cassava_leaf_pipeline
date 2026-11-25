import pickle
from pathlib import Path
import numpy as np

from src.preprocessing import FeatureScaler


def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0.0).astype(np.float32)


def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    y_t = y_true.astype(np.int32).ravel()
    y_p = np.clip(y_prob.astype(np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    y_pred = (y_p >= threshold).astype(np.int32)

    loss = float(-np.mean(y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p)))

    tp = int(np.sum((y_t == 1) & (y_pred == 1)))
    tn = int(np.sum((y_t == 0) & (y_pred == 0)))
    fp = int(np.sum((y_t == 0) & (y_pred == 1)))
    fn = int(np.sum((y_t == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(len(y_t), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)

    order = np.argsort(-y_p)
    y_sorted = y_t[order]
    n_pos = np.sum(y_t == 1)
    n_neg = np.sum(y_t == 0)

    if n_pos == 0 or n_neg == 0:
        roc_auc = 0.5
    else:
        tps = np.concatenate([[0.0], np.cumsum(y_sorted == 1) / n_pos])
        fps = np.concatenate([[0.0], np.cumsum(y_sorted == 0) / n_neg])
        if hasattr(np, "trapezoid"):
            roc_auc = float(np.trapezoid(tps, fps))
        else:
            roc_auc = float(np.trapz(tps, fps))

    return {
        "accuracy": round(float(accuracy), 4),
        "loss": round(float(loss), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


class CassavaNeuralClassifier:
    def __init__(
        self,
        input_dim,
        hidden_dims=(128, 64),
        learning_rate=1e-3,
        l2_lambda=1e-3,
        dropout_rates=(0.30, 0.20),
        seed=101,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rates = dropout_rates
        self.seed = seed
        self.scaler = FeatureScaler()
        self.version = "v1.0.0"
        self.history = {}
        self.params = {}
        self._init_weights()

    def _init_weights(self):
        rng = np.random.default_rng(self.seed)
        h1, h2 = self.hidden_dims
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / self.input_dim), (self.input_dim, h1)).astype(np.float32),
            "b1": np.zeros((1, h1), dtype=np.float32),
            "W2": rng.normal(0, np.sqrt(2.0 / h1), (h1, h2)).astype(np.float32),
            "b2": np.zeros((1, h2), dtype=np.float32),
            "W3": rng.normal(0, np.sqrt(2.0 / h2), (h2, 1)).astype(np.float32),
            "b3": np.zeros((1, 1), dtype=np.float32),
        }

    def _forward(self, X, training=False, rng=None):
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = relu(z1)
        m1 = np.ones_like(a1)
        if training and self.dropout_rates[0] > 0:
            keep1 = 1.0 - self.dropout_rates[0]
            m1 = (rng.random(a1.shape) < keep1).astype(np.float32) / keep1
            a1 = a1 * m1

        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = relu(z2)
        m2 = np.ones_like(a2)
        if training and self.dropout_rates[1] > 0:
            keep2 = 1.0 - self.dropout_rates[1]
            m2 = (rng.random(a2.shape) < keep2).astype(np.float32) / keep2
            a2 = a2 * m2

        z3 = a2 @ self.params["W3"] + self.params["b3"]
        probs = sigmoid(z3)

        cache = {
            "z1": z1, "a1": a1, "m1": m1,
            "z2": z2, "a2": a2, "m2": m2,
            "probs": probs,
        }
        return probs, cache

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=80,
        batch_size=32,
        patience=12,
        lr_decay=0.985,
        fit_scaler=True,
    ):
        rng = np.random.default_rng(self.seed)

        if X_val is None or y_val is None:
            idx = rng.permutation(len(X_train))
            split = max(1, int(0.8 * len(idx)))
            X_tr_raw, y_tr = X_train[idx[:split]], y_train[idx[:split]]
            X_va_raw, y_va = X_train[idx[split:]], y_train[idx[split:]]
        else:
            X_tr_raw, y_tr = X_train, y_train
            X_va_raw, y_va = X_val, y_val

        if fit_scaler:
            X_tr = self.scaler.fit_transform(X_tr_raw)
        else:
            X_tr = self.scaler.transform(X_tr_raw)
        X_va = self.scaler.transform(X_va_raw)

        y_tr_col = y_tr.reshape(-1, 1).astype(np.float32)
        y_va_col = y_va.reshape(-1, 1).astype(np.float32)

        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        best_val_loss = float("inf")
        best_params = {k: val.copy() for k, val in self.params.items()}
        wait = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "best_epoch": 0,
            "stopped_early": False,
        }

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(len(X_tr))
            lr = self.learning_rate * (lr_decay ** (epoch - 1))

            for start in range(0, len(X_tr), batch_size):
                batch_idx = perm[start : start + batch_size]
                xb = X_tr[batch_idx]
                yb = y_tr_col[batch_idx]
                n_b = len(xb)

                probs, cache = self._forward(xb, training=True, rng=rng)

                dz3 = (probs - yb) / n_b
                dW3 = cache["a2"].T @ dz3 + self.l2_lambda * self.params["W3"]
                db3 = np.sum(dz3, axis=0, keepdims=True)

                da2 = dz3 @ self.params["W3"].T
                dz2 = da2 * cache["m2"] * relu_grad(cache["z2"])
                dW2 = cache["a1"].T @ dz2 + self.l2_lambda * self.params["W2"]
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = dz2 @ self.params["W2"].T
                dz1 = da1 * cache["m1"] * relu_grad(cache["z1"])
                dW1 = xb.T @ dz1 + self.l2_lambda * self.params["W1"]
                db1 = np.sum(dz1, axis=0, keepdims=True)

                grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
                step += 1

                for k in self.params:
                    m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                    v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
                    m_hat = m[k] / (1 - beta1 ** step)
                    v_hat = v[k] / (1 - beta2 ** step)
                    self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            tr_probs, _ = self._forward(X_tr, training=False)
            va_probs, _ = self._forward(X_va, training=False)
            tr_m = compute_classification_metrics(y_tr_col, tr_probs)
            va_m = compute_classification_metrics(y_va_col, va_probs)

            history["train_loss"].append(tr_m["loss"])
            history["val_loss"].append(va_m["loss"])
            history["train_acc"].append(tr_m["accuracy"])
            history["val_acc"].append(va_m["accuracy"])

            if va_m["loss"] < best_val_loss - 1e-4:
                best_val_loss = va_m["loss"]
                best_params = {k: val.copy() for k, val in self.params.items()}
                history["best_epoch"] = epoch
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    history["stopped_early"] = True
                    break

        self.params = best_params
        self.history = history
        return history

    def predict_proba(self, X_raw):
        X_scaled = self.scaler.transform(X_raw)
        probs, _ = self._forward(X_scaled, training=False)
        return probs.ravel()

    def predict(self, X_raw, threshold=0.5):
        return (self.predict_proba(X_raw) >= threshold).astype(np.int32)

    def evaluate(self, X_raw, y_true):
        probs = self.predict_proba(X_raw)
        return compute_classification_metrics(y_true, probs)

    def save(self, path):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(self, f)
        return out_path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected CassavaNeuralClassifier, got {type(obj)}")
        return obj


def retrain_from_pretrained(
    pretrained_path,
    X_new,
    y_new,
    epochs=35,
    fine_tune_lr=3e-4,
    save_path=None,
):
    model = CassavaNeuralClassifier.load(pretrained_path)
    model.learning_rate = fine_tune_lr

    history = model.fit(
        X_new,
        y_new,
        epochs=epochs,
        batch_size=16,
        patience=8,
        fit_scaler=False,
    )

    minor = int(model.version.lstrip("v").split(".")[1]) + 1
    model.version = f"v1.{minor}.0"

    target = Path(save_path) if save_path else Path(pretrained_path)
    model.save(target)
    eval_metrics = model.evaluate(X_new, y_new)

    return model, {
        "version": model.version,
        "epochs_ran": len(history["train_loss"]),
        "best_epoch": history["best_epoch"],
        "stopped_early": history["stopped_early"],
        "metrics": eval_metrics,
    }
