import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # BONUS 5: Canh bao lech lac du lieu (Data Drift)
    pos_ratio = float(y_train.mean())
    baseline_ratio = 0.248
    drift_diff = abs(pos_ratio - baseline_ratio)
    if drift_diff > 0.05:
        print(f"[CANH BAO DATA DRIFT] Ty le lop duong ({pos_ratio:.4f}) lech {drift_diff:.4f} (> 5%) so voi tham chieu ({baseline_ratio:.4f})!")
    else:
        print(f"[DATA QUALITY OK] Ty le lop duong: {pos_ratio:.4f} (tham chieu {baseline_ratio:.4f}, chenh lech: {drift_diff:.4f})")

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)
        mlflow.log_metric("pos_class_ratio", pos_ratio)

        # TODO 4: Khoi tao va huan luyen GradientBoostingClassifier
        # Goi y: su dung random_state=42 de dam bao tinh tai tao
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap holdout va tinh chi so
        # Chu y: f1_score o day tinh cho LOP DUONG (target = 1), khong dung average.
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # BONUS 2: Dieu chinh nguong quyet dinh (Decision Threshold Tuning)
        probs = model.predict_proba(X_eval)[:, 1]
        best_threshold = 0.5
        best_f1_tuned = f1
        for thresh in np.arange(0.1, 0.91, 0.05):
            t_preds = (probs >= thresh).astype(int)
            t_f1 = float(f1_score(y_eval, t_preds))
            if t_f1 > best_f1_tuned:
                best_f1_tuned = t_f1
                best_threshold = float(thresh)

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("f1_score_tuned", best_f1_tuned)
        mlflow.sklearn.log_model(model, "model")

        # BONUS 3: Tinh confusion matrix, precision / recall theo lop
        cm = confusion_matrix(y_eval, preds)
        cr = classification_report(y_eval, preds, digits=4)

        # TODO 7: In ket qua ra man hinh
        print(f"F1 (nguong 0.5): {f1:.4f} | Accuracy: {acc:.4f}")
        print(f"F1 toi uu (nguong {best_threshold:.2f}): {best_f1_tuned:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{cr}")

        # TODO 8: Luu metrics ra file outputs/report.json va outputs/detail.txt
        os.makedirs("outputs", exist_ok=True)
        report_data = {
            "f1_score": f1,
            "accuracy": acc,
            "positive_class_ratio": pos_ratio,
            "best_threshold": best_threshold,
            "f1_score_tuned": best_f1_tuned,
        }
        with open("outputs/report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write("=== CONFUSION MATRIX ===\n")
            f.write(str(cm) + "\n\n")
            f.write("=== CLASSIFICATION REPORT ===\n")
            f.write(cr + "\n")
            f.write(f"\nTy le lop duong: {pos_ratio:.4f}\n")
            f.write(f"Nguong toi uu: {best_threshold:.2f} (F1 = {best_f1_tuned:.4f})\n")

        # TODO 9: Luu mo hinh ra file models/model.joblib
        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # TODO 10: Tra ve f1
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
