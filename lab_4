from typing import Tuple, Dict, List
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
    accuracy_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression  # FIX: needed for A2

#Small IO helper
def read_csv_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(axis=1, how="all")
    return df

# Helper (used by A1 and elsewhere)
def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted"
) -> Dict[str, float]:
    #Return weighted precision, recall, f1, and accuracy.
    return {
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred))
    }

# Data loading and split helper (used by multiple questions)
def load_and_split_data(
    csv_path: str,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[label_col])
    y = df[label_col]
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    return X_train, y_train, X_test, y_test

# A1. Evaluate confusion matrix, precision, recall, F1 for both train and test; infer fit by metrics spread
def evaluate_knn_confusion_and_metrics(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k: int = 5
) -> Dict[str, Dict]:
    #Train standardized kNN, compute CMs and metrics for train/test.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    classes = np.unique(np.concatenate([y_train.values, y_test.values]))
    train_cm = confusion_matrix(y_train, y_pred_train, labels=classes)
    test_cm = confusion_matrix(y_test, y_pred_test, labels=classes)

    return {
        "model": pipe,
        "classes": classes.tolist(),
        "train": {
            "confusion_matrix": train_cm,
            "metrics": _classification_metrics(y_train, y_pred_train, average="weighted"),
        },
        "test": {
            "confusion_matrix": test_cm,
            "metrics": _classification_metrics(y_test, y_pred_test, average="weighted"),
        }
    }

# A2. Calculate MSE, RMSE, MAPE and R2 for the price prediction exercise (Lab 02)
def split_purchase_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    #Extract X and y from purchase data.
    feature_cols = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
    target_col = "Payment (Rs)"
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y

def make_linear_regression_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression())
    ])
# MAPE in percent, safe for zero/near-zero targets
def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)
#Compute MSE, RMSE, MAPE% and R2
def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "RMSE": rmse, "MAPE_%": mape, "R2": float(r2)}
#Train Linear Regression
def train_eval_regression(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = make_linear_regression_pipeline()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return {
        "train": regression_scores(y_train, y_pred_train),
        "test": regression_scores(y_test, y_pred_test)
    }

# A3. Generate 20 data points (2D) in [1,10], assign two classes, return DataFrame with colors
def generate_training_points_2d(
    n_points: int = 20,
    low: float = 1.0,
    high: float = 10.0,
    random_state: int = 42
) -> pd.DataFrame:
    # using median(x+y) rule
    rng = np.random.default_rng(random_state)
    X_vals = rng.uniform(low, high, size=n_points)
    Y_vals = rng.uniform(low, high, size=n_points)
    s = X_vals + Y_vals
    thresh = np.median(s)
    labels = np.where(s > thresh, 1, 0)
    colors = np.where(labels == 1, "red", "blue")
    return pd.DataFrame({"X": X_vals, "Y": Y_vals, "label": labels, "color": colors})

# A4. Classify a dense grid [0,10]x[0,10] via kNN (k=3) trained on A3 data
def classify_grid_with_knn(
    train_df: pd.DataFrame,
    k: int = 3,
    step: float = 0.1
) -> pd.DataFrame:
    X_train = train_df[["X", "Y"]].values
    y_train = train_df["label"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    pipe.fit(X_train, y_train)

    grid_vals = np.arange(0.0, 10.0 + 1e-9, step)
    gx, gy = np.meshgrid(grid_vals, grid_vals)
    grid_points = np.c_[gx.ravel(), gy.ravel()]

    preds = pipe.predict(grid_points)
    pred_colors = np.where(preds == 1, "red", "blue")

    return pd.DataFrame({
        "X": grid_points[:, 0],
        "Y": grid_points[:, 1],
        "pred_label": preds,
        "pred_color": pred_colors
    })

# A5. Repeat A4 for various k values; return predictions per k
def classify_grid_multiple_k(
    train_df: pd.DataFrame,
    k_values: List[int],
    step: float = 0.1
) -> Dict[int, pd.DataFrame]:
    out = {}
    for k in k_values:
        out[k] = classify_grid_with_knn(train_df, k=k, step=step)
    return out

# A6. Repeat A3–A5 for project data (two features and two classes)
def project_2d_knn_boundaries(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    feature_pair: Tuple[str, str],
    class_subset: List[str],
    k_values: List[int],
    step: float = 0.1
) -> Dict[int, pd.DataFrame]:
    f1, f2 = feature_pair
    mask = y_all.astype(str).isin([str(c) for c in class_subset])
    Xb = X_all.loc[mask, [f1, f2]].copy()
    yb = y_all.loc[mask].astype(str).copy()

    def _bounds(arr: np.ndarray, pad: float = 0.05):
        lo, hi = np.min(arr), np.max(arr)
        padv = (hi - lo) * pad if hi > lo else 1.0
        return lo - padv, hi + padv

    x_lo, x_hi = _bounds(Xb[f1].values)
    y_lo, y_hi = _bounds(Xb[f2].values)

    gx = np.arange(x_lo, x_hi, step)
    gy = np.arange(y_lo, y_hi, step)
    MX, MY = np.meshgrid(gx, gy)
    grid_points = np.c_[MX.ravel(), MY.ravel()]

    results = {}
    for k in k_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])
        pipe.fit(Xb.values, yb.values)
        preds = pipe.predict(grid_points)
        results[k] = pd.DataFrame({
            f1: grid_points[:, 0],
            f2: grid_points[:, 1],
            "pred_label": preds
        })
    return results

# A7. Hyper-parameter tuning to find ideal ‘k’
def tune_knn_k(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_grid: List[int] = None,
    use_random: bool = False,
    cv: int = 5,
    scoring: str = "f1_weighted",
    random_iter: int = 20,
    random_state: int = 42
) -> Dict:
    #Tune k with GridSearchCV (default) or RandomizedSearchCV.
    if k_grid is None:
        k_grid = list(range(1, 31, 2))  # odd ks reduce ties

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])
    param_grid = {"knn__n_neighbors": k_grid}

    if use_random:
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=min(random_iter, len(k_grid)),
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1
        )

    search.fit(X_train, y_train)
    best_k = int(search.best_params_["knn__n_neighbors"])
    best_score = float(search.best_score_)
    return {"best_k": best_k, "best_score": best_score, "search": search}

#Main program
if __name__ == "__main__":
    crop_csv = "Crop_recommendation.csv"
    purchase_csv = "Purchase_data).csv"

    # Load dataset (utility)
    X_train, y_train, X_test, y_test = load_and_split_data(crop_csv)

    # A1
    a1_out = evaluate_knn_confusion_and_metrics(X_train, y_train, X_test, y_test, k=5)
    print("\nA1: Confusion matrices and metrics (k=5)")
    print("Classes (order):", a1_out["classes"])
    print("Train confusion matrix:\n", a1_out["train"]["confusion_matrix"])
    print("Train metrics:", a1_out["train"]["metrics"])
    print("Test confusion matrix:\n", a1_out["test"]["confusion_matrix"])
    print("Test metrics:", a1_out["test"]["metrics"])

    tr_f1 = a1_out["train"]["metrics"]["f1"]
    te_f1 = a1_out["test"]["metrics"]["f1"]
    gap = tr_f1 - te_f1
    if gap > 0.10:
        fit_infer = "Potential overfit (train >> test)"
    elif gap < -0.02:
        fit_infer = "Potential underfit (test > train; both may be low)"
    else:
        fit_infer = "Likely regular fit (similar train/test)"
    print("A1 Inference (based on train-test F1 gap):", fit_infer)

    # A2
    df_purchase = read_csv_clean(purchase_csv)  # FIX: define cleaner + path
    X_purchase, y_purchase = split_purchase_features_target(df_purchase)
    a2_results = train_eval_regression(X_purchase, y_purchase, test_size=0.3, random_state=42)
    print("\nA2: Payment Prediction (Linear Regression)")
    print("Train Metrics:", a2_results["train"])
    print("Test  Metrics:", a2_results["test"])

    # A3
    train2d = generate_training_points_2d(n_points=20, random_state=7)
    print("\nA3: 2D training set head (X, Y, label, color):")
    print(train2d.head())

    # A4
    grid_pred_k3 = classify_grid_with_knn(train2d, k=3, step=0.2)  # 0.2 keeps runtime light
    print("\nA4: Grid predictions (k=3) sample:")
    print(grid_pred_k3.head())
    print("A4: Total grid points (k=3):", len(grid_pred_k3))

    # A5
    k_list = [1, 3, 5, 9, 15]
    grid_multi = classify_grid_multiple_k(train2d, k_values=k_list, step=0.2)
    print("\nA5: Generated grid predictions for ks:", k_list)
    for k in k_list:
        print(f"k={k}, rows={len(grid_multi[k])}")

    # A6
    feature_pair = ("N", "temperature")
    class_subset = ["rice", "maize"]
    X_all = pd.concat([X_train, X_test], ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True)
    proj_boundaries = project_2d_knn_boundaries(
        X_all, y_all,
        feature_pair=feature_pair,
        class_subset=class_subset,
        k_values=[3, 7, 11],
        step=0.5
    )
    print("\nA6: Project 2D kNN decision grids generated for k in [3, 7, 11].")
    for k, df_k in proj_boundaries.items():
        print(f"k={k}, feature-grid points={len(df_k)}")

    # A7
    tuning = tune_knn_k(X_train, y_train, k_grid=list(range(1, 26, 2)), use_random=False, cv=5)
    print("\nA7: Hyper-parameter tuning (GridSearchCV)")
    print({"best_k": tuning["best_k"], "best_cv_score(f1_weighted)": tuning["best_score"]})
