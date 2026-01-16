import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    GroupShuffleSplit,
    GroupKFold,
)
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    make_scorer,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# ============================================================
# Helper 1: Convert spectrogram values to dB (log scale)
# ============================================================
def _to_db(specs: np.ndarray, spec_kind: str = "power", eps: float = 1e-10) -> np.ndarray:
    """
    Convert spectrograms to dB scale safely.

    Why?
    - Raw spectrogram values (power/magnitude) can have a HUGE numeric range.
    - Taking log compresses the range and makes training more stable.

    Parameters
    ----------
    specs : np.ndarray
        Spectrograms array (N, H, W).
    spec_kind : str
        What the spectrogram values represent:
          - "power"     : values are power -> 10*log10(power)
          - "magnitude" : values are magnitude -> 20*log10(magnitude)
          - "db"        : already in dB -> no change
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    np.ndarray
        Spectrograms in dB scale (float32).
    """
    specs = np.asarray(specs)

    # If already in dB, do nothing
    if spec_kind == "db":
        return specs.astype(np.float32)

    # Ensure everything is >= eps so log() is safe
    specs = np.maximum(specs, eps)

    if spec_kind == "magnitude":
        return (20.0 * np.log10(specs)).astype(np.float32)
    if spec_kind == "power":
        return (10.0 * np.log10(specs)).astype(np.float32)

    raise ValueError("spec_kind must be one of: 'power', 'magnitude', 'db'")


# ============================================================
# Helper 2: Decide which label is the "positive" class (Drone)
# ============================================================
def _infer_positive_encoded_label(classes, positive_class=None):
    """
    Figure out which encoded label should be treated as the "positive" class.

    Why do we need this?
    - We often care most about Drone RECALL (not missing drones).
    - But LabelEncoder may map labels alphabetically, so "Drone" isn't always encoded as 1.
    - We must explicitly know which encoded label corresponds to Drone.

    Parameters
    ----------
    classes : array-like
        le.classes_ from LabelEncoder (original labels in encoder order).
    positive_class : optional
        If you know exactly which label means Drone (e.g., "Drone" or 1), pass it here.
        This overrides all heuristics.

    Returns
    -------
    (pos_encoded, pos_original)
        pos_encoded  : integer encoded label used inside the model
        pos_original : original label name/value (e.g. "Drone")
    """
    classes = np.asarray(classes)

    # Case 1: user explicitly tells us what "Drone" is
    if positive_class is not None:
        # exact match
        for i, c in enumerate(classes):
            if c == positive_class:
                return i, c
        # case-insensitive match for strings
        pc = str(positive_class).strip().lower()
        for i, c in enumerate(classes):
            if str(c).strip().lower() == pc:
                return i, c
        raise ValueError(f"positive_class={positive_class} not found in classes={list(classes)}")

    # Case 2: common numeric labels {0,1} and assume 1 means Drone
    try:
        vals = set(classes.tolist())
        if vals == {0, 1}:
            pos = int(np.where(classes == 1)[0][0])
            return pos, classes[pos]
    except Exception:
        pass

    # Case 3: strings like "Drone" / "Non-Drone" (heuristics)
    lowered = [str(c).lower() for c in classes]

    # Prefer label that contains "drone" but not "non"/"not"
    drone_candidates = []
    for i, s in enumerate(lowered):
        if "drone" in s and ("non" not in s) and ("not" not in s):
            drone_candidates.append(i)
    if len(drone_candidates) == 1:
        pos = drone_candidates[0]
        return pos, classes[pos]

    # If one class looks like non-drone, pick the other (binary only)
    for i, s in enumerate(lowered):
        if "non-drone" in s or "nondrone" in s or ("non" in s and "drone" in s):
            if len(classes) == 2:
                pos = 1 - i
                return pos, classes[pos]

    # Fallback (binary): assume encoded label 1 is Drone
    if len(classes) == 2:
        return 1, classes[1]

    # Multi-class fallback: no single "positive"
    return None, None


# ============================================================
# Main function: Train + Validate + Test a Drone/Non-Drone SVM
# ============================================================
def train_drone_svm_v5(
    spectrograms,
    labels,
    *,
    spec_kind: str = "power",
    n_components=0.95,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    group_ids=None,
    optimize_metric: str = "recall",
    positive_class=None,
    save_path: str = "drone_final_pipeline.pkl",
    show_plot: bool = True,
):
    """
    Train a classifier that predicts: Drone vs Non-Drone from spectrograms.

    Pipeline (what happens to each spectrogram):
      1) Convert spectrogram values to dB (log scale)  [optional, based on spec_kind]
      2) Flatten (H,W) into a long vector (H*W)
      3) Standardize features (mean=0, std=1)
      4) PCA to reduce dimensions and noise
      5) SVM (RBF kernel) classification

    Why optimize_metric="recall"?
    - In detection tasks, missing a real drone (false negative) is usually worse than a false alarm.
    - So we tune hyperparameters to maximize Drone recall by default.

    Parameters
    ----------
    spectrograms : array-like
        Shape (N, H, W).
    labels : array-like
        Shape (N,). Can be [0,1] or strings like ["Drone","Non-Drone"].
    spec_kind : str
        "power", "magnitude", or "db" (already log-scaled).
    n_components : float or int
        PCA components. 0.95 means keep 95% of variance.
    test_size : float
        Fraction reserved for FINAL test set.
    val_size : float
        Fraction reserved for validation (used before final test).
    group_ids : array-like or None
        IMPORTANT if you create many samples from one recording.
        If provided, splits will be done by recording/session to avoid leakage.
    optimize_metric : str
        "recall" (default) or "f1".
    positive_class : optional
        Force which label means "Drone". Example: "Drone" or 1.
    save_path : str
        Path to save the trained model + label encoder.
    show_plot : bool
        If True, show confusion matrix for test set.

    Returns
    -------
    payload : dict
        Contains:
          - model (final sklearn Pipeline)
          - label_encoder (to map labels back/forth)
          - classes_ (original label names)
          - best_params (best C and gamma found)
          - metadata (spec_kind, n_components, etc.)
    """

    # -----------------------------
    # 0) Basic checks
    # -----------------------------
    specs_array = np.asarray(spectrograms)
    if specs_array.ndim != 3:
        raise ValueError(f"spectrograms must have shape (N,H,W). Got {specs_array.shape}")

    y_raw = np.asarray(labels)
    if len(specs_array) != len(y_raw):
        raise ValueError(f"Mismatch: {len(specs_array)} spectrograms vs {len(y_raw)} labels")

    # -----------------------------
    # 1) Encode labels into integers
    # -----------------------------
    # Model works with integers 0..K-1, but we keep the mapping to print nice names later.
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError("Need at least 2 classes.")

    # Decide which class is "Drone" (positive) so recall/f1 are computed correctly
    pos_idx, pos_label_original = _infer_positive_encoded_label(classes, positive_class=positive_class)

    # -----------------------------
    # 2) Preprocess spectrograms
    # -----------------------------
    specs_db = _to_db(specs_array, spec_kind=spec_kind)
    X = specs_db.reshape(len(specs_db), -1).astype(np.float32)  # flatten each spectrogram

    # -----------------------------
    # 3) Split into Train / Val / Test
    # -----------------------------
    # If group_ids exist: avoid leakage across windows from the same recording.
    using_groups = group_ids is not None

    if using_groups:
        groups = np.asarray(group_ids)
        if len(groups) != len(y):
            raise ValueError("group_ids must have same length as labels")

        # Split off TEST set by groups
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        trainval_idx, test_idx = next(gss1.split(X, y, groups=groups))
        X_trainval, y_trainval, g_trainval = X[trainval_idx], y[trainval_idx], groups[trainval_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Split remaining into TRAIN and VAL by groups
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state + 1)
        train_idx, val_idx = next(gss2.split(X_trainval, y_trainval, groups=g_trainval))
        X_train, y_train, g_train = X_trainval[train_idx], y_trainval[train_idx], g_trainval[train_idx]
        X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

        cv = GroupKFold(n_splits=5)

    else:
        # Regular stratified split (keeps label ratio similar across sets)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size, random_state=random_state + 1, stratify=y_trainval
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # -----------------------------
    # 4) Choose scoring for GridSearchCV
    # -----------------------------
    metric = optimize_metric.strip().lower()

    if n_classes == 2:
        # We tune to maximize Drone recall or Drone f1
        if pos_idx is None:
            pos_idx = 1
        if metric == "recall":
            scorer = make_scorer(recall_score, pos_label=pos_idx)
        elif metric == "f1":
            scorer = make_scorer(f1_score, pos_label=pos_idx)
        else:
            raise ValueError("optimize_metric must be 'recall' or 'f1' for binary problems.")
    else:
        # Multi-class fallback
        scorer = "recall_macro" if metric == "recall" else "f1_macro"

    # -----------------------------
    # 5) Build ML pipeline
    # -----------------------------
    # probability=False is faster during grid search. We'll turn it on at the end.
    base_pipe = Pipeline([
        ("scaler", StandardScaler()),                           # normalize features
        ("pca", PCA(n_components=n_components, svd_solver="full")),  # reduce dimensions
        ("svm", SVC(kernel="rbf", class_weight="balanced",
                    probability=False, cache_size=512)),        # classifier
    ])

    # Hyperparameters to try
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],          # regularization strength
        "svm__gamma": ["scale", "auto", 0.01, 0.001],  # RBF kernel width
    }

    # Print summary before training
    print(f"Dataset: {len(X)} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print("Classes (encoder order):", [str(c) for c in classes])
    if n_classes == 2:
        print(f"Positive (Drone) class assumed: {pos_label_original} -> encoded as {pos_idx}")
        print(f"Optimizing metric: {optimize_metric}")
    if using_groups:
        print("[INFO] Using group-based split/CV to prevent recording leakage.")

    # -----------------------------
    # 6) Grid search (train only)
    # -----------------------------
    grid = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    if using_groups:
        grid.fit(X_train, y_train, groups=g_train)
    else:
        grid.fit(X_train, y_train)

    best_params = grid.best_params_
    print("\nBest CV params:", best_params)
    print("Best CV score:", grid.best_score_)

    # -----------------------------
    # 7) Validate on VAL set
    # -----------------------------
    best_cv_model = grid.best_estimator_
    y_val_pred = best_cv_model.predict(X_val)

    print("\n--- VALIDATION SET ---")
    print(f"Val Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(classification_report(y_val, y_val_pred, target_names=[str(c) for c in classes]))
    if n_classes == 2:
        print(f"Val Drone Recall: {recall_score(y_val, y_val_pred, pos_label=pos_idx):.4f}")
        print(f"Val Drone Precision: {precision_score(y_val, y_val_pred, pos_label=pos_idx):.4f}")

    # -----------------------------
    # 8) Final refit on (Train + Val)
    # -----------------------------
    # Now we train once more using the best hyperparameters, with probability=True.
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, svd_solver="full")),
        ("svm", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,       # enables predict_proba()
            cache_size=512,
            C=best_params["svm__C"],
            gamma=best_params["svm__gamma"],
        )),
    ])

    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    final_pipe.fit(X_fit, y_fit)

    # -----------------------------
    # 9) Final evaluation on TEST set (never used in training)
    # -----------------------------
    y_test_pred = final_pipe.predict(X_test)

    print("\n--- FINAL TEST SET ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, target_names=[str(c) for c in classes]))
    if n_classes == 2:
        print(f"Test Drone Recall: {recall_score(y_test, y_test_pred, pos_label=pos_idx):.4f}")
        print(f"Test Drone Precision: {precision_score(y_test, y_test_pred, pos_label=pos_idx):.4f}")
        print(f"Test Drone F1: {f1_score(y_test, y_test_pred, pos_label=pos_idx):.4f}")

    # Confusion matrix (how many correct/incorrect per class)
    cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(n_classes))
    if show_plot:
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (Test)")
        plt.colorbar()
        ticks = np.arange(n_classes)
        plt.xticks(ticks, [str(c) for c in classes], rotation=45, ha="right")
        plt.yticks(ticks, [str(c) for c in classes])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # 10) Save model + label encoder
    # -----------------------------
    # Saving the encoder is important so you can decode predictions later.
    payload = {
        "model": final_pipe,
        "label_encoder": le,
        "classes_": classes,
        "best_params": best_params,
        "spec_kind": spec_kind,
        "n_components": n_components,
        "optimize_metric": optimize_metric,
        "positive_class_original": pos_label_original,
        "positive_class_encoded": pos_idx,
    }
    joblib.dump(payload, save_path)
    print(f"\nSaved to: {save_path}")

    return payload


# ============================================================
# Example usage
# ============================================================
# payload = train_drone_svm_v5(
#     spectrograms=my_specs,          # shape (N,H,W)
#     labels=my_labels,              # 0/1 or "Drone"/"Non-Drone"
#     group_ids=my_recording_ids,    # recommended if you have it
#     optimize_metric="recall",      # default: maximize Drone recall
#     # positive_class="Drone",       # optional: force the Drone label
# )
# model = payload["model"]
# probs = model.predict_proba(X_new)  # available because probability=True in final model
