import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_loader import load_data
from preprocessing import build_preprocessor
from config import (
    TARGET_COLUMN,
    ID_COLUMN,
    RANDOM_STATE,
    CV_FOLDS,
    SCORING_METRIC
)

def get_models(y):
    """Return models with imbalance handling"""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=y.value_counts()[0] / y.value_counts()[1],
            random_state=RANDOM_STATE
        )
    }

def run_pipeline():
    # 1. Load data
    df = load_data()

    # 2. Split features & target
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    # 3. Build preprocessing
    preprocessor = build_preprocessor(X)

    # 4. Cross-validation setup
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = get_models(y)

    best_model = None
    best_score = 0

    print("\n🔍 Model Comparison (ROC-AUC):\n")

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=SCORING_METRIC
        )

        mean_score = scores.mean()
        print(f"{name}: ROC-AUC = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = pipeline

    # 5. Train best model on full data
    best_model.fit(X, y)

    joblib.dump(best_model, "models/best_churn_model.pkl")

    print(f"\n✅ Best model saved (ROC-AUC = {best_score:.4f})")
