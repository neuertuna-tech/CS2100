"""
CS 4700 Final Machine Learning Project
Project: NFL Fantasy Football High-Scoring Game Prediction

Research Question:
Can a machine learning model predict whether an NFL offensive player will have a high fantasy
football performance in a weekly game using player/team information and previous performance trends?

Target:
high_fantasy_game = 1 if fantasy_points_ppr >= 10, else 0

Dataset files expected:
- data/raw/archive.zip OR data/raw/weekly_player_stats_offense.csv

Run:
python src/main.py
"""

import os
import zipfile
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
RAW_DIR = "data/raw"
ZIP_PATH = os.path.join(RAW_DIR, "archive.zip")
CSV_PATH = os.path.join(RAW_DIR, "weekly_player_stats_offense.csv")
TARGET_POINTS_COLUMN = "fantasy_points_ppr"
TARGET_COLUMN = "high_fantasy_game"
HIGH_GAME_THRESHOLD = 10
SAMPLE_SIZE = 5000  # set to None to use the full dataset

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def extract_dataset_if_needed():
    """Extracts the Kaggle archive if the CSV has not been extracted yet."""
    if os.path.exists(CSV_PATH):
        return
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            "Could not find data/raw/archive.zip or data/raw/weekly_player_stats_offense.csv"
        )
    print("Extracting archive.zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(RAW_DIR)


def load_data():
    """Loads selected columns from the weekly player offense dataset."""
    extract_dataset_if_needed()

    selected_columns = [
        "player_name",
        "position",
        "position_group",
        "team",
        "years_exp",
        "season",
        "week",
        "offense_pct",
        "avg_pass_attempts",
        "avg_complete_pass",
        "avg_passing_yards",
        "avg_pass_touchdown",
        "avg_interception",
        "avg_targets",
        "avg_receptions",
        "avg_receiving_yards",
        "avg_yards_after_catch",
        "avg_rush_attempts",
        "avg_rushing_yards",
        "avg_rush_touchdown",
        "avg_touches",
        "avg_total_yards",
        "avg_total_tds",
        "avg_fantasy_points_ppr",
        "prev_4_trend_fantasy_points_ppr",
        "prev_4_trend_touches",
        "prev_4_trend_total_yards",
        "prev_4_trend_total_tds",
        TARGET_POINTS_COLUMN,
    ]

    available_columns = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
    selected_columns = [c for c in selected_columns if c in available_columns]

    df = pd.read_csv(CSV_PATH, usecols=selected_columns)

    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    return df


def inspect_data(df):
    """Prints basic dataset information."""
    print("\n===== DATASET INSPECTION =====")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())


def create_target_and_features(df):
    """Creates the classification target and removes target leakage columns."""
    df = df.dropna(subset=[TARGET_POINTS_COLUMN]).copy()
    df[TARGET_COLUMN] = (df[TARGET_POINTS_COLUMN] >= HIGH_GAME_THRESHOLD).astype(int)

    # Remove direct outcome columns. The model should use prior averages/trends, not the answer.
    drop_columns = [TARGET_POINTS_COLUMN, TARGET_COLUMN]
    X = df.drop(columns=drop_columns)
    y = df[TARGET_COLUMN]

    # player_name is useful for error analysis but not ideal for modeling because it can overfit.
    if "player_name" in X.columns:
        X = X.drop(columns=["player_name"])

    return df, X, y


def run_eda(df):
    """Creates simple EDA visualizations."""
    print("\n===== TARGET CLASS BALANCE =====")
    print(df[TARGET_COLUMN].value_counts())
    print(df[TARGET_COLUMN].value_counts(normalize=True))

    plt.figure(figsize=(6, 4))
    df[TARGET_COLUMN].value_counts().sort_index().plot(kind="bar")
    plt.title("Class Balance: High Fantasy Game")
    plt.xlabel("0 = below 10 PPR points, 1 = 10+ PPR points")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("reports/figures/class_balance.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    df[TARGET_POINTS_COLUMN].hist(bins=40)
    plt.title("Distribution of PPR Fantasy Points")
    plt.xlabel("Fantasy Points PPR")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("reports/figures/fantasy_points_distribution.png")
    plt.close()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_cols = [c for c in numeric_cols if c != TARGET_POINTS_COLUMN]
    corr_cols = corr_cols[:20]
    if len(corr_cols) > 2:
        corr = df[corr_cols].corr()
        plt.figure(figsize=(12, 9))
        plt.imshow(corr, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("reports/figures/correlation_heatmap.png")
        plt.close()


def build_preprocessor(X):
    """Builds preprocessing for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates a classification model and returns a metrics dictionary."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
    }

    print(f"\n===== {model_name} =====")
    for key, value in results.items():
        if key != "Model":
            print(f"{key}: {value:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return results, predictions


def main():
    df = load_data()
    inspect_data(df)
    df, X, y = create_target_and_features(df)
    run_eda(df)

    print("\n===== TRAIN/TEST SPLIT =====")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("Training rows:", X_train.shape[0])
    print("Testing rows:", X_test.shape[0])

    preprocessor = build_preprocessor(X)

    baseline_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    print("\nTraining baseline Logistic Regression model...")
    baseline_model.fit(X_train, y_train)
    baseline_results, baseline_predictions = evaluate_model(
        baseline_model, X_test, y_test, "Baseline Logistic Regression"
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [80, 120],
        "classifier__max_depth": [8, 14],
        "classifier__min_samples_leaf": [1, 3],
    }

    print("\nTuning Random Forest with cross-validation...")
    grid_search = GridSearchCV(
        random_forest,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=1,
    )
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    final_results, final_predictions = evaluate_model(
        best_model, X_test, y_test, "Tuned Random Forest"
    )

    cm = confusion_matrix(y_test, final_predictions)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("Tuned Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png")
    plt.close()

    results_df = pd.DataFrame([baseline_results, final_results])
    results_df.to_csv("reports/model_results.csv", index=False)

    error_df = X_test.copy()
    error_df["actual"] = y_test.values
    error_df["predicted"] = final_predictions
    error_df["correct"] = error_df["actual"] == error_df["predicted"]
    error_df[error_df["correct"] == False].to_csv(
        "reports/misclassified_examples.csv", index=False
    )

    df.to_csv("data/processed/cleaned_weekly_player_stats_sample.csv", index=False)
    joblib.dump(best_model, "models/tuned_random_forest_model.pkl")

    print("\n===== FINAL SUMMARY =====")
    print("Project: NFL Fantasy Football High-Scoring Game Prediction")
    print("Target: 1 if player scored at least 10 PPR fantasy points, else 0")
    print("Baseline model: Logistic Regression")
    print("Final model: Tuned Random Forest")
    print("Results saved to reports/model_results.csv")
    print("Model saved to models/tuned_random_forest_model.pkl")


if __name__ == "__main__":
    main()
