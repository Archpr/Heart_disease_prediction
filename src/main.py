from pathlib import Path
from data_loader import load_data, check_missing_values, target_distribution
from preprocessing import split_data, build_preprocessor, preprocess_data
from model import train_baseline_model, tune_model
from evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "heart.csv"


def main():
    df = load_data(DATA_PATH)
    print(df.columns.tolist())


    check_missing_values(df)
    target_distribution(df, "HeartDisease")

    X_train, X_test, y_train, y_test = split_data(df, "HeartDisease")

    preprocessor = build_preprocessor(X_train)
    X_train_p, X_test_p = preprocess_data(preprocessor, X_train, X_test)

    # Baseline model
    baseline_model = train_baseline_model(X_train_p, y_train)
    y_test_, y_pred, y_proba, metrics = evaluate_model(
        baseline_model, X_test_p, y_test, "Baseline LightGBM"
    )
    plot_confusion_matrix(y_test_, y_pred, "Confusion Matrix (Baseline)")
    plot_roc_curve(y_test_, y_proba, metrics["roc_auc"], "ROC Curve (Baseline)")

    # Tuned model
    print(">>> Entered tune_model()")
    best_model, best_params, best_score = tune_model(X_train_p, y_train)
    print("\nBest Hyperparameters:", best_params)
    print("Best CV ROC AUC:", best_score)
    print(">>> Exited tune_model()")


    y_test_, y_pred, y_proba, metrics = evaluate_model(
        best_model, X_test_p, y_test, "Tuned LightGBM"
    )
    plot_confusion_matrix(y_test_, y_pred, "Confusion Matrix (Tuned)")
    plot_roc_curve(y_test_, y_proba, metrics["roc_auc"], "ROC Curve (Tuned)")

if __name__ == "__main__":
    main()
