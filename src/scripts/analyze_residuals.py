import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.config import FileConfig, DBConfig, StrikeoutModelConfig
    from src.data.utils import DBConnection
except ImportError as e:
    raise SystemExit(f"Failed to import project modules: {e}")


def load_latest_model(model_dir: Path) -> Path | None:
    """Return the most recent LightGBM model path in a directory."""
    model_files = sorted(model_dir.glob("*strikeout_model_*.txt"), reverse=True)
    return model_files[0] if model_files else None


def load_feature_list(model_path: Path) -> list[str]:
    """Load saved feature column list next to the model if available."""
    feat_file = model_path.with_name(
        model_path.name.replace("strikeout_model", "feature_columns")
    ).with_suffix(".pkl")
    if feat_file.exists():
        with open(feat_file, "rb") as f:
            return pickle.load(f)
    booster = lgb.Booster(model_file=str(model_path))
    return list(booster.feature_name())


def compute_residuals(model: lgb.Booster, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    preds = model.predict(X, num_iteration=model.best_iteration)
    res = y - preds
    df = pd.DataFrame({"predicted": preds, "actual": y, "residual": res})
    return df


def plot_residuals(df: pd.DataFrame, output_dir: Path, team_col: str | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.histplot(df["residual"], bins=30, kde=True)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title("Residual Histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_hist.png")
    plt.close()

    if team_col:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=df[team_col], y=df["residual"])
        plt.xticks(rotation=90)
        plt.xlabel(team_col)
        plt.title("Residuals by Team")
        plt.tight_layout()
        plt.savefig(output_dir / "residuals_by_team.png")
        plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df["predicted"], y=df["residual"], alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs. Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_vs_pred.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze residuals on the test set")
    parser.add_argument("--model", type=str, help="Path to LightGBM model file. Defaults to latest model.")
    parser.add_argument("--features", type=str, help="Path to feature list .pkl. Defaults to file next to model.")
    parser.add_argument("--db", type=str, default=DBConfig.PATH, help="Path to SQLite DB with test_features table")
    parser.add_argument("--output-dir", type=str, default=str(FileConfig.PLOTS_DIR), help="Directory to save plots")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else load_latest_model(Path(FileConfig.MODELS_DIR))
    if not model_path or not model_path.exists():
        raise SystemExit("Model file not found. Specify --model or place a model in the models directory.")

    feature_list = None
    if args.features:
        with open(args.features, "rb") as f:
            feature_list = pickle.load(f)
    else:
        feature_list = load_feature_list(model_path)

    booster = lgb.Booster(model_file=str(model_path))

    db_path = Path(args.db)
    with DBConnection(db_path) as conn:
        test_df = pd.read_sql_query("SELECT * FROM test_features", conn)

    features = [f for f in feature_list if f in test_df.columns]
    X_test = test_df[features]
    y_test = test_df[StrikeoutModelConfig.TARGET_VARIABLE]

    res_df = compute_residuals(booster, X_test, y_test)

    team_col = None
    for col in ["team", "opponent_team", "home_team"]:
        if col in test_df.columns:
            team_col = col
            res_df[col] = test_df[col]
            break

    plot_residuals(res_df, Path(args.output_dir), team_col)
    print("Residual plots saved to", args.output_dir)


if __name__ == "__main__":
    main()
