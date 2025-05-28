import argparse
import pickle
from pathlib import Path
import lightgbm as lgb
import pandas as pd

try:
    from src.config import FileConfig
except ImportError as e:
    raise SystemExit(f"Failed to import config modules: {e}")


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
    # Fallback to names stored in the model
    booster = lgb.Booster(model_file=str(model_path))
    return list(booster.feature_name())


def compute_importance(model_path: Path) -> pd.DataFrame:
    booster = lgb.Booster(model_file=str(model_path))
    features = load_feature_list(model_path)
    importance_gain = booster.feature_importance(importance_type="gain")
    importance_split = booster.feature_importance(importance_type="split")
    df = pd.DataFrame(
        {
            "feature": features,
            "importance_gain": importance_gain,
            "importance_split": importance_split,
        }
    )
    return df.sort_values("importance_gain", ascending=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate feature importance table from a trained model"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to LightGBM model file. Defaults to latest in models dir.",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional path to save CSV of feature importance"
    )
    args = parser.parse_args()

    model_path = (
        Path(args.model)
        if args.model
        else load_latest_model(Path(FileConfig.MODELS_DIR))
    )
    if not model_path or not model_path.exists():
        raise SystemExit(
            "Model file not found. Specify --model or place a model in the models directory."
        )

    importance_df = compute_importance(model_path)
    print(importance_df.head(20).to_string(index=False))

    if args.output:
        importance_df.to_csv(args.output, index=False)
        print(f"Saved feature importance to {args.output}")


if __name__ == "__main__":
    main()
