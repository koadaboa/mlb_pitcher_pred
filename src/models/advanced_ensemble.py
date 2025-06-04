import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


class EnsembleStrikeoutModel:
    """Advanced ensemble model for strikeout prediction"""

    def __init__(self) -> None:
        self.models: dict[str, object] = {}
        self.feature_importances: dict[str, np.ndarray] = {}
        self.predictions_cache: dict[str, np.ndarray] = {}
        self.meta_model: Ridge | None = None

    def create_base_models(self) -> None:
        """Create diverse base models"""

        lgb_params_poisson = {
            "objective": "poisson",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 80,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 10,
            "reg_lambda": 3,
            "min_child_samples": 20,
            "seed": 42,
        }
        lgb_params_regression = lgb_params_poisson.copy()
        lgb_params_regression["objective"] = "regression"

        xgb_params = {
            "objective": "count:poisson",
            "eval_metric": "rmse",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 5,
            "reg_lambda": 5,
            "seed": 42,
        }

        cat_params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "verbose": False,
        }

        self.models = {
            "lgb_poisson": lgb.LGBMRegressor(**lgb_params_poisson, n_estimators=500),
            "lgb_regression": lgb.LGBMRegressor(**lgb_params_regression, n_estimators=500),
            "xgb": xgb.XGBRegressor(**xgb_params, n_estimators=500),
            "catboost": CatBoostRegressor(**cat_params, iterations=500),
        }
        self.models["ridge"] = Ridge(alpha=1.0)

    def create_stacked_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> np.ndarray:
        """Create a stacked ensemble"""

        base_predictions = np.zeros((len(X_train), len(self.models)))
        val_predictions = np.zeros((len(X_val), len(self.models)))

        for i, (name, model) in enumerate(self.models.items()):
            if name == "ridge":
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                model.fit(X_train_scaled, y_train)
                base_predictions[:, i] = model.predict(X_train_scaled)
                val_predictions[:, i] = model.predict(X_val_scaled)
            else:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False,
                )
                base_predictions[:, i] = model.predict(X_train)
                val_predictions[:, i] = model.predict(X_val)

        meta_model = Ridge(alpha=0.1)
        meta_model.fit(base_predictions, y_train)

        self.meta_model = meta_model
        return val_predictions

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add important interaction features"""

        interactions = [
            ("swinging_strike_rate_mean_10", "first_pitch_strike_rate_mean_10"),
            ("fastball_pct_mean_5", "breaking_ball_pct_mean_5"),
            ("temp", "breaking_ball_usage"),
            ("opp_ops_mean_10", "pitcher_stuff_plus"),
            ("pitch_count", "strikeout_rate_mean_5"),
            ("leverage_index", "two_strike_approach"),
        ]

        for feat1, feat2 in interactions:
            if feat1 in df.columns and feat2 in df.columns:
                df[f"{feat1}_X_{feat2}"] = df[feat1] * df[feat2]
                df[f"{feat1}_ratio_{feat2}"] = df[feat1] / (df[feat2] + 1e-8)

        return df

    def create_target_transformations(self, y: pd.Series) -> dict[str, np.ndarray]:
        """Create multiple target transformations"""

        transformations = {
            "original": y,
            "log1p": np.log1p(y),
            "sqrt": np.sqrt(y),
            "boxcox": self.boxcox_transform(y),
        }
        return transformations

    def boxcox_transform(self, y: pd.Series) -> np.ndarray:
        from scipy import stats

        transformed, _ = stats.boxcox(y + 1)
        return transformed


class AdvancedFeatureSelection:
    """More sophisticated feature selection"""

    def __init__(self) -> None:
        self.selected_features: list[str] = []
        self.feature_scores: dict[str, float] = {}

    def recursive_feature_elimination_cv(self, X: pd.DataFrame, y: pd.Series, estimator, cv: int = 5) -> list[str]:
        from sklearn.feature_selection import RFECV

        selector = RFECV(estimator, step=1, cv=cv, scoring="neg_mean_squared_error")
        selector = selector.fit(X, y)
        return X.columns[selector.support_].tolist()

    def permutation_importance_selection(
        self, model, X: pd.DataFrame, y: pd.Series, threshold: float = 0.001
    ) -> list[str]:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        important_features = X.columns[result.importances_mean > threshold].tolist()
        return important_features

    def correlation_clustering(self, X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        return [col for col in X.columns if col not in to_drop]


class RobustValidation:
    """More robust validation strategy"""

    def purged_walk_forward_cv(self, df: pd.DataFrame, n_splits: int = 5, gap_days: int = 7) -> list[tuple[pd.Index, pd.Index]]:
        df_sorted = df.sort_values("game_date")
        total_days = (df_sorted["game_date"].max() - df_sorted["game_date"].min()).days
        fold_size = total_days // n_splits

        folds = []
        for i in range(n_splits):
            train_end = df_sorted["game_date"].min() + pd.Timedelta(days=(i + 1) * fold_size)
            test_start = train_end + pd.Timedelta(days=gap_days)
            test_end = test_start + pd.Timedelta(days=fold_size // 2)
            train_idx = df_sorted[df_sorted["game_date"] <= train_end].index
            test_idx = df_sorted[(df_sorted["game_date"] >= test_start) & (df_sorted["game_date"] <= test_end)].index
            folds.append((train_idx, test_idx))

        return folds

    def adversarial_validation(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["is_test"] = 0
        test_df["is_test"] = 1
        combined = pd.concat([train_df, test_df])

        from sklearn.ensemble import RandomForestClassifier

        from sklearn.metrics import roc_auc_score

        feature_cols = [
            col
            for col in combined.columns
            if col not in ["is_test", "game_date", "game_pk", "strikeouts"]
        ]

        X = combined[feature_cols].fillna(0)
        y = combined["is_test"]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])

        if auc > 0.8:
            print(f"WARNING: High AUC ({auc:.3f}) suggests distribution shift")

        return auc
