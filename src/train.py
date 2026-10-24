"""Step 4 - model development and evaluation.

Trains four regression algorithms on the same features:
    Ridge regression (linear baseline), k-nearest neighbors,
    random forest, gradient boosting.
Protocol
  * 80/20 train/test split, stratified by model so every make/model is in both parts
  * hyper-parameters chosen by 5-fold cross-validation on the training set only
  * cross-validated MAE / RMSE / R^2 of the chosen settings, then a single
    evaluation on the held-out test set
High-cardinality columns (model_trim, market) are encoded inside the pipeline,
so any encoding is learned from training folds only (no leakage).
An experiment run first (training data only, 5-fold CV) compares four ways to
encode those columns, and the best one is used for every model.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from config import MODEL_READY_CSV, RANDOM_STATE, RESULTS
from features import HIGH_CARD_FEATURES, LOW_CARD_FEATURES, NUMERIC_FEATURES, TARGET

CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
FEATURES = NUMERIC_FEATURES + LOW_CARD_FEATURES + HIGH_CARD_FEATURES


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its share of the training rows (unseen categories -> 0)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freqs_ = {c: X[c].value_counts(normalize=True) for c in X.columns}
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return np.column_stack([X[c].map(self.freqs_[c]).fillna(0).to_numpy() for c in self.columns_])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.columns_])


def make_preprocessor(scale=True, high_card="target"):
    """Numeric: median impute (+ scale). Low-cardinality: one-hot. High-cardinality: see high_card."""
    num = make_pipeline(SimpleImputer(strategy="median"), *([StandardScaler()] if scale else []))
    low = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False)
    if high_card == "target":
        high = make_pipeline(TargetEncoder(target_type="continuous", cv=5, random_state=RANDOM_STATE),
                             *([StandardScaler()] if scale else []))
    elif high_card == "onehot":
        high = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=3, sparse_output=False)
    elif high_card == "frequency":
        high = make_pipeline(FrequencyEncoder(), *([StandardScaler()] if scale else []))
    else:  # drop the high-cardinality columns
        high = "drop"
    return ColumnTransformer([
        ("num", num, NUMERIC_FEATURES),
        ("cat", low, LOW_CARD_FEATURES),
        ("high", high, HIGH_CARD_FEATURES),
    ], verbose_feature_names_out=True)


# each entry: (builder taking the high-cardinality encoding, hyper-parameter grid)
MODELS = {
    "Ridge Regression": (
        lambda enc: Pipeline([("prep", make_preprocessor(scale=True, high_card=enc)), ("model", Ridge())]),
        {"model__alpha": [0.1, 1.0, 10.0, 30.0]},
    ),
    "K-Nearest Neighbors": (
        lambda enc: Pipeline([("prep", make_preprocessor(scale=True, high_card=enc)),
                              ("model", KNeighborsRegressor())]),
        {"model__n_neighbors": [5, 10, 20], "model__weights": ["uniform", "distance"]},
    ),
    "Random Forest": (
        lambda enc: Pipeline([("prep", make_preprocessor(scale=False, high_card=enc)),
                              ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                                                              n_jobs=-1))]),
        {"model__min_samples_leaf": [1, 3], "model__max_features": [0.33, 0.6]},
    ),
    "Gradient Boosting": (
        lambda enc: Pipeline([("prep", make_preprocessor(scale=False, high_card=enc)),
                              ("model", GradientBoostingRegressor(learning_rate=0.05, subsample=0.8,
                                                                  random_state=RANDOM_STATE))]),
        {"model__n_estimators": [300, 600], "model__max_depth": [3, 4]},
    ),
}
ENCODINGS = ("drop", "onehot", "frequency", "target")


def chosen_encoding():
    """Encoding of model_trim / market with the best mean CV R^2 in the encoding experiment
    (training data only); target encoding if the experiment has not been run."""
    path = RESULTS / "encoding_experiment.csv"
    if not path.exists():
        return "target"
    exp = pd.read_csv(path)
    return exp.groupby("encoding")["cv_R2"].mean().idxmax()


def model_path(name):
    return RESULTS / f"model_{name.lower().replace(' ', '_').replace('-', '_')}.joblib"


def load_split():
    df = pd.read_csv(MODEL_READY_CSV)
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["model"])
    return train, test


def metrics(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred)}


def train_one(name, verbose=True):
    """Grid-search one algorithm with 5-fold CV on the training set, then score it once on the test set."""
    train, test = load_split()
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    build, grid = MODELS[name]
    encoding = chosen_encoding()
    # every metric is cross-validated for every setting; the setting with the lowest CV MAE is refit
    search = GridSearchCV(build(encoding), grid, cv=CV, scoring=SCORING, refit="MAE", n_jobs=-1)
    search.fit(X_tr, y_tr)
    i = search.best_index_
    cvr = search.cv_results_
    y_hat = search.best_estimator_.predict(X_te)
    test_m = metrics(y_te, y_hat)
    row = {
        "model": name,
        "cv_MAE": -cvr["mean_test_MAE"][i], "cv_MAE_sd": cvr["std_test_MAE"][i],
        "cv_RMSE": -cvr["mean_test_RMSE"][i], "cv_RMSE_sd": cvr["std_test_RMSE"][i],
        "cv_R2": cvr["mean_test_R2"][i], "cv_R2_sd": cvr["std_test_R2"][i],
        "test_MAE": test_m["MAE"], "test_RMSE": test_m["RMSE"], "test_R2": test_m["R2"],
        # error converted back to dollars: predicted ratio x base MSRP vs listing price
        "test_MAE_dollars": mean_absolute_error(test["price"], y_hat * test["msrp_base"]),
        "best_params": json.dumps(search.best_params_),
        "high_card_encoding": encoding,
    }
    joblib.dump(search.best_estimator_, model_path(name))
    stem = model_path(name).stem
    pd.DataFrame([row]).to_csv(RESULTS / f"{stem}_metrics.csv", index=False)
    pd.DataFrame({"listing_id": test["listing_id"], name: y_hat}).to_csv(RESULTS / f"{stem}_test_pred.csv", index=False)
    if verbose:
        print(f"{name:<20} params={search.best_params_}  CV R2={row['cv_R2']:.3f}  "
              f"test R2={test_m['R2']:.3f}  test MAE={test_m['MAE']:.4f}", flush=True)
    return row


def collect(verbose=True):
    """Combine the per-model results into one comparison table (+ a mean-ratio baseline)."""
    train, test = load_split()
    table = pd.concat([pd.read_csv(RESULTS / f"{model_path(n).stem}_metrics.csv") for n in MODELS],
                      ignore_index=True)
    base = metrics(test[TARGET], np.full(len(test), train[TARGET].mean()))
    base_row = {"model": "Baseline (mean ratio)", "test_MAE": base["MAE"], "test_RMSE": base["RMSE"],
                "test_R2": base["R2"],
                "test_MAE_dollars": mean_absolute_error(test["price"], train[TARGET].mean() * test["msrp_base"])}
    table = pd.concat([table, pd.DataFrame([base_row])], ignore_index=True).round(4)
    table.to_csv(RESULTS / "model_comparison.csv", index=False)

    preds = test[["listing_id", "source", "make", "model", "year", "msrp_base", "price", TARGET]].copy()
    for n in MODELS:
        p = pd.read_csv(RESULTS / f"{model_path(n).stem}_test_pred.csv")
        preds = preds.merge(p, on="listing_id", how="left")
    preds.to_csv(RESULTS / "test_predictions.csv", index=False)
    pd.Series({"train_rows": len(train), "test_rows": len(test)}, name="rows").to_csv(RESULTS / "split_sizes.csv")
    with open(RESULTS / "best_params.json", "w") as f:
        json.dump({r["model"]: json.loads(r["best_params"]) for _, r in table.dropna(subset=["best_params"]).iterrows()},
                  f, indent=2)
    if verbose:
        print(table.drop(columns=["best_params", "high_card_encoding"]).to_string(index=False))
    return table


def train_all(verbose=True):
    """Encoding experiment first (it picks the encoding), then every model, then the comparison table."""
    encoding_experiment(verbose)
    for name in MODELS:
        train_one(name, verbose)
    return collect(verbose)


def encoding_experiment(verbose=True):
    """5-fold CV on the training set: how should model_trim / market be encoded?"""
    train, _ = load_split()
    X, y = train[FEATURES], train[TARGET]
    rows = []
    for enc in ENCODINGS:
        for name, est, scale in (("Ridge Regression", Ridge(alpha=1.0), True),
                                 ("Gradient Boosting", GradientBoostingRegressor(
                                     n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                     random_state=RANDOM_STATE), False)):
            pipe = Pipeline([("prep", make_preprocessor(scale=scale, high_card=enc)), ("model", est)])
            cv = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
            n_cols = pipe.fit(X, y).named_steps["prep"].transform(X.head(5)).shape[1]
            rows.append({"encoding": enc, "model": name, "n_input_columns": n_cols,
                         "cv_MAE": -cv["test_MAE"].mean(), "cv_RMSE": -cv["test_RMSE"].mean(),
                         "cv_R2": cv["test_R2"].mean()})
    table = pd.DataFrame(rows).round(4)
    table.to_csv(RESULTS / "encoding_experiment.csv", index=False)
    if verbose:
        print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    # python train.py                      -> everything
    # python train.py "Random Forest"      -> one model;  python train.py collect / encoding
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is None:
        train_all()
    elif arg == "collect":
        collect()
    elif arg == "encoding":
        encoding_experiment()
    else:
        train_one(arg)
