"""Step 5 - depreciation analysis and feature ranking.

1. Permutation importance (held-out test set) for every trained model, measured
   on the original columns, so a one-hot encoded column counts as one feature.
2. Impurity importance of the tree ensembles, summed back to original columns.
3. Standardized Ridge coefficients (sign = direction of the effect).
4. A log-linear depreciation model, log(retention) = a + b*age + c*(miles/10k) + controls,
   whose coefficients read directly as "% of value lost per year / per 10k miles",
   fitted for all cars and separately for each make and model.
5. Partial dependence of the best model on age and mileage (the depreciation
   curves the model has learned, other features held at their observed values).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LinearRegression

from config import INTEGRATED_CSV, RANDOM_STATE, RESULTS
from features import HIGH_CARD_FEATURES, LOW_CARD_FEATURES, NUMERIC_FEATURES, TARGET
from train import FEATURES, MODELS, load_split, model_path

ORIGINAL = NUMERIC_FEATURES + LOW_CARD_FEATURES + HIGH_CARD_FEATURES


def original_column(transformed_name):
    """'cat__make_Toyota' -> 'make', 'num__age_years' -> 'age_years', 'high__model_trim' -> 'model_trim'."""
    name = transformed_name.split("__", 1)[1]
    for col in sorted(ORIGINAL, key=len, reverse=True):
        if name == col or name.startswith(col + "_"):
            return col
    return name


def permutation_rankings(test):
    rows = []
    for name in MODELS:
        pipe = joblib.load(model_path(name))
        r = permutation_importance(pipe, test[FEATURES], test[TARGET], scoring="r2",
                                   n_repeats=15, random_state=RANDOM_STATE, n_jobs=-1)
        for col, m, s in zip(FEATURES, r.importances_mean, r.importances_std):
            rows.append({"model": name, "feature": col, "importance_mean": m, "importance_sd": s})
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby("model")["importance_mean"].rank(ascending=False, method="first").astype(int)
    out.sort_values(["model", "rank"]).round(5).to_csv(RESULTS / "permutation_importance.csv", index=False)
    return out


def impurity_rankings():
    rows = []
    for name in ("Random Forest", "Gradient Boosting"):
        pipe = joblib.load(model_path(name))
        names = pipe.named_steps["prep"].get_feature_names_out()
        imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
        agg = imp.groupby(imp.index.map(original_column)).sum().sort_values(ascending=False)
        rows += [{"model": name, "feature": f, "importance": v} for f, v in agg.items()]
    out = pd.DataFrame(rows).round(5)
    out.to_csv(RESULTS / "impurity_importance.csv", index=False)
    return out


def ridge_coefficients():
    pipe = joblib.load(model_path("Ridge Regression"))
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pd.DataFrame({"term": [n.split("__", 1)[1] for n in names],
                         "feature": [original_column(n) for n in names],
                         "coefficient": pipe.named_steps["model"].coef_})
    coef["abs"] = coef["coefficient"].abs()
    coef = coef.sort_values("abs", ascending=False).drop(columns="abs").round(5)
    coef.to_csv(RESULTS / "ridge_coefficients.csv", index=False)
    return coef


def loglinear_depreciation(df):
    """log(ratio) ~ age + miles/10k (+ controls). 1 - exp(b) = share of value lost per unit."""
    df = df.copy()
    df["log_ratio"] = np.log(df[TARGET])
    df["miles_10k"] = df["mileage"] / 10000
    controls = ["source", "condition", "title_status", "seller_type", "model"]

    def fit(sub):
        X = sub[["age_years", "miles_10k"]]
        usable = [c for c in controls if sub[c].nunique() > 1]
        if usable:
            X = pd.concat([X, pd.get_dummies(sub[usable], drop_first=True, dtype=float)], axis=1)
        lr = LinearRegression().fit(X, sub["log_ratio"])
        b_age, b_miles = lr.coef_[0], lr.coef_[1]
        # age alone (mileage not held fixed): the total yearly loss, including typical yearly driving
        X_age = X.drop(columns="miles_10k")
        b_total = LinearRegression().fit(X_age, sub["log_ratio"]).coef_[0]
        return {"n": len(sub), "pct_value_lost_per_year": 100 * (1 - np.exp(b_age)),
                "pct_value_lost_per_10k_miles": 100 * (1 - np.exp(b_miles)),
                "pct_value_lost_per_year_age_only": 100 * (1 - np.exp(b_total)),
                "R2_log_model": lr.score(X, sub["log_ratio"])}

    rows = [{"group": "All cars", "level": "all", **fit(df)}]
    for make, sub in df.groupby("make"):
        rows.append({"group": make, "level": "make", **fit(sub)})
    for model, sub in df.groupby("model"):
        rows.append({"group": f"{sub['make'].iloc[0]} {model}", "level": "model", **fit(sub)})
    out = pd.DataFrame(rows).round(3)
    out.to_csv(RESULTS / "depreciation_rates.csv", index=False)
    return out


def depreciation_curves(train, best_name):
    pipe = joblib.load(model_path(best_name))
    rows = []
    for feat, grid in (("age_years", np.arange(0, 20, dtype=float)),
                       ("mileage", np.arange(0, 220001, 10000, dtype=float))):
        res = partial_dependence(pipe, train[FEATURES], [feat], kind="average", custom_values={feat: grid})
        rows += [{"feature": feat, "value": v, "predicted_ratio": p}
                 for v, p in zip(res["grid_values"][0], res["average"][0])]
    out = pd.DataFrame(rows).round(4)
    out.to_csv(RESULTS / "partial_dependence.csv", index=False)

    # the same age curve computed on each make's own cars
    rows = []
    for make, sub in train.groupby("make"):
        res = partial_dependence(pipe, sub[FEATURES], ["age_years"], kind="average",
                                 custom_values={"age_years": np.arange(0, 16, dtype=float)})
        rows += [{"make": make, "age_years": v, "predicted_ratio": p}
                 for v, p in zip(res["grid_values"][0], res["average"][0])]
    pd.DataFrame(rows).round(4).to_csv(RESULTS / "partial_dependence_age_by_make.csv", index=False)
    return out


def main():
    train, test = load_split()
    comp = pd.read_csv(RESULTS / "model_comparison.csv")
    best_name = comp[comp["model"].isin(MODELS)].sort_values("test_R2", ascending=False)["model"].iloc[0]
    pd.Series({"best_model": best_name}, name="value").to_csv(RESULTS / "best_model.csv")
    print("best model on test R2:", best_name)

    perm = permutation_rankings(test)
    top = perm[perm["model"] == best_name].sort_values("rank").head(12)
    print(top[["feature", "importance_mean", "importance_sd"]].to_string(index=False))
    impurity_rankings()
    ridge_coefficients()
    full = pd.concat([train, test])
    print(loglinear_depreciation(full).to_string(index=False))
    depreciation_curves(train, best_name)

    # empirical medians used in the report
    full["age_band"] = pd.cut(full["age_years"], [-1, 2, 5, 8, 11, 14, 30],
                              labels=["0-2", "3-5", "6-8", "9-11", "12-14", "15+"])
    full.pivot_table(index="make", columns="age_band", values=TARGET, aggfunc="median",
                     observed=False).round(3).to_csv(RESULTS / "median_ratio_by_make_age.csv")
    integ = pd.read_csv(INTEGRATED_CSV)
    integ.groupby("title_status")[TARGET].agg(["count", "median"]).round(3).to_csv(
        RESULTS / "median_ratio_by_title_status.csv")


if __name__ == "__main__":
    main()
