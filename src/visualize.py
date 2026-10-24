"""Step 6 - figures for the report (saved to figures/ as PNG)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIGURES, MAKES_MODELS, MODEL_READY_CSV, RESULTS  # noqa: E402
from features import TARGET  # noqa: E402

# palette: fixed categorical order (blue, orange, aqua, yellow), muted text and grid
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
INK, INK2, GRID, POINT = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
SOURCE_COLORS = {"craigslist": BLUE, "carmax": ORANGE}
SOURCE_LABELS = {"craigslist": "Craigslist", "carmax": "CarMax"}
MAKE_ORDER = list(MAKES_MODELS)  # Toyota, Honda, Ford
MODEL_ORDER = ["Ridge Regression", "K-Nearest Neighbors", "Random Forest", "Gradient Boosting"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "text.color": INK,
})

PRETTY = {
    "age_years": "Vehicle age (years)", "mileage": "Mileage", "miles_per_year": "Miles per year",
    "msrp_base": "Base MSRP", "msrp_spread": "MSRP spread (top/base trim)", "cylinders": "Cylinders",
    "distance_miles": "Distance from Philadelphia", "days_listed": "Days listed",
    "desc_word_count": "Description length", "premium_feature_count": "Premium features mentioned",
    "desc_one_owner": "Text: one owner", "desc_no_accidents": "Text: no accidents",
    "desc_needs_work": "Text: needs work", "desc_recent_service": "Text: recent service",
    "desc_warranty": "Text: warranty", "desc_prior_rental_fleet": "Text: rental/fleet/lease",
    "source": "Source (Craigslist/CarMax)", "make": "Make", "model": "Model", "body_type": "Body type",
    "drivetrain": "Drivetrain", "transmission": "Transmission", "fuel_type": "Fuel type",
    "exterior_color": "Exterior color", "condition": "Condition (seller-stated)",
    "title_status": "Title status", "seller_type": "Seller type",
    "model_trim": "Model + trim (target-encoded)", "market": "Market (target-encoded)",
    "retention_ratio": "Retention ratio", "price": "Price",
}


def best_model_name():
    return pd.read_csv(RESULTS / "best_model.csv", index_col=0).iloc[0, 0]


def save(fig, name):
    fig.savefig(FIGURES / name, facecolor="white")
    plt.close(fig)
    print("saved", name)


def fig_data_overview(df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    specs = [("price", "Listing price (USD)", np.arange(0, 70001, 2500), axes[0, 0], "Price distribution by source"),
             ("mileage", "Mileage", np.arange(0, 300001, 10000), axes[0, 1], "Mileage distribution by source"),
             ("age_years", "Vehicle age (years)", np.arange(-0.5, 20.5, 1), axes[1, 0], "Vehicle age by source"),
             (TARGET, "Retention ratio (price / base MSRP)", np.arange(0, 2.01, 0.05), axes[1, 1],
              "Value retention ratio (target)")]
    for col, label, bins, ax, title in specs:
        for src in ("craigslist", "carmax"):
            vals = df.loc[df["source"] == src, col].clip(upper=bins[-1])
            med = f"{vals.median():,.2f}" if col == TARGET else f"{vals.median():,.0f}"
            ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=SOURCE_COLORS[src],
                    label=f"{SOURCE_LABELS[src]} (median {med})")
        ax.set_title(title)
        ax.set_xlabel(label)
        ax.set_ylabel("Listings")
        ax.legend(loc="best", fontsize=9)
        if col == "age_years":
            ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        if col in ("price", "mileage"):
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    fig.tight_layout()
    save(fig, "fig01_data_overview.png")


def fig_make_model_counts(df):
    counts = df.groupby(["make", "model", "source"]).size().unstack(fill_value=0)
    counts = counts.reindex(sorted(counts.index, key=lambda i: (MAKE_ORDER.index(i[0]), i[1])))
    labels = [f"{mk} {md}" for mk, md in counts.index]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    left = np.zeros(len(labels))
    for src in ("craigslist", "carmax"):
        vals = counts[src].to_numpy() if src in counts else np.zeros(len(labels))
        ax.barh(y, vals, left=left, height=0.7, color=SOURCE_COLORS[src], edgecolor="white", linewidth=2,
                label=SOURCE_LABELS[src])
        left += vals
    for yi, total in zip(y, left):
        ax.text(total + 3, yi, f"{int(total)}", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, left.max() * 1.1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Listings after cleaning")
    ax.set_title("Listings per make and model, by source")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig02_make_model_counts.png")


def fig_retention_vs_age_mileage(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    ax = axes[0]
    ax.scatter(df["age_years"] + np.random.default_rng(0).uniform(-0.25, 0.25, len(df)), df[TARGET],
               s=8, color=POINT, alpha=0.5, linewidths=0, label="Individual listings")
    for color, make in zip(SERIES, MAKE_ORDER):
        sub = df[df["make"] == make]
        med = sub.groupby("age_years")[TARGET].median()
        med = med[sub.groupby("age_years").size() >= 5]
        ax.plot(med.index, med.values, color=color, linewidth=2, marker="o", markersize=4, label=f"{make} (median)")
    ax.set_xlabel("Vehicle age (years)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.set_ylabel("Retention ratio (price / base MSRP)")
    ax.set_title("Retention falls with age")
    ax.legend(fontsize=8.5, loc="upper right")

    ax = axes[1]
    ax.scatter(df["mileage"] / 1000, df[TARGET], s=8, color=POINT, alpha=0.5, linewidths=0, label="Individual listings")
    bins = np.arange(0, 260001, 20000)
    mid = (bins[:-1] + bins[1:]) / 2 / 1000
    for src in ("craigslist", "carmax"):
        sub = df[df["source"] == src]
        grp = sub.groupby(pd.cut(sub["mileage"], bins), observed=False)[TARGET]
        med, n = grp.median().to_numpy(), grp.size().to_numpy()
        ok = n >= 8
        ax.plot(mid[ok], med[ok], color=SOURCE_COLORS[src], linewidth=2, marker="o", markersize=4,
                label=f"{SOURCE_LABELS[src]} (median per 20k mi)")
    ax.set_xlabel("Mileage (thousands)")
    ax.set_title("... and with mileage")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_ylim(0, 2.0)
    fig.tight_layout()
    save(fig, "fig03_retention_vs_age_mileage.png")


def fig_correlations(df):
    cols = ["retention_ratio", "price", "age_years", "mileage", "miles_per_year", "msrp_base", "msrp_spread",
            "premium_feature_count", "desc_word_count", "days_listed", "distance_miles"]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
    names = [PRETTY.get(c, c) for c in cols]
    ax.set_xticks(range(len(cols)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), names)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else INK)
    ax.grid(False)
    ax.set_title("Spearman correlation of numeric attributes")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "fig04_correlation_heatmap.png")


def fig_model_comparison():
    comp = pd.read_csv(RESULTS / "model_comparison.csv")
    models = comp[comp["model"] != "Baseline (mean ratio)"]
    base = comp[comp["model"] == "Baseline (mean ratio)"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(models))
    for ax, metric, better in zip(axes, ["MAE", "RMSE", "R2"], ["lower", "lower", "higher"]):
        ax.bar(x - 0.19, models[f"cv_{metric}"], width=0.36, color=BLUE, edgecolor="white", linewidth=2,
               yerr=models[f"cv_{metric}_sd"], error_kw={"ecolor": INK2, "capsize": 3, "linewidth": 1},
               label="5-fold CV (train)")
        ax.bar(x + 0.19, models[f"test_{metric}"], width=0.36, color=ORANGE, edgecolor="white", linewidth=2,
               label="Held-out test")
        for xi, v in zip(x, models[f"test_{metric}"]):
            ax.text(xi + 0.19, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.axhline(base[f"test_{metric}"], color=INK2, linestyle="--", linewidth=1)
        ax.text(-0.55, base[f"test_{metric}"], " mean-ratio baseline (test)", ha="left",
                va="top" if metric == "R2" else "bottom", fontsize=8, color=INK2)
        if metric == "R2":
            ax.set_ylim(min(-0.1, base["test_R2"] - 0.1), 1.05)
        ax.set_xticks(x, [m.replace(" ", "\n", 1) for m in models["model"]], fontsize=8.5)
        ax.set_title(f"{metric.replace('R2', 'R²')} ({better} is better)")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "fig05_model_comparison.png")


def fig_pred_vs_actual():
    best = best_model_name()
    pred = pd.read_csv(RESULTS / "test_predictions.csv")
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for src in ("craigslist", "carmax"):
        sub = pred[pred["source"] == src]
        ax.scatter(sub[TARGET], sub[best], s=14, color=SOURCE_COLORS[src], alpha=0.6, linewidths=0,
                   label=SOURCE_LABELS[src])
    lim = [0, max(pred[TARGET].max(), pred[best].max()) * 1.05]
    ax.plot(lim, lim, color=INK2, linewidth=1, linestyle="--", label="Perfect prediction")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual retention ratio")
    ax.set_ylabel("Predicted retention ratio")
    ax.set_title(f"{best}: predicted vs actual (test set)")
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "fig06_predicted_vs_actual.png")


def fig_feature_importance():
    best = best_model_name()
    perm = pd.read_csv(RESULTS / "permutation_importance.csv")
    sub = perm[perm["model"] == best].sort_values("importance_mean", ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh([PRETTY.get(f, f) for f in sub["feature"]], sub["importance_mean"], xerr=sub["importance_sd"],
            color=BLUE, edgecolor="white", linewidth=2, height=0.7,
            error_kw={"ecolor": INK2, "capsize": 2, "linewidth": 1})
    pad = sub["importance_mean"].max() * 0.015
    for yi, (v, sd) in enumerate(zip(sub["importance_mean"], sub["importance_sd"])):
        ax.text(v + sd + pad, yi, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("Drop in test R² when the feature is shuffled")
    ax.set_title(f"Feature ranking: permutation importance ({best})")
    ax.grid(axis="y", visible=False)
    save(fig, "fig07_feature_importance.png")


def fig_importance_agreement():
    """Importance of the best model's top-10 features under every model."""
    perm = pd.read_csv(RESULTS / "permutation_importance.csv")
    best = best_model_name()
    top = perm[perm["model"] == best].nsmallest(10, "rank")["feature"].tolist()
    models = [m for m in MODEL_ORDER if m in perm["model"].unique()]
    colors = dict(zip(MODEL_ORDER, SERIES))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(models)
    x = np.arange(len(top))
    for i, m in enumerate(models):
        vals = perm[perm["model"] == m].set_index("feature").loc[top, "importance_mean"]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width=width, color=colors[m],
               edgecolor="white", linewidth=1.5, label=m)
    short = [PRETTY.get(f, f).split(" (")[0] for f in top]
    ax.set_xticks(x, short, rotation=35, ha="right", rotation_mode="anchor", fontsize=8.5)
    ax.set_ylabel("Permutation importance (drop in R²)")
    ax.set_title(f"Top-10 features of the {best} model, importance in every model")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, ncol=4, loc="upper right")
    save(fig, "fig08_importance_all_models.png")


def fig_ridge_coefficients():
    coef = pd.read_csv(RESULTS / "ridge_coefficients.csv").head(16)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = [BLUE if c > 0 else ORANGE for c in coef["coefficient"]]
    ax.barh(coef["term"].str.replace("_", " "), coef["coefficient"], color=colors, edgecolor="white",
            linewidth=2, height=0.7)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Standardized coefficient (change in retention ratio)")
    ax.set_title("Ridge regression: largest coefficients")
    ax.text(0.99, 0.02, "blue = raises value   orange = lowers value", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=INK2)
    ax.grid(axis="y", visible=False)
    save(fig, "fig09_ridge_coefficients.png")


def fig_depreciation_curves():
    pdp = pd.read_csv(RESULTS / "partial_dependence.csv")
    by_make = pd.read_csv(RESULTS / "partial_dependence_age_by_make.csv")
    best = best_model_name()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    age = pdp[pdp["feature"] == "age_years"]
    axes[0].plot(age["value"], age["predicted_ratio"], color=BLUE, linewidth=2, marker="o", markersize=4)
    axes[0].set_xlabel("Vehicle age (years)")
    axes[0].set_ylabel("Average predicted retention ratio")
    axes[0].set_title("Learned effect of age")
    miles = pdp[pdp["feature"] == "mileage"]
    axes[1].plot(miles["value"] / 1000, miles["predicted_ratio"], color=BLUE, linewidth=2, marker="o", markersize=4)
    axes[1].set_xlabel("Mileage (thousands)")
    axes[1].set_title("Learned effect of mileage")
    for color, make in zip(SERIES, MAKE_ORDER):
        sub = by_make[by_make["make"] == make]
        axes[2].plot(sub["age_years"], sub["predicted_ratio"], color=color, linewidth=2, label=make)
    axes[2].set_xlabel("Vehicle age (years)")
    axes[2].set_title("Learned effect of age, by make")
    axes[2].legend(fontsize=8.5)
    axes[2].set_xlim(0, 15)
    fig.suptitle(f"Partial dependence of the {best} model (other features at their observed values)",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig10_depreciation_curves.png")


def fig_depreciation_rates():
    rates = pd.read_csv(RESULTS / "depreciation_rates.csv")
    sub = rates[rates["level"] == "model"].sort_values("pct_value_lost_per_year")
    overall = rates[rates["level"] == "all"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    make_color = dict(zip(MAKE_ORDER, SERIES))
    colors = [make_color[g.split()[0]] for g in sub["group"]]
    for ax, col, label in ((axes[0], "pct_value_lost_per_year", "% of value lost per extra year of age"),
                           (axes[1], "pct_value_lost_per_10k_miles", "% of value lost per extra 10,000 miles")):
        ax.barh(sub["group"], sub[col], color=colors, edgecolor="white", linewidth=2, height=0.7)
        ax.axvline(overall[col], color=INK2, linestyle="--", linewidth=1)
        ax.text(overall[col], len(sub) - 0.4, f" all cars {overall[col]:.1f}%", fontsize=8.5, color=INK2)
        for yi, v in enumerate(sub[col]):
            ax.text(max(v, 0) + 0.1, yi, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=make_color[m]) for m in MAKE_ORDER]
    fig.legend(handles, MAKE_ORDER, fontsize=9, loc="lower center", ncol=len(MAKE_ORDER), bbox_to_anchor=(0.5, -0.06))
    axes[0].set_title("Depreciation rates from the log-linear model, by model")
    fig.tight_layout()
    save(fig, "fig11_depreciation_rates.png")


def main():
    df = pd.read_csv(MODEL_READY_CSV)
    fig_data_overview(df)
    fig_make_model_counts(df)
    fig_retention_vs_age_mileage(df)
    fig_correlations(df)
    fig_model_comparison()
    fig_pred_vs_actual()
    fig_feature_importance()
    fig_importance_agreement()
    fig_ridge_coefficients()
    fig_depreciation_curves()
    fig_depreciation_rates()


if __name__ == "__main__":
    main()
