"""Step 3 - feature engineering.

Adds numeric features derived from the listing (age, miles per year, MSRP
spread, days listed), keyword features extracted from the free-text
description, and defines which columns are treated as numeric, low-cardinality
categorical (one-hot) and high-cardinality categorical (target encoding).
Output: data/processed/model_ready.csv
"""
import re

import pandas as pd

from config import INTEGRATED_CSV, MODEL_READY_CSV, RESULTS, SCRAPE_DATE, SCRAPE_YEAR

# Features extracted from listing descriptions (regular expressions, case-insensitive).
# Craigslist descriptions are the seller's own text; CarMax descriptions are the
# listing's highlight badges, prior-use notes and feature list (see integrate.py).
TEXT_FEATURES = {
    "desc_one_owner": r"\b(?:one|1|single)[\s-]*owner\b",
    "desc_no_accidents": r"no accidents?|accident[\s-]*free|clean (?:carfax|autocheck|history|vehicle history)"
                         r"|never (?:been )?(?:in an )?accident",
    "desc_needs_work": r"needs? (?:some |minor |a )?(?:work|repairs?|engine|transmission|tlc|brakes)"
                       r"|mechanic.?s special|\bas[\s-]is\b|check engine|not running|does ?n.?t run"
                       r"|won.?t start|project car|\bblown\b|bad (?:engine|transmission)|knocking",
    "desc_recent_service": r"new (?:tires|brakes|battery|rotors|timing|transmission|engine|pads)"
                           r"|recently (?:serviced|replaced|inspected)|just (?:serviced|inspected)"
                           r"|service records|maintenance records|well[\s-]maintained|fresh (?:oil|inspection)",
    "desc_warranty": r"warranty",
    "desc_prior_rental_fleet": r"\brental\b|\bfleet\b|\bloaner\b|previously leased|lease return",
}
PREMIUM_PATTERNS = [
    r"leather", r"sun\s?roof|moon\s?roof|panoramic", r"navigation|\bnav\b",
    r"heated (?:front )?seats?|seat heaters", r"back\s?up camera|rear[\s-]?view camera",
    r"apple carplay|android auto", r"blind spot", r"remote start", r"premium (?:sound|audio)|\bbose\b|\bjbl\b",
]

NUMERIC_FEATURES = [
    "age_years", "mileage", "miles_per_year", "msrp_base", "msrp_spread", "cylinders",
    "distance_miles", "days_listed", "desc_word_count", "premium_feature_count",
    *TEXT_FEATURES.keys(),
]
LOW_CARD_FEATURES = [
    "source", "make", "model", "body_type", "drivetrain", "transmission", "fuel_type",
    "exterior_color", "condition", "title_status", "seller_type",
]
HIGH_CARD_FEATURES = ["model_trim", "market"]
TARGET = "retention_ratio"


def add_features(df):
    df = df.copy()
    df["age_years"] = (SCRAPE_YEAR - df["year"]).clip(lower=0)
    df["miles_per_year"] = df["mileage"] / df["age_years"].clip(lower=0.5)
    df["msrp_spread"] = df["msrp_top"] / df["msrp_base"]
    listed = pd.to_datetime(df["listed_date"], errors="coerce")
    df["days_listed"] = (pd.Timestamp(SCRAPE_DATE) - listed).dt.days.clip(lower=0)
    df["days_listed"] = df["days_listed"].fillna(df["days_listed"].median())

    text = df["description"].fillna("") + " " + df["title"].fillna("")
    for name, pat in TEXT_FEATURES.items():
        df[name] = text.str.contains(pat, flags=re.I, regex=True).astype(int)
    df["premium_feature_count"] = sum(text.str.contains(p, flags=re.I, regex=True).astype(int)
                                      for p in PREMIUM_PATTERNS)
    df["desc_word_count"] = df["description"].fillna("").str.split().str.len()

    df["model_trim"] = df["model"] + " " + df["trim"]
    df["market"] = df["market"].replace("", "unknown").fillna("unknown")
    df["distance_miles"] = df["distance_miles"].fillna(df["distance_miles"].median())
    return df


def main():
    df = add_features(pd.read_csv(INTEGRATED_CSV))
    keep = (["listing_id", "url", "year", "trim", "price"] + NUMERIC_FEATURES + LOW_CARD_FEATURES
            + HIGH_CARD_FEATURES + [TARGET])
    df[keep].to_csv(MODEL_READY_CSV, index=False)

    # how often each description feature fires, per source (for the report)
    rates = df.groupby("source")[list(TEXT_FEATURES) + ["premium_feature_count", "desc_word_count"]].mean().T.round(3)
    rates.to_csv(RESULTS / "text_feature_rates.csv")
    effect = pd.DataFrame({f: df.groupby(f)[TARGET].median() for f in TEXT_FEATURES}).T
    effect = effect.rename(columns={0: "median_ratio_without", 1: "median_ratio_with"}).round(3)
    effect.to_csv(RESULTS / "text_feature_median_ratio.csv")
    card = {c: df[c].nunique() for c in LOW_CARD_FEATURES + HIGH_CARD_FEATURES}
    pd.Series(card, name="n_unique").to_csv(RESULTS / "categorical_cardinality.csv")
    print(f"model-ready table: {df.shape[0]} rows, {len(NUMERIC_FEATURES)} numeric + "
          f"{len(LOW_CARD_FEATURES)} low-card + {len(HIGH_CARD_FEATURES)} high-card features")
    print(rates.to_string())
    print(effect.to_string())
    print("cardinality:", card)


if __name__ == "__main__":
    main()
