"""Step 2 - preprocessing and integration.

Reads the first-parsed CSV of each source, maps both onto one schema
(same column names, formats and units), removes listings that are out of
scope or duplicated, merges the U.S. News MSRP table and computes the target

    retention_ratio = listing price / original (base) MSRP

Every filtering step is logged so the report can show how many listings each
rule removed. Output: data/processed/integrated_listings.csv
"""
import re

import numpy as np
import pandas as pd

from config import (CARMAX_CSV, CRAIGSLIST_CSV, INTEGRATED_CSV, MAKES_MODELS,
                    MSRP_CSV, PHILLY_LAT, PHILLY_LON, RESULTS, SCRAPE_YEAR)

UNIFIED_COLUMNS = [
    "source", "listing_id", "url", "vin", "make", "model", "year", "trim",
    "price", "mileage", "body_type", "drivetrain", "transmission", "fuel_type",
    "cylinders", "exterior_color", "condition", "title_status", "seller_type",
    "market", "latitude", "longitude", "distance_miles", "listed_date",
    "title", "description",
]

# ---------------------------------------------------------------- vocab ---
# Pattern that must appear for a listing to count as that model, and patterns
# that mean it is actually a different model (e.g. Corolla Cross, Rogue Sport).
MODEL_PATTERNS = {
    "Camry": (r"\bcamry\b", r"solara"),
    "Corolla": (r"\bcorolla\b", r"\bcross\b|\bim\b"),
    "RAV4": (r"\brav\s?-?4\b", None),
    "Civic": (r"\bcivic\b", None),
    "Accord": (r"\baccord\b", r"crosstour"),
    "CR-V": (r"\bcr\s?-?v\b", None),
    "F-150": (r"\bf\s?-?150\b", None),
    "Escape": (r"\bescape\b", None),
    "Explorer": (r"\bexplorer\b", r"sport\s?trac"),
    "Altima": (r"\baltima\b", None),
    "Rogue": (r"\brogue\b", r"rogue\s+sport|\bselect\b"),
    "Sentra": (r"\bsentra\b", None),
}
MODEL_TO_MAKE = {m: mk for mk, ms in MAKES_MODELS.items() for m in ms}

# Trim names used by these 12 models (checked longest/most specific first).
TRIM_VOCAB = [
    "King Ranch", "Sport Touring", "Type R", "EX-L", "XSE", "XLE", "TRD", "Nightshade",
    "Adventure", "Woodland", "Prime", "Limited", "Platinum", "Touring", "Raptor", "Tremor",
    "Lariat", "XLT", "STX", "XL", "Timberline", "King Cab", "Titanium", "SEL", "ST-Line",
    "ST", "Active", "Midnight", "SR", "SL", "SV", "Si", "LX", "EX", "Sport", "SE", "LE",
    "S", "L",
]
TRIM_ALIASES = {"exl": "EX-L", "ex l": "EX-L", "ex-l": "EX-L", "st line": "ST-Line"}

COLOR_GROUPS = {
    "white": r"white|pearl|ivory",
    "black": r"black|ebony|midnight|onyx",
    "silver": r"silver",
    "gray": r"gr[ae]y|charcoal|graphite|gunmetal|magnetic|steel",
    "blue": r"blue|navy",
    "red": r"red|burgundy|maroon",
}

HIGHLIGHT_TEXT = {  # CarMax "highlights" badges -> the phrase shown on the listing
    "singleOwner": "One owner.",
    "lowMiles": "Low miles.",
    "lowMilesPerYear": "Low miles per year.",
    "warranty": "Remaining manufacturer warranty.",
    "fuelEfficient": "Fuel efficient.",
    "advancedFeatures": "Advanced features.",
}

# VIN position 10 -> model year (30-year cycle)
_VIN_CODES = "ABCDEFGHJKLMNPRSTVWXY123456789"
VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")


# -------------------------------------------------------------- helpers ---
def to_number(s):
    """'$13,425' -> 13425.0, '116,700' -> 116700.0, junk -> NaN"""
    if pd.isna(s):
        return np.nan
    digits = re.sub(r"[^\d.]", "", str(s))
    try:
        return float(digits) if digits else np.nan
    except ValueError:
        return np.nan


def vin_model_year(vin, reference_year=None):
    """Decode the model year from a 17-character VIN (None if not decodable)."""
    if not isinstance(vin, str):
        return None
    vin = vin.strip().upper()
    if not VIN_RE.match(vin) or vin[9] not in _VIN_CODES:
        return None
    base = 1980 + _VIN_CODES.index(vin[9])
    candidates = [base, base + 30]
    if reference_year and not pd.isna(reference_year):
        return min(candidates, key=lambda y: abs(y - reference_year))
    # North-American VINs: a letter in position 7 means the 2010+ cycle
    return base + 30 if vin[6].isalpha() else base


def haversine_miles(lat, lon, lat0=PHILLY_LAT, lon0=PHILLY_LON):
    lat, lon = np.radians(lat.astype(float)), np.radians(lon.astype(float))
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def find_trim(text):
    if not isinstance(text, str) or not text.strip():
        return "Base/Unknown"
    t = " " + re.sub(r"[_/,()]", " ", text) + " "
    for alias, trim in TRIM_ALIASES.items():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", t, re.I):
            return trim
    for trim in TRIM_VOCAB:
        flags = 0 if len(trim) <= 2 else re.I  # 1-2 letter trims must be upper case ("LE", "S")
        if re.search(rf"(?<![A-Za-z0-9-]){re.escape(trim)}(?![A-Za-z0-9-])", t, flags):
            return trim
    return "Base/Unknown"


def match_model(text, model):
    pat, exclude = MODEL_PATTERNS[model]
    text = text if isinstance(text, str) else ""
    return bool(re.search(pat, text, re.I)) and not (exclude and re.search(exclude, text, re.I))


def norm_body(s):
    s = str(s).lower() if not pd.isna(s) else ""
    if not s:
        return np.nan
    for key, pat in [("truck", r"pickup|truck|crew|cab"), ("suv", r"sport utility|suv|offroad"),
                     ("sedan", r"sedan"), ("hatchback", r"hatch"), ("coupe", r"coupe|convertible"),
                     ("wagon", r"wagon"), ("van", r"van")]:
        if re.search(pat, s):
            return key
    return "other"


def norm_drive(s):
    s = str(s).lower() if not pd.isna(s) else ""
    if not s:
        return np.nan
    if re.search(r"front|fwd", s):
        return "fwd"
    if re.search(r"rear|rwd", s):
        return "rwd"
    if re.search(r"all|awd|4wd|4x4|four", s):
        return "awd/4wd"
    return np.nan


def norm_transmission(s):
    s = str(s).lower() if not pd.isna(s) else ""
    if not s:
        return np.nan
    return "manual" if "manual" in s else "automatic"  # 'other' on Craigslist = CVT/automated


def norm_fuel(s):
    s = str(s).lower() if not pd.isna(s) else ""
    if not s:
        return np.nan
    if "plug" in s:
        return "plug-in hybrid"
    for key in ("hybrid", "electric", "diesel"):
        if key in s:
            return key
    return "gas"


def norm_color(s):
    s = str(s).lower() if not pd.isna(s) else ""
    if not s:
        return "unknown"
    for group, pat in COLOR_GROUPS.items():
        if re.search(pat, s):
            return group
    return "other"


def title_year(title):
    m = re.search(r"\b(19[89]\d|20[0-2]\d)\b", str(title))
    return float(m.group(1)) if m else np.nan


# ------------------------------------------------------- source mapping ---
def map_craigslist(raw):
    df = pd.DataFrame(index=raw.index)
    df["source"] = "craigslist"
    df["listing_id"] = "cl-" + raw["post_id"].astype(str)
    df["url"] = raw["url"]
    df["vin"] = raw["attr_vin"].str.upper().str.strip()
    df["search_model"] = raw["search_model"]
    df["make_model_text"] = raw["make_model"].fillna("") + " | " + raw["title"].fillna("")
    df["listed_year"] = pd.to_numeric(raw["year"], errors="coerce")
    df["title_year"] = raw["title"].map(title_year)
    df["trim_text"] = raw["title"].fillna("") + " " + raw["make_model"].fillna("")
    df["price"] = raw["price"].map(to_number)
    df["mileage"] = raw["attr_odometer"].map(to_number)
    df["body_type"] = raw["attr_type"].map(norm_body)
    df["drivetrain"] = raw["attr_drive"].map(norm_drive)
    df["transmission"] = raw["attr_transmission"].map(norm_transmission)
    df["fuel_type"] = raw["attr_fuel"].map(norm_fuel)
    df["cylinders"] = raw["attr_cylinders"].map(to_number)
    df["exterior_color"] = raw["attr_paint_color"].map(norm_color)
    df["condition"] = raw["attr_condition"].fillna("not stated").str.lower()
    df["title_status"] = raw["attr_title_status"].fillna("not stated").str.lower()
    df["seller_type"] = raw["seller_type"].fillna("owner")
    df["market"] = raw["area"].fillna("").str.replace(">", "").str.strip().str.lower()
    df["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    df["distance_miles"] = haversine_miles(df["latitude"], df["longitude"])
    df["listed_date"] = pd.to_datetime(raw["posted_at"], errors="coerce", utc=True).dt.date
    df["title"] = raw["title"]
    df["description"] = raw["description"].fillna("")
    return df


def carmax_description(row):
    parts = []
    for code in str(row.get("highlights") or "").split("; "):
        if code and code != "nan":
            parts.append(HIGHLIGHT_TEXT.get(code, re.sub(r"(?<!^)([A-Z])", r" \1", code).capitalize() + "."))
    if isinstance(row.get("priorUseText"), str):
        parts.append(row["priorUseText"])
    if isinstance(row.get("features"), str) and row["features"]:
        parts.append("Features: " + row["features"].replace(";", ",") + ".")
    if isinstance(row.get("packages"), str) and row["packages"]:
        parts.append("Packages: " + row["packages"].replace(";", ",") + ".")
    return " ".join(parts)


def map_carmax(raw):
    df = pd.DataFrame(index=raw.index)
    df["source"] = "carmax"
    df["listing_id"] = "cm-" + raw["stockNumber"].astype(str)
    df["url"] = raw["url"]
    df["vin"] = raw["vin"].str.upper().str.strip()
    df["search_model"] = raw["search_model"]
    df["make_model_text"] = raw["make"].fillna("") + " " + raw["model"].fillna("")
    df["listed_year"] = pd.to_numeric(raw["year"], errors="coerce")
    df["title_year"] = np.nan
    df["trim_text"] = raw["trim"].fillna("")
    df["price"] = pd.to_numeric(raw["basePrice"], errors="coerce")
    df["mileage"] = pd.to_numeric(raw["mileage"], errors="coerce")
    df["body_type"] = raw["body"].map(norm_body)
    df["drivetrain"] = raw["driveTrain"].map(norm_drive)
    df["transmission"] = raw["transmission"].map(norm_transmission)
    fuel = raw["engineType"].fillna(raw["fuelType"])
    fuel = fuel.where(~raw["model"].str.contains("hybrid", case=False, na=False), "Hybrid")
    df["fuel_type"] = fuel.map(norm_fuel)
    df["cylinders"] = pd.to_numeric(raw["cylinders"], errors="coerce")
    df["exterior_color"] = raw["exteriorColor"].map(norm_color)
    df["condition"] = "not stated"     # CarMax does not grade condition on listings
    df["title_status"] = "clean"       # CarMax does not retail salvage/flood-title cars
    df["seller_type"] = "dealer"
    df["market"] = raw["storeCity"].fillna("").str.lower() + ", " + raw["stateAbbreviation"].fillna("").str.lower()
    df["latitude"] = np.nan
    df["longitude"] = np.nan
    df["distance_miles"] = pd.to_numeric(raw["distance"], errors="coerce")
    df["listed_date"] = pd.to_datetime(raw["lastMadeSaleableDate"], errors="coerce", utc=True).dt.date
    df["title"] = raw["year"].astype(str) + " " + raw["make"] + " " + raw["model"] + " " + raw["trim"].fillna("")
    df["description"] = raw.apply(carmax_description, axis=1)
    return df


# ------------------------------------------------------------ pipeline ---
class StepLog:
    """Records how many listings of each source a cleaning step removed."""

    def __init__(self):
        self.rows = []

    def __call__(self, step, df_before, df_after):
        for src in ("craigslist", "carmax"):
            b = int((df_before["source"] == src).sum())
            a = int((df_after["source"] == src).sum())
            self.rows.append({"step": step, "source": src, "before": b, "removed": b - a, "after": a})
        return df_after.copy()

    def table(self):
        return pd.DataFrame(self.rows)


def integrate(verbose=True):
    log = StepLog()
    cl_raw = pd.read_csv(CRAIGSLIST_CSV, dtype=str)
    cm_raw = pd.read_csv(CARMAX_CSV, dtype={"vin": str, "trim": str})
    msrp = pd.read_csv(MSRP_CSV)

    df = pd.concat([map_craigslist(cl_raw), map_carmax(cm_raw)], ignore_index=True)

    # 0. study scope (see config.py: Nissan was collected but has no complete MSRP data)
    df = log("make outside the study scope (Nissan, see config.py)", df,
             df[df["search_model"].isin(list(MODEL_TO_MAKE))])

    # 1. model check: the listing must really be the make/model that was searched
    #    (all 12 model names are unique to their make, so the model name is enough)
    df["model"] = df["search_model"]
    df["make"] = df["model"].map(MODEL_TO_MAKE)
    ok = pd.Series([match_model(t, m) for t, m in zip(df["make_model_text"], df["model"])], index=df.index)
    wanted = df["title"].fillna("").str.contains(
        r"\b(?:we buy|i buy|buying|wanted|cash for|looking for|parting out)\b", case=False, regex=True)
    df = log("wrong or different model / wanted ads", df, df[ok & ~wanted])

    # 2. model year: VIN decode > year in the title > listed year attribute
    vin_year = [vin_model_year(v, y) for v, y in zip(df["vin"], df["listed_year"])]
    df["vin_year"] = pd.to_numeric(pd.Series(vin_year, index=df.index), errors="coerce")
    df["year"] = df["vin_year"].fillna(df["title_year"]).fillna(df["listed_year"])
    df["year_conflict"] = (df["year"] != df["listed_year"]) & df["listed_year"].notna()
    df = log("missing model year", df, df[df["year"].notna()])

    # 3. duplicates: same listing id, then same VIN (within or across sources)
    df = log("duplicate listing id", df, df.drop_duplicates("listing_id"))
    has_vin = df["vin"].fillna("").str.match(VIN_RE)
    dup_vin = has_vin & df.duplicated("vin", keep="first")
    df = log("duplicate VIN (reposts / cross-source)", df, df[~dup_vin])
    # Craigslist reposts without a VIN: same model, year, price and mileage
    key = ["source", "model", "year", "price", "mileage"]
    df = log("duplicate repost (same model/year/price/miles)", df, df[~df.duplicated(key, keep="first")])

    # 4. price and mileage sanity
    df = log("price missing or outside $1,000-$120,000", df, df[df["price"].between(1000, 120000)])
    age = SCRAPE_YEAR - df["year"]
    bad_miles = df["mileage"].isna() | (df["mileage"] > 400000) | ((df["mileage"] < 100) & (age >= 2))
    df = log("odometer missing or implausible", df, df[~bad_miles])
    down_payment = df["description"].str.contains(r"down payment|\$\s?\d[\d,]*\s*down\b|\bdown\s*payment",
                                                  case=False, regex=True) & (df["price"] < 5000)
    df = log("price is a down payment, not the car price", df, df[~down_payment])
    df = log("title 'parts only'", df, df[df["title_status"] != "parts only"])

    # 5. merge MSRP (inner join: U.S. News covers model years 2007+)
    msrp = msrp.rename(columns={"msrp_low": "msrp_base", "msrp_high": "msrp_top"})
    for c in ("msrp_base", "msrp_top"):
        msrp[c] = msrp[c].map(to_number)
    msrp = msrp[["make", "model", "year", "msrp_base", "msrp_top"]].drop_duplicates(["make", "model", "year"])
    merged = df.merge(msrp, on=["make", "model", "year"], how="left")
    df = log("no MSRP for that make/model/year", merged, merged[merged["msrp_base"].notna()])

    # 6. target and outlier removal
    df["retention_ratio"] = df["price"] / df["msrp_base"]
    df = log("retention ratio outside 0.03-2.0", df, df[df["retention_ratio"].between(0.03, 2.0)])

    # 7. fill structural attributes from the model when the seller left them blank
    df["trim"] = df["trim_text"].map(find_trim)
    for col in ("body_type", "drivetrain", "transmission", "fuel_type", "cylinders"):
        mode = df.groupby("model")[col].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
        df[col] = df[col].fillna(df["model"].map(mode))

    out = df[UNIFIED_COLUMNS + ["msrp_base", "msrp_top", "retention_ratio", "year_conflict"]].copy()
    out["year"] = out["year"].astype(int)
    out = out.sort_values(["source", "make", "model", "year"]).reset_index(drop=True)
    out.to_csv(INTEGRATED_CSV, index=False)

    steps = log.table()
    steps.to_csv(RESULTS / "cleaning_steps.csv", index=False)
    write_eda(cl_raw, cm_raw, out)
    if verbose:
        print(steps.to_string(index=False))
        print(f"\nintegrated dataset: {len(out)} rows -> {INTEGRATED_CSV}")
        print(out["source"].value_counts().to_string())
    return out


def write_eda(cl_raw, cm_raw, after):
    """Attribute statistics used in the report (Data Exploration section)."""
    # share of missing values in each source's own (raw) attributes, before any cleaning
    for name, raw in (("craigslist", cl_raw), ("carmax", cm_raw)):
        raw.isna().mean().round(3).rename("missing_share").to_csv(RESULTS / f"eda_missing_raw_{name}.csv")

    num_cols = ["price", "mileage", "year", "msrp_base", "retention_ratio", "cylinders", "distance_miles"]
    desc = after.groupby("source")[num_cols].describe().T.round(2)
    desc.to_csv(RESULTS / "eda_numeric_by_source.csv")
    after[num_cols].describe().T.round(2).to_csv(RESULTS / "eda_numeric_all.csv")

    cats = ["make", "model", "body_type", "drivetrain", "transmission", "fuel_type",
            "exterior_color", "condition", "title_status", "seller_type", "trim"]
    rows = []
    for c in cats:
        vc = after[c].value_counts()
        rows.append({"attribute": c, "n_unique": after[c].nunique(),
                     "top_values": "; ".join(f"{k} ({v})" for k, v in vc.head(6).items())})
    pd.DataFrame(rows).to_csv(RESULTS / "eda_categorical.csv", index=False)
    counts = after.groupby(["make", "model", "source"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts.to_csv(RESULTS / "eda_make_model_counts.csv")


if __name__ == "__main__":
    integrate()
