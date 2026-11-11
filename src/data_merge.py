"""Clean, merge and feature-engineer the two customer tables.

Output: data/processed/merged_customer_data.csv
    one row per transaction: customer_id, transaction_id, <features>, product_category

Run:  python src/data_merge.py
"""
import pandas as pd

from config import MERGED_CSV, SOCIAL_RAW, TARGET, TRANSACTIONS_RAW

SENTIMENT_SCORE = {"Negative": -1, "Neutral": 0, "Positive": 1}
PLATFORMS = ["Facebook", "Instagram", "LinkedIn", "TikTok", "Twitter"]


def load_raw():
    social = pd.read_excel(SOCIAL_RAW)
    transactions = pd.read_excel(TRANSACTIONS_RAW)
    return social, transactions


# ------------------------------------------------------------------ social
def clean_social(social: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and turn 'A178' into the integer 178."""
    s = social.drop_duplicates().copy()
    for col in ["customer_id_new", "social_media_platform", "review_sentiment"]:
        s[col] = s[col].astype(str).str.strip()
    s["customer_id"] = s["customer_id_new"].str.lstrip("Aa").astype(int)
    s["sentiment_score"] = s["review_sentiment"].map(SENTIMENT_SCORE)
    return s.drop(columns="customer_id_new")


def aggregate_social(s: pd.DataFrame) -> pd.DataFrame:
    """Collapse social to ONE row per customer.

    A customer can have several profiles (one per platform / review). Joining the raw
    rows onto transactions would multiply each purchase by the number of profiles
    (150 purchases -> 213 rows) and count the same purchase several times.
    """
    g = s.groupby("customer_id")
    agg = pd.DataFrame({
        "n_social_profiles": g.size(),
        "n_platforms": g["social_media_platform"].nunique(),
        "engagement_mean": g["engagement_score"].mean(),
        "engagement_max": g["engagement_score"].max(),
        "interest_mean": g["purchase_interest_score"].mean(),
        "interest_max": g["purchase_interest_score"].max(),
        "sentiment_mean": g["sentiment_score"].mean(),
        "pct_positive": g["review_sentiment"].apply(lambda x: (x == "Positive").mean()),
        "pct_negative": g["review_sentiment"].apply(lambda x: (x == "Negative").mean()),
    })
    # platform where the customer is most engaged (their "main" platform)
    top = s.sort_values("engagement_score", ascending=False).drop_duplicates("customer_id")
    agg["top_platform"] = top.set_index("customer_id")["social_media_platform"]
    # one flag per platform the customer is active on
    on = pd.crosstab(s["customer_id"], s["social_media_platform"]).clip(upper=1)
    on = on.reindex(columns=PLATFORMS, fill_value=0).add_prefix("on_")
    on.columns = on.columns.str.lower()
    agg = agg.join(on)
    agg["engagement_x_interest"] = agg["engagement_mean"] * agg["interest_mean"] / 100
    return agg.reset_index()


# ------------------------------------------------------------ transactions
def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Rename the ID, parse dates and impute the 10 missing customer_rating values.

    Missing ratings get the overall median (not the per-category median, which would
    leak the target into a feature) plus a flag so the model knows it was imputed.
    """
    t = transactions.rename(columns={"customer_id_legacy": "customer_id"}).copy()
    t["customer_id"] = t["customer_id"].astype(int)
    t["purchase_date"] = pd.to_datetime(t["purchase_date"])
    t[TARGET] = t[TARGET].astype(str).str.strip()
    t["rating_missing"] = t["customer_rating"].isna().astype(int)
    t["customer_rating"] = t["customer_rating"].fillna(t["customer_rating"].median())
    return t


def add_transaction_features(t: pd.DataFrame) -> pd.DataFrame:
    t = t.sort_values(["customer_id", "purchase_date", "transaction_id"]).copy()
    t["purchase_month"] = t["purchase_date"].dt.month
    t["purchase_dayofweek"] = t["purchase_date"].dt.dayofweek
    t["is_weekend"] = (t["purchase_dayofweek"] >= 5).astype(int)
    g = t.groupby("customer_id")
    # history features use only EARLIER purchases of the same customer -> no leakage
    t["prior_purchases"] = g.cumcount()
    t["prev_category"] = g[TARGET].shift(1).fillna("first_purchase")
    t["prev_amount"] = g["purchase_amount"].shift(1)
    t["prev_amount"] = t["prev_amount"].fillna(t["purchase_amount"].median())
    return t.sort_values("transaction_id").reset_index(drop=True)


# ------------------------------------------------------------------- merge
def merge(transactions: pd.DataFrame, social_agg: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    """Join on customer_id.

    INNER on purpose: the model predicts what a customer buys *when visiting our social
    media*, so every training row needs real social features. Only 61 customers are in
    both tables; a left join would keep 33 more purchases but their social columns
    would be 100% imputed (made up), which adds noise rather than signal.
    """
    m = transactions.merge(social_agg, on="customer_id", how=how, validate="many_to_one")
    front = ["customer_id", "transaction_id", "purchase_date"]
    rest = [c for c in m.columns if c not in front + [TARGET]]
    return m[front + rest + [TARGET]]


def build_merged(how: str = "inner", verbose: bool = True) -> pd.DataFrame:
    social_raw, txn_raw = load_raw()
    social = clean_social(social_raw)
    social_agg = aggregate_social(social)
    txn = add_transaction_features(clean_transactions(txn_raw))
    merged = merge(txn, social_agg, how=how)
    if verbose:
        print(f"social:       {len(social_raw)} rows -> {len(social)} after dropping "
              f"{len(social_raw) - len(social)} duplicates -> {len(social_agg)} customers")
        print(f"transactions: {len(txn)} rows, {txn['customer_id'].nunique()} customers, "
              f"{int(txn['rating_missing'].sum())} ratings imputed")
        print(f"customers in both tables: {merged['customer_id'].nunique()}")
        print(f"merged ({how}): {merged.shape[0]} rows x {merged.shape[1]} columns")
    return merged


def main():
    merged = build_merged()
    MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_CSV, index=False)
    print(f"saved -> {MERGED_CSV}")


if __name__ == "__main__":
    main()
