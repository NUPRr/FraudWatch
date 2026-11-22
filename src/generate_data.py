"""Creates a fake card-transaction file with fraud patterns and some messy data."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 50000

cities = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Chennai", "Hyderabad", "Kolkata", "Ahmedabad"]
normal_cats = ["groceries", "fuel", "restaurants", "clothing", "electronics",
               "travel", "jewelry", "gift cards", "pharmacy", "utilities"]
normal_cat_p = [0.22, 0.12, 0.17, 0.10, 0.05, 0.06, 0.02, 0.02, 0.12, 0.12]
fraud_cats = ["electronics", "jewelry", "gift cards", "travel", "clothing"]
fraud_cat_p = [0.35, 0.25, 0.20, 0.12, 0.08]

is_fraud = rng.random(N) < 0.02

# Time of day: normal spending is mostly daytime, fraud skews to the night
normal_h = np.array([1]*6 + [3]*3 + [7]*12 + [4]*3, dtype=float)
fraud_h = np.array([12]*6 + [3]*3 + [3]*12 + [5]*3, dtype=float)
normal_h /= normal_h.sum()
fraud_h /= fraud_h.sum()
hour = np.where(is_fraud, rng.choice(24, N, p=fraud_h), rng.choice(24, N, p=normal_h))

days = rng.integers(0, 300, N)
minute = rng.integers(0, 60, N)
ts = (pd.Timestamp("2026-01-01") + pd.to_timedelta(days, unit="D")
      + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m"))

amount = np.where(is_fraud, rng.lognormal(5.5, 0.9, N), rng.lognormal(3.8, 0.9, N)).round(2)
category = np.where(is_fraud,
                    rng.choice(fraud_cats, N, p=fraud_cat_p),
                    rng.choice(normal_cats, N, p=normal_cat_p))
channel = np.where(is_fraud,
                   rng.choice(["pos", "online", "atm"], N, p=[0.10, 0.85, 0.05]),
                   rng.choice(["pos", "online", "atm"], N, p=[0.60, 0.30, 0.10]))
is_foreign = np.where(is_fraud, rng.random(N) < 0.40, rng.random(N) < 0.03)
new_device = np.where(is_fraud, rng.random(N) < 0.60, rng.random(N) < 0.05)
city = np.where(is_foreign, "Overseas", rng.choice(cities, N))

df = pd.DataFrame({
    "transaction_id": [f"T{i:06d}" for i in range(N)],
    "customer_id": rng.integers(1000, 6000, N),
    "timestamp": ts,
    "amount": amount,
    "merchant_category": category,
    "channel": channel,
    "city": city,
    "is_foreign": is_foreign.astype(int),
    "new_device": new_device.astype(int),
    "is_fraud": is_fraud.astype(int),
})

# ---- Make the data messy on purpose, so we can clean it with SQL later ----
df.loc[rng.choice(N, 500, replace=False), "city"] = None                 # missing cities
bad = rng.choice(N, 250, replace=False)
df.loc[bad, "amount"] = -df.loc[bad, "amount"]                            # negative amounts
messy = rng.choice(N, 1500, replace=False)
df.loc[messy, "merchant_category"] = " " + df.loc[messy, "merchant_category"].str.upper() + " "
dups = df.sample(200, random_state=1)                                     # duplicate rows
df = pd.concat([df, dups]).sample(frac=1, random_state=2).reset_index(drop=True)
import os; os.makedirs("data", exist_ok=True)
df.to_csv("data/transactions_raw.csv", index=False)
print("Rows:", len(df))
print("Fraud rate:", round(df["is_fraud"].mean() * 100, 2), "%")
print("Saved to data/transactions_raw.csv")