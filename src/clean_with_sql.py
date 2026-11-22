"""Loads the raw CSV into a SQLite database, cleans it with SQL, and finds fraud patterns."""
import os
import sqlite3

import pandas as pd

os.makedirs("data", exist_ok=True)
raw = pd.read_csv("data/transactions_raw.csv")
con = sqlite3.connect("data/fraud.db")
raw.to_sql("transactions_raw", con, if_exists="replace", index=False)


def one(sql):
    return con.execute(sql).fetchone()[0]


print("--- Problems found in the raw data ---")
print("Rows:", one("SELECT COUNT(*) FROM transactions_raw"))
print("Duplicate rows:", one("SELECT COUNT(*) - COUNT(DISTINCT transaction_id) FROM transactions_raw"))
print("Negative amounts:", one("SELECT COUNT(*) FROM transactions_raw WHERE amount < 0"))
print("Missing cities:", one("SELECT COUNT(*) FROM transactions_raw WHERE city IS NULL"))
print("Untidy category names:", one(
    "SELECT COUNT(*) FROM transactions_raw WHERE merchant_category <> LOWER(TRIM(merchant_category))"))

# ---- The cleaning, written in SQL ----
con.executescript("""
DROP TABLE IF EXISTS transactions_clean;

CREATE TABLE transactions_clean AS
SELECT
    transaction_id,
    customer_id,
    timestamp,
    ABS(amount)                              AS amount,             -- negative amounts are sign errors
    LOWER(TRIM(merchant_category))           AS merchant_category,  -- ' GIFT CARDS ' -> 'gift cards'
    channel,
    COALESCE(NULLIF(TRIM(city), ''), 'Unknown') AS city,            -- missing city -> Unknown
    is_foreign,
    new_device,
    CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
    is_fraud
FROM transactions_raw
WHERE rowid IN (SELECT MIN(rowid) FROM transactions_raw GROUP BY transaction_id);  -- keep one copy of each
""")
con.commit()

print("\n--- After cleaning ---")
print("Rows:", one("SELECT COUNT(*) FROM transactions_clean"))
print("Negative amounts:", one("SELECT COUNT(*) FROM transactions_clean WHERE amount < 0"))
print("Missing cities:", one("SELECT COUNT(*) FROM transactions_clean WHERE city IS NULL"))
print("Overall fraud rate: %.2f%%" % one("SELECT 100.0 * SUM(is_fraud) / COUNT(*) FROM transactions_clean"))


def fraud_by(title, sql):
    print(f"\n--- Fraud rate by {title} ---")
    for name, total, rate in con.execute(sql):
        print(f"{str(name):<14} {total:>6} transactions   {rate:>5}% fraud")


fraud_by("channel", """SELECT channel, COUNT(*), ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2)
                       FROM transactions_clean GROUP BY channel ORDER BY 3 DESC""")
fraud_by("category", """SELECT merchant_category, COUNT(*), ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2)
                        FROM transactions_clean GROUP BY merchant_category ORDER BY 3 DESC LIMIT 5""")
fraud_by("time of day", """SELECT CASE WHEN hour < 6 THEN 'night (0-5)' ELSE 'daytime' END,
                                  COUNT(*), ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2)
                           FROM transactions_clean GROUP BY 1 ORDER BY 3 DESC""")
fraud_by("new device", """SELECT CASE WHEN new_device = 1 THEN 'new device' ELSE 'known device' END,
                                 COUNT(*), ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2)
                          FROM transactions_clean GROUP BY 1 ORDER BY 3 DESC""")

pd.read_sql("SELECT * FROM transactions_clean", con).to_csv("data/transactions_clean.csv", index=False)
con.close()
print("\nSaved: data/fraud.db and data/transactions_clean.csv")