# Fraud Watch

Fraud Watch is a fraud detection pipeline for card transactions. It cleans messy data with SQL, trains a model to score each transaction, and shows the results on an interactive dashboard where a non-technical person can see what was flagged and why.

**Live demo:** https://fraudwatch-aarg3mjjdwgjrtzvcmbh4d.streamlit.app/

## The problem

Banks handle millions of card transactions, and only a tiny share are fraud. Checking every transaction by hand is impossible, and blocking too many good transactions annoys customers. The goal is to flag the risky ones so a person only reviews those.

## How it works

1. **Generate data** (`src/generate_data.py`): 50,000 fake card transactions with a built-in 2% fraud rate. The data is deliberately messy (duplicates, negative amounts, missing cities, untidy category names).
2. **Clean with SQL** (`src/clean_with_sql.py`): loads the data into SQLite and fixes the problems with SQL queries. It also measures fraud rates by channel, shop type, time of day and device.
3. **Train models** (`src/train_model.py`): compares Logistic Regression and Random Forest on a held-out test set, then saves the best one.
4. **Dashboard** (`app.py`): built with Streamlit. A sensitivity control lets you see the trade-off between catching fraud and flagging good customers.

## Results

On 12,500 unseen test transactions (249 of them fraud):

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.439 | 0.972 | 0.605 | 0.922 |
| Random Forest | 0.688 | 0.920 | 0.787 | 0.904 |

At the default setting, the best model caught 242 of 249 fraud cases (98.9% of the fraud value) and raised 309 false alarms.

Accuracy is not used as the main score, because a model that always says "not fraud" would be about 98% accurate and catch nothing. Precision, recall and PR-AUC show the real picture.

## Dashboard features

- Sensitivity control with presets, plus a chart of the trade-off between fraud caught and false alarms
- Filterable list of flagged transactions with a "why was it flagged?" view
- A page to try your own transaction and see its fraud score
- Cost calculator for the price of a false alarm (an assumption you can change)
- Works on phone and laptop

## Run it locally

You need Python 3.12.

```
pip install -r requirements.txt
python src/generate_data.py
python src/clean_with_sql.py
python src/train_model.py
streamlit run app.py
```

## Project structure

```
app.py                  Streamlit dashboard
requirements.txt        Python packages
src/generate_data.py    Creates the messy transaction data
src/clean_with_sql.py   Cleans the data with SQL and explores fraud patterns
src/train_model.py      Trains, compares and saves the models
data/                   Generated data files
outputs/                Scored transactions, metrics, saved model and charts
```

## Limitations

- **The data is generated.** I built the fraud patterns into it, so the model finds them easily. Real bank data is harder, and real-world scores would be lower.
- Negative amounts were assumed to be sign errors and made positive.
- The cost of a false alarm (Rs 100) is an assumption, not a measured figure.
- The model uses only a few simple signals (amount, hour, channel, shop type, foreign, new device). A real system would also use customer history.

## Built with

Python, pandas, NumPy, scikit-learn, SQL (SQLite), Streamlit, Altair, matplotlib