"""Fraud Watch - interactive dashboard for the fraud detection pipeline."""
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Watch", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg,#10243a,#1b4f9c); color:#fff; padding:26px 28px;
       border-radius:16px; margin-bottom:18px;}
.hero h1 {margin:0; font-size:2.1rem; color:#fff;}
.hero p {margin:6px 0 0; opacity:.92; font-size:1.05rem;}
.kpi {border-radius:14px; padding:16px 18px; background:#fff; border:1px solid #e1e8f0;
      border-left:8px solid var(--c); box-shadow:0 1px 2px rgba(16,36,58,.08);}
.kpi .l {color:#5b6b7c; font-size:.9rem; font-weight:600;}
.kpi .v {font-size:1.9rem; font-weight:800; color:#10243a; line-height:1.25;}
.kpi .s {color:#5b6b7c; font-size:.85rem;}
.tag {display:inline-block; padding:4px 12px; border-radius:20px; font-size:.9rem; font-weight:700;
      margin:3px 6px 3px 0; background:#fbe5e2; color:#c0392b;}
.good {background:#e3f5ef; color:#0f8b6d;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("outputs/scored_transactions.csv")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    return df, metrics


@st.cache_resource
def load_model():
    return joblib.load("outputs/fraud_model.joblib")


df, metrics = load_data()
model = load_model()
real = df["is_fraud"] == 1
total_fraud = int(real.sum())
value_total = float(df.loc[real, "amount"].sum())
big_amount = float(df["amount"].quantile(0.95))


def kpi(col, label, value, sub, color):
    col.markdown(
        f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div>'
        f'<div class="v">{value}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True)


def warning_signs(r):
    out = []
    if r["new_device"] == 1:
        out.append("New device")
    if r["hour"] < 6:
        out.append("Night-time purchase")
    if r["is_foreign"] == 1:
        out.append("Made abroad")
    if r["channel"] == "online":
        out.append("Online purchase")
    if r["merchant_category"] in ("jewelry", "gift cards", "electronics"):
        out.append(f"High-risk shop type ({r['merchant_category']})")
    if r["amount"] > big_amount:
        out.append("Unusually large amount")
    return out


# ------------------------------ Sidebar controls ------------------------------
if "sens" not in st.session_state:
    st.session_state.sens = 50


def set_sens(v):
    st.session_state.sens = v


with st.sidebar:
    st.header("Controls")
    st.write("**How sensitive should the system be?**")
    st.button("Catch more fraud", on_click=set_sens, args=(80,), use_container_width=True)
    st.button("Balanced", on_click=set_sens, args=(50,), use_container_width=True)
    st.button("Fewer false alarms", on_click=set_sens, args=(20,), use_container_width=True)
    st.slider("Fine-tune sensitivity (%)", 5, 95, step=5, key="sens",
              help="Higher = more transactions are flagged. You catch more fraud, "
                   "but more good customers get flagged by mistake.")
    cost = st.number_input("Cost of one false alarm (Rs)", min_value=0, value=100, step=50,
                           help="Assumption: a customer call and a card replacement. Change it to see the effect.")

sens = st.session_state.sens
threshold = (100 - sens) / 100
flagged = df["fraud_score"] >= threshold
tp = int((flagged & real).sum())
fp = int((flagged & ~real).sum())
fn = total_fraud - tp
value_caught = float(df.loc[flagged & real, "amount"].sum())
net = value_caught - fp * cost

# ------------------------------ Header ------------------------------
st.markdown(
    '<div class="hero"><h1>Fraud Watch</h1>'
    '<p>Spots card transactions that look like fraud, so a person only has to review the risky ones. '
    'Use the controls on the left and watch the results change.</p></div>',
    unsafe_allow_html=True)

tab_over, tab_review, tab_try, tab_how = st.tabs(
    ["Overview", "Review flagged", "Try a transaction", "How it works"])

# ------------------------------ Overview ------------------------------
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Fraud caught", f"{tp} of {total_fraud}", f"{100 * tp / total_fraud:.0f}% of all fraud", "#0f8b6d")
    kpi(c2, "Fraud missed", f"{fn}", "slipped through", "#c0392b")
    kpi(c3, "False alarms", f"{fp}", "good customers flagged by mistake", "#b36b00")
    kpi(c4, "Fraud value stopped", f"Rs {value_caught:,.0f}", f"of Rs {value_total:,.0f} total", "#1b4f9c")

    st.write("")
    st.info(
        f"At **{sens}% sensitivity** the system flags **{int(flagged.sum())}** of {len(df):,} transactions. "
        f"**{tp}** are real fraud. **{fp}** are good customers who would get a call or a blocked card. "
        f"Counting each false alarm as Rs {cost}, the estimated net benefit is **Rs {net:,.0f}**.")

    sweep = []
    for s in range(5, 100, 5):
        fl = df["fraud_score"] >= (100 - s) / 100
        sweep.append({"s": s, "caught_pct": 100 * (fl & real).sum() / total_fraud,
                      "false_alarms": int((fl & ~real).sum())})
    sw = pd.DataFrame(sweep)
    base = alt.Chart(sw).encode(x=alt.X("s:Q", title="Sensitivity (%)"))
    line1 = base.mark_line(color="#1b4f9c", strokeWidth=3).encode(
        y=alt.Y("caught_pct:Q", title="Fraud caught (%)", axis=alt.Axis(titleColor="#1b4f9c")))
    line2 = base.mark_line(color="#c0392b", strokeWidth=3).encode(
        y=alt.Y("false_alarms:Q", title="False alarms", axis=alt.Axis(titleColor="#c0392b", orient="right")))
    mark = alt.Chart(pd.DataFrame({"s": [sens]})).mark_rule(strokeDash=[6, 4], color="#888").encode(x="s:Q")
    trade = alt.layer(line1, line2, mark).resolve_scale(y="independent").properties(height=300)

    hours = (df[real].groupby("hour").size().reindex(range(24), fill_value=0)
             .reset_index(name="cases"))
    bars = alt.Chart(hours).mark_bar(color="#c0392b").encode(
        x=alt.X("hour:O", title="Hour of day"), y=alt.Y("cases:Q", title="Fraud cases"),
        tooltip=["hour", "cases"]).properties(height=300)

    left, right = st.columns(2)
    left.subheader("The trade-off")
    left.caption("Blue: fraud caught. Red: false alarms. The dotted line is your current setting.")
    left.altair_chart(trade, use_container_width=True)
    right.subheader("When does fraud happen?")
    right.caption("Real fraud cases in the test data, by hour of the day.")
    right.altair_chart(bars, use_container_width=True)

# ------------------------------ Review flagged ------------------------------
with tab_review:
    st.write("Transactions the system flagged at your current setting. Narrow them down with the filters.")
    f1, f2, f3 = st.columns(3)
    ch = f1.multiselect("Channel", sorted(df["channel"].unique()), default=sorted(df["channel"].unique()))
    cat = f2.multiselect("Shop type", sorted(df["merchant_category"].unique()),
                         default=sorted(df["merchant_category"].unique()))
    min_amt = f3.number_input("Minimum amount (Rs)", min_value=0, value=0, step=500)

    sel = df[flagged & df["channel"].isin(ch) & df["merchant_category"].isin(cat) & (df["amount"] >= min_amt)]
    sel = sel.sort_values("fraud_score", ascending=False)
    st.write(f"**{len(sel)}** transactions match. Showing the {min(len(sel), 300)} most suspicious.")

    top = sel.head(300)
    view = pd.DataFrame({
        "Transaction": top["transaction_id"],
        "When": pd.to_datetime(top["timestamp"]).dt.strftime("%d %b %Y, %H:%M"),
        "Amount": top["amount"],
        "Shop type": top["merchant_category"].str.title(),
        "Channel": top["channel"].str.title(),
        "City": top["city"],
        "New device": np.where(top["new_device"] == 1, "Yes", "No"),
        "Fraud score": top["fraud_score"],
        "Really fraud?": np.where(top["is_fraud"] == 1, "Yes", "No"),
    })
    st.dataframe(view, use_container_width=True, hide_index=True, height=360, column_config={
        "Amount": st.column_config.NumberColumn("Amount (Rs)", format="%.0f"),
        "Fraud score": st.column_config.ProgressColumn("Fraud score", min_value=0.0, max_value=1.0, format="%.2f"),
    })
    st.caption("'Really fraud?' is known only because this is labelled test data. A live system would not know it.")
    st.download_button("Download this list (CSV)", view.to_csv(index=False), "flagged_transactions.csv", "text/csv")

    if len(top):
        st.subheader("Why was it flagged?")
        pick = st.selectbox("Pick a transaction to inspect", top["transaction_id"].tolist())
        r = top[top["transaction_id"] == pick].iloc[0]
        a, b = st.columns([1, 2])
        a.metric("Fraud score", f"{r['fraud_score']:.2f}")
        a.write(f"Rs {r['amount']:,.0f} at a {r['merchant_category']} shop, {r['channel']}, "
                f"{r['city']}, hour {int(r['hour'])}.")
        signs = warning_signs(r)
        b.write("**Warning signs present:**")
        b.markdown("".join(f'<span class="tag">{s}</span>' for s in signs)
                   or "No obvious single sign.", unsafe_allow_html=True)
        b.caption("These are common warning signs found in the data, shown to help a reviewer. "
                  "The model weighs many signals together.")

# ------------------------------ Try a transaction ------------------------------
categories = sorted(df["merchant_category"].unique())
EXAMPLES = {
    "Everyday grocery shop": dict(t_amt=450.0, t_hr=17, t_ch="pos", t_cat="groceries", t_for=False, t_dev=False),
    "Suspicious night purchase": dict(t_amt=30000.0, t_hr=2, t_ch="online", t_cat="jewelry", t_for=True, t_dev=True),
    "Large online electronics": dict(t_amt=15000.0, t_hr=21, t_ch="online", t_cat="electronics", t_for=False, t_dev=False),
}
for k, v in EXAMPLES["Everyday grocery shop"].items():
    st.session_state.setdefault(k, v)


def load_example(name):
    for k, v in EXAMPLES[name].items():
        st.session_state[k] = v


with tab_try:
    st.write("Pick an example or enter your own details to see how suspicious the system finds it.")
    ex_cols = st.columns(len(EXAMPLES))
    for col, name in zip(ex_cols, EXAMPLES):
        col.button(name, on_click=load_example, args=(name,), use_container_width=True)

    a, b = st.columns(2)
    amount = a.number_input("Amount (Rs)", min_value=1.0, step=500.0, key="t_amt")
    hour = b.slider("Hour of day (0 to 23)", 0, 23, key="t_hr")
    channel = a.selectbox("Channel", ["pos", "online", "atm"], key="t_ch")
    category = b.selectbox("Shop type", categories, key="t_cat")
    foreign = a.checkbox("Made abroad", key="t_for")
    new_dev = b.checkbox("New device", key="t_dev")

    row = pd.DataFrame([{
        "log_amount": np.log1p(amount), "hour": hour, "is_night": int(hour < 6),
        "is_foreign": int(foreign), "new_device": int(new_dev),
        "channel": channel, "merchant_category": category}])
    score = float(model.predict_proba(row)[0, 1])

    st.write("")
    s1, s2 = st.columns([1, 2])
    s1.metric("Fraud score", f"{score:.2f}")
    s2.progress(min(score, 1.0))
    if score >= threshold:
        s2.error("Flagged: this looks like possible fraud.")
    else:
        s2.success("Looks normal.")
    signs = warning_signs({"new_device": int(new_dev), "hour": hour, "is_foreign": int(foreign),
                           "channel": channel, "merchant_category": category, "amount": amount})
    st.write("**Warning signs present:**")
    st.markdown("".join(f'<span class="tag">{s}</span>' for s in signs)
                or '<span class="tag good">None</span>', unsafe_allow_html=True)

# ------------------------------ How it works ------------------------------
with tab_how:
    h1, h2, h3 = st.columns(3)
    h1.markdown("### 1. Learn\nThe model studied 37,500 past transactions and which ones turned out to be fraud.")
    h2.markdown("### 2. Score\nIt gave each of 12,500 new transactions a fraud score from 0 to 1.")
    h3.markdown("### 3. Flag\nTransactions above your sensitivity setting are sent to a person for review.")
    left, right = st.columns(2)
    left.image("outputs/feature_importance.png", caption="Signals the model relies on most")
    right.image("outputs/pr_curve.png", caption="Comparison of the two models")
    table = pd.DataFrame(metrics["models"]).T.round(3)
    table.columns = ["Precision (flags that were right)", "Recall (fraud caught)", "F1 (balance)",
                     "ROC-AUC", "PR-AUC"]
    st.dataframe(table, use_container_width=True)
    st.warning("The data was generated for this project and the fraud patterns were built in, "
               "so real-world results would be lower.")