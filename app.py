import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cohere
import joblib

# ---- Custom Styling ----
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.markdown-text-container {
    font-family: 'Segoe UI', sans-serif !important;
}

h1 {
    color: #1E5D22;
    text-align: center;
    font-weight: 700;
}

h2, h3, h4 {
    color: #387F3E;
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    background-color: #F2FBF6;
    font-family: 'Segoe UI', sans-serif;
}

.css-1d391kg {
    color: #205F24;
    font-weight: 600;
}

.insight-box {
    background-color: #f2fbf6;
    padding: 15px;
    border-radius: 10px;
    font-size: 16px;
    border-left: 5px solid #205F24;
    line-height: 1.6;
    color: #333;
    font-family: 'Segoe UI', sans-serif !important;
}

thead tr th {
    background-color: #DAEADB;
    color: #205F24;
    font-weight: bold;
}
tbody tr:nth-child(even),
tbody tr:nth-child(odd) {
    background-color: #F2FBF6 !important;
}
tbody td {
    padding: 6px 10px;
}
</style>
""", unsafe_allow_html=True)

# ---- Load Data and Model ----
final_df = pd.read_csv("final_df.csv", parse_dates=["Date"])
xgb_model = joblib.load("xgboost_model.pkl")

# ---- Cohere Setup ----
co = cohere.Client("gKOrxLTQztkU7rLIUra670tfDHOUVdSpdt5DJ8w9")  # Replace with your key

# ---- App Title ----
st.title("InsightEdge : The Market Analyst AI")

# ---- Sidebar Navigation ----
st.sidebar.header("📌 Navigation")
page = st.sidebar.radio("Go to", [
    "📈 Insight & Prediction", 
    "📉 Price & Sentiment Trend", 
    "🧾 Selected Date Data", 
    "ℹ️ About"
])

# ---- About Page ----
if page == "ℹ️ About":
    st.header("About InsightEdge")
    st.markdown("""
    **InsightEdge** is a professional AI-powered dashboard that combines **technical analysis**, **news sentiment**, and **machine learning** to help you understand and anticipate stock movements.

    - Built using **Streamlit**, **XGBoost**, and **Cohere AI**
    - Sentiment scores derived from real financial news
    - Generates professional financial insights and investment outlooks
    """)
    st.stop()

# ---- Common Date Picker for Relevant Pages ----
if page in ["📈 Insight & Prediction", "🧾 Selected Date Data"]:
    selected_date = st.sidebar.date_input("Select a date", final_df["Date"].max().date())
    selected_row = final_df[final_df["Date"] == pd.to_datetime(selected_date)]

# ---- Insight & Prediction Page ----
if page == "📈 Insight & Prediction":
    if selected_row.empty:
        st.warning("No data available for the selected date.")
        st.stop()

    # Prediction
    features = selected_row.drop(columns=["Date", "target"], errors='ignore')
    pred = xgb_model.predict(features)[0]
    direction = "📈 Rise" if pred == 1 else "📉 Fall"

    # Insight generation
    row = selected_row.iloc[0]
    prompt = f"""
    Generate a detailed financial insight based on the following data:

    - Date: {row['Date']}
    - Closing Price: {row['Close']:.2f}
    - Sentiment: {row['sentiment_label']} with average score {row['avg_sentiment']:.2f} from {int(row['headline_count'])} news headlines
    - RSI: {row['rsi']:.2f}
    - MACD: {row['macd']:.2f}
    - Model Prediction: {'📈 Rise' if pred == 1 else '📉 Fall'}

    Include market interpretation, technical outlook, and investor recommendation in a professional tone.
    """

    cohere_response = co.generate(
        model='command-r-plus',
        prompt=prompt,
        max_tokens=300,
        temperature=0.6,
    )

    # Display Insight
    st.subheader(f"📘 Insight for {selected_date}")
    st.markdown(f"<div class='insight-box'>{cohere_response.generations[0].text.strip()}</div>", unsafe_allow_html=True)

    st.subheader("📌 Tomorrow’s Price Direction Prediction:")
    st.success(f"Model predicts: **{direction}**")

# ---- Price & Sentiment Trend Page ----
elif page == "📉 Price & Sentiment Trend":
    st.subheader("📉 Price and Sentiment Trend")

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(final_df["Date"], final_df["Close"], label="Close Price", color="#205F24", linewidth=2)
    ax1.set_ylabel("Close Price", color="#205F24", fontsize=11)
    ax1.tick_params(axis='y', labelcolor="#205F24")

    ax2 = ax1.twinx()
    ax2.plot(final_df["Date"], final_df["avg_sentiment"], label="Sentiment Score", color="#7BAE7F", linestyle="--", linewidth=2)
    ax2.set_ylabel("Sentiment Score", color="#7BAE7F", fontsize=11)
    ax2.tick_params(axis='y', labelcolor="#7BAE7F")

    plt.title("📊 Close Price vs. Sentiment Score Over Time", fontsize=14, color="#205F24", fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig)

# ---- Selected Date Data Page ----
elif page == "🧾 Selected Date Data":
    if selected_row.empty:
        st.warning("No data available for the selected date.")
    else:
        st.subheader(f"🧾 Data for {selected_date}")
        st.dataframe(selected_row.T)

# ---- Footer ----
st.caption("🚀 Built using Streamlit, XGBoost, and Cohere")
