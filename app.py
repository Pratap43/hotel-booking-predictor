import streamlit as st
import joblib
import pandas as pd
import time

# Page config
st.set_page_config(page_title="Hotel Predictor", layout="wide")

# CSS
st.markdown("""
<style>
body {background-color: #0f172a; color: white;}
.title {text-align:center; font-size:40px; font-weight:bold;}
.footer {text-align:center; font-size:14px; color:gray;}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>🏨 Hotel Booking Cancellation Predictor</div>", unsafe_allow_html=True)

# Load model safely
try:
    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
except:
    st.error("Model files not found. Please check model.pkl & columns.pkl")
    st.stop()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Booking Details")
    lead_time = st.number_input("Lead Time", min_value=0)
    adr = st.number_input("ADR", min_value=0.0)
    nights = st.number_input("Stay Nights", min_value=1)
    requests = st.number_input("Special Requests", min_value=0)
    repeat = st.selectbox("Repeated Guest", [0,1])

with col2:
    st.subheader("🏨 Hotel Info")
    hotel = st.selectbox("Hotel Type", ["Resort Hotel","City Hotel"])
    meal = st.selectbox("Meal", ["BB","HB","FB","SC"])
    segment = st.selectbox("Market Segment", ["Online TA","Offline TA/TO","Direct","Corporate"])

# Input processing
input_data = {
    'lead_time': lead_time,
    'adr': adr,
    'total_stay_nights': nights,
    'total_of_special_requests': requests,
    'is_repeated_guest': repeat
}

df_input = pd.DataFrame([input_data])
df_input = pd.get_dummies(df_input)

for col in columns:
    if col not in df_input:
        df_input[col] = 0

df_input = df_input[columns]

# Predict button
st.markdown("---")

if st.button("🔍 Predict"):

    with st.spinner("Analyzing booking..."):
        time.sleep(1)

    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    if pred == 1:
        st.error(f"❌ High Risk of Cancellation ({prob*100:.1f}%)")
        st.warning("💡 Suggestion: Offer discount or confirm booking early.")
    else:
        st.success(f"✅ Booking Likely Confirmed ({(1-prob)*100:.1f}%)")
        st.info("💡 Customer looks reliable.")

# Chatbot
st.markdown("---")
st.subheader("🤖 AI Assistant")

q = st.text_input("Ask about booking behavior...")

if q:
    q = q.lower()
    if "lead" in q:
        st.info("Long lead time increases cancellation chances.")
    elif "price" in q or "adr" in q:
        st.info("Higher price bookings may cancel more often.")
    else:
        st.info("This factor influences booking behavior.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Built by You 🚀 | ML Project</div>", unsafe_allow_html=True)

