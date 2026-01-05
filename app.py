import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="Naive Bayes Spam Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Detection")
st.write("Naive Bayes Classification (BernoulliNB)")

# ------------------------------------
# Load Dataset
# ------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("NBA_Spam.csv")

df = load_data()

st.success("Dataset loaded successfully")
st.dataframe(df.head(), use_container_width=True)

# ------------------------------------
# Features & Target
# ------------------------------------
X = df.drop("Spam", axis=1)
y = df["Spam"]

# ------------------------------------
# Train-Test Split
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------
# Train Naive Bayes Model
# ------------------------------------
model = BernoulliNB()
model.fit(X_train, y_train)

# ------------------------------------
# Model Accuracy
# ------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Accuracy")
st.write(f"Accuracy: **{accuracy:.2f}**")

# ------------------------------------
# User Input
# ------------------------------------
st.subheader("✉️ Check New Email")

st.write("Select words present in the email:")

free = st.checkbox("Free")
win = st.checkbox("Win")
offer = st.checkbox("Offer")
money = st.checkbox("Money")
click = st.checkbox("Click")

# ------------------------------------
# Prediction
# ------------------------------------
if st.button("Predict Spam"):
    input_data = np.array([[int(free), int(win), int(offer), int(money), int(click)]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("🚨 This email is SPAM")
    else:
        st.success("✅ This email is NOT SPAM")
