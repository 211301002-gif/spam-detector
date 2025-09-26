import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Spam Detector", page_icon="📧")

@st.cache_data(show_spinner=False)
def load_data():
    url = "https://raw.githubusercontent.com/codebasics/py/master/ML/14_naive_bayes/spam.csv"
    df = pd.read_csv(url, encoding="latin-1")
    
    # Automatically detect columns
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].rename(columns={'v1': 'Category', 'v2': 'Message'})
    elif 'Category' in df.columns and 'Message' in df.columns:
        df = df[['Category', 'Message']]
    else:
        st.error(f"CSV does not have expected columns. Found columns: {list(df.columns)}")
        st.stop()
    
    # Convert spam/non-spam to numeric
    df['spam'] = df['Category'].apply(lambda x: 1 if str(x).lower() == 'spam' else 0)
    return df

@st.cache_resource(show_spinner=False)
def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['spam'],
                                                        test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_count = vectorizer.fit_transform(X_train.values)
    
    model = MultinomialNB()
    model.fit(X_train_count, y_train)
    
    # Test accuracy
    X_test_count = vectorizer.transform(X_test.values)
    accuracy = model.score(X_test_count, y_test)
    return model, vectorizer, accuracy

# Train model and get vectorizer
model, vectorizer, test_accuracy = train_model()

# ---- Streamlit UI ----
st.title("📧 Spam Detector Web App")
st.write("Type or paste a message below and click **Predict** to check if it's spam or not.")

st.markdown(f"**Model Test Accuracy:** `{test_accuracy:.3f}`")

user_text = st.text_area("Enter your message:", height=150)

if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please enter a message first.")
    else:
        X_input = vectorizer.transform([user_text])
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0].max()
        if pred == 1:
            st.error(f"🚨 Spam (Confidence: {proba:.2f})")
        else:
            st.success(f"✅ Not Spam (Confidence: {proba:.2f})")

