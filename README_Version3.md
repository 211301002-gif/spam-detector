# 📧 Spam Detector Web App

A simple Streamlit app to classify messages as spam or not spam using a Naive Bayes classifier.

## Features

- Paste any message and check if it's spam or not
- Shows model test accuracy
- Optionally view some sample data

## Local Run

Clone this repo, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Online Run

Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud).

- Click "New app"
- Point to your repo and set `app.py` as the main file
- Deploy

---

Built using [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/).