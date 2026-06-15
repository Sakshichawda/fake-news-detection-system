import streamlit as st
import pickle

model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake News Detection System")

news = st.text_area("Enter News Article")

if st.button("Check News"):
    data = vectorizer.transform([news])
    prediction = model.predict(data)

    if prediction[0] == 0:
        st.error("Fake News")
    else:
        st.success("Real News")