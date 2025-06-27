import streamlit as st
from summarizer import TextSummarizer

st.set_page_config(page_title="Text Summarizer", layout="centered")

st.title("📝 Text Summarization Tool")

text = st.text_area("Enter text to summarize:", height=300)
method = st.selectbox("Choose summarization method", ["both", "extractive", "abstractive"])

if st.button("Summarize"):
    summarizer = TextSummarizer()
    result = summarizer.summarize(text, method=method)

    if "extractive" in result:
        st.subheader("Extractive Summary")
        st.write(result["extractive"])
    
    if "abstractive" in result:
        st.subheader("Abstractive Summary")
        st.write(result["abstractive"])

