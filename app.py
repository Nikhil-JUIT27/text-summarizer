import streamlit as st
from summarizer import TextSummarizer

st.title("🧠 Text Summarization Tool")

sample_text = st.text_area("Enter text to summarize")

if st.button("Summarize"):
    summarizer = TextSummarizer()
    result = summarizer.summarize(sample_text, method="both")
    
    st.subheader("📄 Extractive Summary")
    st.write(result["extractive"])
    
    st.subheader("🧠 Abstractive Summary")
    st.write(result["abstractive"])

