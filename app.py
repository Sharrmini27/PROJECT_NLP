import streamlit as st
from transformers import pipeline
from newspaper import Article
import time

# 1. Page Configuration (Requirement 1.ii: Specify features)
st.set_page_config(page_title="News Article Summarizer", page_icon="📝")

# 2. Load NLP Model (Requirement 1.iii: Use appropriate technologies)
# We use BART because it is excellent for Abstractive Summarization
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

st.title("🤖 News Article Summarizer")
st.markdown("Enter a news URL to get an AI-generated summary using the **BART Transformer model**.")

# 3. User Input (Requirement 1.ii)
url = st.text_input("Paste News Article URL here:")

if st.button("Summarize Article"):
    if url:
        try:
            with st.spinner('AI is reading and summarizing...'):
                start_time = time.time()
                
                # Fetch and Parse the article
                article = Article(url)
                article.download()
                article.parse()
                
                # Execute Summarization (Abstractive)
                # We limit length to ensure a concise paragraph
                summary_output = summarizer(article.text, max_length=130, min_length=30, do_sample=False)
                summary_text = summary_output[0]['summary_text']
                
                end_time = time.time()
                
                # 4. Display Results (Requirement 1.iv: Performance measurement)
                st.subheader(f"Title: {article.title}")
                st.write("### AI Summary:")
                st.success(summary_text)
                
                # Measurements for your Report
                orig_len = len(article.text.split())
                summ_len = len(summary_text.split())
                duration = round(end_time - start_time, 2)
                
                st.write("---")
                st.info(f"**Performance Metrics:**")
                st.write(f"- Original Length: {orig_len} words")
                st.write(f"- Summary Length: {summ_len} words")
                st.write(f"- Processing Time: {duration} seconds")
                
        except Exception as e:
            st.error(f"Error: {e}. Check if the URL is valid.")
    else:
        st.warning("Please enter a URL first.")

# Sidebar for Report Info (Requirement 2: Clarity)
st.sidebar.title("Project Info")
st.sidebar.write("**Student Task:** JIE43303 NLP")
st.sidebar.write("**Model:** BART (Abstractive)")
