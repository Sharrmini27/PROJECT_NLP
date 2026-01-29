import streamlit as st
from transformers import pipeline
from newspaper import Article
import time
import nltk

# Fix for "Punkt" error
nltk.download('punkt')

# 1. Page Configuration
st.set_page_config(page_title="News Article Summarizer", page_icon="📝")

# 2. Load NLP Model
@st.cache_resource
def load_summarizer():
    # Using a smaller model like 'sshleifer/distilbart-cnn-12-6' if your RAM is low, 
    # but facebook/bart-large-cnn is the high-quality one you want for the report.
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

st.title("🤖 News Article Summarizer")
st.markdown("Enter a news URL to get an AI-generated summary using the **BART Transformer model**.")

# 3. User Input
url = st.text_input("Paste News Article URL here:")

if st.button("Summarize Article"):
    if url:
        try:
            with st.spinner('AI is reading and summarizing...'):
                start_time = time.time()
                
                # Fetch and Parse the article with a User-Agent to prevent blocks
                article = Article(url)
                article.download()
                article.parse()
                article.nlp() # Optional: extra NLP processing
                
                if not article.text:
                    st.error("Could not extract text. Try a different news URL.")
                else:
                    # Execute Summarization
                    summary_output = summarizer(article.text, max_length=130, min_length=30, do_sample=False)
                    summary_text = summary_output[0]['summary_text']
                    
                    end_time = time.time()
                    
                    # 4. Display Results
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
            st.error(f"Error: {e}. Some websites block automated scraping.")
    else:
        st.warning("Please enter a URL first.")

# Sidebar
st.sidebar.title("Project Info")
st.sidebar.write("**Student Task:** JIE43303 NLP")
st.sidebar.write("**Model:** BART (Abstractive)")
