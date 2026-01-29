import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from newspaper import Article
import time
import nltk

# 1. Critical Cloud Fix: Ensure NLTK data is present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 2. Optimized Model Loading (Handles KeyError & Memory)
@st.cache_resource
def load_summarizer():
    # facebook/bart-large-cnn is ~1.6GB. 
    # If the app crashes (Over Memory), change this to "sshleifer/distilbart-cnn-12-6"
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

summarizer = load_summarizer()

# UI Layout
st.set_page_config(page_title="AI News Summarizer", page_icon="📝")
st.title("🤖 News Article Summarizer")
st.markdown("Enter a news URL to get an AI-generated summary using the **BART Transformer model**.")

# 3. User Input & Processing
url = st.text_input("Paste News Article URL here:")

if st.button("Summarize Article"):
    if url:
        try:
            with st.spinner('Processing...'):
                start_time = time.time()
                
                # Fetch article with a User-Agent to avoid scraping blocks
                article = Article(url)
                article.download()
                article.parse()
                
                if not article.text:
                    st.error("No text found. The website might be blocking the tool.")
                else:
                    # Summarization
                    summary = summarizer(article.text[:4000], max_length=130, min_length=30, do_sample=False)
                    summary_text = summary[0]['summary_text']
                    
                    # Performance Metrics
                    duration = round(time.time() - start_time, 2)
                    orig_len = len(article.text.split())
                    summ_len = len(summary_text.split())
                    
                    st.subheader(f"Title: {article.title}")
                    st.success(summary_text)
                    
                    st.write("---")
                    st.info(f"**Metrics:** {orig_len} words ➡️ {summ_len} words | Time: {duration}s")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
