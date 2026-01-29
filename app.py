import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from newspaper import Article
import time
import nltk

# Fix for NLTK errors in cloud environments
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 1. Page Configuration
st.set_page_config(page_title="AI News Summarizer", page_icon="📝", layout="wide")

# 2. Optimized Model Loading (Bypasses KeyError)
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Using explicit task setup to prevent 'KeyError: summarization'
    return pipeline("summarization", model=model, tokenizer=tokenizer)

summarizer = load_model()

# UI Setup
st.title("🤖 AI-Powered News Summarizer")
st.markdown("---")

# 3. User Input
url = st.text_input("🔗 Paste News Article URL here:", placeholder="https://www.channelnewsasia.com/...")

if st.button("Generate Summary ✨"):
    if url:
        try:
            with st.spinner('AI is analyzing the article... please wait.'):
                start_time = time.time()
                
                # Fetch and Parse
                article = Article(url)
                article.download()
                article.parse()
                
                if not article.text or len(article.text) < 100:
                    st.error("❌ Extraction failed. This website might be blocking AI access or has too little text.")
                else:
                    # Summarization Logic
                    summary_output = summarizer(article.text, max_length=150, min_length=40, do_sample=False)
                    summary_text = summary_output[0]['summary_text']
                    
                    end_time = time.time()
                    
                    # 4. Display Results
                    st.subheader(f"📄 Article Title: {article.title}")
                    
                    col_res, col_space = st.columns([2, 1])
                    with col_res:
                        st.success(summary_text)
                    
                    # Performance Metrics Calculation
                    orig_len = len(article.text.split())
                    summ_len = len(summary_text.split())
                    duration = round(end_time - start_time, 2)
                    reduction = round((1 - (summ_len / orig_len)) * 100, 1)
                    
                    # Display Metrics
                    st.write("---")
                    st.markdown("### 📊 Performance Metrics")
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Original Words", orig_len)
                    m_col2.metric("Summary Words", summ_len)
                    m_col3.metric("Reduction", f"{reduction}%")
                    m_col4.metric("Time Taken", f"{duration}s")
                    
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please provide a URL first.")

# Sidebar
st.sidebar.title("Project Details")
st.sidebar.info("Model: BART-Large-CNN")
st.sidebar.markdown("This app uses Abstractive Summarization to rewrite news articles concisely.")
