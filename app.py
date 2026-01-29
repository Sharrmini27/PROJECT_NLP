import streamlit as st
from transformers import pipeline
from newspaper import Article
import time
import nltk

# Requirement 1.iii: Initialize necessary NLP resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 1. Page Configuration (Requirement 1.ii: Specify features)
st.set_page_config(page_title="News Article Summarizer", page_icon="📝")

# 2. Load NLP Model (Requirement 1.iii: Use appropriate technologies)
# We use BART because it is excellent for Abstractive Summarization
@st.cache_resource
def load_summarizer():
    # Fixed: Explicitly defining task="summarization" to resolve the KeyError
    return pipeline(task="summarization", model="facebook/bart-large-cnn")

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
                
                # Fetch and Parse the article using Newspaper3k
                article = Article(url)
                article.download()
                article.parse()
                
                if not article.text:
                    st.error("Could not extract text. Some sites block automated access.")
                else:
                    # Execute Summarization (Abstractive)
                    # Requirement 1.iv: Model configuration for optimal summary length
                    summary_output = summarizer(article.text, max_length=130, min_length=30, do_sample=False)
                    summary_text = summary_output[0]['summary_text']
                    
                    end_time = time.time()
                    
                    # 4. Display Results (Requirement 1.iv: Performance measurement)
                    st.subheader(f"Title: {article.title}")
                    st.write("### AI Summary:")
                    st.success(summary_text)
                    
                    # Performance Metrics for the Report
                    orig_len = len(article.text.split())
                    summ_len = len(summary_text.split())
                    duration = round(end_time - start_time, 2)
                    reduction = round((1 - (summ_len / orig_len)) * 100, 1)
                    
                    st.write("---")
                    st.info(f"**Performance Metrics:**")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Original Words", orig_len)
                    col2.metric("Summary Words", summ_len)
                    col3.metric("Reduction", f"{reduction}%")
                    st.write(f"⏱ **Processing Time:** {duration} seconds")
                    
        except Exception as e:
            st.error(f"Error: {e}. Please ensure the URL is valid and accessible.")
    else:
        st.warning("Please enter a URL first.")

# Sidebar for Report Info (Requirement 2: Clarity)
st.sidebar.title("Project Info")
st.sidebar.write("**Course:** JIE43303 NLP")
st.sidebar.write("**Model:** BART (Abstractive)")
st.sidebar.write("**Student Project Task**")
