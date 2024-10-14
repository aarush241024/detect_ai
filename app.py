import streamlit as st
import os
import openai
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import inf
from googletrans import Translator

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Download NLTK data
nltk.download('punkt', quiet=True)

# Initialize translator
translator = Translator()

def detect_language(text):
    return translator.detect(text).lang

def translate_to_english(text, source_lang):
    if source_lang != 'en':
        return translator.translate(text, src=source_lang, dest='en').text
    return text

def calculate_perplexity(text):
    tokens = word_tokenize(text)
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens)

def calculate_burstiness(text):
    tokens = word_tokenize(text)
    token_counts = Counter(tokens)
    max_count = max(token_counts.values())
    return max_count / len(tokens)

def detect_ai_content(text, lang):
    english_text = translate_to_english(text, lang)
    
    prompt = f"""Analyze the following text and determine the likelihood of it being AI-generated or human-written. 
    Provide your analysis as percentages for both categories. 
    Additionally, explain the key indicators or sources that led to your conclusion. 
    Format your response as follows:
    AI-generated: X%
    Human-written: Y%
    Key Indicators:
    1. [First indicator]
    2. [Second indicator]
    3. [Third indicator]
    Potential Sources:
    - [List potential sources or models that might have generated this content]

    Text: {english_text}"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "You are an AI detector. Analyze the given text and provide percentages for AI-generated and human-written content, along with key indicators and potential sources."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    
    return response.choices[0].message['content']

def plot_repeated_words(text):
    tokens = word_tokenize(text.lower())
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    
    words, counts = zip(*top_words)
    
    fig, ax = plt.subplots()
    ax.bar(words, counts)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 10 Most Repeated Words')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

st.title("AI Content Detector")

languages = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de'
}

user_input = st.text_area("Enter the text you want to analyze:", height=200)

if user_input:
    detected_lang = detect_language(user_input)
    detected_lang_name = next((name for name, code in languages.items() if code == detected_lang), "Other")
    
    st.write(f"Detected language: {detected_lang_name}")
    
    use_detected = st.checkbox("Use detected language", value=True)
    
    if use_detected:
        selected_lang = detected_lang_name
    else:
        selected_lang = st.selectbox("Select input language:", list(languages.keys()))

if st.button("Analyze"):
    if user_input:
        with st.spinner("Analyzing..."):
            lang_code = languages[selected_lang]
            result = detect_ai_content(user_input, lang_code)
            
            perplexity = calculate_perplexity(user_input)
            burstiness = calculate_burstiness(user_input)
            
            st.subheader("Analysis Result:")
            st.write(result)
            
            st.subheader("Additional Metrics:")
            st.write(f"Perplexity Score: {perplexity:.4f}")
            st.write(f"Burstiness Score: {burstiness:.4f}")
            
            # Extract AI-generated percentage
            ai_percentage = float(result.split('\n')[0].split(':')[1].strip().rstrip('%'))
            
            if ai_percentage > 55:
                st.subheader("Most Repeated Words:")
                fig = plot_repeated_words(user_input)
                st.pyplot(fig)
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("This app analyzes text and estimate the likelihood of it being AI-generated or human-written, along with potential sources and key indicators. It supports English, Spanish, French, and German inputs, and can auto-detect the input language.")

st.markdown("---")
st.subheader("Note on Scores and Interpretation:")
st.markdown("""
- **AI-generated and Human-written Percentages**: These percentages indicate the likelihood of the text being AI-generated or human-written. They are complementary and sum to 100%. A higher AI-generated percentage suggests a greater likelihood of AI involvement in creating the text.

- **Perplexity Score**: Ranges from 0 to 1. A lower score indicates more repetitive and predictable text, which might suggest AI generation. A higher score suggests more diverse vocabulary and potentially human-written content.

- **Burstiness Score**: Ranges from 0 to 1. A higher score indicates the presence of "bursty" patterns where certain words or phrases are repeated in clusters, which can be characteristic of human writing. A lower score might suggest more uniformly distributed text, potentially indicating AI generation.

Please note that these scores are indicators and not definitive proof of AI or human authorship. The interpretation should consider the context and nature of the text being analyzed.
""")
