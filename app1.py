import streamlit as st
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from collections import Counter
import re

nltk.download('punkt', quiet=True)

@st.cache_resource
def load_models():
    gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')
    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    roberta_model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
    
    return gpt2_model, gpt2_tokenizer, roberta_model, roberta_tokenizer

gpt2_model, gpt2_tokenizer, roberta_model, roberta_tokenizer = load_models()

def calculate_perplexity(text):
    input_ids = gpt2_tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        outputs = gpt2_model(input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss).item()

def get_roberta_score(text):
    inputs = roberta_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)
    return scores[0][1].item()  # Probability of being AI-generated

def calculate_statistical_features(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    avg_sentence_length = np.mean([len(word_tokenize(sentence)) for sentence in sentences])
    unique_words_ratio = len(set(words)) / len(words)
    
    word_freq = Counter(words)
    common_words_ratio = sum(count for word, count in word_freq.most_common(10)) / len(words)
    
    punctuation_ratio = len(re.findall(r'[^\w\s]', text)) / len(words)
    
    return avg_sentence_length, unique_words_ratio, common_words_ratio, punctuation_ratio

def analyze_text(text):
    perplexity = calculate_perplexity(text)
    roberta_score = get_roberta_score(text)
    avg_sent_len, unique_words_ratio, common_words_ratio, punctuation_ratio = calculate_statistical_features(text)
    
    # Normalize and combine scores
    perplexity_score = min(1, max(0, (perplexity - 10) / 100))
    statistical_score = (
        (avg_sent_len / 20) +  # Assume average human sentence length is around 20 words
        (1 - unique_words_ratio) +
        common_words_ratio +
        (1 - punctuation_ratio)
    ) / 4
    
    combined_score = (perplexity_score + roberta_score + statistical_score) / 3
    
    ai_probability = combined_score * 100
    human_probability = (1 - combined_score) * 100

    return ai_probability, human_probability, perplexity, roberta_score, avg_sent_len, unique_words_ratio, common_words_ratio, punctuation_ratio

def display_score_bar(label, value, color):
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 150px; margin-right: 10px;">{label}</div>
            <div style="flex-grow: 1; background-color: #f0f0f0; border-radius: 5px; height: 25px;">
                <div style="width: {value}%; height: 100%; background-color: {color}; border-radius: 5px; text-align: right; padding-right: 5px; color: white;">
                    {value:.1f}%
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.title("AI Content Detector")

user_input = st.text_area("Enter the text you want to analyze:", height=200)

if st.button("Analyze"):
    if user_input:
        with st.spinner("Analyzing..."):
            ai_prob, human_prob, perplexity, roberta_score, avg_sent_len, unique_words_ratio, common_words_ratio, punctuation_ratio = analyze_text(user_input)
            
            st.subheader("Analysis Result:")
            display_score_bar("AI-generated", ai_prob, "#FF4B4B")
            display_score_bar("Human-written", human_prob, "#4BB543")
            
            st.subheader("Detailed Metrics:")
            st.write(f"Perplexity: {perplexity:.2f}")
            st.write(f"RoBERTa AI Detection Score: {roberta_score:.2f}")
            st.write(f"Average Sentence Length: {avg_sent_len:.2f}")
            st.write(f"Unique Words Ratio: {unique_words_ratio:.2f}")
            st.write(f"Common Words Ratio: {common_words_ratio:.2f}")
            st.write(f"Punctuation Ratio: {punctuation_ratio:.2f}")
            
            st.markdown("---")
            st.markdown("""
            **Interpretation:**
            - This ensemble method combines perplexity, a pre-trained AI detector (RoBERTa), and statistical features.
            - Higher AI-generated probability suggests a greater likelihood of AI involvement in creating the text.
            - The detailed metrics provide insights into different aspects of the text's structure and complexity.
            
            Please note that while this method is more comprehensive, it's still not infallible. Always consider the context and use human judgment alongside these automated tools.
            """)
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("""
This app uses an ensemble approach to estimate the likelihood of text being AI-generated or human-written. 
It combines perplexity calculation, a pre-trained AI detector model, and statistical analysis of the text.
This multi-faceted approach aims to provide a more robust assessment than single-method detectors.
""")
