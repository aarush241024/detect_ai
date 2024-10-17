import streamlit as st
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt_tab')
# Set page config at the very beginning
st.set_page_config(page_title="Advanced AI Content Detector", page_icon="🤖", layout="wide")

# Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

# Load RoBERTa model and tokenizer
@st.cache_resource
def load_roberta_model():
    model_name = "roberta-base-openai-detector"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

class SimpleFeatureExtractor:
    def extract_features(self, text):
        words = text.lower().split()
        return {
            'word_count': len(words),
            'unique_word_count': len(set(words)),
            'average_word_length': np.mean([len(word) for word in words]) if words else 0,
        }

class SimpleStyleAnalyzer:
    def analyze_style(self, text):
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        return {
            'sentence_count': len(sentences),
            'average_sentence_length': np.mean([len(word_tokenize(sentence)) for sentence in sentences]) if sentences else 0,
            'vocabulary_richness': len(set(words)) / len(words) if words else 0,
        }

class SimpleAITextDetector:
    def __init__(self):
        self.feature_extractor = SimpleFeatureExtractor()
        self.style_analyzer = SimpleStyleAnalyzer()

    def detect(self, text):
        features = {}
        features.update(self.feature_extractor.extract_features(text))
        features.update(self.style_analyzer.analyze_style(text))
        
        # Simple heuristic
        score = (features['unique_word_count'] / features['word_count'] if features['word_count'] > 0 else 0) * 100
        
        is_ai_generated = score < 60  # Arbitrary threshold
        
        return {
            'is_ai_generated': is_ai_generated,
            'confidence': abs(score - 60) / 60,  # Simple confidence measure
            'score': score,
            'features': features
        }

def predict_roberta(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    ai_probability = probabilities[0][1].item()
    return ai_probability

def main():
    st.title("Advanced AI Content Detector")
    
    st.markdown("""
    This app uses two different approaches to detect AI-generated content:
    1. A simple heuristic-based detector
    2. A fine-tuned RoBERTa model
    
    Enter your text below and click 'Analyze' to get the results from both methods.
    """)
    
    tokenizer, model = load_roberta_model()
    simple_detector = SimpleAITextDetector()
    
    text = st.text_area("Enter your text:", height=200)
    
    if st.button("Analyze", type="primary"):
        if text:
            with st.spinner("Analyzing text..."):
                simple_result = simple_detector.detect(text)
                roberta_probability = predict_roberta(text, tokenizer, model)
            
            st.subheader("Analysis Results:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Simple Heuristic Detector")
                st.metric("AI-Generated Score", f"{simple_result['score']:.2f}")
                st.progress(simple_result['score'] / 100)
                
                if simple_result['is_ai_generated']:
                    st.error("🤖 The text is likely AI-generated.")
                else:
                    st.success("✅ The text is likely human-written.")
                
                st.markdown("#### Features:")
                for feature, value in simple_result['features'].items():
                    st.text(f"{feature}: {value:.2f}")
            
            with col2:
                st.markdown("### RoBERTa Model Detector")
                st.metric("AI-Generated Probability", f"{roberta_probability:.2%}")
                st.progress(roberta_probability)
                
                if roberta_probability < 0.3:
                    st.success("✅ The text is likely human-written.")
                elif roberta_probability > 0.7:
                    st.error("🤖 The text is likely AI-generated.")
                else:
                    st.warning("⚠️ The results are inconclusive.")
            
            st.info("""
            Note: 
            - The simple heuristic detector uses basic text features and may not be as accurate as the RoBERTa model.
            - The RoBERTa model is based on a fine-tuned version trained to distinguish between human and AI-generated text.
            - While generally accurate, these methods may not catch the latest AI text generation techniques.
            - Always use critical thinking alongside automated detection tools.
            """)
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
