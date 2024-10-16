import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "roberta-base-openai-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def predict_ai_content(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    ai_probability = probabilities[0][1].item()
    return ai_probability

def main():
    st.set_page_config(page_title="LLM-based AI Content Detector", page_icon="🤖", layout="wide")
    
    st.title("LLM-based AI Content Detector")
    
    st.markdown("""
    This app uses a fine-tuned RoBERTa model to detect AI-generated content.
    Enter your text below and click 'Analyze' to get the results.
    """)
    
    tokenizer, model = load_model()
    
    text = st.text_area("Enter your text:", height=200)
    
    if st.button("Analyze", type="primary"):
        if text:
            with st.spinner("Analyzing text..."):
                ai_probability = predict_ai_content(text, tokenizer, model)
            
            st.subheader("Analysis Result:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("AI-Generated Probability", f"{ai_probability:.2%}")
                st.progress(ai_probability)
            
            with col2:
                if ai_probability < 0.3:
                    st.success("✅ The text is likely human-written.")
                elif ai_probability > 0.7:
                    st.error("🤖 The text is likely AI-generated.")
                else:
                    st.warning("⚠️ The results are inconclusive.")
            
            st.info("""
            Note: 
            - This model is based on a fine-tuned RoBERTa model trained to distinguish between human and AI-generated text.
            - While generally accurate, it may not catch the latest AI text generation techniques.
            - Always use critical thinking alongside automated detection tools.
            """)
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()