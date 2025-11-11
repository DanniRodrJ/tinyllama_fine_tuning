import streamlit as st
from src.pipeline import TravelAssistantPipeline
from config.config import TravelAssistantConfig as Config
import os
from src.utils.logger import app_logger


st.set_page_config(page_title="Intelligent Travel Assistant", layout="wide")
st.title("✈️ Intelligent Travel Assistant LLM")
st.subheader("Supervised Fine-Tuning (SFT) with LoRA on TinyLlama-1.1B")

st.sidebar.header("MLOps Configuration")
st.sidebar.markdown(f"""
- **Base Model:** `{Config.MODEL_NAME}`
- **Technique:** LoRA (PEFT)
- **Trained Records:** {Config.TRAIN_RECORDS_LIMIT}
- **Adapter Path:** `./models/tinyllama_travel_adapter`
""")

st.sidebar.markdown("---")
st.sidebar.warning("""
**Language Warning:** This model was fine-tuned exclusively on an English Q&A dataset. 
It performs best on English queries and may generate inconsistent or incorrect text if asked in other languages.
""")
st.sidebar.markdown("---")
st.sidebar.info("💡 Project conceived as a Capstone Challenge on the DataCamp Platform.")

@st.cache_resource
def load_llm_pipeline():
    """Carga y ejecuta el pipeline de inferencia condicionalmente."""
    app_logger.info("Starting pipeline initialization and model loading for Streamlit...")
    try:
        pipeline = TravelAssistantPipeline()
        pipeline.run_or_load(force_train=False) 
        return pipeline
    except Exception as e:
        app_logger.error(f"Critical failure while loading the model: {e}")
        st.error(f"FATAL ERROR: Could not load the model. Check if adapters exist in {Config.ADAPTER_OUTPUT_DIR}. Detail: {e}")
        st.stop()
        
if not os.path.exists(Config.ADAPTER_OUTPUT_DIR):
    st.error("Error: Trained adapters not found. Please run 'python main.py' first or copy the Colab files to the designated 'models/' folder.")
    st.stop()
    
travel_pipeline = load_llm_pipeline()

st.markdown("---")
st.markdown(f"The model is a specialized assistant for **{Config.TRAIN_RECORDS_LIMIT}** travel inquiry types.")

user_input = st.text_area("Enter your Travel Query (in English):", 
                          value="I want to know the weight limit for my carry-on baggage.", 
                          height=100)

if st.button("Generate Response", key="generate_btn", type="primary"):
    if user_input:
        with st.spinner("Processing query with TinyLlama..."):
            try:
                response = travel_pipeline.generate_response(user_input)
                
                st.subheader("Generated Response")
                st.code(response, language='markdown')
            except Exception as e:
                st.error(f"Error during response generation: {e}")
    else:
        st.warning("Please enter a query to proceed.")