import streamlit as st
import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Set up page configurations
st.set_page_config(page_title="தமிழ் Speech to Text", page_icon="🎙️", layout="wide")
st.title("🎙️ தமிழ் Speech to Text")
st.markdown("IIT Madras fine-tuned Whisper model for accurate Tamil transcriptions.")

# 1. Initialize hardware configuration
# Streamlit Cloud's free tier is CPU-only, so we optimize for CPU processing
device = "cpu"
torch_dtype = torch.float32

# 2. Load model and processor (Cached so it only loads once on startup)
@st.cache_resource
def load_speech_pipeline():
    model_id = "vasista22/whisper-tamil-large-v2"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,  # Splits long audio into manageable chunks
        batch_size=4,       # Low batch size to prevent running out of RAM on Streamlit Cloud
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

# Safely initialize the pipeline
try:
    with st.spinner("Loading AI Model into memory... This may take a couple of minutes on first boot."):
        pipe = load_speech_pipeline()
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. Create the UI Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Tamil Audio Input")
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "ogg"])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        submit_btn = st.button("Transcribe / மொழியாக்கம் செய்", type="primary", use_container_width=True)

with col2:
    st.subheader("Transcription Result / தமிழ் உரை")
    if uploaded_file is not None and 'submit_btn' in locals() and submit_btn:
        with st.spinner("Processing audio... Please wait."):
            temp_filename = "temp_audio_file"
            try:
                # Save uploaded file bytes to a temporary file
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Run the pipeline explicitly forcing Tamil text output
                result = pipe(
                    temp_filename, generate_kwargs={"language": "tamil", "task": "transcribe"}
                )

                # Display final transcription output
                st.text_area(
                    label="Transcribed Text",
                    value=result["text"],
                    height=300,
                    label_visibility="collapsed"
                )

                # Add a quick download button for the text output
                st.download_button(
                    label="Download Text File / பதிவிறக்கம் செய்",
                    data=result["text"],
                    file_name="tamil_transcription.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")
            finally:
                # Clean up temporary audio file from server memory
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
    else:
        st.info("Upload an audio file on the left and click 'Transcribe' to view the text output here.")
