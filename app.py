import streamlit as st
import whisper
from pydub import AudioSegment
import tempfile
import os
import torch

# Load Whisper model (choose 'base' for fast processing or 'large' for better accuracy)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(DEVICE)

def convert_audio_to_wav(audio_path, input_format):
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            audio = AudioSegment.from_file(audio_path, format=input_format)
            audio = audio.set_frame_rate(16000).set_channels(1)  
            audio.export(temp_wav.name, format="wav")
            return temp_wav.name  
    except Exception as e:
        return f"Error converting audio: {e}"

def speech_to_text(audio_path):
   
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error processing audio: {e}"

def main():
    st.title("🎙️ Speech to Text Converter")
    st.write("Upload an audio file and convert it to text..")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg"])

    if uploaded_file is not None:
        st.write(f"📂 **Uploaded File:** {uploaded_file.name}")

        # Save uploaded file temporarily
        file_extension = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path = temp_audio.name

        #  Convert audio to WAV if necessary
        if file_extension != "wav":
            st.write("🔄 Converting to WAV format...")
            temp_audio_path = convert_audio_to_wav(temp_audio_path, file_extension)

        # Convert speech to text
        st.write("🎧 Processing audio .....")
        text = speech_to_text(temp_audio_path)

        #  Display result
        st.subheader("📝 Converted Text:")
        st.write(text)

        #  Cleanup temporary files
        os.remove(temp_audio_path)

if __name__ == "__main__":
    main()
