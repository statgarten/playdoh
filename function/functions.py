import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def load_model(model_name):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model

def preprocess_audio(file):
    audio_input, _ = sf.read(file)
    
    # Convert stereo audio to mono if necessary
    if audio_input.ndim == 2:
        audio_input = audio_input.mean(axis=1)

    return audio_input

def convert_audio_to_text(audio_input, processor, model):
    input_values = processor(audio_input, return_tensors="pt", padding=True, sampling_rate=16000).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription
