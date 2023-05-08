from fastapi import FastAPI, File, UploadFile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
import librosa

app = FastAPI()
tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def stt_pipeline(input_file):
    speech, sample_rate = sf.read(input_file)
    
    # Convert to mono if the audio has multiple channels
    if len(speech.shape) > 1 and speech.shape[1] > 1:
        speech = speech.mean(axis=1)
    
    # Resample the audio to the expected sampling rate
    target_sampling_rate = 16000
    if sample_rate != target_sampling_rate:
        speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=target_sampling_rate)
        sample_rate = target_sampling_rate

    input_values = tokenizer(speech, return_tensors="pt", padding="longest", sampling_rate=sample_rate).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    return {"transcription": stt_pipeline(file.file)}