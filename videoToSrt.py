import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import configparser
from datetime import datetime

# Function to load configuration
def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

config = load_config()

# Load audio file (change this to your MP3 file path)
audio_file = input("Please input mp3 filepath: ").strip()

# Set device and data type
device = 0 if torch.cuda.is_available() else -1  # -1 for CPU
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model ID from config
model_id = config['WHISPER']['model']  # Ensure this matches the desired model

# Initialize processor and model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    # Removed device_map and low_cpu_mem_usage
)

if torch.cuda.is_available():
    model.to(torch.device('cuda'))
else:
    model.to(torch.device('cpu'))

# Initialize the pipeline without the language parameter
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,  # Adjust based on your device
    device=device
)
# Use the pipeline with the generate_kwargs parameter for translation
# result = asr_pipeline(audio_file) # NOTE: Use if do not need translation, will have timestamps.
result = asr_pipeline(audio_file, generate_kwargs={"task": "translate"}) # NOTE: the translation is just so off.

# Get current timestamp in MM_DD_HHMM format
timestamp = datetime.now().strftime("%m_%d_%H%M")

# Save the transcription as an SRT file with timestamp
output_file = f"{os.path.splitext(audio_file)[0]}_{timestamp}.srt"

try:
    segments = result.get("chunks", [])  # Adjust based on the actual output structure

    if not segments:
        raise ValueError("No segments with timestamps found.")

    with open(output_file, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(segments, start=1):
            # Access the start and end times from the 'timestamp' tuple
            start_time, end_time = segment['timestamp']
            text = segment['text'].strip()

            # Convert time to SRT format (hours:minutes:seconds,milliseconds)
            start_time_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
            end_time_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"

            # Write to SRT file
            srt_file.write(f"{i}\n")
            srt_file.write(f"{start_time_srt} --> {end_time_srt}\n")
            srt_file.write(f"{text}\n\n")

except ValueError as e:
    error_message = f"Error: {str(e)}"
    print(error_message)
    with open(output_file, "w", encoding="utf-8") as srt_file:
        srt_file.write(f"{error_message}\n\n")
        srt_file.write("Full result:\n")
        srt_file.write(str(result))