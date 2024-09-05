import subprocess
import json
import os
import yt_dlp
import whisper
import torch
from pyannote.audio import Pipeline
import configparser
import datetime
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Function to load configuration
def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

# Function to download YouTube video as MP3
def download_youtube_audio(youtube_url, output_filename):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_filename,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return f"{output_filename}.mp3"

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path, config):
    """
    Transcribes the audio using faster-whisper with batched inference.
    
    Parameters:
    - audio_path: str, path to the audio file.
    - config: dict, configuration dictionary containing model settings.
    
    Returns:
    - List of segment dictionaries containing start time, end time, and text.
    """
    model_size = config['WHISPER']['model']
    device = config['WHISPER'].get("device", "cpu")  # e.g., "cuda", "cpu"
    compute_type = config['WHISPER'].get("compute_type", "int8_float16")  # e.g., "float16", "int8_float16", "int8"

    # Initialize the Whisper model with the given settings
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # Initialize the batched inference pipeline
    batched_model = BatchedInferencePipeline(model=model)
    
    # Perform the transcription using batched inference
    batch_size = config['WHISPER'].get("batch_size", 16)  # Default batch size is 16
    segments, info = batched_model.transcribe(audio_path, batch_size=batch_size)

    # Convert list of segments into the format that your application needs
    transcriptions = []
    for segment in segments:
        transcriptions.append({
            "start": segment.start,          # Start time of the segment (in seconds)
            "end": segment.end,              # End time of the segment (in seconds)
            "text": segment.text             # Transcribed text for this segment
        })
        # Debug/verbose output if needed:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    return transcriptions

# Function to perform diarization using Pyannote
def perform_diarization(audio_path, config):
    auth_token = config['PYANNOTE']['auth_token']
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token=auth_token)
    diarization = pipeline(audio_path)
    return diarization

# Function to combine transcription and diarization
def combine_transcription_and_diarization(transcription, diarization):
    # This is a simplified version. You may need to adjust based on your specific needs.
    combined_output = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_text = transcription[int(turn.start * 1000):int(turn.end * 1000)]
        combined_output.append(f"[{turn.start:.2f} - {turn.end:.2f}] {speaker}: {segment_text}")
    return "\n".join(combined_output)

def main():
    config = load_config()

    input_type = input("Enter input type, [1] for YouTube URL, [2] for filepath: ").strip().lower()
    diarization_input = input("Enable diarization for multiple speakers? [y/n]: ").strip().lower()

    if input_type == "1":
        url = input("Enter URL: ").strip()
        filename = input("Enter filename: ").strip()
        output_filename, _ = os.path.splitext(filename)
        audio_path = download_youtube_audio(url, output_filename)
    elif input_type == "2":
        audio_path = input("Enter filepath: ").strip()
        output_filename, _ = os.path.splitext(os.path.basename(audio_path))
    else:
        print("Invalid input type.")
        return

    # Transcribe audio
    transcription = transcribe_audio(audio_path, config)

    if diarization_input == 'y':
        # Perform diarization
        diarization = perform_diarization(audio_path, config)
        # Combine transcription and diarization
        final_output = combine_transcription_and_diarization(transcription, diarization)
    else:
        final_output = transcription

    # Format the timestamp as "AbbrOfMonth/Day/Year_HHMMSS"
    now = datetime.datetime.now()
    timestamp = now.strftime("%b%d%Y_%H%M%S")  # Example: Sep052024_150450

    # Write output to file
    output_filepath = f"transcripts/{output_filename}_{timestamp}.txt"
    with open(output_filepath, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"Transcription completed. Output saved to {output_filepath}")

if __name__ == "__main__":
    main()