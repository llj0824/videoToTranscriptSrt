import subprocess
import json
import os
import yt_dlp
import whisper
import torch
from pyannote.audio import Pipeline
import configparser

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
    model_name = config['WHISPER']['model']
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path,verbose=True)
    return result["text"]

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

    # Write output to file
    output_filepath = f"transcripts/{output_filename}.txt"
    with open(output_filepath, 'w') as f:
        f.write(final_output)

    print(f"Transcription completed. Output saved to {output_filepath}")

if __name__ == "__main__":
    main()