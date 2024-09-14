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
import logging

# Function to load configuration
def load_config():
    # Set up logging
    logging.basicConfig()
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

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

# Function to convert MP4 to MP3
def convert_mp4_to_mp3(input_file):
    output_file = os.path.splitext(input_file)[0] + ".mp3"
    command = f"ffmpeg -i '{input_file}' -vn -acodec libmp3lame -q:a 2 '{output_file}'"
    result = os.system(command)
    if result == 0:
        print(f"Successfully converted {input_file} to {output_file}")
        return output_file
    else:
        print(f"Error converting {input_file} to MP3")
        return None

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
    # https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html#ctranslate2.models.Whisper.compute_type
    compute_type = "float32"

    # Initialize the Whisper model with the given settings
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # Initialize the batched inference pipeline
    batched_model = BatchedInferencePipeline(model=model)
    
    # Perform the transcription using batched inference
    batch_size = config['WHISPER'].get("batch_size", 16)  # Default batch size is 16
    segments, info = batched_model.transcribe(audio_path, batch_size=batch_size, log_progress=True)

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


# Assume transcribe_audio and perform_diarization are already defined elsewhere
# Helper function to format transcription into SRT format
def convert_to_srt(transcription):
    srt_output = []
    for i, entry in enumerate(transcription):
        start_time = format_timestamp(entry["start"])
        end_time = format_timestamp(entry["end"])
        text = entry["text"]
        srt_output.append(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n")
    return "\n".join(srt_output)

# Helper function to format seconds into SRT timestamp format
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

# Updated main function
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
        filepath = input("Enter filepath: ").strip()
        file_extension = os.path.splitext(filepath)[1].lower()
        
        if file_extension == '.mp4':
            print("Converting MP4 to MP3...")
            filepath = convert_mp4_to_mp3(filepath)
            if filepath is None:
                print("Conversion failed. Exiting.")
                return
        elif file_extension != '.mp3':
            print("Unsupported file format. Please provide an MP3 or MP4 file.")
            return
        
        # Extract filename without extension
        audio_path = filepath
        output_filename, _ = os.path.splitext(os.path.basename(filepath))
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
    timestamp = now.strftime("%b%d%Y_%H%M%S")  # Example: Sep142024_161331

    # Write output to TXT file
    txt_output_filepath = f"transcripts/{output_filename}_{timestamp}.txt"
    with open(txt_output_filepath, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"Transcription completed. TXT output saved to {txt_output_filepath}")

    # Convert transcription to SRT
    srt_output = convert_to_srt(final_output)

    # Write output to SRT file
    srt_output_filepath = f"transcripts/{output_filename}_{timestamp}.srt"
    with open(srt_output_filepath, 'w') as f:
        f.write(srt_output)
    print(f"SRT output saved to {srt_output_filepath}")

    return txt_output_filepath, srt_output_filepath
    
if __name__ == "__main__":
    main()