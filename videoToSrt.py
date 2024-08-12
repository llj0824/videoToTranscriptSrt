import whisperx
import configparser

# Function to load configuration
def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

config = load_config()

# Load audio file (change this to your MP3 file path)
audio_file = input("Please Input mp3 filepath: ").strip()

# Load WhisperX model
model_name = config['WHISPER']['model']
model = whisperx.load_model(model_name, device="cpu")  # Use "cpu" if you don't have a GPU


# Transcribe and align the audio
transcription = model.transcribe(audio_file)

# Align with word-level timestamps
aligned_transcription = whisperx.align(transcription["segments"], transcription["text"], audio_file)

# Save the transcription as an SRT file. 
# Generate the output file path by replacing the extension with '.srt'
output_file = os.path.splitext(audio_file)[0] + ".srt"

with open(output_file, "w") as srt_file:
    for segment in aligned_transcription["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        # Convert time to SRT format (hours:minutes:seconds,milliseconds)
        start_time_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
        end_time_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"

        # Write to SRT file
        srt_file.write(f"{segment['id']}\n")
        srt_file.write(f"{start_time_srt} --> {end_time_srt}\n")
        srt_file.write(f"{text}\n\n")