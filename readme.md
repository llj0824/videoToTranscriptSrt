# Video to Transcript with Diarization

This project allows users to generate a transcript from a video file or a YouTube URL, with optional speaker annotation. The transcript is saved as an SRT or text file, which is widely used for subtitles.

## Features
- **Speaker annotation (Diarization)**: Supports multiple speakers and diarization for accurate transcription.
- **Video and YouTube Support**: Process local video files or YouTube URLs for transcription.
- **FFmpeg Integration**: Automatically converts MP4 files to MP3 for audio extraction.

## Requirements

To get started, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have `ffmpeg` installed on your system. For macOS, you can install it via Homebrew:

```bash
brew install ffmpeg
```

## Configuration

Before running the script, update the `config.ini` file to include your credentials:

```ini
[WHISPER]
model = distil-large-v3

[PYANNOTE]
auth_token = YOUR_HUGGING_FACE_TOKEN

[OPENAI]
api_key = YOUR_OPENAI_API_KEY
```

- **WHISPER `model`**: Specifies the model used for transcription.
- **PYANNOTE `auth_token`**: Replace with your Hugging Face token to enable speaker diarization.
- **OPENAI `api_key`**: Add your OpenAI API key for Whisper integration.

## How to Use

1. **Run the script**:
    ```bash
    python main.py
    ```

2. **Input type**: 
    - You will be prompted to choose between a YouTube URL (`1`) or a local file (`2`).
    - If choosing a file, provide the path to an MP4 or MP3 file.

3. **Diarization**: 
    - Choose `y` or `n` to enable or disable speaker diarization.

4. **Output**:
    - The transcription will be saved in the `transcripts/` folder with a timestamped filename.

## Example Usage

- For a YouTube URL:
  ```bash
  python main.py
  Enter input type, [1] for YouTube URL, [2] for filepath: 1
  Enter URL: https://youtube.com/example
  Enter filename: example_transcript
  Enable diarization for multiple speakers? [y/n]: n
  ```

- For a local MP4 file:
  ```bash
  python main.py
  Enter input type, [1] for YouTube URL, [2] for filepath: 2
  Enter filepath: /path/to/video.mp4
  Enable diarization for multiple speakers? [y/n]: y
  ```

## Output Example

Below is a sample output from a transcription run:

```json
[
  {
    "start": 6.411,
    "end": 33.459,
    "text": " Everybody's asleep, man, or getting home after a long night..."
  },
  {
    "start": 33.883,
    "end": 58.316,
    "text": " What's not always open is the opportunity to check the box in life..."
  },
  {
    "start": 58.774,
    "end": 88.303,
    "text": " Everyone else is probably getting home from some party right now..."
  },
  ...
]
```

The full transcription can be found in the `transcripts/` folder. For example, `transcripts/{input_filename}_{timestamp}.txt` 

## Notes
- The script converts MP4 files to MP3 for audio extraction using `ffmpeg`.
- Ensure your Hugging Face and OpenAI API keys are correct in `config.ini` to enable full functionality.
- The output is saved in the `transcripts/` directory in a timestamped text file.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

This README now includes the correct output format, demonstrating the structure and content users can expect.