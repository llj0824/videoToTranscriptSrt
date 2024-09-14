# Video to Transcript with Diarization

This project provides scripts to generate a transcript with speaker diarization from a video file or a YouTube URL. The result is saved as an SRT (SubRip Subtitle) file, which is commonly used for subtitles.

## Features
- **Speaker Diarization**: Detect and differentiate between multiple speakers in the transcript.
- **Supports Video Files & YouTube Links**: Can process local video files or YouTube URLs.

## Requirements

To install the required dependencies, ensure you have Python 3.x installed, then run:

```bash
pip install -r requirements.txt
```

## Configuration

Before running the script, you need to configure some settings in the `config.ini` file:

- `video_source`: Path to the video file or YouTube URL.
- `output_path`: Path where the SRT file will be saved.
- `language`: Language of the transcript.

## How to Use

1. Clone the repository:
    ```bash
    git clone https://github.com/llj0824/videoToTranscriptSrt
    cd videoToTranscriptSrt
    ```

2. Modify the `config.ini` file to specify the video source and output path.

3. Run the main script to process the video:
    ```bash
    python main.py
    ```

4. The transcript with speaker diarization will be generated and saved as an SRT file in the specified output directory.

### Example Usage

- For a local video file:
  - Set `video_source = /path/to/video.mp4` in `config.ini`.
  - Run `python main.py`.

- For a YouTube URL:
  - Set `video_source = https://youtube.com/example_video` in `config.ini`.
  - Run `python main.py`.

## Files Description

- **main.py**: This script orchestrates the transcription process. It reads the video source, invokes the transcription and diarization engine, and saves the result.
  
- **videoToSrt.py**: This script handles the core functionality of converting the audio into text and creating the SRT file.

## Dependencies

Make sure you have the following dependencies installed:

- `whisper` (for transcription)
- `pytube` (for YouTube download)
- `pydub` (for audio processing)
- Additional packages as listed in `requirements.txt`


## To Run

### Create Virtual Environment
python -m venv env

### Activate Virtual Environment
source env/bin/activate

## Notes

- Ensure your video file is accessible, or the YouTube URL is valid.
- The transcript might take some time depending on the length of the video and complexity of the speaker diarization.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

This README file outlines the basic steps and information needed for users to run the scripts. If there are specific functionalities or nuances in the code, feel free to adjust it accordingly.