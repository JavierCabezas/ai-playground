from transformers import pipeline

WAV_FILE = 'file.wav'

def extract_lyrics_from_audio(wav_file):
    # Initialize the ASR pipeline with a suitable model for Spanish and enable return_timestamps
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        return_timestamps=True
    )

    with open(wav_file, "rb") as f:
        audio_bytes = f.read()

    # Get the transcription
    result = asr_pipeline(audio_bytes)
    return result["text"], result["chunks"]


def save_as_srt(chunks, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            # Assuming each chunk is a dictionary with 'timestamp' field containing (start, end) tuple
            start_time, end_time = chunk["timestamp"]
            text = chunk["text"]

            # Format timestamps to SRT format
            start_srt = format_timestamp(start_time)
            end_srt = format_timestamp(end_time)

            # Write SRT entry
            f.write(f"{i + 1}\n")
            f.write(f"{start_srt} --> {end_srt}\n")
            f.write(f"{text}\n\n")


def format_timestamp(seconds):
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


if __name__ == "__main__":
    wav_file = WAV_FILE
    output_file = "subtitles.srt"

    # Extract lyrics and chunks
    lyrics, chunks = extract_lyrics_from_audio(wav_file)

    # Save transcription as SRT file
    save_as_srt(chunks, output_file)
    print(f"Subtitles saved to {output_file}")
