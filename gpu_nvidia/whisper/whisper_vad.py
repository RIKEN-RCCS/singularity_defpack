import os
import argparse
import torch
import torchaudio
from datetime import timedelta
from pyannote.audio import Pipeline
from silero_vad import collect_chunks, get_speech_timestamps, load_silero_vad, read_audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np

from df.enhance import enhance, init_df, load_audio, save_audio

THRESHOLD = 0.3
MIN_SPEECH_DURATION_MS = 250
MIN_SILENCE_DURATION_MS = 100
SPEECH_PAD_MS = 120
CHUNK_LENGTH = 30 # time in second
MODEL_REMOTE = "openai/whisper-large-v3"
MODEL_LOCAL = "./whisper-large-v3-ja-final"

#MODEL = "remote"
MODEL = "local"

TEMPERATURE = 0.2
NUM_BEAMS = 10
MAX_NEW_TOKENS = 440
REPETITION_PENALTY=1.2
NO_REPEAT_NGRAM_SIZE=3

def apply_deepfilternet(audio_file, sample_rate=16000, chunk_duration_sec=600):
    """
    DeepFilterNet noise suppression processing with chunking support for long audio files

    Parameters:
        audio_file (str): Input audio file (.wav)
        sample_rate (int): Target sample rate for resampling (default: 16kHz)
        chunk_duration_sec (int): Chunk duration (in seconds) (default: 600sec. = 10min.)

    Returns:
        str: File path to save the denoised audio
    """
    print("[INFO] Applying DeepFilterNet3 noise suppression with chunking...")
    model, df_state, _ = init_df()

    waveform, _ = load_audio(audio_file, sr=df_state.sr())

    chunk_size = sample_rate * chunk_duration_sec
    total_frames = waveform.shape[1]

    enhanced_chunks = []
    with torch.no_grad():
        for i in range(0, total_frames, chunk_size):
            chunk = waveform[:, i:i + chunk_size].contiguous()
            enhanced_chunk = enhance(model, df_state, chunk)
            enhanced_chunks.append(enhanced_chunk)

    enhanced_audio = torch.cat(enhanced_chunks, dim=-1)

    enhanced_path = "denoised.wav"
    save_audio(enhanced_path, enhanced_audio, df_state.sr())
    return enhanced_path

def remove_silence(audio_file, sampling_rate):
    print("[INFO] Detecting silent segments (Silero VAD)...")
    model = load_silero_vad(onnx=False)
    audio = read_audio(audio_file, sampling_rate=sampling_rate)
    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sampling_rate,
        threshold=THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )
    processed_audio = collect_chunks(speech_timestamps, audio)
    return processed_audio, sampling_rate

def chunk_audio(audio, sample_rate, chunk_length_sec=30):
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    chunk_size = chunk_length_sec * sample_rate
    total_length = audio.shape[-1]
    chunks = []
    for i in range(0, total_length, chunk_size):
        chunk = audio[:, i:i + chunk_size]
        chunks.append((i // sample_rate, (i + chunk.shape[-1]) // sample_rate, chunk))
    return chunks

def assign_speaker_labels(segments, diarization):
    labeled_segments = []
    for seg in segments:
        whisper_start = seg['start']
        whisper_end = seg['end']
        text = seg['text'].strip()
        if not text or text in ["...", "…"]:
            continue
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if whisper_start < turn.end and whisper_end > turn.start:
                labeled_segments.append({
                    "start": whisper_start,
                    "end": whisper_end,
                    "speaker": speaker,
                    "text": text
                })
                break
    return labeled_segments

def main():
    parser = argparse.ArgumentParser(description="Transcription with Whisper + PyAnnote + Silero VAD")
    parser.add_argument("input_audio", help="Input audio file path (e.g., meeting.wav)")
    parser.add_argument("output_text", help="Output text file path (e.g., result.txt)")
    args = parser.parse_args()

    input_audio = args.input_audio
    output_text = args.output_text

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Remove noise
    denoised_audio = apply_deepfilternet(input_audio, sample_rate=16000)

    # Remove silent segment
    processed_waveform, sample_rate = remove_silence(denoised_audio, sampling_rate=16000)
    chunks = chunk_audio(processed_waveform, sample_rate, chunk_length_sec=CHUNK_LENGTH)

    # Diarization
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token=hf_token).to(device)
    print("[INFO] Diarization (PyAnnote)...")
    original_waveform, sr = torchaudio.load(input_audio)
    diarization = pipeline({"waveform": original_waveform, "sample_rate": sr})

    if MODEL == "local":
        # Load Whisper model from local directory
        processor = WhisperProcessor.from_pretrained(MODEL_LOCAL, language="Japanese", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_LOCAL).to(device)
    else:
        # Load Whisper model from Hugging Face
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        processor = WhisperProcessor.from_pretrained(MODEL_REMOTE, use_auth_token=hf_token, language="ja", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_REMOTE, use_auth_token=hf_token).to(device)

    model.eval()

    temperature=TEMPERATURE

    # Transcribe each chunk
    print("[INFO] Diarization (PyAnnote)...")
    segments = []
    for start_sec, end_sec, chunk in chunks:
        print(f"start_sec: {start_sec}")
        inputs = processor(
            chunk.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            language="ja",
            task="transcribe",
        )
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                return_timestamps=True,
                temperature=TEMPERATURE,
                do_sample=True if temperature > 0 else False,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            )

        result = processor.batch_decode(generated_ids, skip_special_tokens=True)
        result = [t.encode("utf-8", errors="ignore").decode("utf-8") for t in result]
        transcription = result[0]

        segments.append({
            "start": start_sec,
            "end": end_sec,
            "text": transcription.strip()
        })

    labeled = assign_speaker_labels(segments, diarization)

    # Output text
    print(f"[INFO] Output text file: {output_text}")
    with open(output_text, "w", encoding="utf-8") as f:
        f.write("# Transcrition\n\n")

        prev_speaker = None
        prev_start = None
        prev_end = None
        buffer = ""

        for seg in labeled:
            start = int(seg['start'])
            end = int(seg['end'])
            speaker = seg['speaker']
            text = seg['text'].strip()

            if speaker == prev_speaker:
                buffer += "\n" + text
                prev_end = end
            else:
                if prev_speaker is not None:
                    s = str(timedelta(seconds=prev_start))
                    e = str(timedelta(seconds=prev_end))
                    f.write(f"#### [{s} - {e}] {prev_speaker}\n")
                    f.write(f"{buffer.strip()}\n\n")

                prev_speaker = speaker
                prev_start = start
                prev_end = end
                buffer = text

        if prev_speaker is not None:
            s = str(timedelta(seconds=prev_start))
            e = str(timedelta(seconds=prev_end))
            f.write(f"#### [{s} - {e}] {prev_speaker}\n")
            f.write(f"{buffer.strip()}\n\n")

    print("[INFO] Finished")

if __name__ == "__main__":
    main()
