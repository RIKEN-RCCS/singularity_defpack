import os
import argparse
import torch
import torchaudio
from pyannote.audio import Pipeline
from datetime import timedelta
import whisper

parser = argparse.ArgumentParser(description="Whisper + PyAnnote による話者付き文字起こし")
parser.add_argument("input_audio", help="Input audio file path (e.g., meeting.wav)")
parser.add_argument("output_text", help="Output text file path (e.g., result.txt)")
args = parser.parse_args()

input_audio = args.input_audio
output_text = args.output_text

hf_token = os.getenv("HUGGING_FACE_TOKEN")

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
	use_auth_token=hf_token,
).to(torch.device("cuda"))

waveform, sample_rate = torchaudio.load(input_audio)
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

model = whisper.load_model("large-v3-turbo").to(torch.device("cuda"))
result = model.transcribe(
    input_audio,
    verbose=False,
    condition_on_previous_text=True,
    temperature=0.0,
    beam_size=20,
    best_of=5
)
segments = result['segments']

def assign_speaker_labels(segments, diarization):
    labeled_segments = []
    for seg in segments:
        whisper_start = seg['start']
        whisper_end = seg['end']
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if whisper_start < turn.end and whisper_end > turn.start:
                labeled_segments.append({
                    "start": whisper_start,
                    "end": whisper_end,
                    "speaker": speaker,
                    "text": seg['text']
                })
                break
    return labeled_segments

labeled = assign_speaker_labels(segments, diarization)

with open(output_text, "w", encoding="utf-8") as f:
    f.write("## Diarized text\n\n")

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
            # concatenate speak if same speaker
            buffer += "\n" + text
            prev_end = end
        else:
            # output speak when speaker changed
            if prev_speaker is not None:
                s = str(timedelta(seconds=prev_start))
                e = str(timedelta(seconds=prev_end))
                f.write(f"#### [{s} - {e}] {prev_speaker}\n")
                f.write(f"{buffer.strip()}\n\n")

            # Save latest speak as buffer
            prev_speaker = speaker
            prev_start = start
            prev_end = end
            buffer = text

    # Output last speak
    if prev_speaker is not None:
        s = str(timedelta(seconds=prev_start))
        e = str(timedelta(seconds=prev_end))
        f.write(f"#### [{s} - {e}] {prev_speaker}\n")
        f.write(f"{buffer.strip()}\n\n")
