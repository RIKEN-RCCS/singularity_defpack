# Transcribe

## Extract audio file

Since `whisper` supports only WAV format, extract the WAV audio using `ffmpeg` from meeting videos in MP4 format.
To improve speech recognition accuracy, the audio is converted to mono, resampled to 16 kHz, and a 1 kHz high-pass filter is applied. Additionally, the audio sample format is converted to 16-bit PCM.

```bash
ffmpeg -y -i $INPUT -ac 1 -ar 16000 -vn -af "highpass=f=1000" -sample_fmt s16 $OUTPUT
```

Here, `$INPUT` is the input file, and `$OUTPUT` is the output file.
Make sure the output file has the ".wav" extension.

## Transcription script

This script performs speech transcription.
First, to improve transcription accuracy, noise and silence segments are removed.
`DeepFilterNet` is used for noise suppression, and `Silero VAD` is used to remove silent segments.
Next, speaker diarization is performed using `pyannote.audio`.
Finally, transcription is performed using the `OpenAI Whisper` model.

To use a model downloaded from Hugging Face, set the variable `MODEL` to `remote`.  
To use a locally fine-tuned model, set `MODEL` to `local`.

```python
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
```

# [Option] Cleanse transcribed text by LLM

Using the method introduced in [this web page](https://qiita.com/hann-solo/items/e51892ca485fca99a184), you can post-process transcribed text from Whisper with an LLM to improve its quality.  
The key components are the **"dictionary"** and the **"prompt"**.  
Prepare a dictionary containing technical terms specific to your area of expertise in advance, using a file named `dictionary.md`.
By incorporating this dictionary into the system prompt and providing appropriate summarization instructions through the user prompt, you can generate high-quality summaries even from transcription text with low accuracy.

```python
import os
import argparse
from llama_cpp import Llama
from pathlib import Path
import re

BASE_PATH = "/home/users/hikaru.inoue/.cache/llama.cpp/"
MODEL_PATH = "unsloth_gemma-3-27b-it-GGUF_gemma-3-27b-it-Q4_0.gguf"
LLM_MODEL_PATH = BASE_PATH + MODEL_PATH
TEMPERATURE = 0.1
N_CTX = 131072
MAX_TOKENS = 131072
N_GPU_LAYERS = -1

DICTIONARY_PATH = "dictionary.md"

SYSTEM_PROMPT_TEMPLATE = """
## 役割：
あなたは「質の荒いトランスクリプトを一発で整える熟練の議事録職人」である。
添付のトランスクリプトと用語集を用いて、**Markdown形式の発言記録風議事録**を作成せよ。
以下の要件を厳守すること。

【入力ファイル】
- 用語集：{dictionary_content}

【出力要件】
- 議論の内容を分析して、**必要なだけのトピックに分割してください（数に制限はありません）**。
- Markdownのトップレベル見出しは「##」、二階層目は「###」。
- 構成は必ずトピックごとに下記の順序とラベルを用いる。
- トピック数は会話内容に応じて、1つでも5つでも20以上でも構いません。

【出力フォーマット例】
## 【トピック1】
（このトピックの要点を3〜6行程度の箇条書き）
### 【議論内容】
- 発言者A
  - ○○に関して説明。
- 発言者B
  - その点について意見。

## 【トピック2】
（このトピックの要点を3〜6行程度の箇条書き）
### 【議論内容】
- 発言者A
  - ××について質問。
- 発言者C
  - それに対する回答。

...

**トピックは必要なだけ、適切に続けてください**

【書式詳細】
- 「議論内容」では発言者を一次リスト、内容を二次リストで表現する。
  - 例：
    - 進行役
      - 本日のアジェンダですが…
    - 田中さん
      - その件について質問があります。
- リストは * ではなく **-** を使用し、インデントは半角スペース2つ。
- 太字は必要最小限（固有名詞・キーワードなど）とする。過剰装飾は禁止。
- 文体は**だ・である調**かつ**体言止め**を交えて簡潔に。
- 言い淀み（例：「あのー」「えっと」）、重複語、無意味な相槌は削除し、自然で読みやすい日本語に整形する。
- トランスクリプトに存在しない情報を加筆しない。
- 用語集の内容をそのまま引用しない。
- 進行役やファシリテーターの発言は要点を簡潔にまとめ、主要な発言者やゲストの発言は丁寧かつ網羅的に記録する。

【品質基準】
- ビジネス文書としてチーム内で共有可能なレベルを目指す。手作業による修正は、全体の2割程度に収まることが望ましい。
- 見出しの粒度や順序は、会話の流れと重要性を最もよく反映するように、あなたが判断して最適化してよい。

【備考（このセクションはユーザーが適宜調整する）】
- 発言者の名称（例：進行役、田中さん）は、トランスクリプトからLLMが識別できる場合はそれを、不明な場合は「発言者A」「発言者B」などの一般的な呼称を使用するよう指示します。
- チームやプロジェクト特有のルールがあれば、【書式詳細】や【品質基準】に追記してください。

出力は以上の条件を満たすMarkdown形式の出力のみとする。
"""

TOKEN_LIMIT = int(MAX_TOKENS * 0.7)

def estimate_tokens(text):
    encoded = llm.tokenize(text.encode("utf-8"))
    return len(encoded)

def correct_batch(blocks, dictionary_content):
    combined_text = "\n".join(f"{header}\n{body}" for header, body in blocks)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(dictionary_content=dictionary_content)
    prompt = system_prompt + "\n## 元の文章:\n" + combined_text.strip() + "\n\n## 修正した文章:\n"
    res = llm(prompt=prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    return res["choices"][0]["text"].strip()

llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=32,
    n_gpu_layers=N_GPU_LAYERS,
    use_mlock=False,
    use_mmap=True,
    verbose=False,
)

def parse_blocks(text: str):
    pattern = r"(####\s*\[\d{1,2}:\d{2}:\d{2} - \d{1,2}:\d{2}:\d{2}\]\s*SPEAKER_\d{1,2})\n(.*?)\n(?=####|\Z)"
    return re.findall(pattern, text, flags=re.DOTALL)

def main():
    parser = argparse.ArgumentParser(description="Cleansing text generated by Whisper")
    parser.add_argument("input_text", help="Path of input text file (e.g., meeting.txt)")
    parser.add_argument("output_text", help="Path of output text file (e.g., result.txt)")
    parser.add_argument("--dictionary", default=DICTIONARY_PATH, help="Path to dictionary file (default: dictionary.md)")
    parser.add_argument("--token_limit", type=int, default=TOKEN_LIMIT, help="Max tokens per batch (default: 70% of MAX_TOKENS)")
    args = parser.parse_args()

    input_text = Path(args.input_text).read_text(encoding='utf-8')
    dictionary_content = Path(args.dictionary).read_text(encoding='utf-8')
    system_prompt_text = SYSTEM_PROMPT_TEMPLATE.format(dictionary_content=dictionary_content)
    prompt_tokens = estimate_tokens(system_prompt_text)
    print(f"Prompt Tokes: {prompt_tokens}")

    blocks = parse_blocks(input_text)
    print(f"Total blocks: {len(blocks)}")

    output_lines = []
    current_batch = []
    current_tokens = prompt_tokens

    for i, (header, body) in enumerate(blocks):
        block_text = f"{header}\n{body}\n"
        block_tokens = estimate_tokens(block_text)
        print(f"[{i}] Block Tokes: {block_tokens}")

        if current_tokens + block_tokens > args.token_limit:
            print(f"[{i+1}] Processing batch with {len(current_batch)} blocks, tokens: {current_tokens}")
            corrected = correct_batch(current_batch, dictionary_content)
            output_lines.append(corrected + "\n")
            current_batch = []
            current_tokens = prompt_tokens

        current_batch.append((header, body))
        current_tokens += block_tokens

    if current_batch:
        print(f"Processing final batch with {len(current_batch)} blocks, tokens: {current_tokens}")
        corrected = correct_batch(current_batch, dictionary_content)
        output_lines.append(corrected + "\n")

    Path(args.output_text).write_text("\n".join(output_lines), encoding='utf-8')
    print(f"Completed: {args.output_text}")

if __name__ == "__main__":
    main()
```

# Fine-tuning whisper model

## Prepare dataset

For fine-tuning, a dataset consisting of pairs of audio and corresponding text is prepared, with the audio file names and text listed in `metadata.csv`.

```
file,sentence
custom_dataset/audio-1.wav,こんにちは、わたしは理研で働いています
custom_dataset/ausio-2.wav,富岳の運用技術の仕事をしています
```

Initially, meeting recordings were used, but it was observed that fine-tuning with poor-quality audio degraded transcription accuracy.
Therefore, text-to-speech tools such as [VOICEVOX](https://voicevox.hiroshiba.jp/) were used to synthesize clean audio from text, and this synthetic data was used for fine-tuning.

## Fine-tuning script

This script fine-tunes the `whisper-large-v3` model.
Empirically, excessive training tends to degrade transcription accuracy.
Therefore, it is recommended to limit `max_steps` to around 40.

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Audio
from dataclasses import dataclass

DEBUG = 1

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="ja", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

dataset = load_dataset("csv", data_files="./custom_dataset/metadata.csv", split="train")
dataset = dataset.cast_column("file", Audio(sampling_rate=16000))

split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

def preprocess(batch):
    audio = batch["file"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

train_dataset = train_dataset.map(preprocess)
eval_dataset = eval_dataset.map(preprocess)

@dataclass
class MyCollator:
    processor: WhisperProcessor
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = self.processor.tokenizer.pad(label_features, return_tensors="pt")["input_ids"]
        batch["labels"] = labels
        return batch

import evaluate
metric = evaluate.load("wer")

def compute_metrics(pred):
    import spacy
    import ginza
    nlp = spacy.load("ja_ginza")
    ginza.set_split_mode(nlp, "C")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [" ".join([str(i) for i in nlp(j)]) for j in pred_str]
    label_str = [" ".join([str(i) for i in nlp(j)]) for j in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

if DEBUG == 1:
    WARMUP_STEPS = 5
    MAX_STEPS = 40
    SAVE_STEPS = 10
    EVAL_STEPS = 10
else:
    WARMUP_STEPS = 50
    MAX_STEPS = 400
    SAVE_STEPS = 100
    EVAL_STEPS = 100

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-v3-ja",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=False,
    #save_strategy="epoch",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=10,
    report_to=["tensorboard"],
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=MyCollator(processor),
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer,
)

trainer.train()

trainer.save_model("./whisper-large-v3-ja-final")
processor.save_pretrained("./whisper-large-v3-ja-final")
```
