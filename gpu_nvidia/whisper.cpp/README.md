# Setup

## Downloading the Model

Download the `whisper.cpp` model from HuggingFace and place it in `$HOME/whisper.cpp/models`.

```bash
wget https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-ggml/resolve/main/ggml-large-v3-turbo.bin -P $HOME/whisper.cpp/models
```

## Preparing the Audio File

Since `whisper.cpp` supports only WAV format, extract the WAV audio using `ffmpeg` from meeting videos in MP4 format.
Use the `ffmpeg` installed in the container.

```bash
singularity run --nv $SIFFILE ffmpeg -i $INPUT $OUTPUT
```

Here, `$SIFFILE` is the container file, `$INPUT` is the input file, and `$OUTPUT` is the output file.
Make sure the output file has the ".wav" extension.

## Preparing the Script

The output from `whisper.cpp` often contains incorrect or overly fragmented line breaks.
To improve the formatting, use spaCy to adjust line break positions.
Note, however, that the adjustment is not perfect.
Save the script below as `whisper_flow.py`.

```python
import spacy
import re

nlp = spacy.load("ja_ginza")

def split_by_byte_limit(text, byte_limit=49000):
    chunks = []
    current = ""
    for char in text:
        current_bytes = current.encode("utf-8")
        next_bytes = char.encode("utf-8")
        if len(current_bytes) + len(next_bytes) > byte_limit:
            chunks.append(current)
            current = char
        else:
            current += char
    if current:
        chunks.append(current)
    return chunks

def reflow_japanese(text):
    text = re.sub(r"(?<!\n)\n(?!\n)", "", text)

    chunks = split_by_byte_limit(text)
    sentences = []

    for chunk in chunks:
        doc = nlp(chunk)
        for sent in doc.sents:
            sentence = str(sent).strip()
            if sentence:
                sentences.append(sentence)

    return "\n".join(sentences)

if __name__ == "__main__":
    import sys
    from pathlib import Path

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".reflow.txt")

    text = input_path.read_text(encoding="utf-8")
    result = reflow_japanese(text)
    output_path.write_text(result, encoding="utf-8")
```

# Running Transcription

Use `whisper.cpp` to transcribe the audio, adjust line breaks using spaCy, and perform proofreading with `textlint`.

```bash
time singularity run --nv $SIFFILE whisper-cli --model $DIR/$MODEL --flash-attn --language ja --file $INPUT --output-txt --output-file .tmp
time singularity run --nv $SIFFILE python whisper_flow.py .tmp.txt $INPUT.txt
time singularity run --nv $SIFFILE textlint --fix $INPUT.txt
```

Here, `$SIFFILE` is the container file, `$INPUT` is the input WAV file, `$MODEL` is the model file, and `$DIR` is the model directory.
The final output will be saved as `$INPUT.txt`.


**The best balance of speed and accuracy was achieved with `ggml-large-v3-turbo.bin`.
On GH200, transcribing about 1 hour and 30 minutes of audio took only about 2 minutes, and the final output closely matched the original audio.
Kotoba Technologies' Kotoba-Whisper was also tested, but it showed issues in accuracy, such as omissions and misordered segments.**


# Benchmark Test

The execution time of various models was compared using the `whisper-bench` command.
The tests were run on a GH200, with flash attention enabled via `whisper-bench` options.

## Model: ggml-kotoba-whisper-v2.0.bin

```
     load time =   372.64 ms  
     fallbacks =   0 p /   0 h  
      mel time =     0.00 ms  
   sample time =     0.00 ms /     1 runs (     0.00 ms per run)  
   encode time =    20.07 ms /     1 runs (    20.07 ms per run)  
   decode time =   131.92 ms /   256 runs (     0.52 ms per run)  
   batchd time =    57.14 ms /   320 runs (     0.18 ms per run)  
   prompt time =    18.24 ms /  4096 runs (     0.00 ms per run)  
    total time =   227.71 ms  
```

## Model: ggml-large-v3.bin

```
     load time =   726.07 ms  
     fallbacks =   0 p /   0 h  
      mel time =     0.00 ms  
   sample time =     0.00 ms /     1 runs (     0.00 ms per run)  
   encode time =    24.01 ms /     1 runs (    24.01 ms per run)  
   decode time =  1448.48 ms /   256 runs (     5.66 ms per run)  
   batchd time =   508.82 ms /   320 runs (     1.59 ms per run)  
   prompt time =   181.13 ms /  4096 runs (     0.04 ms per run)  
    total time =  2162.92 ms  
```

## Model: ggml-large-v3-q5\_0.bin

```
     load time =   287.27 ms  
     fallbacks =   0 p /   0 h  
      mel time =     0.00 ms  
   sample time =     0.00 ms /     1 runs (     0.00 ms per run)  
   encode time =    32.89 ms /     1 runs (    32.89 ms per run)  
   decode time =  1334.45 ms /   256 runs (     5.21 ms per run)  
   batchd time =   379.44 ms /   320 runs (     1.19 ms per run)  
   prompt time =   244.48 ms /  4096 runs (     0.06 ms per run)  
    total time =  1991.70 ms  
```

## Model: ggml-large-v3-turbo.bin

```
     load time =   403.21 ms  
     fallbacks =   0 p /   0 h  
      mel time =     0.00 ms  
   sample time =     0.00 ms /     1 runs (     0.00 ms per run)  
   encode time =    20.53 ms /     1 runs (    20.53 ms per run)  
   decode time =   220.04 ms /   256 runs (     0.86 ms per run)  
   batchd time =    87.87 ms /   320 runs (     0.27 ms per run)  
   prompt time =    30.50 ms /  4096 runs (     0.01 ms per run)  
    total time =   359.36 ms  
```

## Model: ggml-large-v3-turbo-q5\_0.bin

```
     load time =   215.24 ms  
     fallbacks =   0 p /   0 h  
      mel time =     0.00 ms  
   sample time =     0.00 ms /     1 runs (     0.00 ms per run)  
   encode time =    27.92 ms /     1 runs (    27.92 ms per run)  
   decode time =   196.38 ms /   256 runs (     0.77 ms per run)  
   batchd time =    60.63 ms /   320 runs (     0.19 ms per run)  
   prompt time =    37.56 ms /  4096 runs (     0.01 ms per run)  
    total time =   322.87 ms  
```

## Model: ggml-large-v3-turbo-q8\_0.bin

```
     load time =   275.33 ms  
     fallbacks =   0 p /   0 h  
      mel time =     0.00 ms  
   sample time =     0.00 ms /     1 runs (     0.00 ms per run)  
   encode time =    26.34 ms /     1 runs (    26.34 ms per run)  
   decode time =   221.24 ms /   256 runs (     0.86 ms per run)  
   batchd time =    60.84 ms /   320 runs (     0.19 ms per run)  
   prompt time =    37.65 ms /  4096 runs (     0.01 ms per run)  
    total time =   346.39 ms  
```

