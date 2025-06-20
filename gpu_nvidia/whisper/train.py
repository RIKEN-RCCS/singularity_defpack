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
