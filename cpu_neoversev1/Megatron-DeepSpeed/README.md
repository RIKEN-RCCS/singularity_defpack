# Megatron-DeepSpeed Container

Creating a new Megatron-DeepSpeed container based on the PyTorch version 2.5.0 container.

## What is Megatron-DeepSpeed?

### Overview
Megatron-DeepSpeed is a framework optimized for training ultra-large transformer models. It integrates the model parallelism technology of Megatron-LM with DeepSpeed's distributed optimization features, enabling efficient training of models with billions to trillions of parameters.

### Features
#### 1. Megatron-LM  
A large-scale transformer model training framework developed by NVIDIA.
- Supports model parallelism (Tensor Parallelism, Pipeline Parallelism), enabling efficient training of models with hundreds of billions to trillions of parameters.
- Utilizes high-speed custom CUDA kernels.

#### 2. DeepSpeed  
A large-scale distributed learning framework developed by Microsoft.
- Reduces memory usage with ZeRO (Zero Redundancy Optimizer), allowing larger models to be trained on a single GPU.
- Provides features such as CPU offloading, communication optimization, and mixed precision training (FP16, BF16).

### Key Benefits of Megatron-DeepSpeed
#### 1. Integration of Model Parallelism (MP) and Data Parallelism (DP)
  Combines Megatron's Tensor Parallelism & Pipeline Parallelism with DeepSpeed's ZeRO for efficient training of ultra-large models.

#### 2. Improved Memory Efficiency  
DeepSpeed's ZeRO-3 distributes memory usage for parameters, gradients, and optimizers.

#### 3. Large-Scale Cluster Support  
Optimized for multi-node distributed training using InfiniBand / NVLink.

#### 4. Mixed Precision & Low-Precision Training  
Leverages FP16, BF16, and 8-bit quantization (DeepSpeed-AIO) to reduce computation costs.

#### 5. Dynamic Batch Size Adjustment  
Supports ZeRO-infinity, which dynamically adjusts the batch size based on available memory.

### Use Cases
- Training large language models such as GPT-3, LLaMA, and BLOOM.
- Distributed training of ultra-large transformer models.
- Maximizing computational resource utilization for AI model development.

## Creating the Container

### Base Model
The container will use a locally available image created with the PyTorch version 2.5.0 definition file published in this repository.

### Installing Python Libraries
Using the `pip` command, install necessary libraries for large language model training and inference, including:
- `transformers`
- `deepspeed`
- `trl` (reinforcement learning library)
- `peft` (fine-tuning library)

### Installing Megatron-DeepSpeed
Download from Microsoft's repository and build the helper libraries.

## Usage Examples

### Fine-tuning

```python
import torch
import datasets
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import cProfile, pstats

def prompt_format(example, tokenizer):
    output = []

    if isinstance(example["output"], str):
        example["output"] = [example["output"]]
    elif not isinstance(example["output"], list):
        raise ValueError(f"example['output'] should be a list or string, but got {type(example['output'])}")

    if isinstance(example["instruction"], str):
        example["instruction"] = [example["instruction"]]

    for i in range(len(example['instruction'])):
        text = (
            "以下はタスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n"
            f"### 指示:\n{example['instruction'][i]}\n"
            f"### 応答:\n{example['output'][i]}\n"
            + tokenizer.eos_token
        )
        output.append(text)
    return output

def finetuning(model_name):
    print(f"Model {model_name}")
    save_path = "./trained_model/"+model_name
    checkpoint_path = "./checkpoint/"+model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float32)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    datadic = datasets.load_dataset("kunishou/databricks-dolly-15k-ja")
    dataset = datadic['train']
    dataset = dataset.filter(lambda data: data['input'] == "")

    response_template = "### 応答:\n"

    print(dataset[0])

    trainer = trl.SFTTrainer(
        model=model,
        data_collator=trl.DataCollatorForCompletionOnlyLM(
            response_template,
            tokenizer=tokenizer
        ),
        args=TrainingArguments(
            output_dir=checkpoint_path,
            num_train_epochs=1,
            per_device_eval_batch_size=1,
            learning_rate=5e-5,
        ),
        train_dataset=dataset,
        formatting_func=lambda example: prompt_format(example, tokenizer)
    )

    trainer.train()
    trainer.save_model(save_path)

def main():
    profiler = cProfile.Profile()
    profiler.enable()

    # GPTNeoX 160M
    model_name = "cyberagent/open-calm-small"
    finetuning(model_name)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime').print_stats(50)

if __name__ == "__main__":
    main()
```

### Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import cProfile, pstats

def infer(model_name, base_model_name):

    print(f"Model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_text = "スマホが人間に悪い理由"

    prompt = (
        "以下はタスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n"
        f"### 指示:\n{input_text}\n"
        "### 応答:\n"
    )

    for j in range(2):
        if j == 0:
            inputs = tokenizer(input_text, return_tensors="pt")
        else:
            inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
                **inputs,
                max_length=100,
                do_sample=True,
                top_k=50,
                temperature=0.7,
                num_return_sequences=3
        )

        for i in range(len(outputs)):
            print("%d:" % (i+1))
            output = tokenizer.decode(outputs[i], skip_special_tokens=True)
            print(output)
        print('-----')

def main():
    profiler = cProfile.Profile()
    profiler.enable()

    # GPTNeoX
    base_model_name = "cyberagent/open-calm-small"
    infer(base_model_name, base_model_name)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime').print_stats(50)

if __name__ == "__main__":
    main()
```


