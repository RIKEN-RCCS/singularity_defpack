# **Megatron-DeepSpeed Container**

Creating a new Megatron-DeepSpeed container based on the PyTorch version 2.5.0 container.

## **What is Megatron-DeepSpeed?**

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
#### 1. **Integration of Model Parallelism (MP) and Data Parallelism (DP)**
Combines Megatron's Tensor Parallelism & Pipeline Parallelism with DeepSpeed's ZeRO for efficient training of ultra-large models.

#### 2. **Improved Memory Efficiency**
DeepSpeed's ZeRO-3 distributes memory usage for parameters, gradients, and optimizers.

#### 3. **Large-Scale Cluster Support**
Optimized for multi-node distributed training using InfiniBand / NVLink.

#### 4. **Mixed Precision & Low-Precision Training**
Leverages FP16, BF16, and 8-bit quantization (DeepSpeed-AIO) to reduce computation costs.

#### 5. **Dynamic Batch Size Adjustment**
Supports ZeRO-infinity, which dynamically adjusts the batch size based on available memory.

----

## **Building Megatron-DeepSpeed container**

### definition file

The container will use a locally available image created with the PyTorch version 2.5.0 definition file published in this repository.

```
Bootstrap: localimage
From: pytorch_2.5.0.sif
```

Create a Python virtual environment named `llm` and install the full set of PyTorch, TorchVision, and TorchAudio packages from pytorch_2.5.0.sif.

```
  # Install PyTorch, TorchVision and TorchAudio
  cd /opt
  python3 -m venv llm
  . /opt/llm/bin/activate
  python3 -m pip install -r /opt/requirements.txt
```

Install the Python packages for LLMs using Megatron-DeepSpeed.

```
  python3 -m pip install accelerate transformers deepspeed bitsandbytes datasets evaluate hjson huggingface-hub sentencepiece tokenizers wandb ninja packaging pybind11 six trl optimum peft regex tensorboard mpi4py
```

Clone the Megatron-DeepSpeed from Git, and build helper library.

```
  # Build Megatron DeepSpeed
  cd /opt
  git clone https://github.com/microsoft/Megatron-DeepSpeed
  cd Megatron-DeepSpeed

  cd /opt/Megatron-DeepSpeed/megatron/data/
  make
```

### Package List

The container created on May 11, 2025, for Graviton3E contains the following packages.
```
Package                 Version
----------------------- ------------------
absl-py                 2.1.0
accelerate              1.4.0
aiohappyeyeballs        2.5.0
aiohttp                 3.11.13
aiosignal               1.3.2
annotated-types         0.7.0
astunparse              1.6.3
attrs                   25.1.0
bitsandbytes            0.42.0
certifi                 2025.1.31
charset-normalizer      3.4.1
click                   8.1.8
datasets                3.3.2
deepspeed               0.16.4
dill                    0.3.8
docker-pycreds          0.4.0
einops                  0.8.1
evaluate                0.4.3
expecttest              0.3.0
filelock                3.17.0
frozenlist              1.5.0
fsspec                  2024.12.0
gitdb                   4.0.12
GitPython               3.1.44
grpcio                  1.71.0
hjson                   3.1.0
huggingface-hub         0.29.2
hypothesis              6.127.9
idna                    3.10
Jinja2                  3.1.6
lintrunner              0.12.7
Markdown                3.7
markdown-it-py          3.0.0
MarkupSafe              3.0.2
mdurl                   0.1.2
mpi4py                  4.0.3
mpmath                  1.3.0
msgpack                 1.1.0
multidict               6.1.0
multiprocess            0.70.16
networkx                3.4.2
ninja                   1.11.1.3
numpy                   2.2.3
optimum                 1.24.0
optree                  0.14.1
packaging               24.2
pandas                  2.2.3
peft                    0.14.0
pillow                  11.1.0
pip                     24.0
pkgconfig               1.5.5
platformdirs            4.3.6
propcache               0.3.0
protobuf                5.29.3
psutil                  7.0.0
py-cpuinfo              9.0.0
pyarrow                 19.0.1
pybind11                2.13.6
pydantic                2.10.6
pydantic_core           2.27.2
Pygments                2.19.1
python-dateutil         2.9.0.post0
pytz                    2025.1
PyYAML                  6.0.2
regex                   2024.11.6
requests                2.32.3
rich                    13.9.4
safetensors             0.5.3
scipy                   1.15.2
SCons                   4.9.0
sentencepiece           0.2.0
sentry-sdk              2.22.0
setproctitle            1.3.5
setuptools              68.1.2
six                     1.17.0
smmap                   5.0.2
sortedcontainers        2.4.0
sympy                   1.13.1
tensorboard             2.19.0
tensorboard-data-server 0.7.2
tokenizers              0.21.0
torch                   2.5.0a0+git32f585d
torchaudio              2.6.0a0+c670ad8
torchvision             0.22.0a0+124dfa4
tqdm                    4.67.1
transformers            4.49.0
trl                     0.15.2
types-dataclasses       0.6.6
typing_extensions       4.12.2
tzdata                  2025.1
urllib3                 2.3.0
wandb                   0.19.8
Werkzeug                3.1.3
wheel                   0.42.0
xxhash                  3.5.0
yarl                    1.18.3
```

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


