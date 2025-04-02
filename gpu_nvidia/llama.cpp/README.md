# **ChatGPT-like service: Open WebUI + llama.cpp**

Implement a ChatGPT-like service in a local environment using OpenWebUI fot UI and llama.cpp for inference with the large language model.
1. Install llama.cpp and related modules in a Singularity container, some large language models in a local environment, then start llama-server.
2. Start the OpenWebUI server in a local environment.
3. Launch a web browser in the local environment and connect to llama-server.

## **1. Start llama-server**

The necessary packages are installed in a Singularity container created with llama.cpp.def, so the container image should be pre-built with the name `llama.cpp.sif`.
Start the llama-server using the Singularity container.
- the number of layers to store in VRAM is 99
- port to listen is 1000
- enable Flash Attention
- LLM is fine-tuned model in Japanese by CyberAgent on deepseek-ai/DeepSeek-R1-Distill-Qwen-32B, and the 8-bit quantized model converted to the GGUF format by mmnga.
- Specify the path to llama.cpp.sif.

```bash
#!/bin/sh

cat << EOF > .server.sh
llama-server \
  --n-gpu-layers 99 \
  --port 10000 \
  --flash-attn  \
  -hf mmnga/cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese-gguf:Q8_0
EOF

SIFFILE=/path/to/llama.cpp.sif
export SINGULARITY_BIND=/lvs0
singularity run --nv $SIFFILE sh ./.server.sh
```

## **2. Start ollama server**

Start the ollama server using the Singularity container.
- Specify the path to llama.cpp.sif.

```bash
#!/bin/sh

cat << EOF > .ollama.sh
cd /opt/ollama && go run . serve
EOF

SIFFILE=/path/to/llama.cpp.sif
export SINGULARITY_BIND=/lvs0
singularity run --nv $SIFFILE sh ./.ollama.sh
```

## **3. Start Tika server**

Start the Tika server using the Singularity container.
- Specify the path to llama.cpp.sif.

```bash
#!/bin/sh

cat << EOF > .tika.sh
cd /opt/tika && java -jar tika-server/tika-server-standard/target/tika-server-standard-4.0.0-SNAPSHOT.jar
EOF

SIFFILE=/path/to/llama.cpp.sif
export SINGULARITY_BIND=/lvs0
singularity run --nv $SIFFILE sh ./.tika.sh
```

## **4. Start OpenWebUI server**

Start the OpenWebUI server using the Singularity container.
- Python version is 3.10 in this container
- Specify the path to llama.cpp.sif.

```bash
#!/bin/sh

cat << EOF > .openwebui.sh
DATA_DIR=~/.open-webui /opt/uv/uvx --python 3.10 open-webui@latest serve
EOF

SIFFILE=/path/to/llama.cpp.sif
export SINGULARITY_BIND=/lvs0
singularity run --nv $SIFFILE sh ./.openwebui.sh
```

## **5. Generate responses with web browser**

Connect llama-server to OpenWebUI.
- Go to Admin Settings in Open WebUI.
- Navigate to Connections > OpenAI Connections.
- Add the following details for the new connection:
  - URL: http://127.0.0.1:10000/v1
  - API Key: none

  <img src="./images/openwebui1.jpg" width="640">

You can now use Open WebUI’s chat interface to interact with the 

  <img src="./images/openwebui2.jpg" width="640">

# **Benchmark test**

Measure performance using llama-bench provided by llama.cpp.
The models used for benchmarking are DeepSeek-R1-Distill-Qwen-14B and DeepSeek-R1-Distill-Qwen-32B.
The benchmarking environment is **NVIDIA GH200**.

## **DeepSeek-R1-Distill-Qwen-14B**

### Condition
   - Model : [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/mmnga/DeepSeek-R1-Distill-Qwen-14B-gguf)
   - Quantization : Q8_0, Q6_K, Q5_K_S, Q5_0, Q5_K_M, IQ4_XS, Q4_K_S, IQ4_NL, Q4_0, Q4_K_M
   - Batch-size : 128, 256, 512, 1024, 2048, 4096
   - Flash attention : ON
### Result

  <img src="./images/14b.png" width="640">

## **DeepSeek-R1-Distill-Qwen-32B**

### Condition
   - Model : [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/mmnga/DeepSeek-R1-Distill-Qwen-32B-gguf)
   - Quantization : Q8_0, Q6_K, Q5_K_S, Q5_0, Q5_K_M, IQ4_XS, Q4_K_S, IQ4_NL, Q4_0, Q4_K_M
   - Batch-size : 128, 256, 512, 1024, 2048, 4096
   - Flash attention : ON
### Result

  <img src="./images/32b.png" width="640">
