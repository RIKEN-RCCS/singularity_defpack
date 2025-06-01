import os
import json
import argparse
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter

# === AI Model ===
BASE_PATH = "/home/users/hikaru.inoue/.cache/llama.cpp/"
#MODEL_PATH = "mmnga_cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese-gguf_cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese-Q4_0.gguf"
#MODEL_PATH = "mmnga_cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese-gguf_cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese-Q8_0.gguf"
#MODEL_PATH = "unsloth_Mistral-Small-3.1-24B-Instruct-2503-GGUF_Mistral-Small-3.1-24B-Instruct-2503-Q4_0.gguf"
#MODEL_PATH = "unsloth_Qwen3-30B-A3B-128K-GGUF_BF16_Qwen3-30B-A3B-128K-BF16-00001-of-00002.gguf"
#MODEL_PATH = "unsloth_gemma-3-27b-it-GGUF_gemma-3-27b-it-Q4_0.gguf"
#MODEL_PATH = "unsloth_gemma-3-27b-it-GGUF_gemma-3-27b-it-Q8_0.gguf"
MODEL_PATH = "unsloth_gemma-3-27b-it-qat-GGUF_BF16_gemma-3-27b-it-qat-BF16-00001-of-00002.gguf"
#MODEL_PATH = "unsloth_Llama-4-Scout-17B-16E-Instruct-GGUF_Q4_0_Llama-4-Scout-17B-16E-Instruct-Q4_0-00001-of-00002.gguf"
LLM_MODEL_PATH = BASE_PATH + MODEL_PATH

TOP_K = 40
TOP_P = 0.95
TEMPERATURE = 0.1
N_CTX = 131072
MAX_TOKENS = 1024
N_GPU_LAYERS = -1
N_BATCH = 128
REPEAT_PENALTY = 1.4
REPEAT_LAST_N = 256

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# === Template based on JSON ===
'''
SLIDE_TEMPLATE = """
以下は、スライドの発表内容の記録です。JSON形式で提供されます。
スライドの内容点を日本語で10行程度に要約してください。
※ 出力には繰り返しや冗長な表現を含めず、内容に集中してください。
※ 出力には語られていないことや推測を含めないでください。
※ 出力フォーマットは以下の形式に従ってください：

### {slide_title}: 発表内容

------
{json_block}
------
"""
'''

SLIDE_TEMPLATE = """
以下は、スライドの発表内容の記録です。JSON形式で提供されます。
スライドの内容点を日本語で10行程度に要約してください。
※ 出力には繰り返しや冗長な表現を含めず、内容に集中してください。
※ 出力には語られていないことや推測を含めないでください。

------
{json_block}
------
"""

SLIDE_CHUNK_TEMPLATE = """
下記の内容を日本語で要約し、要約のみを出力する。
------
{text}
------
"""

# === LangChain ===
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    top_k=TOP_K,
    top_p=TOP_P,
    n_ctx=N_CTX,
    repeat_penalty=REPEAT_PENALTY,
    repeat_last_n = REPEAT_LAST_N,
    n_gpu_layers=N_GPU_LAYERS,
    n_batch=N_BATCH,
    callback_manager=callback_manager,
    verbose=False,
)

slide_prompt = PromptTemplate(
    input_variables=["slide_title", "json_block"],
    template=SLIDE_TEMPLATE,
)
slide_chain = slide_prompt | llm

slide_chunk_prompt = PromptTemplate(
    input_variables=["text"],
    template=SLIDE_CHUNK_TEMPLATE,
)
slide_chunk_chain = slide_chunk_prompt | llm

# === input JSON and summarize ===
def summarize_from_json(json_path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, target_field: str = "qa") -> str:
    import json
    from textwrap import wrap

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    summaries = []

    for entry in data:
        slide_title = entry.get("slide_title", "")
        slide_texts = [s.get("text", "").strip() for s in entry.get(target_field, []) if s.get("text", "").strip()]

        # 1. split into short chunk
        full_text = "\n".join(slide_texts)
        chunks = splitter.split_text(full_text)

        # 2. summarize chunk (map)
        chunk_summaries = []
        for chunk in chunks:
            response = slide_chunk_chain.invoke({"text":chunk})
            chunk_summaries.append(response.strip())

        # 3. summarize chunks (reduce)
        if chunk_summaries:
            combined_summary = "\n".join(chunk_summaries)
            final_json_block = json.dumps({
                "slide": [{"text": combined_summary}]
            }, ensure_ascii=False, indent=2)

            final_response = slide_chain.invoke({
                "slide_title": slide_title,
                "json_block": final_json_block
            })

            summaries.append(final_response.strip() + "\n")

    return "\n\n".join(summaries), "\n\n".join(chunk_summaries), slide_title

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize slide contents and QA")
    parser.add_argument("target", help="Summarize slide contents or QA")
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("output", help="Output markdown file path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"No input file: {args.input}")

    if args.target not in ("slide", "qa"):
        print("target must be slide or qa")
        sys.exit(1)

    result, chunk_result, result_title = summarize_from_json(args.input, target_field=args.target)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n\n-----\n")
        f.write(f"# {result_title}\n")
        f.write("## Summary\n")
        f.write(result)
        f.write("\n\n-----\n")
        f.write("## Chunk-level Summaries\n")
        f.write(chunk_result)
